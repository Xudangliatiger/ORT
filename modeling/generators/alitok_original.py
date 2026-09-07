""" Adapted from:
    https://github.com/bytedance/1d-tokenizer/blob/main/modeling/rar.py
    https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
"""

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.modules import BaseModel
from functools import partial


#################################################################################
#                      Rotary Positional Embedding Functions                    #
################################################################################# 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 

def precompute_freqs_cis_2d(start, grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(start, start + grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1)  
    cache = cache_grid.flatten(0, 1)
    if cls_token_num > 0:
        cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) 
    else:
        cond_cache = cache
    return cond_cache 

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2) 
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2) 
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

# util function
def build_causal_mask(seq_length):
    mask = torch.empty(seq_length, seq_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FeedForward(nn.Module):
    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3) 
        hidden_dim = find_multiple(hidden_dim, 256)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(proj_drop)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

    
# attention layer with KV cache supported
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, freqs_cis, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k) # [4, 16, 273, 48] 
        q = rearrange(q, 'b h seq dim -> b seq h dim')
        k = rearrange(k, 'b h seq dim -> b seq h dim')
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        
        q = rearrange(q, 'b seq h dim -> b h seq dim')
        k = rearrange(k, 'b seq h dim -> b h seq dim')
        
        if self.kv_cache:
            if self.k_cache is None and self.v_cache is None:
                k_cache = k
                v_cache = v
            else:
                assert N in [1, 2], f"x.shape {x.shape}"
                k_cache = torch.cat([self.k_cache, k], dim=-2)
                v_cache = torch.cat([self.v_cache, v], dim=-2)

            self.k_cache = k_cache
            self.v_cache = v_cache

            k = k_cache
            v = v_cache

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x 

# basic transformer block
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0., 
            norm_layer: nn.Module = nn.LayerNorm, 
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = RMSNorm(dim)  
        self.mlp = FeedForward(dim, proj_drop) 

    def forward(self, x: torch.Tensor, freqs_cis, attn_mask=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ARModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        # parse the configs
        embed_dim = config.model.generator.hidden_size
        depth = config.model.generator.num_hidden_layers
        num_heads = config.model.generator.num_attention_heads 
        mlp_ratio = 4

        image_seq_len = config.model.generator.image_seq_len
        target_codebook_size = config.model.vq_model.codebook_size
        condition_num_classes = config.model.generator.condition_num_classes
        norm_layer=partial(RMSNorm)

        dropout_rate = config.model.generator.dropout
        attn_dropout_rate = config.model.generator.attn_drop
    
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_norm=True,
                proj_drop=dropout_rate,
                attn_drop=attn_dropout_rate,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.embeddings = nn.Embedding(
            target_codebook_size + 1 + condition_num_classes + 1, embed_dim)  
         
        self.norm = RMSNorm(embed_dim, eps=1e-5)
        self.output = nn.Linear(embed_dim,
                                 target_codebook_size, bias=True)
        self.condition_num_classes = condition_num_classes
        self.image_seq_len = image_seq_len
        self.target_codebook_size = target_codebook_size
        self.none_condition_id = self.condition_num_classes + self.target_codebook_size + 1 

        attn_mask = build_causal_mask(self.image_seq_len + 1024) # include condition
        self.register_buffer('attn_mask', attn_mask, persistent=False)

        self.use_checkpoint = config.model.generator.get("use_checkpoint", False)
        self.tok_dropout = nn.Dropout(config.model.generator.tok_dropout)
        
        # 2d rotary pos embedding
        self.freqs_cis_img = precompute_freqs_cis_2d(17, 16, embed_dim // num_heads, 10000, 0)
        
        self.freqs_cis_globle = precompute_freqs_cis(17, embed_dim // num_heads, 10000, 1)
        
        self.freqs_cis = torch.cat([self.freqs_cis_globle, self.freqs_cis_img], 0)
        
        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights) 
        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def enable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = True
            block.attn.reset_kv_cache()

    def disable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = False
            block.attn.reset_kv_cache() 

    def preprocess_condition(self, condition, cond_drop_prob=0.0):
        # Set class condition to None condition
        drop_label_mask = torch.rand_like(condition, dtype=torch.float) < cond_drop_prob
        condition = condition + self.target_codebook_size + 1  # [0, 999] -> [codebook_size + 1, codebook_size + 999]
        condition[drop_label_mask] = self.none_condition_id
        return condition

    def get_none_condition(self,
                           condition
                           ):
        return torch.full_like(condition, self.none_condition_id)
    
    def forward(self, input_ids, condition, return_labels=False):
         
        return self.forward_fn(input_ids, condition, return_labels)

    def forward_fn(self, input_ids, condition,
                   return_labels=False, 
                   is_sampling=False): 
         
        labels = input_ids.clone() # [batch ,exit_patches]
        # prepend condition token
        
        input_ids = torch.cat([condition.view(condition.shape[0], -1), # [batch]
                               input_ids.view(input_ids.shape[0], -1),], dim=1) # [batch ,exit_patches]
                              # [batch ,exit_patches+1]
        x = self.embeddings(input_ids) # [batch, exit_patches+1, 1024]  
        x = self.tok_dropout(x)
        self.freqs_cis = self.freqs_cis.to(x.device) 
        freqs_cis = self.freqs_cis[:x.shape[1]]
        
        # causal attention masking
        attn_mask = self.attn_mask[:x.shape[1], :x.shape[1]]  # [exit_patches+2, exit_patches+2]  
        # seperate condition token for each step, at generation, we start from 1 to seq len 

        if self.blocks[0].attn.kv_cache:
            if self.blocks[0].attn.k_cache is not None and self.blocks[0].attn.v_cache is not None:
                # only need to process the last token 
                attn_mask = None 
                freqs_cis = freqs_cis[-1:]
                x = x[:, -1:]

        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                        blk.forward, x, freqs_cis, attn_mask, use_reentrant=False)
            else:
                x = blk(x, freqs_cis, attn_mask=attn_mask)  

        x = self.norm(x)
        x = self.output(x)

        if return_labels:
            return x, labels
        return x
    
    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale,
                 randomize_temperature,
                 guidance_scale_pow,
                 kv_cache=True,
                 **kwargs): 
        condition = self.preprocess_condition(
            condition, cond_drop_prob=0.0)
        device = condition.device
        num_samples = condition.shape[0]
        ids = torch.full((num_samples, 0), -1, device=device)  # ids是已采样的token在codebook里的值
        cfg_scale = 0.

        if kv_cache:
            self.enable_kv_cache()
 
        for step in range(self.image_seq_len): 
            # ref: https://github.com/sail-sg/MDT/blob/441d6a1d49781dbca22b708bbd9ed81e9e3bdee4/masked_diffusion/models.py#L513C13-L513C23
            scale_pow = torch.ones((1), device=device) * guidance_scale_pow
            scale_step = (1 - torch.cos(
                ((step / self.image_seq_len) ** scale_pow) * torch.pi)) * 1/2
            cfg_scale = (guidance_scale - 1) * scale_step + 1 
            if guidance_scale != 0: 
                logits = self.forward_fn(
                    torch.cat([ids, ids], dim=0),
                    torch.cat([condition, self.get_none_condition(condition)], dim=0), is_sampling=True)
                cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits = self.forward_fn(
                    ids, condition, is_sampling=True
                )

            # keep the logit of last token
            logits = logits[:, -1]
            logits = logits / randomize_temperature
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, sampled), dim = -1) 
            
        self.disable_kv_cache()
        return ids

