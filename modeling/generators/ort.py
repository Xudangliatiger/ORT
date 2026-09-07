""" Adapted from:
    https://alitok/modeling/rar.py
    https://github.com/bytedance/1d-tokenizer/blob/main/modeling/rar.py
    https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
"""

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from modeling.modules import BaseModel
from functools import partial
import random
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from utils.registry import register_model


#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache])  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(start, grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(start, start + grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
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
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)  # (1, seq_len, 1, head_dim//2, 2)
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
        q, k = self.q_norm(q), self.k_norm(k)  # [4, 16, 273, 48]
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

@register_model("ort")
class ORTModel(BaseModel):
    def __init__(self, config, logger=None):
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
        norm_layer = partial(RMSNorm)

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
        # self.norm_ = RMSNorm(embed_dim, eps=1e-5)
        # self.output_ = nn.Linear(embed_dim,
        #                         target_codebook_size, bias=True)
        self.condition_num_classes = condition_num_classes
        self.image_seq_len = image_seq_len
        self.target_codebook_size = target_codebook_size
        self.none_condition_id = self.condition_num_classes + self.target_codebook_size + 1

        attn_mask = build_causal_mask(self.image_seq_len + 1024)  # include condition
        self.register_buffer('attn_mask', attn_mask, persistent=False)

        self.use_checkpoint = config.model.generator.get("use_checkpoint", False)
        self.tok_dropout = nn.Dropout(config.model.generator.tok_dropout)

        # 2d rotary pos embedding
        freqs_cis_img = precompute_freqs_cis_2d(17, 16, embed_dim // num_heads, 10000, 0)
        freqs_cis_globle = precompute_freqs_cis(17, embed_dim // num_heads, 10000, 1)
        freqs_cis = torch.cat([freqs_cis_globle, freqs_cis_img], dim=0)

        # ✅ 用 register_buffer 直接注册，不要先 self.freqs_cis = ...
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.pos_embed = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, image_seq_len + 1024, embed_dim)), 0., 0.02)

        self.target_aware_pos_embed = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, image_seq_len + 1024, embed_dim)), 0., 0.02)

        self.initialize_weights()

        self.random_ratio = 0.0

        self.alpha_weight_random = config.model.loss.alpha_weight_random
        self.beta_weight_random = config.model.loss.beta_weight_random
        self.alpha_weight_raster = config.model.loss.alpha_weight_raster_start
        self.beta_weight_raster = config.model.loss.beta_weight_raster_start

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

    def get_rar_random_ratio(self, config, cur_step):
        randomness_anneal_start = config.model.generator.randomness_anneal_start
        randomness_anneal_end = config.model.generator.randomness_anneal_end
        if cur_step < randomness_anneal_start:
            return 1.0
        elif cur_step >= randomness_anneal_end:
            return 0
        else:
            return 1.0 - (cur_step - randomness_anneal_start) / (randomness_anneal_end - randomness_anneal_start)

    def get_rar_alpha_beta_weight(self, config, cur_step):
        weight_anneal_start = config.model.loss.weight_anneal_start
        weight_anneal_end = config.model.loss.weight_anneal_end
        max_train_steps = config.training.max_train_steps
        alpha_weight_1 = config.model.loss.alpha_weight_raster_start
        beta_weight_1 = config.model.loss.beta_weight_raster_start
        alpha_weight_2 = config.model.loss.alpha_weight_raster_end
        beta_weight_2 = config.model.loss.beta_weight_raster_end

        if cur_step < weight_anneal_start:
            return alpha_weight_1, beta_weight_1
        elif cur_step >= weight_anneal_end:
            return alpha_weight_2, beta_weight_2
        else:
            ans1 = alpha_weight_1 - (alpha_weight_1 - alpha_weight_2) * (cur_step - weight_anneal_start) / (
                    weight_anneal_end - weight_anneal_start)
            ans2 = beta_weight_1 - (beta_weight_1 - beta_weight_2) * (cur_step - weight_anneal_start) / (
                    weight_anneal_end - weight_anneal_start)
            return ans1, ans2

    def set_alpha_beta_weight(self, new_alpha_weight, new_beta_weight):
        self.alpha_weight_raster = new_alpha_weight
        self.beta_weight_raster = new_beta_weight

    def enable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = True
            block.attn.reset_kv_cache()

    def disable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = False
            block.attn.reset_kv_cache()


    @torch.compiler.disable
    def sample_orders(self, x):
        batch_size = x.shape[0]
        shuffled_orders = []
        shuffled_weight = []

        for _ in range(batch_size):
            if torch.rand(()) < self.random_ratio:
                # random order
                shuffled_orders.append(torch.randperm(self.image_seq_len, device=x.device))
                shuffled_weight.append(
                    torch.linspace(self.alpha_weight_random, self.beta_weight_random, steps=self.image_seq_len,
                                   device=x.device))
            else:
                # raster order
                shuffled_orders.append(torch.arange(self.image_seq_len, device=x.device))
                shuffled_weight.append(
                    torch.linspace(self.alpha_weight_raster, self.beta_weight_raster, steps=self.image_seq_len,
                                   device=x.device))

        shuffled_orders = torch.stack(shuffled_orders)
        shuffled_weight = torch.stack(shuffled_weight)
        return shuffled_orders.to(x.device), shuffled_weight.to(x.device)

    def set_random_ratio(self, new_ratio):
        self.random_ratio = new_ratio

    def get_raster_orders(self, x):
        batch_size = x.shape[0]
        shuffled_orders = torch.stack([torch.arange(self.image_seq_len, device=x.device) for _ in range(batch_size)])
        return shuffled_orders

    @torch.compiler.disable
    def shuffle(self, x, orders):
        batch_size, seq_len = x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        shuffled_x = x[batch_indices, orders]
        return shuffled_x

    def unshuffle(self, shuffled_x, orders):
        # Unshuffle the tensor based on the original orders
        batch_size, seq_len = shuffled_x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        unshuffled_x = torch.zeros_like(shuffled_x)
        unshuffled_x[batch_indices, orders] = shuffled_x
        return unshuffled_x

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
        orders, weight = self.sample_orders(input_ids)
        return self.forward_fn(input_ids, condition, return_labels, orders, weight)

    def forward_fn(self, input_ids, condition,
                   return_labels=False,
                   orders=None,
                   weight=None,
                   is_sampling=False):

        labels = input_ids.clone()  # [batch ,exit_patches]
        # prepend condition token
        if orders is None:
            orders = self.get_raster_orders(input_ids)

        input_ids = torch.cat([condition.view(condition.shape[0], -1),  # [batch]
                               input_ids.view(input_ids.shape[0], -1), ], dim=1)  # [batch ,exit_patches]
        # [batch ,exit_patches+1]
        x = self.embeddings(input_ids)  # [batch, exit_patches+1, 1024]
        x = self.tok_dropout(x)
        # condition_token = x[:, 0]

        # prepare positional embeddings.
        # shuffle pos embed
        pos_embed = self.pos_embed.repeat(input_ids.shape[0], 1, 1)
        # condition, the permute does not impact these prefix tokens.
        prefix = 1
        pos_embed_prefix = pos_embed[:, :prefix]
        pos_embed_postfix = self.shuffle(pos_embed[:, prefix:prefix + self.image_seq_len], orders)

        # prepare target-aware positional embeddings.
        target_aware_pos_embed = self.target_aware_pos_embed.repeat(input_ids.shape[0], 1, 1)
        # target_aware_pos_embed_prefix = target_aware_pos_embed[:, :prefix]
        target_aware_pos_embed_postfix = self.shuffle(target_aware_pos_embed[:, prefix:prefix + self.image_seq_len],
                                                      orders)


        if not is_sampling:
            # shuffle labels
            labels = self.shuffle(labels, orders)
            # randomized permutation: during training, we need to shuffle the input_ids's order but not for sampling
            x = torch.cat([x[:, :1], self.shuffle(x[:, 1:], orders)], dim=1)

        # add original pos embed
        x = x + torch.cat([pos_embed_prefix, pos_embed_postfix], dim=1)[:, :x.shape[1]]

        # add target-aware pos embed
        target_aware_pos_embed = torch.cat(
            [torch.zeros_like(x[:, :prefix - 1]), target_aware_pos_embed_postfix, torch.zeros_like(x[:, -1:])],
            dim=1
        )
        x = x + target_aware_pos_embed[:, :x.shape[1]]

        # self.freqs_cis = self.freqs_cis.to(x.device)
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
            # if idx == 7:
            #     x_ = self.norm_(x)
            #     x_ = self.output_(x_)

        x = self.norm(x)
        x = self.output(x)


        if return_labels:
            return {'x':x, 'x_': None}, labels, weight


        return {'x':x, 'x_': None}

    def proj(self, v, u, eps=1e-8):
        # v,u: [B, 1, D] (or [B, L, D])
        vu = (v * u).sum(dim=(-1, -2), keepdim=True)  # [B,1,1]
        uu = (u * u).sum(dim=(-1, -2), keepdim=True) + eps  # [B,1,1]
        return (vu / uu) * u  # [B,1,D]

    def residual(self, v, u, eps=1e-8):
        return v - self.proj(v, u, eps)

    import torch

    def guidance_norm(
            self,
            v: torch.Tensor,
            ref: torch.Tensor | None = None,
            mode: str = "base",  # "base" or "self"
            eps: float = 1e-8,
            clip: float | None = None,  # e.g., 10.0 to prevent extreme scaling
    ) -> torch.Tensor:
        """
        Normalize a guidance vector v (e.g., [B, 1, D] or [B, D]).

        - mode="self":     v_hat = v / (||v|| + eps)
        - mode="base":     v_hat = (v / (||v|| + eps)) * ||ref||   (ref must be provided)

        clip: optional clamp on scaling factor (||ref|| / ||v||) or on ||ref|| depending on mode.
        """
        if v is None:
            return v
        if not torch.is_tensor(v):
            return v  # allow v=0 or other sentinels

        # Compute norm over last dim, keep dims for broadcasting
        v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)

        if mode == "self":
            out = v / v_norm
            return out

        if mode == "base":
            if ref is None:
                raise ValueError('guidance_norm(mode="base") requires ref.')
            ref_norm = torch.linalg.vector_norm(ref, dim=-1, keepdim=True).clamp_min(eps)

            # Scale v to have magnitude similar to ref
            scale = ref_norm / v_norm

            if clip is not None:
                scale = scale.clamp(max=clip)

            out = v * scale
            return out

        raise ValueError(f"Unknown mode: {mode}. Use 'self' or 'base'.")

    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale,
                 randomize_temperature,
                 guidance_scale_pow,
                 kv_cache=True,
                 internal_guidance_scale=1,
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
                ((step / self.image_seq_len) ** scale_pow) * torch.pi)) * 1 / 2
            cfg_scale = (guidance_scale - 1) * scale_step
            ig_scale = (internal_guidance_scale - 1) * scale_step

            if guidance_scale != 0:
                logits = self.forward_fn(
                    torch.cat([ids, ids], dim=0),
                    torch.cat([condition, self.get_none_condition(condition)], dim=0), is_sampling=True)
                logits, logits_mid = logits['x'], logits['x_']
                if logits_mid is not None:
                    internal_guidance = logits[:num_samples] - logits_mid[:num_samples]
                    # logits[:num_samples] = logits[:num_samples] + internal_guidance * ig_scale
                else:
                    internal_guidance = 0
                # print((logits - logits_mid).shape)
                cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
                classifier_free_guidance = cond_logits - uncond_logits
                # diff = internal_guidance - classifier_free_guidance   # [B,1,D]
                # classifier_free_guidance = self.residual(classifier_free_guidance, internal_guidance, eps=1e-8)
                #
                # internal_guidance = self.guidance_norm(internal_guidance, cond_logits)
                # classifier_free_guidance = self.guidance_norm(classifier_free_guidance, cond_logits)

                logits = cond_logits + classifier_free_guidance * cfg_scale
            else:
                logits = self.forward_fn(
                    ids, condition, is_sampling=True
                )
                logits, logits_mid = logits['x'], logits['x_']

            # keep the logit of last token
            logits = logits[:, -1]
            logits = logits / randomize_temperature
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, sampled), dim=-1)

        self.disable_kv_cache()
        return ids