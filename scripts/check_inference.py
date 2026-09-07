"""Bounded GPU forward/backward, checkpoint loading and token/image generation."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from PIL import Image
from modeling.generators import ORTModel
from modeling.losses import ORTARLoss
from modeling.tokenizers import AliTok

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--tokenizer-weights')
p.add_argument('--output', required=True)
p.add_argument('--seed', type=int, default=2)
p.add_argument('--backward-check', action='store_true', help='Run one synthetic optimizer update after inference')
a = p.parse_args()
c = OmegaConf.load(a.config)
torch.manual_seed(a.seed)
m = ORTModel(c)
s = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
s = s.get('model', s)
s = {k.removeprefix('_orig_mod.'): v for k,v in s.items()}
m.load_state_dict(s, strict=True)
m.cuda().eval()
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    tokens = m.generate(torch.tensor([1], device='cuda'), guidance_scale=1.0, guidance_scale_pow=1.0, randomize_temperature=1.0)
assert tokens.shape == (1,273)
assert tokens.min() >= 0 and tokens.max() < c.model.vq_model.codebook_size
r = Path(a.output);r.mkdir(parents=True, exist_ok=True)
torch.save(tokens.cpu(), r / 'tokens.pt')
print('PASS strict checkpoint load and 273-token autoregressive inference', flush=True)
if a.tokenizer_weights:
    d = AliTok()
    d.load_state_dict(torch.load(a.tokenizer_weights, map_location='cpu', weights_only=True), strict=True)
    d.cuda().eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        pixels = d.decode_tokens(tokens)
    assert pixels.shape == (1,3,256,256) and torch.isfinite(pixels).all()
    image = (pixels[0].clamp(0,1)*255).permute(1,2,0).to('cpu',torch.uint8).numpy()
    Image.fromarray(image).save(r / 'sample.png')
    print('PASS tokenizer load and 256x256 image decode', flush=True)

if a.backward_check:
    tokens = tokens.clone()
    m.train()
    m.set_random_ratio(1.0)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-5)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits, labels, weights = m(tokens, m.preprocess_condition(torch.tensor([1], device='cuda')), return_labels=True)
        loss, _ = ORTARLoss(c)(logits, labels, weights)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [v.grad for v in m.parameters() if v.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    optimizer.step()
    print(f'PASS full model backward and optimizer update: loss={loss.item():.6f}', flush=True)
