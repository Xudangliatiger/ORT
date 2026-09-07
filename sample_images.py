"""Decode balanced class samples to the uint8 NHWC archive consumed by ADM evaluation.

Requires matching tokenizer weights, supplied separately.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch
from modeling.factory import build_model, build_tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer-weights', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, required=True, help='Sampling seed, independent of training seed')
    p.add_argument('--count', type=int, default=50000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    c = OmegaConf.load(a.config)
    classes = int(c.model.generator.condition_num_classes)
    if a.count <= 0 or a.count % classes or a.batch_size <= 0:
        p.error('count must be a positive multiple of class count; batch size must be positive')
    output = Path(a.output)
    if output.exists():
        raise ValueError('Output exists; choose a new path')
    decoder = build_tokenizer(c, a.tokenizer_weights)
    model = build_model(c)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    state = state.get('model', state)
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(a.device).eval()
    decoder.to(a.device).eval()
    torch.manual_seed(a.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.with_suffix('.images.npy')
    labels = np.arange(a.count, dtype=np.int64) % classes
    images = None
    with torch.inference_mode():
        for start in range(0, a.count, a.batch_size):
            tokens = model.generate(condition=torch.tensor(labels[start:start+a.batch_size], device=a.device),
                guidance_scale=c.model.generator.guidance_scale,
                guidance_scale_pow=c.model.generator.guidance_scale_pow,
                randomize_temperature=c.model.generator.randomize_temperature,
                **({'internal_guidance_scale': c.model.generator.get("internal_guidance_scale", 1)} if c.model.generator.type == 'ort' else {}))
            batch = decoder.decode_tokens(tokens.reshape(tokens.shape[0], -1)).clamp(0, 1)
            batch = (batch * 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            if batch.shape[1:] != (256, 256, 3):
                raise ValueError(f'Expected ImageNet256 RGB decoder output, got {batch.shape}')
            if images is None:
                images = np.lib.format.open_memmap(scratch, mode='w+', dtype=np.uint8, shape=(a.count,256,256,3))
            images[start:start+len(batch)] = batch
            print(f'{start+len(batch)}/{a.count}', flush=True)
    images.flush()
    np.savez(output, arr_0=images)
    output.with_suffix('.json').write_text(json.dumps({'sampling_seed':a.seed, 'training_seed':int(c.training.seed), 'count':a.count, 'batch_size':a.batch_size, 'config':OmegaConf.to_container(c, resolve=True)}, indent=2))
    del images
    scratch.unlink()


if __name__ == '__main__':
    main()
