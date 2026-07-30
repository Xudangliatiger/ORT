"""Sample class-conditional AliTok token sequences from an ORT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from ort import ORTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ort_alitok_xl.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="ort_token_samples.npz")
    parser.add_argument("--labels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = ORTModel(config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    labels = torch.tensor(args.labels, device=device, dtype=torch.long)
    tokens = model.generate(
        condition=labels,
        guidance_scale=config.model.generator.guidance_scale,
        guidance_scale_pow=config.model.generator.guidance_scale_pow,
        randomize_temperature=config.model.generator.randomize_temperature,
        internal_guidance_scale=config.model.generator.internal_guidance_scale,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        tokens=tokens.cpu().numpy().astype(np.int32),
        labels=labels.cpu().numpy().astype(np.int32),
    )
    print(f"Saved {tokens.shape[0]} token sequences to {output}")


if __name__ == "__main__":
    main()
