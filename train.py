"""Minimal distributed trainer for ORT on pretokenized image tokens."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from accelerate import Accelerator
from datasets import load_from_disk
from omegaconf import OmegaConf
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from ort import ORTARLoss, ORTModel


class PretokenizedDataset(Dataset):
    """Reads a Hugging Face dataset with ``label`` and ``tokens`` columns."""

    def __init__(self, path: str):
        self.dataset = load_from_disk(path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        tokens = item["tokens"]
        if tokens and isinstance(tokens[0], list):
            tokens = random.choice(tokens)
        return {
            "label": torch.tensor(item["label"], dtype=torch.long),
            "tokens": torch.tensor(tokens, dtype=torch.long),
        }


def optimizer_for(model: torch.nn.Module, config) -> AdamW:
    params = list(model.named_parameters())
    no_decay = lambda name, value: (
        value.ndim < 2
        or "bias" in name
        or "norm" in name
        or "embed" in name
    )
    return AdamW(
        [
            {
                "params": [p for n, p in params if no_decay(n, p) and p.requires_grad],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in params if not no_decay(n, p) and p.requires_grad],
                "weight_decay": config.optimizer.params.weight_decay,
            },
        ],
        lr=config.optimizer.params.learning_rate,
        betas=(config.optimizer.params.beta1, config.optimizer.params.beta2),
    )


def cosine_scheduler(optimizer: AdamW, config):
    warmup = int(config.lr_scheduler.params.warmup_steps)
    total = int(config.training.max_train_steps)
    min_ratio = float(config.lr_scheduler.params.end_lr) / float(
        config.lr_scheduler.params.learning_rate
    )

    def scale(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def save_checkpoint(accelerator: Accelerator, model: torch.nn.Module, output_dir: Path, step: int):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir = output_dir / f"checkpoint-{step:07d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = accelerator.get_state_dict(model)
        accelerator.save(state, checkpoint_dir / "pytorch_model.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ort_alitok_xl.yaml")
    parser.add_argument("--dataset", help="Override dataset.params.pretokenization")
    parser.add_argument("--output", help="Override experiment.output_dir")
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if args.dataset:
        config.dataset.params.pretokenization = args.dataset
    if args.output:
        config.experiment.output_dir = args.output
    if args.max_steps:
        config.training.max_train_steps = args.max_steps

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
    )
    torch.manual_seed(int(config.training.seed) + accelerator.process_index)
    random.seed(int(config.training.seed) + accelerator.process_index)

    output_dir = Path(config.experiment.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, output_dir / "config.yaml")

    dataset = PretokenizedDataset(config.dataset.params.pretokenization)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.dataset.params.num_workers_per_gpu,
        pin_memory=True,
    )

    model = ORTModel(config)
    loss_module = ORTARLoss(config)
    optimizer = optimizer_for(model, config)
    scheduler = cosine_scheduler(optimizer, config)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    step = 0
    while step < config.training.max_train_steps:
        for batch in dataloader:
            if step >= config.training.max_train_steps:
                break

            unwrapped = accelerator.unwrap_model(model)
            unwrapped.set_random_ratio(unwrapped.get_rar_random_ratio(config, step))
            alpha, beta = unwrapped.get_rar_alpha_beta_weight(config, step)
            unwrapped.set_alpha_beta_weight(alpha, beta)

            with accelerator.accumulate(model):
                condition = unwrapped.preprocess_condition(
                    batch["label"],
                    cond_drop_prob=config.model.generator.class_label_dropout,
                )
                logits, labels, weight = model(batch["tokens"], condition, return_labels=True)
                loss, metrics = loss_module(logits, labels, weight)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if accelerator.is_main_process and step % config.experiment.log_every == 0:
                accuracy = accelerator.gather(metrics["correct_tokens"].detach()).mean().item()
                print(
                    f"step={step} loss={loss.detach().item():.5f} "
                    f"token_accuracy={accuracy:.5f} random_ratio={unwrapped.random_ratio:.4f}"
                )
            if step % config.experiment.save_every == 0:
                save_checkpoint(accelerator, model, output_dir, step)

    save_checkpoint(accelerator, model, output_dir, step)


if __name__ == "__main__":
    main()
