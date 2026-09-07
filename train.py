"""Minimal distributed trainer for ORT on pretokenized image tokens."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from datasets import load_from_disk
from omegaconf import OmegaConf
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from modeling.losses import ORTARLoss
from modeling.factory import build_model, training_outputs, configure_order, build_loss


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
        or "ln" in name
        or "bias" in name
        or "latent_tokens" in name
        or "mask_token" in name
        or "gamma" in name
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
            return step / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def save_checkpoint(accelerator, model, output_dir, step, epoch, batch_offset):
    accelerator.wait_for_everyone()
    checkpoint_dir = output_dir / f"checkpoint-{step:07d}"
    accelerator.save_state(str(checkpoint_dir))
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save(state, checkpoint_dir / "pytorch_model.bin")
        (checkpoint_dir / "progress.json").write_text(json.dumps({
            "step": step, "epoch": epoch, "batch_offset": batch_offset,
            "world_size": accelerator.num_processes,
        }))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ort_l_alitok_xl_300.yaml")
    parser.add_argument("--dataset", help="Override dataset.params.pretokenization")
    parser.add_argument("--output", help="Override experiment.output_dir")
    parser.add_argument("--stop-after", type=int, help="Stop at this optimizer update without changing the recipe/scheduler horizon")
    parser.add_argument("--resume", help="Full checkpoint directory; same recipe and world size required")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if args.dataset:
        config.dataset.params.pretokenization = args.dataset
    if args.output:
        config.experiment.output_dir = args.output
    stop_at = int(config.training.max_train_steps)
    if args.stop_after is not None:
        if not 0 < args.stop_after <= stop_at:
            raise ValueError("--stop-after must be within the configured training horizon")
        stop_at = args.stop_after
    if config.training.get("use_ema", False) or config.training.get("torch_compile", False):
        raise ValueError("This trainer does not implement EMA or torch_compile")
    torch.backends.cuda.matmul.allow_tf32 = bool(config.training.get("enable_tf32", False))

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=int(config.training.seed)),
    )
    set_seed(int(config.training.seed))
    expected = config.training.get("global_batch_size", 2048)
    actual = accelerator.num_processes * config.training.per_gpu_batch_size * config.training.gradient_accumulation_steps
    if actual != expected:
        raise ValueError(f"Global batch {actual} differs from recipe {expected}; adjust per-device batch/accumulation")

    output_dir = Path(config.experiment.output_dir)
    if args.resume:
        previous = OmegaConf.load(Path(args.resume).parent / "config.yaml")
        if OmegaConf.to_container(previous, resolve=True) != OmegaConf.to_container(config, resolve=True):
            raise ValueError("Resume config differs from recorded config; use the same recipe")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is nonempty; select a fresh run directory or use --resume")
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
        generator=torch.Generator().manual_seed(int(config.training.seed)),
    )

    model = build_model(config)
    loss_module = build_loss(config)
    optimizer = optimizer_for(model, config)
    scheduler = cosine_scheduler(optimizer, config)
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    accelerator.register_for_checkpointing(scheduler)
    set_seed(int(config.training.seed), device_specific=True)

    step = 0
    epoch = 0
    batch_offset = 0
    if args.resume:
        accelerator.load_state(args.resume)
        progress = json.loads((Path(args.resume) / "progress.json").read_text())
        step, epoch, batch_offset = progress["step"], progress["epoch"], progress["batch_offset"]
    if args.resume and progress.get("world_size") != accelerator.num_processes:
        raise ValueError("Checkpoint world size differs or is missing")
    if len(dataloader) % int(config.training.gradient_accumulation_steps):
        raise ValueError("Distributed epoch batches must be divisible by accumulation steps; partial updates would change global batch")
    if not len(dataloader):
        raise ValueError("Dataset does not contain one complete distributed batch")
    while step < stop_at:
        dataloader.set_epoch(epoch)
        active_loader = accelerator.skip_first_batches(dataloader, batch_offset)
        for batch in active_loader:
            if step >= stop_at:
                break

            unwrapped = accelerator.unwrap_model(model)
            configure_order(unwrapped, config, step)

            with accelerator.accumulate(model):
                condition = unwrapped.preprocess_condition(
                    batch["label"],
                    cond_drop_prob=config.model.generator.class_label_dropout,
                )
                logits, labels, weight = training_outputs(model, batch["tokens"], condition, config)
                loss, metrics = loss_module(logits, labels, weight)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            batch_offset += 1
            if not accelerator.sync_gradients or accelerator.optimizer_step_was_skipped:
                continue
            step += 1
            if step % config.experiment.log_every == 0:
                accuracy = accelerator.gather(metrics["correct_tokens"].detach()).mean().item()
                accelerator.print(
                    f"step={step} loss={loss.detach().item():.5f} "
                    f"token_accuracy={accuracy:.5f} random_ratio={getattr(unwrapped, 'random_ratio', 0.0):.4f}"
                )
            if step % config.experiment.save_every == 0:
                save_checkpoint(accelerator, model, output_dir, step, epoch, batch_offset)
        if batch_offset >= len(dataloader):
            epoch += 1
            batch_offset = 0

    save_checkpoint(accelerator, model, output_dir, step, epoch, batch_offset)
    accelerator.end_training()


if __name__ == "__main__":
    main()
