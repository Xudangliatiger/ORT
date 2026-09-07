"""CPU integration check: mid-epoch resume with two accumulated microbatches."""
import os
from pathlib import Path
import subprocess
import sys

from datasets import Dataset
from omegaconf import OmegaConf
import torch


def test_resume_matches_uninterrupted(tmp_path):
    root = Path(__file__).resolve().parents[1]
    Dataset.from_dict({'label': [i % 10 for i in range(16)], 'tokens': [[i % 32] * 273 for i in range(16)]}).save_to_disk(str(tmp_path / 'data'))
    c = OmegaConf.load(root / 'configs/ort_l_alitok_xl_300.yaml')
    for k, v in dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=128, image_seq_len=273, condition_num_classes=10).items():
        c.model.generator[k] = v
    c.model.vq_model.codebook_size = 32
    c.model.vq_model.num_latent_tokens = 273
    c.training.global_batch_size = 4
    c.training.per_gpu_batch_size = 2
    c.training.gradient_accumulation_steps = 2
    c.training.mixed_precision = 'no'
    c.training.max_train_steps = 4
    c.dataset.params.num_workers_per_gpu = 0
    c.dataset.params.pretokenization = str(tmp_path / 'data')
    c.lr_scheduler.params.warmup_steps = 1
    c.experiment.log_every = 1
    c.experiment.save_every = 2
    c.experiment.output_dir = str(tmp_path / 'resumed')
    config = tmp_path / 'config.yaml'
    OmegaConf.save(c, config)
    def run(*args):
        subprocess.run([sys.executable, str(root / 'train.py'), '--config', str(config), *map(str, args)], check=True, env={**os.environ, 'OMP_NUM_THREADS':'1', 'ACCELERATE_USE_CPU':'true'})
    run('--stop-after', 2)
    run('--resume', tmp_path / 'resumed/checkpoint-0000002')
    run('--output', tmp_path / 'full')
    states = [torch.load(tmp_path / name / 'checkpoint-0000004/pytorch_model.bin', weights_only=True) for name in ('resumed', 'full')]
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
