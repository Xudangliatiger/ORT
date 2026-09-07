"""Create synthetic token data for a bounded train/resume smoke test."""
import argparse
from pathlib import Path
from datasets import Dataset
from omegaconf import OmegaConf

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--world-size', type=int, default=2)
p.add_argument('--base-config', default='configs/ort_alitok_xl.yaml')
a = p.parse_args()
r = Path(a.output).resolve()
r.mkdir(parents=True, exist_ok=True)
c = OmegaConf.load(Path(__file__).resolve().parents[1] / a.base_config)
c.model.generator.hidden_size = 64
c.model.generator.num_hidden_layers = 2
c.model.generator.num_attention_heads = 4
c.model.generator.condition_num_classes = 10
c.model.vq_model.codebook_size = 32
c.model.generator.randomness_anneal_start = 1
c.model.generator.randomness_anneal_end = 3
c.training.global_batch_size = a.world_size * 4
c.training.per_gpu_batch_size = 2
c.training.gradient_accumulation_steps = 2
c.training.max_train_steps = 4
c.lr_scheduler.params.warmup_steps = 1
c.dataset.params.num_workers_per_gpu = 0
c.dataset.params.pretokenization = str(r / 'data')
c.experiment.output_dir = str(r / 'run')
c.experiment.save_every = 2
c.experiment.log_every = 1
Dataset.from_dict({'label': [i % 10 for i in range(32)], 'tokens': [[i % 32] * int(c.model.generator.image_seq_len) for i in range(32)]}).save_to_disk(str(r / 'data'))
OmegaConf.save(c, r / 'config.yaml')
