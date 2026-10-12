"""Regression: every epoch must reshuffle and re-draw crops; a mid-epoch resume must reproduce the order."""
from pathlib import Path
import random
import sys

from accelerate import Accelerator, DataLoaderConfiguration
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import epoch_loader


class Crops(Dataset):
    """Same RNG use as PretokenizedDataset: random.choice over 10 stored crops."""
    def __len__(self):
        return 256

    def __getitem__(self, index):
        return {'idx': torch.tensor(index), 'crop': torch.tensor(random.choice(range(10)))}


def make_loader():
    accelerator = Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=42))
    loader = DataLoader(Crops(), batch_size=32, shuffle=True, drop_last=True, num_workers=2,
                        generator=torch.Generator().manual_seed(42))
    return accelerator, accelerator.prepare(loader)


def run_epoch(accelerator, loader, epoch, batch_offset=0):
    order, crops = [], {}
    for batch in epoch_loader(accelerator, loader, epoch, batch_offset):
        order += batch['idx'].tolist()
        crops.update(zip(batch['idx'].tolist(), batch['crop'].tolist()))
    return order, crops


def test_each_epoch_reshuffles_and_redraws_crops():
    accelerator, loader = make_loader()
    epochs = [run_epoch(accelerator, loader, e) for e in range(4)]
    for previous, current in zip(epochs, epochs[1:]):
        assert current[0] != previous[0], 'shuffle order repeated across epochs'
        same = sum(current[1][i] == previous[1][i] for i in previous[1]) / len(previous[1])
        assert same < 0.5, f'crop choice frozen across epochs ({same:.0%} identical)'


def test_mid_epoch_resume_reproduces_order():
    accelerator, loader = make_loader()
    full = [run_epoch(accelerator, loader, e)[0] for e in range(3)]
    accelerator, loader = make_loader()
    assert run_epoch(accelerator, loader, 2, batch_offset=3)[0] == full[2][3 * 32:]
