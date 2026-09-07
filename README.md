# Ordinal-Biased Random Training (ORT)

Minimal ORT training and inference code extracted from the GNN codebase.
Includes the original ORT generator, ordinal cross-entropy loss, AliTok tokenizer,
pretokenized trainer, full-state resume, token sampling and image decoding.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

Use a CUDA-compatible PyTorch build on GPU machines. Existing HPC module
installations may be reused in an isolated environment. Weights and data are
supplied separately.

## Train

The dataset is a Hugging Face `save_to_disk` dataset with `label` and `tokens`
columns. Tokens are 273 AliTok IDs in [0,4095], optionally multiple crop sequences.

```bash
accelerate launch --num_processes 32 train.py --config configs/ort_e_alitok_xl_400.yaml --dataset /path/to/tokens --output outputs/ort_e_seed42
```

Maintain global batch 2048 = world size × per-GPU batch × gradient accumulation.
For multiple hosts, configure Accelerate machine count, ranks and rendezvous.
The 400-epoch recipe has 250000 optimizer updates; the separate 300-epoch recipe
has a full 187500-update cosine horizon. Epoch names follow the nominal source
convention of 625 updates per epoch.

`--stop-after N` stops early without changing the schedule. Resume with the same
recipe and topology using `--resume outputs/ort_e_seed42/checkpoint-0000010`.
Training seed is in the YAML; sampling seed is a separate CLI argument.

## Inference

```bash
python sample_tokens.py --config configs/ort_e_alitok_xl_400.yaml --checkpoint /path/to/ort.bin --seed 2 --labels 1 7 --output outputs/tokens.npz
python scripts/check_inference.py --config configs/ort_e_alitok_xl_400.yaml --checkpoint /path/to/ort.bin --tokenizer-weights /path/to/AliTok.pth --output outputs/image
```

For a balanced 50k archive compatible with the
[ADM evaluator](https://github.com/openai/guided-diffusion/tree/main/evaluations):

```bash
python sample_images.py --config configs/ort_e_alitok_xl_400.yaml --checkpoint /path/to/ort.bin --tokenizer-weights /path/to/AliTok.pth --seed 2 --count 50000 --output outputs/samples.npz
python /path/to/guided-diffusion/evaluations/evaluator.py /path/to/VIRTUAL_imagenet256_labeled.npz outputs/samples.npz
```

Archive generation is serial and uses a temporary disk array; allow about 20 GB
free disk for a 50k run. Hold seed, batch size, CFG and class ordering fixed when
comparing checkpoints. No paper metric reproduction is claimed by a smoke test.

See [validation status](docs/RELEASE_STATUS.md) and [source notes](docs/SOURCE_PROVENANCE.md).
Inherited licenses and third-party notices are retained; AliTok-specific
redistribution terms remain unresolved in the inspected upstream source.
