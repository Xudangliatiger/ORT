# Models and downloads

| Model | Epochs | FID ↓ (paper) | Download | Recipe |
|---|---:|---:|---|---|
| ORT-E / AliTok-XL | 400 | **1.18** | [Weights](https://huggingface.co/donglixu/ORT/resolve/main/ort-e-alitok-xl-400.bin) | [Config](https://huggingface.co/donglixu/ORT/blob/main/configs/ort-e-alitok-xl-400.yaml) |
| ORT-E / AliTok-XL | 300 | **1.26** | [Weights](https://huggingface.co/donglixu/ORT/resolve/main/ort-e-alitok-xl-300.bin) | [Config](https://huggingface.co/donglixu/ORT/blob/main/configs/ort-e-alitok-xl-300.yaml) |
| ORT-L / AliTok-XL | 300 | **1.34** | [Weights](https://huggingface.co/donglixu/ORT/resolve/main/ort-l-alitok-xl-300.bin) | [Config](https://huggingface.co/donglixu/ORT/blob/main/configs/ort-l-alitok-xl-300.yaml) |
| ORT-L / AliTok-XL | 400 | **1.31** | [Weights](https://huggingface.co/donglixu/ORT/resolve/main/ort-l-alitok-xl-400.bin) | [Config](https://huggingface.co/donglixu/ORT/blob/main/configs/ort-l-alitok-xl-400.yaml) |
| AliTok-XL\* baseline | 300 | 1.42 | [Weights](https://huggingface.co/donglixu/ORT/resolve/main/alitok-xl-baseline-300.bin) | [Config](https://huggingface.co/donglixu/ORT/blob/main/configs/alitok-xl-baseline-300.yaml) |
| AliTok-XL\* baseline | 400 | 1.35 | [Weights](https://huggingface.co/donglixu/ORT/resolve/main/alitok-xl-baseline-400.bin) | [Config](https://huggingface.co/donglixu/ORT/blob/main/configs/alitok-xl-baseline-400.yaml) |

FID values are paper-reported on ImageNet 256×256 (current manuscript, Tables 2 and 7). These rows correspond to the authors’ AliTok*-XL recipes. The released checkpoint-to-result mapping has not been independently verified by a new 50k-image evaluation. Exact per-row evaluation CFG and sampling seed remain unverified; config defaults and the example seed are not a claim to reproduce these FIDs.

## Download

Choose one of the six model IDs below; the helper downloads its matching config and verifies SHA-256.

```bash
python scripts/download_models.py --model alitok-xl-baseline-300 --output weights
python scripts/download_models.py --model alitok-xl-baseline-400 --output weights
python scripts/download_models.py --model ort-e-alitok-xl-400 --output weights
python scripts/download_models.py --model ort-e-alitok-xl-300 --output weights
python scripts/download_models.py --model ort-l-alitok-xl-300 --output weights
python scripts/download_models.py --model ort-l-alitok-xl-400 --output weights
```

For a single model with the Hub CLI:

```bash
hf download donglixu/ORT ort-e-alitok-xl-400.bin configs/ort-e-alitok-xl-400.yaml --local-dir weights
```

All files are final generator state dictionaries (2,661,983,138 bytes each); optimizer states are excluded. Training seed is 42. The 300- and 400-epoch recipes use 187,500 and 250,000 updates respectively.

## Integrity and recipe provenance

The machine-readable [manifest](../configs/pretrained.json) records SHA-256, size and matching config for every file.

- `ort-e-alitok-xl-400.bin`: `9fec7c815728ab9bec5f41c39151c4da6786d1f7ed4eb916f98a555c7b0d1094`
- `ort-e-alitok-xl-300.bin`: `4a99f02de89bd35ca4629a3df61a6f85484a1bd9139e5ac3512a7136fba1dc3c`
- `ort-l-alitok-xl-300.bin`: `ddfc6561d9c344ea0ff9415cfb48c433715a0db0d152aa50ae3fea72f7f38c34`
- `ort-l-alitok-xl-400.bin`: `3fba113df943ed8dd21cb1d1b2ddaa4d061356285378ddb908f5f9196c83ba37`

ORT-L 400’s standalone saved config was stale. Its released portable recipe was reconstructed from the training log: alpha=0.75, beta=1.25, 250,000 updates, random-order annealing from 125,000 to 187,500. Every tensor in the released final state dictionary matches the final step-250000 checkpoint after normalizing compilation prefixes. Portable configs disable compilation and use placeholder paths.

## Tokenizer and execution checks

Obtain the matching AliTok tokenizer from the [official project](https://github.com/ali-vilab/alitok#-usage). Expected tokenizer SHA-256 for the validated inference is `154843c6ee4bdb9c04ba6db0a51cf68fc18f41f1bc28097e23f76d1aee4ef6a6`. See [original baseline downloads](BASELINES.md).

ORT-E 400 passed strict loading, 273-token generation, 256×256 RGB decoding and a full-model optimizer update on GH200. Other released files have source-to-Hub hash verification; that does not imply each has passed GPU inference or reproduced its paper FID.

## Reimplemented AliTok-XL baselines

AliTok-XL\* denotes our reimplementation and training with randomized pretraining and uniform token loss weights (alpha=beta=1). Both use `model.generator.type: ort` to preserve the trained parameter layout; use the downloaded config, not the original raster AliTok config. Each released state dictionary matches every tensor of its final training checkpoint, and its parameter names and shapes match the verified ORT layout. This is an integrity and layout check, not a fresh GPU inference or FID evaluation.

- `alitok-xl-baseline-300.bin`: SHA-256 `bcfb13aecda05026d09c4235102f3b234c02737c45761c2a9bde0e20f15bbaa6`.
- `alitok-xl-baseline-400.bin`: SHA-256 `276d4e5771865aeeb9f22941ba8214efc213b42faf200859ab3ca9bf1b03f3af`.
