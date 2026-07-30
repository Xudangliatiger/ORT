# Ordinal-Biased Random Training (ORT)

This is an anonymized, minimal implementation of Ordinal-Biased Random Training
for class-conditional visual autoregressive models. It contains only the code needed
to inspect, train, and sample the ORT token generator; unrelated experiments, logs,
checkpoints, cluster scripts, and author metadata are intentionally excluded.

## Method in one paragraph

During randomized-order training, ORT assigns a linear position-dependent weight
to each token loss:

```text
w(t; alpha, beta) = alpha + (beta - alpha) * t / (T - 1).
```

The provided configuration uses biased weights in the randomized phase. The
probability of using a randomized path is then annealed from one to zero, so training
transitions to uniformly weighted raster paths. No architecture or inference-time
change is required.

## Contents

- `ort/model.py`: autoregressive transformer, randomized/raster order sampling,
  ordinal weights, annealing schedule, and generation.
- `ort/loss.py`: weighted token-level cross entropy.
- `configs/ort_alitok_xl.yaml`: the 300-epoch AliTok-XL ORT configuration.
- `train.py`: minimal Accelerate trainer for pretokenized datasets.
- `sample_tokens.py`: class-conditional token sampling from a checkpoint.
- `tests/test_smoke.py`: CPU forward/backward and schedule tests.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data format

`train.py` expects a Hugging Face dataset saved with `Dataset.save_to_disk`.
Every record must contain:

- `label`: ImageNet class index.
- `tokens`: one sequence of 273 AliTok token IDs, or a list of crop-specific
  sequences from which one is sampled during training.

AliTok tokenizer/decoder code and weights should be obtained from the official
`ali-vilab/alitok` release. They are deliberately not duplicated here.

## Training

Edit the dataset path in `configs/ort_alitok_xl.yaml`, or override it:

```bash
accelerate launch train.py \
  --config configs/ort_alitok_xl.yaml \
  --dataset /path/to/pretokenized_imagenet \
  --output outputs/ort_alitok_xl
```

The paper's 300-epoch setting corresponds to 187,500 optimizer steps at global
batch size 2,048. The randomized-path probability is 1 through step 62,500,
anneals to 0 by step 125,000, and remains 0 afterward.

The default configuration contains the main ORT-L setting
`(alpha, beta) = (0.75, 1.25)`. Other paper settings can be selected without
code changes, for example:

```bash
# Early bias used in ORT-E experiments
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('configs/ort_alitok_xl.yaml'); c.model.loss.alpha_weight_random=1.0; c.model.loss.beta_weight_random=0.0; OmegaConf.save(c, 'configs/ort_e.yaml')"
```

## Sampling

The sampling script produces token IDs:

```bash
python sample_tokens.py \
  --config configs/ort_alitok_xl.yaml \
  --checkpoint outputs/ort_alitok_xl/checkpoint-0187500/pytorch_model.bin \
  --labels 0 1 2 3 \
  --output outputs/token_samples.npz
```

Decode the `tokens` array with the official AliTok decoder, then use the standard
ADM ImageNet evaluation protocol to compute FID, IS, sFID, precision, and recall.

## Verification

```bash
pytest -q
```

The smoke test instantiates a small CPU model while preserving the 273-token AliTok
layout, exercises randomized ordinal weighting, computes the weighted loss, runs
backpropagation, and checks the random-to-raster schedule.

## Anonymity

This package was rebuilt without version-control history. It contains no author
names, usernames, email addresses, affiliations, machine names, absolute paths,
experiment dashboards, API keys, or pretrained weights.

## License and attribution

See `LICENSE` and `NOTICE.md`. The implementation retains attribution to the
upstream open-source projects from which the transformer/RAR components were
adapted.
