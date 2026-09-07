# Dataset preparation

The portable trainer reads **pretokenized ImageNet-1K training data**, not JPEGs,
WebDataset tar shards, or JSONL directly. Prepare the data once and reuse the same
records, class mapping and crop variants across baseline and ORT experiments.

## Choose the matching tokenizer

| Generator | Tokenizer | Tokens per image/crop | Token ID range |
|---|---|---:|---|
| `rar_original`, `rar_ort` | MaskGIT VQGAN, ImageNet f16-256 | 256 | 0–1023 |
| `alitok_original`, `ort` | AliTok | 273 | 0–4095 |

RAR-XL baseline 300, RAR-XL ORT-L 400 and RAR-XL ORT-E 400 must use the **same
MaskGIT dataset**. Changing the tokenizer setting does not convert existing IDs.
Do not truncate AliTok codes to 256 tokens or substitute a different VQGAN.
Tokenizer download references are in [BASELINES.md](BASELINES.md).

## Download pretokenized data

**AliTok:** [ORT ImageNet AliTok dataset](https://huggingface.co/datasets/donglixu/ORT-ImageNet-AliTok)
contains 1,281,167 images, ten 273-token crops per image, in 26 Parquet shards
(about 5.07 GB compressed). Downloads are public and do not require approval.
The source ImageNet terms still apply.

```python
from datasets import load_dataset
train = load_dataset("donglixu/ORT-ImageNet-AliTok", split="train")
train.save_to_disk("data/imagenet_alitok_train")
```

**RAR:** use the [official MaskGIT token file](https://huggingface.co/yucornetto/RAR/blob/main/maskgitvq.jsonl),
linked by the upstream RAR training guide. It is about 16.5 GB and uses `class_id`
and `tokens` fields. Download it on your training system and convert it:

```python
from huggingface_hub import hf_hub_download
hf_hub_download("yucornetto/RAR", "maskgitvq.jsonl", local_dir="data/rar_source")
```

```bash
python scripts/prepare_dataset.py --jsonl data/rar_source/maskgitvq.jsonl \
  --label-key class_id --tokenizer maskgit --output data/imagenet_maskgit_train
```

The official file's complete record count and crop grouping have not been audited
here; do not impose the AliTok dataset's count on it without checking. Its first
32 records passed real-token training/resume checks with the new RAR adapters.

## Starting from images

Obtain ImageNet-1K through its authorized distribution and keep the training and
validation splits separate. The standard training split contains 1,281,167 images.
Preserve the original mapping from class synset to integer label (0–999) used by
the tokenizer export and generator evaluation. Save that mapping with your local
experiment records; the range check below cannot verify its semantic correctness.

Use the matching tokenizer's image preprocessing and frozen pretrained weights:
RGB images, the intended 256×256 crop policy, and pixel values in [0, 1] for the
included tokenizer implementations. Do not apply ImageNet mean/std normalization.
The historical AliTok export supports an ADM-style center crop, or ten crops
from a center-cropped 281×281 image; these are different data recipes. Preserve
the chosen recipe and all variants when comparing methods. The exact crop policy
of each historical released run has not been independently verified.

The tokenizer APIs after preprocessing a batch `images` are:

```python
# images: [B, 3, 256, 256], RGB float values in [0, 1]
# tokenizer must be loaded with matching weights, in eval mode, on the same device.
with torch.inference_mode():
    # MaskGIT:
    codes = tokenizer.encode(images).reshape(images.shape[0], 256)
    # OR AliTok:
    _, result = tokenizer.encode(images, tokenizer.latent_tokens)
    codes = result['min_encoding_indices'].reshape(images.shape[0], 273)
```

Export one record per original image. With multiple crops, collect the encoded
sequences into that image's `tokens` list. Do not export every crop as a separate
image: that changes dataset length and training exposure. Image loading, GPU
sharding and preprocessing must be supplied by your preprocessing pipeline; this
release does not yet provide a validated end-to-end image encoding launcher.
Training with already encoded IDs does not load the tokenizer. Image sampling
does require its weights. Decode a few prepared examples and inspect them before
committing to a full training run.

## Record format

A record has an integer `label` and either one sequence or multiple crop sequences:

```text
{"label": 17, "tokens": [id_0, ..., id_255]}
{"label": 17, "tokens": [[crop0_id_0, ..., crop0_id_255], [crop1_id_0, ..., crop1_id_255]]}
```

These are schematic examples, not literal JSON. Use 273 IDs for AliTok. The
trainer randomly selects one saved crop per image visit. It does not perform
online image crops or flips; preprocessing fields in YAML do not augment token IDs.

## Convert existing JSONL

From the repository root:

```bash
python scripts/prepare_dataset.py \
  --jsonl /path/to/maskgit_tokens.jsonl \
  --tokenizer maskgit --expected-count 1281167 \
  --output data/imagenet_maskgit_train
```

Multiple JSONL shards can follow `--jsonl`. If your export uses different column
names, provide `--label-key` and `--tokens-key`; inspect its actual schema first.
Conversion preserves image order, labels, token IDs and all crop variants. It
normalizes single sequences to one-crop lists and saves only the training columns.
Existing output directories are never overwritten. Temporary Arrow data uses the
Hugging Face Datasets cache; configure `HF_DATASETS_CACHE` on spacious storage when
needed. Perform conversion where the large source files already reside.

For AliTok, use `--tokenizer alitok` and a separate output directory.

## Validate an existing HF dataset

```bash
python scripts/prepare_dataset.py \
  --dataset data/imagenet_maskgit_train \
  --tokenizer maskgit --expected-count 1281167
```

This scans every record for integer labels, sequence length and token-ID range,
checks the requested image count, and reports represented classes and crop counts.
It cannot establish the tokenizer identity from IDs alone, find duplicate images,
or verify class semantics. Preserve tokenizer checkpoint identity, preprocessing
recipe, class mapping and shard inventory as provenance. Check that all 1000
classes are present for the full training split.

The directory must be a single `datasets.Dataset.save_to_disk` output with
`label` and `tokens` columns, not a `DatasetDict` or a parent of unmerged GPU parts.
If needed, merge known disjoint parts in a documented order using
`concatenate_datasets([load_from_disk(part) for part in parts]).save_to_disk(output)`;
verify all ranks completed and no shards are missing or duplicated first.

## Train

```bash
accelerate launch --num_processes 32 train.py \
  --config configs/rar_xl_original_300.yaml \
  --dataset data/imagenet_maskgit_train
```

Select `configs/ort_l_rar_xl_400.yaml` or `configs/ort_e_rar_xl_400.yaml` for ORT-L or ORT-E,
using the same dataset. This is a launch example, not a complete multi-node Slurm
script. At batch 64 per GPU and accumulation 1, 32 processes give global batch
2048, which the trainer checks before loading data. Full ImageNet encoding and
training remain unverified in this release; converter tests use synthetic records.
