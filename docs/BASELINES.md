# Original baselines

The original generator files are copied byte-for-byte from these pinned upstream
revisions; the shared trainer adapts output signatures without wrapping the model
or changing checkpoint keys.

| Baseline | Original source | Local file | Tokenizer |
|---|---|---|---|
| AliTok-XL | [AliTok, 15410ba](https://github.com/ali-vilab/alitok/blob/15410babdf944c957dcde96cc6d09e246bad9c3e/modeling/ar.py) | `modeling/generators/alitok_original.py` | AliTok, 273 tokens / 4096 vocabulary |
| RAR-XL | [RAR, 942a96f](https://github.com/bytedance/1d-tokenizer/blob/942a96fbdd873780179d1b78d5462911528bf8c8/modeling/rar.py) | `modeling/generators/rar_original.py` | MaskGIT, 256 tokens / 1024 vocabulary |

AliTok uses its original raster-order generator and unweighted AR loss.
RAR uses its original randomized-order generator, with the original 400-epoch
order schedule (200–300 epoch annealing). Neither is the uniform-weight ORT model.
The RAR MaskGIT tokenizer class is extracted from upstream titok.py; only unrelated
TiTok classes/imports were removed. Its maskgit_vqgan.py implementation is retained.

```bash
accelerate launch --num_processes 32 train.py --config configs/alitok_xl_original_400.yaml --dataset /path/to/alitok_tokens
accelerate launch --num_processes 32 train.py --config configs/rar_xl_original_400.yaml --dataset /path/to/maskgit_tokens
```

The portable configs use per-GPU batch 64 on 32 GPUs, global batch 2048. Official
AliTok uses 32 on 64 GPUs; this changes topology, not the global batch. Our trainer
is a shared portable entry point, not the complete upstream trainer. Official RAR
JSONL data must be converted to the documented HF dataset format, preserving all
labels, token IDs and crop variants. Full-training equivalence is unverified.

Use `sample_tokens.py` or `sample_images.py` with the selected config and matching
weights. Download original checkpoints from the [AliTok model table](https://github.com/ali-vilab/alitok#-usage)
or [RAR model page](https://huggingface.co/yucornetto/RAR). The MaskGIT tokenizer is
[available here](https://huggingface.co/fun-research/TiTok/blob/main/maskgit-vqgan-imagenet-f16-256.bin).
Do not load an AliTok checkpoint with the RAR config, or vice versa.

Original generator CPU forward/backward and complete token generation tests pass.
Two-GPU GH200 synthetic train/resume/token-generation checks also pass for both.
Official pretrained-baseline weights and full-image metrics have not been tested
in this extraction. Third-party licensing notes remain in NOTICE.md.

## RAR with ORT

`rar_ort` subclasses original RAR, preserving parameter names and shapes. RAR
already supports random permutations and target-aware position embeddings. The
adapter adds ordinal loss weights and the shared ORT curriculum; raster inference
remains original RAR. Weights follow generation position, not spatial token ID.
Loss is `mean(weight * token_cross_entropy)`, without weight renormalization.
Random examples use `(alpha, beta)`; raster examples use `(1, 1)`, including in the
mixed-order phase. Both variants use MaskGIT (256 tokens, vocabulary 1024).

| Recipe | Total updates | Order annealing updates | Random loss endpoints |
|---|---:|---|---|
| `configs/rar_xl_original_300.yaml` | 187500 | 62500–125000 | uniform |
| `configs/ort_l_rar_xl_400.yaml` (ORT-L) | 250000 | 125000–187500 | 0.75, 1.25 |
| `configs/ort_e_rar_xl_400.yaml` (ORT-E) | 250000 | 125000–187500 | 1, 0 |

The two primary recipes are baseline 300 and ORT 400. The E variant changes only
alpha/beta and experiment/output names so both runs can coexist. Epoch labels
follow the historical convention of 625 optimizer updates per epoch with global
batch 2048. These are new experiments, not verified reconstructions of missing
historical checkpoints. CFG 16 / power 2.75 is inherited from the existing RAR
recipe and is not established as optimal for XL.

Set `--dataset` to an HF dataset directory with MaskGIT token IDs and labels. The
portable trainer does not directly read historical JSONL. Tokenizer weights are
needed for image decoding, not pretokenized training. YAML holds recipe settings;
cluster scripts must separately configure resources, environments, ranks,
rendezvous and caches. Short launch examples are not multi-node Slurm scripts.

CPU tests verify strict state-dict interchange with original RAR, equal logits
under identical orders, L/E weighted loss, raster uniform loss, backward, full
256-token generation and MaskGIT dispatch. The new adapter also passed two-GPU GH200 tiny-model training/resume and token
generation. Full-size 32-GPU training and full ImageNet evaluation remain unverified.
