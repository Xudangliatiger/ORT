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
