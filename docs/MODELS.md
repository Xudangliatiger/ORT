# Models and downloads

The [ORT Hugging Face model repository](https://huggingface.co/donglixu/ORT) is public.
The ORT-E 400 file has been uploaded and its remote SHA-256 matches the original.

```bash
python scripts/download_models.py --model ort-e-alitok-xl-400 --output weights
# Equivalent Hub CLI download:
hf download donglixu/ORT ort-e-alitok-xl-400.bin configs/ort-e-alitok-xl-400.yaml --local-dir weights
```

[Direct checkpoint download](https://huggingface.co/donglixu/ORT/resolve/main/ort-e-alitok-xl-400.bin) · [Matching recipe](https://huggingface.co/donglixu/ORT/blob/main/configs/ort-e-alitok-xl-400.yaml)

| Checkpoint | Epochs | Random-phase weights (alpha, beta) | State |
|---|---:|---|---|
| ORT-E / AliTok-XL | 400 | (1, 0) | Available on Hub; strict load, image decode and backward verified |
| Historical file labeled ORT-L 400 | Conflicting metadata | Saved config says ORT-E 300 | Not released; directory/config mismatch |
| ORT-E / AliTok-XL | 300 | (1, 0) | Final checkpoint found; upload pending |
| ORT-L / AliTok-XL | 300 | (0.75, 1.25) | Final checkpoint found; upload pending |
| Historical 400-epoch run | 400 | (0.5, 1) | Final checkpoint found; not released |
| ORT-E IG extension | 300 | (1, 0) | Different parameter layout; archival only, standard loader unsupported |

Release only final generator state dictionaries with sanitized recipe configs and
SHA-256 hashes. Intermediate optimizer checkpoints are not inference downloads.
Tokenizer weights are obtained from upstream rather than re-hosted without clear
redistribution terms. See [baseline downloads](BASELINES.md).

Verified ORT-E 400 checkpoint SHA-256:
`9fec7c815728ab9bec5f41c39151c4da6786d1f7ed4eb916f98a555c7b0d1094`.

Expected AliTok tokenizer SHA-256 for the validated inference:
`154843c6ee4bdb9c04ba6db0a51cf68fc18f41f1bc28097e23f76d1aee4ef6a6`.

The current tests validate execution and compatibility, not published FID/IS.
