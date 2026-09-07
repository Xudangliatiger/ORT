# Validation status

Five CPU regression tests passed: ORT forward/backward with ordinal weights,
order schedule, and accumulated mid-epoch checkpoint/resume. Final resumed
weights were bitwise identical to uninterrupted training.

Two-GPU GH200 validation passed in 52 seconds: synthetic tiny-model training
for two updates, full-state resume to four updates, and 273-token generation.
The run used PyTorch 2.9.1 and bf16.

The historical ORT-E XL 400-epoch checkpoint passed strict loading, 273-token
autoregressive generation, AliTok 256x256 image decode, and one full-model
backward/AdamW update on GH200 in 59 seconds. The synthetic update loss was
2.706696; this is a smoke-test loss, not an evaluation metric.

Original AliTok and RAR generator forward/backward and complete token generation
passed on CPU. Two-GPU GH200 training/resume and token inference for both original baselines
passed in a combined 111-second run. These used tiny models and synthetic data;
official pretrained-baseline weights and full-image metrics remain unverified.
Full-dataset training, throughput, and 50k-image metrics have not been validated.
Multiworker crop RNG equivalence across resume is not guaranteed.

AliTok-specific redistribution terms remain unresolved in inspected upstream
revision 15410babdf944c957dcde96cc6d09e246bad9c3e; inherited notices alone do not
establish permission for those adaptations.

New RAR ORT-L/ORT-E adapters and the RAR baseline passed two-GPU GH200 tiny-model
training (2 updates), full-state resume (4 updates), strict checkpoint loading
and 256-token generation in a combined 142-second job. These use synthetic tokens.
The new trainer also passed a CPU train/resume check on 16 real AliTok token
records with the full 4096 vocabulary and original scheduler horizon retained.
Neither check establishes full-size 32-GPU training or paper FID reproduction.
