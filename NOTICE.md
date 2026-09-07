# Third-party notices

The transformer and randomized-order generator implementation is adapted from:

- `bytedance/1d-tokenizer` (Apache-2.0), including the RAR implementation.
- `FoundationVision/LlamaGen` (MIT).

The released configuration targets AliTok tokens. AliTok tokenizer code is retained from the GNN source snapshot. Tokenizer weights
are not redistributed; obtain them from the official `ali-vilab/alitok` repository.

This release candidate contains no pretrained model weights or training data.
The LlamaGen MIT license is included as LICENSE-LlamaGen. The AliTok adaptation's
exact upstream revision and redistribution terms remain to be verified before
public release; these notices do not substitute for that check.

On 2026-09-07, the official AliTok checkout at
15410babdf944c957dcde96cc6d09e246bad9c3e contained no LICENSE, COPYING, or
NOTICE file. Its attribution headers alone do not establish redistribution
permission for AliTok-specific adaptations. Public release remains blocked.

Original AliTok and RAR generator files are included from the pinned revisions
in docs/BASELINES.md. MaskGIT VQGAN code retains Google, Hugging Face and Bytedance
attribution and Apache-2.0 notices. The pretrained MaskGIT tokenizer wrapper was
extracted from upstream titok.py with unrelated imports/classes removed.
