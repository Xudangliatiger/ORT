# Source and adaptation notes

The generator, loss, tokenizer, base model and registry are copied from the GNN
source snapshot with the original package layout. Unrelated model registrations,
networks, experiment configurations and cluster scripts are excluded.
The portable trainer retains fixes for collective-safe logging, accumulation,
optimizer-update cosine scheduling and checkpoint/resume RNG isolation.
It is a reduced training entry point, not a byte-identical copy of the full trainer.

The 400-epoch ORT-E configuration was recovered alongside a completed historical
checkpoint: alpha=1, beta=0, 250000 updates, 62500 warmup, order annealing
125000–187500, global batch 2048, seed 42. Paths and logging fields were removed.
The original enabled torch_compile; this portable trainer disables compilation.
The source config's CFG is retained; a historical evaluation used other CFG
settings. Exact published metric equivalence is not established here.

[AliTok](https://github.com/ali-vilab/alitok/tree/15410babdf944c957dcde96cc6d09e246bad9c3e)
and [RAR](https://github.com/bytedance/1d-tokenizer/tree/942a96fbdd873780179d1b78d5462911528bf8c8)
are upstream sources. LlamaGen MIT and inherited Apache notices are retained.
No weights, datasets, internal handoff notes or machine-specific paths are included.

## Retained source file hashes

SHA-256 identifies the unmodified implementation files in this extraction.

- `modeling/generators/ort.py`: `492943002ca8d47033e30bdce940e39b00ed4a1bdcf2bda51de221c244a1ae91`
- `modeling/losses/ort_ar_loss.py`: `714d5c11ecfc0ed9a2f4e84f3ef679bfc799720a4d83884b06d9149c3b5a6014`
- `modeling/tokenizers/alitok.py`: `949a00563300931cecd0c7b436872725c1a4b22e24eaccd5751908d6df1b07b7`
- `modeling/modules/base_model.py`: `a68ec3bf8ec971406f80bcb5bee9b56f6117e7832cdc1a05b3a18b068a3c0ea0`
- `utils/registry.py`: `144473521fed583aa17e6fcdbdea226d823dc9fcf5603eb4cd7c780fcadc273c`

## Original baselines

Original generator files are retained byte-for-byte from the pinned official
revisions listed in BASELINES.md. Shared adapters are in modeling/factory.py.

- `modeling/generators/alitok_original.py`: `f170c0c007bbd112d0569653a02ad798b07ae42e89c46a67a39429a6320857df`
- `modeling/generators/rar_original.py`: `130886ecde547592c0bfc347ebefa36473d2452b4cc362e892496a1c322c7d37`
- `modeling/losses/ar_loss.py`: `a4ea916929e785751b49f9447cd999c99fc4613869ec84cef0a44d6de98b21ea`
- `modeling/modules/maskgit_vqgan.py`: `c4c633759215d8c611735fc7dfc946a1073e8e10269bf5c10cfbedc1e83545f3`
