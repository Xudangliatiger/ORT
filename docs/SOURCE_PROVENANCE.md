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
