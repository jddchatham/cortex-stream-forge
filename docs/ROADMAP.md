# Cortex Stream Forge roadmap

**North star:** Run the complete approximately 97.5 GB Qwen3.8 Flash-Next Abliterated IQ4_XS GGUF model on low-memory CPU hardware by streaming weights from disk. Maintain one mathematically correct engine across hardware profiles.

## Milestone 1 — Bounded selection planning

- Verified strict plans for `[0,1)` and `[1,2)` under a 2 GiB/zero-swap gate.
- Draft [PR #3](https://github.com/jddchatham/cortex-stream-forge/pull/3) contains the proposed implementation and reference-contract documentation.
- Planning success is **not** evidence that model inference fits or computes correctly.

## Milestone 2 — One numerically verified owned-layer transaction

- Capture byte-identical input, recurrent/KV state, PLE rows, positions and model/executable hashes.
- Execute through the existing owned-layer streaming engine.
- Compare full output to a matching reference boundary and report numerical error.
- Measure real allocation peaks, bytes read, wall time, and OOM events.

## Milestone 3 — Full layer traversal

- Stream weights and state across all 48 layers, including PLE and sparse-MoE dependencies.
- Verify layer handoffs and preserve mathematical equivalence.
- Do not assume identical tensor requirements for all layers.

## Milestone 4 — Consecutive tokens

- Complete the output projection and decoding.
- Run at least two consecutive tokens while preserving recurrent and attention state.
- Verify results against reference computation.

## Milestone 5 — Usability and portability

- End-to-end prompt-to-output demonstration under the reference memory budget.
- Report real tokens/s and disk I/O; optimize only after correctness.
- Introduce minimal hardware-profile configuration, optional caches, and future compute backends without duplicating the engine.

Contributions should target the **smallest verifiable milestone**, not broad rewrites or unmeasured performance claims.
