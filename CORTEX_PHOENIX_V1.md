# Cortex Phoenix v1

This document freezes the completed second-token milestone at the Phoenix
checkpoint. It records verified results and the safe migration boundary back
to the original Cortex fork.

## Checkpoint

- Phoenix source checkpoint: current `experiment/deepseek-state-replay` HEAD
- First-token protected hash: `1420f43caf5e1e05`
- Second-token input: `180671`
- Second-token final activation: `603239d1601b8e30`
- Final HC normalization hash: `1a5132b4d7347f8d`
- Vocabulary projection: Q6_K, 248,320 logits
- Second-token prediction: token `60810`, logit `3.55325484`
- Second-token logits hash: `bf808ab0707a89fb`
- Process exit: `0`

The protected run completed layers 0 through 47 for both token positions.
Graph A, sparse MoE, Graph B, state commit, and zero unselected-expert reads
were reported for the completed transactions. The final measured run used
`MemoryMax=4G` and `MemorySwapMax=4G`, with peak RSS `923280 KiB` and swap
`0 KiB`.

## Proven engineering discoveries

- Diagnostic captures must be scoped to an execution/transaction rather than
  first-capture-wins tensor names.
- Tensor comparisons use one canonical byte-wise hash, avoiding word-wise
  versus byte-wise hash disagreements.
- Graph-to-transaction handoffs require owned buffers with verified lifetime;
  producer, owned, and consumer hashes are checked at the copy boundary.
- Qwen4Exp embedding batches require all three M-RoPE position planes. The
  legacy one-plane allocation caused out-of-bounds/uninitialized position data
  and nondeterministic state replay.
- Hybrid sequence state is serialized/restored as an opaque state snapshot;
  state hashes and byte counts are recorded at token boundaries.
- Layers and selected MoE experts are streamed within bounded memory, with
  completed-layer resources released and unselected expert reads kept at zero.

## Limitations

- Independent numerical reference comparison for the second-token logits was
  not run (`REFERENCE_COMPARISON=NOT_RUN`).
- Token generation beyond the second token and a complete multi-token quality
  evaluation were not established.
- The full 48-layer continuation uses the existing per-layer streaming and
  opaque state handoff; it is not proof that arbitrary internal state can be
  independently reconstructed outside that protocol.
- The Phoenix experiment branch contains unrelated uncommitted files in the
  worktree; they are intentionally excluded from the milestone commit.

## Migration plan to the original Cortex fork

Do not merge Phoenix wholesale. Port and review only the minimum techniques:

1. Cherry-pick or manually port the execution-scoped diagnostic identity and
   canonical hashing changes, keeping them disabled by default.
2. Port explicit owned activation handoffs and lifetime checks at existing
   graph boundaries; do not add graph nodes solely for diagnostics.
3. Port the three-plane M-RoPE batch allocation/fill fix, then rerun the
   original first-token regression before any continuation test.
4. Port bounded state spooling only if the fork has the same state ownership
   and rollback contract; retain durable-write and cleanup behavior behind an
   explicit option.
5. Revalidate one layer, then a bounded layer range, then the first-token
   frozen hash. Keep Phoenix-specific continuation flags and generated runtime
   directories out of the fork.
6. Preserve the protected milestone tags and compare output, state, memory,
   and unselected-read invariants at every migration step.

The migration target is the original fork's development branch, not a
wholesale merge of experimental harnesses, logs, model data, or build trees.
