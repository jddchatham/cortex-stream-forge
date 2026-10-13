# Cortex Stream Forge — Iteration 122

The test-gated CPU boundary capture observes the completed destination of the
layer-2 gate `GGML_OP_MUL_MAT_ID` node `ffn_moe_gate-2` after the CPU worker
barrier and before the higher-level post-node observer.

Confirmed capture:

- Destination shape: `[640,10,9,1]`
- Destination size: `230400` bytes (`57600` F32 values)
- All `57600` values are finite and zero
- Bounded replay mismatch: `57586/57600`
- Maximum absolute replay error: `2.94772`
- RMS replay error: `0.322075`

This isolates the first observed divergence at the completed CPU destination.
The root cause is not identified by this iteration. Full regressions and
deployment validation remain pending. The capture is test-gated by
`CORTEX_MOE_KERNEL_CAPTURE`; production arithmetic is unchanged.
