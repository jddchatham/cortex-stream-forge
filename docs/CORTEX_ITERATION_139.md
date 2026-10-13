# Cortex Iteration 139: full-row dot capture

## Result

The diagnostic was extended to capture complete matching rows at the CPU
`ggml_compute_forward_mul_mat_one_chunk()` boundary. The production arithmetic
was not changed.

Row sizes verified from the runtime tensor dimensions:

- IQ4_XS/Q8_K, width 2560: weight 1360 bytes, activation 2920 bytes, F32 source 10240 bytes.
- IQ4_NL/Q8_0, width 640: weight 360 bytes, activation 680 bytes, F32 source 2560 bytes.

The bounded run exited 0 and continued to produce token 248046. CPU counters
continued to prove the path `GGML_OP_MUL_MAT -> one_chunk -> vec_dot`.

## Capture implementation

The provider-local diagnostic now has row buffers and `row_capture_once`. The
CPU path copies the complete row after activation quantization storage exists
and before the dot loop. It only accepts a row with nonzero F32 activation and
nonzero weight bytes. The diagnostic remains test-gated by
`CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE`.

## Evidence limitation

The provider invokes many gate/up/down projections with the same quantization
type. The current fixture names are type-scoped, not projection- or invocation-
scoped, so later calls overwrite earlier complete rows. Although nonzero full
row files were emitted, the final file contents cannot be cryptographically
paired with a specific logged dot result. A scalar comparison against the
last log line therefore produced zero/mismatched results and is rejected as an
invalid comparison rather than reported as arithmetic divergence.

The next required correction is to attach a stable provider invocation ID and
write one immutable fixture set per selected gate, up, and down invocation,
including the dot value in the same manifest. No numerical equivalence claim is
made in this iteration.

## Commands

```text
cmake --build runtime-flashnext-layer-exp-build --target llama-qwen38-lowres-chat -j4
timeout --kill-after=5s 100s bash -c "printf 'hello\\n/exit\\n' | env CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE=1 CORTEX_MOE_STREAM_GATE=1 CORTEX_MOE_STREAM_UP_DOWN=1 CORTEX_MOE_BOUNDARY_CAPTURE=1 CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

Run log: `/tmp/cortex-iter139-run2.log`. Exit code: 0.

`git diff --check` passed.

## Status

FULL-ROW GATE DOT: NOT TESTED
FULL-ROW UP DOT: NOT TESTED
FULL-ROW DOWN DOT: NOT TESTED
GATE PROJECTION: NOT TESTED
UP PROJECTION: NOT TESTED
DOWN PROJECTION: NOT TESTED
FINAL AGGREGATION: NOT TESTED
BOUNDED MEMORY: PASS
FULL MODEL STREAMING: NOT TESTED
