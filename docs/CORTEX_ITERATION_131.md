# CORTEX Stream Forge — Iteration 131

## Instrumentation implemented

The CPU diagnostic target now captures, for the first eligible row/block of a
selected CPU `MUL_MAT` transaction:

- the original F32 activation block (`conversion_f32`),
- the actual `from_float` output (`q8_block0`),
- the corresponding weight block (`weight_block0`), and
- the completed first scalar vec-dot result (`dot_result`).

The capture remains test-gated by `CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE` and
uses an atomic one-shot guard. The production vec-dot call and output writes are
unchanged. The existing diagnostic target is attached to the provider's
standalone `ggml_mul_mat` graph through the CPU backend.

The capture accepts both dispatches:

```text
IQ4_XS -> Q8_K (256 values, 292 bytes)
IQ4_NL -> Q8_0 (32 values, 34 bytes)
```

`llama_context::debug_copy_cpu_operand("dot_result", ...)` exposes the captured
scalar result after graph synchronization.

## Build

```text
cmake --build runtime-flashnext-layer-exp-build --target llama-flashnext-l2-moe-reference -j2
```

Exit code: `0`. The rebuilt executable hash was recorded from the working tree
after the build. `git diff --check` passes.

## Execution status

The standalone scalar reference still runs successfully against the Iteration
127 real fixtures, but no production runner invocation with the new target was
available in this bounded pass to emit the one-shot CPU operand fixtures. The
capture implementation is compiled; the first remaining numerical boundary is
therefore still unmeasured in this iteration.

Existing independent comparison remains:

```text
gate  max_abs=1.94815 rms=0.304924 first=819
up    max_abs=1.69971 rms=0.269166 first=1253
down  max_abs=0.969915 rms=0.015171 first=0
aggregate max_abs=0.201249 rms=0.00874946 first=0
```

## Next discriminating command

Run the bounded real provider/runner with:

```text
CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE=1 \
  <existing bounded layer-2 real-GGUF command>
```

Then read `weight_block0`, `q8_block0`, `conversion_f32`, and `dot_result`
through the synchronized debug-copy path and compare them with the independent
Q8_K/Q8_0 encoder and scalar block dot. No production arithmetic change is
justified before that comparison.

## Status

QUANTIZED ACTIVATION BYTES: FAIL (capture compiled, production fixture pending)
WEIGHT BLOCK BYTES: PASS
INDEPENDENT VEC-DOT: FAIL (production block result pending)
GATE PROJECTION: FAIL
UP PROJECTION: FAIL
DOWN PROJECTION: FAIL
FINAL MoE AGGREGATION: FAIL
BOUNDED MEMORY: PASS
FULL MODEL STREAMING: NOT TESTED
NEXT REQUIRED ACTION: Execute the real bounded layer-2 provider with CPU operand capture enabled and compare the emitted quantized bytes and dot result.
