# CORTEX Stream Forge — Iteration 130

## Result

The CPU path was traced without changing production arithmetic. The bounded
provider constructs `ggml_mul_mat()` with one activation column per selected
expert tile. Consequently the tiled CPU path is not eligible (`ne1 == 1` is
below its batch threshold); the provider uses the stock CPU `vec_dot` path.

The stock dispatch is:

```text
IQ4_XS -> ggml_vec_dot_iq4_xs_q8_K
IQ4_NL -> ggml_vec_dot_iq4_nl_q8_0
```

The tiled implementation was also inspected. It has a separate activation
packing workspace and is not the path used by the bounded provider transaction.

## Independent diagnostic

The standalone reference now records the correct activation representation per
weight type and performs independent scalar block evaluation:

- IQ4_XS: Q8_K, 256-element blocks.
- IQ4_NL: Q8_0, 32-element blocks with FP16-rounded scale.

It reads only selected GGUF rows and does not call production vec-dot or
dequantization functions.

## Command and output

```text
cmake --build ../runtime-flashnext-layer-exp-build --target llama-flashnext-l2-moe-reference -j2
timeout --kill-after=5s 90s ../runtime-flashnext-layer-exp-build/bin/llama-flashnext-l2-moe-reference \
  /tmp/cortex-iter120-layer2-index.json \
  /tmp/cortex-iter120-l2-gate-activation.bin \
  /tmp/cortex-iter120-l2-gate-ids.bin \
  /tmp/cortex-iter127-l2-up-output.bin \
  /tmp/cortex-iter127-l2-down-output.bin \
  /tmp/cortex-iter127-l2-out.bin
```

Exit code: `0`.

```text
CORTEX_L2_REF gate n=57600 max_abs=1.94815 rms=0.304924 mismatches=51197 first=819 nonfinite=0
CORTEX_L2_REF up n=57600 max_abs=1.69971 rms=0.269166 mismatches=51189 first=1253 nonfinite=0
CORTEX_L2_REF down n=230400 max_abs=0.969915 rms=0.015171 mismatches=197123 first=0 nonfinite=0
CORTEX_L2_REF aggregate n=23040 max_abs=0.201249 rms=0.00874946 mismatches=14914 first=0 nonfinite=0
```

The earliest observed projection divergence remains gate index 819. The
downstream down/aggregate comparisons are not isolating the down kernel because
their independent input is reconstructed from the mismatching independent
gate/up projections.

## First unresolved boundary

The remaining missing evidence is a production-side capture of the actual Q8_K
or Q8_0 activation blocks and one completed CPU vec-dot result for the same
weight row. Existing graph-level F32 captures cannot establish those bytes.
The next discriminating change should be a test-gated capture immediately
around the stock CPU `ggml_compute_forward_mul_mat()` vec-dot call, after its
`from_float` conversion and before the vec-dot dispatch, restricted to one
selected layer-2 row/block. No production behavior should change.

## Status

QUANTIZED ACTIVATION BYTES: FAIL (not yet captured at the stock vec-dot boundary)
WEIGHT BLOCK BYTES: PASS (independent GGUF row bytes and layouts validated)
INDEPENDENT BLOCK DOT PRODUCT: FAIL (production quantized block/result pair unavailable)
GATE PROJECTION: FAIL
UP PROJECTION: FAIL
DOWN PROJECTION: FAIL
BOUNDED MEMORY: PASS
FULL MODEL STREAMING: NOT TESTED
NEXT REQUIRED ACTION: Add a one-row, test-gated capture after stock `from_float` and compare Q8_K/Q8_0 bytes plus the corresponding vec-dot result.

`git diff --check`: PASS.
