# CORTEX Stream Forge — Iteration 129

## Scope

This iteration traced the CPU quantized dot dispatch and updated the independent
layer-2 MoE evaluator. Production arithmetic and the bounded provider were not
changed.

## Dispatch evidence

The checkout's CPU type-trait table selects:

| Weight type | CPU vec-dot | Activation representation |
|---|---|---|
| IQ4_XS | `ggml_vec_dot_iq4_xs_q8_K` | Q8_K, 256-value blocks |
| IQ4_NL | `ggml_vec_dot_iq4_nl_q8_0` | Q8_0, 32-value blocks |

The generic implementations in `ggml/src/ggml-cpu/quants.c` and the x86
implementations in `ggml/src/ggml-cpu/arch/x86/quants.c` agree on the weight
lookup order and block traversal. IQ4_XS consumes two 32-value sub-blocks per
scale pair; IQ4_NL consumes the low and high nibbles against the 32-byte Q8_0
activation block.

## Independent evaluator change

`examples/flashnext-l2-moe-reference.cpp` now has separate scalar activation
round trips:

- IQ4_XS gate/up: Q8_K, 256 values, signed-byte scale and block dequantization.
- IQ4_NL down: Q8_0, 32 values, FP16-rounded block scale and signed-byte
  dequantization.

The evaluator continues to decode IQ4_XS and IQ4_NL weight rows independently
from GGUF bytes and does not call production dequantizers or vec-dot kernels.

## Executed command

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

Build and execution exit code: `0`.

## Results

With Q8_K for IQ4_XS and Q8_0 for IQ4_NL:

```text
CORTEX_L2_REF gate n=57600 max_abs=1.94815 rms=0.304924 mismatches=51197 first=819 nonfinite=0
CORTEX_L2_REF up n=57600 max_abs=1.69971 rms=0.269166 mismatches=51189 first=1253 nonfinite=0
CORTEX_L2_REF down n=230400 max_abs=0.969915 rms=0.015171 mismatches=197123 first=0 nonfinite=0
CORTEX_L2_REF aggregate n=23040 max_abs=0.201249 rms=0.00874946 mismatches=14914 first=0 nonfinite=0
```

The prior all-Q8_K run was:

```text
CORTEX_L2_REF down n=230400 max_abs=0.764412 rms=0.0117262 mismatches=227108 first=0 nonfinite=0
CORTEX_L2_REF aggregate n=23040 max_abs=0.152755 rms=0.00640988 mismatches=22358 first=0 nonfinite=0
```

Thus the dispatch correction is real, but it does not establish numerical
equivalence. The earliest observed projection divergence remains gate output
index 819; the evaluator's down result also cannot yet be interpreted as an
isolated down-kernel comparison because its input is reconstructed from the
independent gate/up results rather than a captured production post-gate
activation.

## Memory and safety

The evaluator reads one selected expert row at a time and uses bounded vectors;
it does not materialize any complete expert tensor. No production model math,
routing, or streaming behavior was changed. `git diff --check` passed.

## Status

INDEPENDENT IQ4_XS DECODING: PASS
INDEPENDENT IQ4_NL DECODING: PASS
GATE PROJECTION: FAIL
UP PROJECTION: FAIL
DOWN PROJECTION: FAIL
FINAL MoE AGGREGATION: FAIL
BOUNDED MEMORY: PASS
FULL MODEL STREAMING: NOT TESTED
NEXT REQUIRED ACTION: Capture the production CPU quantized activation blocks and the exact post-gate/post-up activation, then compare block bytes and one vec-dot accumulation before changing the independent evaluator.
