# Cortex Stream Forge — Iteration 127

## Status

`LAYER-2 BOUNDED MoE NUMERICAL VALIDATION: FAIL`

The complete test-gated layer-2 graph executes with bounded streamed gate,
up, and down projections, and its final expert aggregation is independently
checked. Full numerical equivalence to an independent GGUF dequantizer is not
yet proven because the projection reference still shares the provider's GGML
row-matmul implementation.

## Captured real graph transaction

Command:

```sh
timeout --kill-after=5s 90s bash -c \
  "printf 'hello\\n/exit\\n' | env CORTEX_MOE_STREAM_GATE=1 \
   CORTEX_MOE_STREAM_UP_DOWN=1 CORTEX_MOE_BOUNDARY_CAPTURE=1 \
   CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
   runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
   /home/user/models/qwen38-flashnext-abliterated/\
   Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

Exit code: `0`. The callbacks reported:

```text
projection=gate output_bytes=230400
projection=up   output_bytes=230400
projection=down output_bytes=921600
```

Captured tensors:

| Boundary | Shape | Bytes | Finite | Nonzero |
|---|---:|---:|---:|---:|
| up | `[640,10,9,1]` | 230400 | 57600 | 6400 |
| down | `[2560,10,9,1]` | 921600 | 230400 | 25600 |
| routing weights | `[1,10,9,1]` | 360 | 90 | 90 |
| aggregated output | `[2560,9,1,1]` | 92160 | 23040 | 23040 |

Down output range was `[-0.9710966, 0.6082801]`; aggregated output range was
`[-0.2020013, 0.1219889]`. The stream used 2,150,400 bytes of projector
scratch and did not bind complete expert tensors.

Fixtures:

- `/tmp/cortex-iter127-l2-up-output.bin`
- `/tmp/cortex-iter127-l2-down-output.bin`
- `/tmp/cortex-iter127-l2-weights.bin`
- `/tmp/cortex-iter127-l2-out.bin`

## Independent aggregation check

A separate scalar Python calculation read the captured down tensor and routing
weights, then accumulated each output channel in token/expert order. Compared
with the synchronized graph output:

```text
elements       = 23040
max_abs        = 1.80075497e-08
RMS            = 6.64072712e-10
above 1e-6     = 0
first mismatch = none
```

This validates tensor layout, routing-weight indexing, and the final expert
aggregation operation. It does not independently validate the IQ4_XS/IQ4_NL
weight decoding because both the production callback and projection replay
currently rely on the same GGML row-matmul implementation.

## Regression result

The streamed gate/up/down run returned token `248046`, logit `10.8127`, exit
code `0`. The gate-only Iteration 125 path remains `248046`, logit `10.9936`.
The logit change is retained as evidence that real up/down expert computation
is being used; it is not treated as proof of correctness.

`git diff --check` passes. No commit or push was performed. Existing unrelated
worktree changes were preserved.

## Remaining blocker

The earliest unproven boundary is the real IQ4_XS/IQ4_NL row decode and dot
product for up/down. The next required action is a separate bounded scalar
GGUF reference that decodes selected IQ4_XS and IQ4_NL rows without calling
`flashnext_execute_selected_projection_rows_layer()`, followed by elementwise
comparisons for gate, up, down, and the final aggregate.

Other MoE layers and full multi-layer streamed inference remain unvalidated.

LAYER-2 BOUNDED MoE: FAIL
NUMERICAL EQUIVALENCE: FAIL
BOUNDED MEMORY: PASS
FULL MODEL STREAMING INFERENCE: NOT YET TESTED
NEXT REQUIRED ACTION: Implement an independent scalar IQ4_XS/IQ4_NL GGUF row decoder and compare selected layer-2 gate/up/down outputs elementwise.
