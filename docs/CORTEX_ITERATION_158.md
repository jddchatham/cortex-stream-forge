# Cortex Iteration 158 — conventional layer-4 weight binding forensics

## Scope and preservation

Work continued on the dirty Iteration 157 state in branch
`cortex-iteration-158`. No production arithmetic, routing, quantization, or
streamed-layer selection was changed. The new CPU trace is test-gated by
`CORTEX_CONVENTIONAL_WEIGHT_TRACE`.

## Executed graph identity

The conventional A run completed normally (exit 0, token `248046`, selected
logit `10.7801`). The CPU dispatch trace reached stock
`GGML_OP_MUL_MAT_ID` for physical layer 4 with these actual graph leaves:

```text
blk.4.ffn_gate_exps.weight  type=23 IQ4_XS
  ne=[2560,640,512,1] nb=[136,1360,870400,445644800]
  vec_dot_type=15 Q8_K
blk.4.ffn_up_exps.weight    type=23 IQ4_XS
  ne=[2560,640,512,1] nb=[136,1360,870400,445644800]
  vec_dot_type=15 Q8_K
blk.4.ffn_down_exps.weight  type=20 IQ4_NL
  ne=[640,2560,512,1] nb=[18,360,921600,471859200]
  vec_dot_type=8 Q8_0
```

Each node had a non-null tensor data pointer and backend buffer, and the same
selected-expert ID tensor was supplied as `src[2]`. Therefore the conventional
nodes are active graph nodes, not disconnected diagnostic metadata.

The capture path also retained selected tensors as graph outputs and read them
only after scheduler synchronization. The streamed B run produced finite,
nonzero captures, confirming the backend read path is functional.

## Weight residency result

The new CPU-side trace sampled the first 64 bytes of every active expert row
before `MUL_MAT_ID` computation. For physical layer 4, all sampled conventional
rows had:

```text
sample_bytes=64 sample_nonzero=0 fnv1a=eebe4e4e51c6c683
```

This occurred for gate, up, and down leaves. The runtime buffers were therefore
allocated but zero-filled/unpopulated; non-null storage was not valid GGUF
weight residency.

For the same model index and expert 108, direct bounded reads from the
canonical GGUF descriptor produced nonzero bytes:

| tensor | descriptor offset | runtime sample | GGUF sample | GGUF nonzero / 64 |
|---|---:|---|---|---:|
| `blk.4.ffn_gate_exps.weight` | 35839149344 | all zero | `9600f0cc056f9767...` | 63 |
| `blk.4.ffn_up_exps.weight` | 36290917664 | all zero | `8d003cfff0876d6e...` | 63 |
| `blk.4.ffn_down_exps.weight` | 35366368544 | all zero | `6f0c051ab5b6b5a5...` | 64 |

The GGUF row address was computed as descriptor absolute offset plus expert
index times the verified per-expert plane stride (`870400` bytes for IQ4_XS
and `921600` bytes for IQ4_NL). This is a byte-level mismatch, not a numerical
rounding issue.

## First proven zero boundary and cause

The first proven invalid transition is the conventional model tensor storage
presented to the CPU kernel: scheduler-owned buffers exist, but the canonical
GGUF payload was never populated into them. The CPU kernel then consumes valid
tensor metadata and zero bytes, producing the zero gate/up/down results seen in
the conventional captures. The graph is executed and the capture lifetime is
valid.

This is consistent with the existing residency design: expert tensors are
classified as externally streamed and excluded from `cortex_rebind_layer()`;
the conventional fallback still reaches stock `MUL_MAT_ID` with its canonical
objects unbound to real GGUF payloads. No repair was applied in this iteration.

## A/B validity

| check | status | result |
|---|---|---|
| conventional layer-4 graph identity | PASS | stock `MUL_MAT_ID` reached for all three leaves |
| conventional tensor buffers allocated | PASS | non-null data and backend buffers |
| conventional payload equals GGUF | FAIL | active samples zero versus nonzero GGUF rows |
| conventional capture synchronization | PASS | graph-output retention plus synchronized backend reads |
| streamed provider regression | PASS | provider test exit 0; prior max errors remain within limits |
| A/B arithmetic equivalence | NOT TESTED | conventional weights are invalid |
| full logits equivalence | NOT TESTED | no trustworthy conventional reference exists |
| memory impact | NOT TESTED | no ownership or allocation correction attempted |

## Smallest safe next action

Trace the model-load/residency boundary for layer 4 and either (a) populate a
bounded, valid conventional reference buffer before the conventional graph is
scheduled, or (b) keep layer 4 on the validated streamed provider. Do not
attach a selected-expert-sized buffer to a canonical full tensor and do not
claim A/B equivalence until the CPU trace shows GGUF-matching bytes.

## Validation commands

```text
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-qwen38-lowres-chat llama-flashnext-iq4xs-provider-test -j4
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  /tmp/cortex-iter153-layer2-4-index.json 4
git diff --check
```

Build, provider regression, and whitespace checks completed successfully.
