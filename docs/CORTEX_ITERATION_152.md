# Cortex Iteration 152 — Trace Missing Layer-3 Streaming Callback

Date: 2026-10-13
Branch: `cortex-iteration-152`
Base: preserved dirty Iteration 151 state; no commit or push.

## Preservation

The Iteration 151 worktree was preserved with a new branch. Existing unrelated
changes and untracked evidence were not reset, cleaned, stashed, or overwritten.

## Root cause

The layer-3 graph was not absent. Source and runtime tracing show that
`llama_model_qwen4exp::graph::build_layer_ffn(cur, 3)` calls
`llm_graph_context::build_moe_ffn()` with valid layer-3 gate, up, and down
canonical tensors. The earlier run had no unambiguous layer field in the
callback configuration and used a descriptor index containing only layer 3
while the callback implementation still passed hard-coded layer 2 to the
provider. Consequently, the real provider path was not proven for layer 3 and
the earlier output change was not attributable.

The correction adds an explicit `layer` field to the provider-local callback
configuration and passes the physical layer to
`flashnext_execute_selected_projection_rows_layer()`. The streaming condition
is deliberately limited to layers 2 and 3. No quantization, routing,
aggregation, or conventional fallback arithmetic was changed.

## Construction and execution evidence

Trace flag:

```text
CORTEX_MOE_TRACE_LAYERS=1
```

The trace records for layer 3 show:

```text
CORTEX_MOE_TRACE qwen4exp_build_layer_ffn layer=3 ...
CORTEX_MOE_TRACE construction layer=3 ... gate=... up=... down=... gate_up=(nil) stream_gate=1 stream_up_down=1
```

The same real graph transaction then reports:

```text
CORTEX_MOE_STREAM result=PASS layer=3 projection=0 tokens=9 ids=10 output_bytes=230400
CORTEX_MOE_STREAM result=PASS layer=3 projection=1 tokens=9 ids=10 output_bytes=230400
CORTEX_MOE_STREAM result=PASS layer=3 projection=2 tokens=9 ids=10 output_bytes=921600
```

Layer 2 also reports all three PASS records in the same run. The trace log is
`/tmp/cortex-iter152-run.log`.

Layer-3 graph reservation was:

```text
CPU compute buffer size = 3.45 MiB
graph: nodes = 173, splits = 1, input objects = 6, input tensors = 13
```

This confirms the shape-tensor path is active for layer 3 and avoids the
full-sized canonical expert leaves in that context.

## Numerical tests

Layer-aware deterministic provider test:

```text
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  /tmp/cortex-iter151-layer3-index.json 3
```

Exit: 0.

```text
one token:   max_abs=3.19351e-6, RMS=7.9655e-7
three token: max_abs=3.86965e-6, RMS=9.53667e-7
```

Layer-2 regression remained passing:

```text
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  flashnext-tensor-index.json
```

Exit: 0, with the established Iteration 150 one- and three-token bounds.

## Real runner

Command:

```text
printf 'hello\n/exit\n' | env CORTEX_MOE_TRACE_LAYERS=1 \
  CORTEX_MOE_STREAM_GATE=1 CORTEX_MOE_STREAM_UP_DOWN=1 \
  CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter151-layer2-3-index.json \
  runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
  /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1
```

Exit: 0. Output: token `248046`, logit `10.7801`, scratch `2,150,400` bytes.
The trace-enabled run proves both layer-2 and layer-3 callbacks execute.

The independently monitored equivalent run exited 0 with:

```text
Peak VmRSS: 5,874,712 kB
VmHWM:      5,874,712 kB
Peak VmSwap: 517,332 kB
```

Scheduler reservations still include conventional contexts at approximately
`1308.27 MiB` and `1301.89 MiB`; the layer-3 streamed context is `3.45 MiB`.
Whole-process RSS reduction is not claimed because the remaining prompt
contexts still allocate full expert leaves.

## Status

| Stage | Status | Evidence |
|---|---|---|
| Preserve Iteration 151 | PASS | isolated branch, dirty state retained |
| Layer-3 graph construction | PASS | Qwen4Exp and generic MoE traces |
| Physical layer ID propagation | PASS | callback records layer 3 |
| Layer-3 callback registration/invocation | PASS | gate/up/down PASS records |
| Layer-2 regression | PASS | provider exit 0 and tolerance bounds |
| Layer-3 numerical provider validation | PASS | max abs ≤ `3.87e-6`, RMS ≤ `9.54e-7` |
| Layer-3 graph reservation reduction | PASS | `3.45 MiB` streamed context |
| Whole-run scheduler reduction | FAIL | other contexts remain 1300 MiB-class |
| Whole-process RSS reduction | NOT DEMONSTRATED | peak remains ~5.87 GiB |
| Full-model streaming inference | NOT TESTED | only layers 2 and 3 are streamed |
| `git diff --check` | PASS | clean |

## Next action

The missing-callback boundary is resolved for layer 3. The next safe milestone
is to compare the layer-2/layer-3 real-runner output against an unchanged
baseline with identical state and prompt, then build per-layer descriptor
coverage before enabling any further layers. Conventional layers remain
unchanged.
