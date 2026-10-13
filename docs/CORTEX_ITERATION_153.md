# Cortex Iteration 153 — Descriptor-Driven Multilayer MoE Streaming

Date: 2026-10-13
Branch: `cortex-iteration-153`
Base: preserved dirty Iteration 152 state; no commit or push.

## Descriptor registry

The existing JSON tensor-index format is now used as a physical-layer registry.
Each entry is keyed by the `blk.<layer>` tensor name and carries the shard,
absolute payload offset, byte count, GGML type, and shape. Provider resolution
still rejects missing or incompatible descriptors before PREAD execution.

The graph streaming selection is controlled by `CORTEX_MOE_STREAM_LAYERS` and
defaults to the previously validated layers `2,3`. This iteration tested
`2,3,4`; conventional paths remain unchanged for all other layers. The
callback carries the selected physical layer into the provider rather than
using a hardcoded layer number.

## Layer-4 descriptor evidence

| Tensor | Type | Shape | Offset | Bytes | Row bytes |
|---|---|---|---:|---:|---:|
| `blk.4.ffn_gate_exps.weight` | IQ4_XS | `[2560,640,512]` | 35,839,149,344 | 445,644,800 | 1,360 |
| `blk.4.ffn_up_exps.weight` | IQ4_XS | `[2560,640,512]` | 36,290,917,664 | 445,644,800 | 1,360 |
| `blk.4.ffn_down_exps.weight` | IQ4_NL | `[640,2560,512]` | 35,366,368,544 | 471,859,200 | 360 |

The three descriptors were read from the real GGUF and added to the combined
test registry `/tmp/cortex-iter153-layer2-4-index.json`. The bounded tensor
window tests for all three layer-4 tensors report `TENSOR-WINDOW PROOF: PASS`
with `max_abs=0` and `max_rel=0`.

## Independent layer-4 provider validation

```text
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  /tmp/cortex-iter153-layer2-4-index.json 4
```

Exit: 0.

```text
one token:   max_abs=3.19351e-6, RMS=7.9655e-7
three token: max_abs=3.86965e-6, RMS=9.53667e-7
```

These satisfy the established limits `max_abs <= 1e-5` and `RMS <= 2e-6`.

## Real graph evidence

Command:

```text
printf 'hello\n/exit\n' | env CORTEX_MOE_TRACE_LAYERS=1 \
  CORTEX_MOE_STREAM_LAYERS=2,3,4 \
  CORTEX_MOE_STREAM_GATE=1 CORTEX_MOE_STREAM_UP_DOWN=1 \
  CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter153-layer2-4-index.json \
  runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
  /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1
```

Exit: 0. The trace contains three PASS callbacks each for layers 2, 3, and 4:

```text
layer=2: projection 0/1/2, output bytes 230400/230400/921600
layer=3: projection 0/1/2, output bytes 230400/230400/921600
layer=4: projection 0/1/2, output bytes 230400/230400/921600
```

Layer-4 streamed graph reservation is `9.59 MiB`; the trace log is
`/tmp/cortex-iter153-run.log`. Remaining conventional contexts reserve
approximately `1301.89–1308.27 MiB`.

## Cross-layer result

The real runner returns token `248046` but logit `11.2143`. The prior layer-2/3
run returned logit `10.7801`. Since this is a material logit change and no
unchanged conventional reference was executed with identical graph/state
inputs, cross-layer numerical equivalence is not established. Matching token
IDs are insufficient evidence. The changed logit is recorded as a validation
failure requiring investigation before enabling further layers.

## Memory

The monitored layer-2/3/4 run exited 0:

```text
Peak VmRSS: 5,873,552 kB
VmHWM:      5,873,552 kB
Peak VmSwap: 0 kB
```

No whole-process RSS reduction is claimed. The large conventional prompt
contexts remain present, and the no-swap result is an environmental observation
rather than proof that the scheduler reservation was eliminated.

## Status

| Requirement | Status | Evidence |
|---|---|---|
| Preserve Iteration 152 | PASS | isolated branch and dirty work retained |
| Descriptor-driven layer selection | PASS | `CORTEX_MOE_STREAM_LAYERS`, per-layer registry entries |
| Layer-4 descriptor bounds/type validation | PASS | real GGUF metadata and tiled proof |
| Layer-4 independent provider regression | PASS | max abs `3.19e-6`, RMS `7.97e-7` |
| Layer-2/3/4 callback execution | PASS | nine callback PASS records |
| Layer-4 streamed reservation | PASS | `9.59 MiB` local context |
| Cross-layer logit equivalence | FAIL | logit changed to `11.2143` |
| Whole-run scheduler reduction | NOT DEMONSTRATED | other contexts remain large |
| Whole-process RSS reduction | NOT DEMONSTRATED | no comparable reduction established |
| Full-model streaming inference | NOT TESTED | only layers 2–4 streamed |
| `git diff --check` | PASS | clean |

## Next action

Compare the conventional and layers-2/3/4 streamed graphs at the first layer-4
boundary using identical captured activation, expert IDs, routing weights, and
state. The first divergent layer must be identified before enabling another
layer or claiming multilayer end-to-end equivalence.
