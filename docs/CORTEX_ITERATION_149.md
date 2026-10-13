# Cortex Iteration 149 — Safe External Expert Tensor Ownership

Date: 2026-10-13
Branch: `cortex-iteration-149`
Base checkpoint: `71746490e17b7c55eabebfd0bae1472e0e1d7399`

## Scope and preservation

The experiment was performed on an isolated branch. The worktree already contained
unrelated Synthesis diagnostics and generated files; none were reset, cleaned,
staged, committed, or pushed.

The focused change is in `runtime-flashnext-layer-exp/src/llama-graph.cpp`.

## Consumer and ownership audit

The layer-2 Qwen4Exp MoE graph has three canonical expert leaves:

| Projection | GGML tensor | Type | Declared payload |
|---|---|---:|---:|
| gate | `blk.2.ffn_gate_exps.weight` | IQ4_XS | 445,644,800 bytes |
| up | `blk.2.ffn_up_exps.weight` | IQ4_XS | 445,644,800 bytes |
| down | `blk.2.ffn_down_exps.weight` | IQ4_NL | 471,859,200 bytes |

In the streamed path, `cortex_stream_gate_callback()` invokes the bounded
selected-expert PREAD provider. The provider reads only the selected GGUF rows
and writes the output tensor; it does not read canonical expert-leaf storage.

Before this iteration, `build_moe_ffn()` still called `ggml_mul_mat_id()` as a
shape probe for each streamed projection. That retained the canonical full-size
expert leaf in the graph, even though the callback supplied the actual result.
The leaves entered scheduler planning with `data=NULL` and `buffer=NULL`, so
GGML allocated their declared payloads.

## Focused change

For the layer-2 streamed gate, up, and down branches, the shape probe is now an
explicit F32 shape tensor constructed from model metadata:

```cpp
ggml_new_tensor_3d(ctx0, GGML_TYPE_F32,
    projection_weight->ne[1], selected_experts->ne[0], cur->ne[2]);
```

The selected-expert axis is `ne[0]`, matching the observed IDs shape
`[10,9,1,1]` and projection output shapes `[640,10,9,1]` or
`[2560,10,9,1]`. No canonical expert leaf is attached to these three callback
outputs. No fake pointer, undersized tensor buffer, arithmetic change, routing
change, or scheduler validation bypass was added.

## Commands and evidence

Build:

```text
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-qwen38-lowres-chat llama-flashnext-iq4xs-provider-test -j4
```

Result: exit 0. Existing compiler warnings in `llama-graph.cpp` concern older
diagnostic one-line `if` statements and are unrelated to this shape change.

Numerical regression:

```text
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  flashnext-tensor-index.json
```

Result: exit 0.

```text
tokens=1 max_abs=3.09898e-06 rms=7.49719e-07
tokens=3 max_abs=3.67127e-06 rms=8.08196e-07
```

Both satisfy the established limits `max_abs <= 1e-5` and `RMS <= 2e-6`.

Real runner:

```text
printf 'hello\n/exit\n' | env \
  CORTEX_MOE_STREAM_GATE=1 CORTEX_MOE_STREAM_UP_DOWN=1 \
  CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
  runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
  /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1
```

Result: exit 0. The three callbacks reported `result=PASS` for layer 2 and
produced output sizes 230,400, 230,400, and 921,600 bytes. The real response
was token `248046`, logit `10.8127`, with provider scratch `2,150,400` bytes.

Allocator profile:

```text
CORTEX_ALLOC_PROFILE=1 <same runner and environment>
```

Evidence log: `/tmp/cortex-iter149-alloc.log`.

The layer-2 context reservation was reduced to `9.59 MiB`; the full run also
constructs other physical-layer contexts whose reservations remain:

```text
layer-2 streaming context: 9.59 MiB
other prompt contexts:     1308.03–1308.27 MiB
generation context:        0.08 MiB
```

The largest observed scheduler allocation events remain the three full expert
leaves in other contexts. Therefore the total run-level scheduler reservation
did not decrease from the approximately `1308.27 MiB` baseline.

Monitored run evidence: `/tmp/cortex-iter149-rss2.log` and
`/tmp/cortex-iter149-rss2.tsv`.

Observed peak values in this run were approximately:

```text
VmRSS/VmHWM: 5,700,444 / 5,706,596 kB
VmSwap:      1,495,324 kB
```

The sampling environment showed higher residency and swap than the historical
Iteration 147 run (`VmHWM 5,647,288 kB`, process swap `364,436 kB`). This is not
an attributable memory improvement; environmental variation and the existing
multi-context execution path remain unresolved. `/usr/bin/time` was unavailable
in the environment, so `/proc/<pid>/status` sampling was used for the reported
RSS/HWM/swap values.

## Status

| Check | Status | Evidence |
|---|---|---|
| Consumer/ownership audit | PASS | `llama-graph.cpp`, provider path, allocator trace |
| Safe layer-2 non-leaf shape representation | PASS | explicit F32 shape tensors; real callbacks execute |
| Layer-2 streamed gate/up/down execution | PASS | three `CORTEX_MOE_STREAM result=PASS` records |
| One-token numerical regression | PASS | max abs `3.09898e-6`, RMS `7.49719e-7` |
| Three-token numerical regression | PASS | max abs `3.67127e-6`, RMS `8.08196e-7` |
| Layer-2 scheduler reservation reduction | PASS | `1308.27` to `9.59 MiB` for the focused context |
| Whole-run scheduler reservation reduction | FAIL | other contexts still reserve `1308.03–1308.27 MiB` |
| Actual peak RSS reduction | FAIL | measured run peaked at ~`5.70 GiB`, above historical baseline |
| Full-model streaming inference | NOT TESTED | only bounded layer-2 path exercised |

## Conclusion and next action

The experiment proves a safe local graph-dependency reduction: the layer-2
streaming context no longer needs the three canonical expert leaves for its
callback outputs, while numerical regressions remain within tolerance. It does
not yet reduce the total scheduler reservation or resident memory because the
runner builds additional contexts whose graphs still contain full expert leaves.

The smallest next experiment is to apply the same dependency audit selectively
to the remaining prompt-context construction path, beginning with the first
non-layer-2 context that emits the `1308.27 MiB` reservation. Do not change
global scheduler allocation policy until that context's graph consumers and
ownership are proven equivalent.

No commit or push was performed.
