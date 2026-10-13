# Cortex Iteration 151 — Second-Layer External MoE Streaming

Date: 2026-10-13
Branch: `cortex-iteration-151`
Base: preserved dirty Iteration 150 state; no commit or push.

## Preservation

The existing dirty worktree was preserved with `git switch -c
cortex-iteration-151`. No reset, clean, stash, merge, force operation, or
unrelated-file overwrite was performed.

## Layer-3 GGUF descriptor evidence

The model contains compatible layer-3 expert tensors in shard 1:

| Tensor | Type | Shape | Absolute payload offset | Bytes | Row bytes |
|---|---|---|---:|---:|---:|
| `blk.3.ffn_gate_exps.weight` | IQ4_XS (23) | `[2560,640,512]` | 34,431,135,520 | 445,644,800 | 1,360 |
| `blk.3.ffn_up_exps.weight` | IQ4_XS (23) | `[2560,640,512]` | 34,882,903,840 | 445,644,800 | 1,360 |
| `blk.3.ffn_down_exps.weight` | IQ4_NL (20) | `[640,2560,512]` | 33,958,354,720 | 471,859,200 | 360 |

The effective row counts are 327,680, 327,680, and 1,310,720 respectively.
Offsets and sizes were obtained from the actual GGUF metadata and placed in
`/tmp/cortex-iter151-layer3-index.json`.

## Independent layer-3 validation

The bounded tensor-window executable was extended only to accept an explicit
tensor name and correct three-dimensional row count. It reads selected tiles
from GGUF, compares tiled CPU GGML matmul against a full-row reference, and
uses an 8 MiB arena.

Command pattern:

```text
runtime-flashnext-layer-exp-build/bin/llama-qwen38-pread-tensor-window \
  /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf \
  8388608 blk.3.ffn_gate_exps.weight
```

Results for gate, up, and down were all:

```text
TENSOR-WINDOW PROOF: PASS max_abs=0 max_rel=0
```

The layer-aware provider regression was extended to accept a layer argument and
run against the layer-3 index:

```text
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  /tmp/cortex-iter151-layer3-index.json 3
```

Exit: 0.

Results:

```text
one token:   max_abs=3.19351e-6, RMS=7.9655e-7
three token: max_abs=3.86965e-6, RMS=9.53667e-7
```

Both satisfy `max_abs <= 1e-5` and `RMS <= 2e-6`. The tested input is
deterministic, nonzero, and uses real layer-3 GGUF rows, selected expert IDs,
IQ4_XS/Q8_K gate and up paths, and IQ4_NL/Q8_0 down paths.

## Graph integration attempt

The graph callback configuration was extended narrowly from `il == 2` to
`il == 2 || il == 3`, with an explicit layer field passed to the provider.
The combined layer-2/layer-3 descriptor file is:

```text
/tmp/cortex-iter151-layer2-3-index.json
```

The full runner built successfully, but the executed log did not contain the
expected `CORTEX_MOE_STREAM result=PASS layer=3` records. Therefore layer-3
graph integration is not proven. The run did exit after producing token
`248046` with logit `10.7801`, but the changed logit and missing callback
evidence make this an unsuccessful integration validation, not a correctness
claim. No production arithmetic was altered.

The relevant run log is `/tmp/cortex-iter151-run.log`; build logs are
`/tmp/cortex-iter151-build3.log` and `/tmp/cortex-iter151-build2.log`.

## Memory and reservations

The attempted full run still showed approximately `1308.27 MiB` reservations
for conventional contexts and `0.08 MiB` for the final generation context.
Because layer-3 callback execution was not proven and no safe end-to-end
comparison exists, no memory improvement is claimed. Peak RSS/swap was not
repeated for this failed integration attempt.

## Status

| Stage | Status | Evidence |
|---|---|---|
| Layer-3 GGUF metadata and bounds | PASS | three real descriptor records |
| Layer-3 tiled PREAD vs full-row reference | PASS | max abs/rel both zero |
| Layer-3 independent gate/up/down provider | PASS | max abs ≤ `3.87e-6`, RMS ≤ `9.54e-7` |
| Layer-2 regression preservation | PASS | prior Iteration 150 values preserved in source/build |
| Layer-3 graph callback execution | FAIL | no layer-3 callback records |
| Layer-2 + layer-3 end-to-end equivalence | NOT TESTED | missing trustworthy graph captures |
| Scheduler reservation reduction | NOT TESTED | integration was not accepted |
| Whole-process RSS reduction | NOT TESTED | no valid end-to-end comparison |
| Full-model streaming | NOT TESTED | not demonstrated |

## First remaining boundary and next action

The first unresolved boundary is graph construction/execution: the standalone
layer-aware provider receives valid layer-3 descriptors, but the real runner
does not demonstrate that its layer-3 MoE graph reaches the streaming callback.
The next smallest experiment is to add one bounded construction/execution trace
around `build_moe_ffn()` for `il == 3`, including the selected tensor names and
callback graph node, then determine whether layer 3 is skipped, uses a different
MoE representation, or is executed through a separate context. Do not broaden
the streaming condition again until that boundary is identified.

`git diff --check` was run after documentation and source changes.
