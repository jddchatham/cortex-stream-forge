# Cortex Stream Forge — Iteration 126

## Scope

This iteration extends the test-gated bounded layer-2 MoE path from gate to
up and down projections. Canonical full expert tensors remain metadata-only;
no undersized buffer is attached to them.

The provider now exposes a projection-generic bounded row path. It selects
the real descriptor and quantization by projection:

- gate: IQ4_XS, `[640, 2560]` rows;
- up: IQ4_XS, `[640, 2560]` rows;
- down: IQ4_NL, `[2560, 640]` rows.

The layer-2 graph uses test-only `GGML_OP_MAP_CUSTOM3` nodes at the original
up and down boundaries. Their dependencies are the actual graph activation
and selected-ID tensors. Down gathers the per-expert post-SwiGLU activation
from `[640,10,n_tokens]`; gate and up consume the shared `[2560,1,n_tokens]`
activation. Each callback uses one task and a bounded 16 KiB tile arena.

## Verification

Checkout and existing worktree changes were preserved. No reset, clean,
stash, commit, or push was performed.

Build:

```sh
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-qwen38-lowres-chat -j2
```

Build exit code: `0`.

Gate-only regression:

```sh
timeout --kill-after=5s 60s bash -c \
  "printf 'hello\\n/exit\\n' | env CORTEX_MOE_STREAM_GATE=1 \
   CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
   runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
   /home/user/models/qwen38-flashnext-abliterated/\
   Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

Result: exit `0`, gate callback PASS, output shape `[640,10,9,1]`, 230400
bytes, token `248046`, logit `10.9936`.

Gate/up/down execution:

```sh
timeout --kill-after=5s 90s bash -c \
  "printf 'hello\\n/exit\\n' | env CORTEX_MOE_STREAM_GATE=1 \
   CORTEX_MOE_STREAM_UP_DOWN=1 \
   CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
   runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
   /home/user/models/qwen38-flashnext-abliterated/\
   Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

All three callbacks executed and the process exited `0`:

```text
CORTEX_MOE_STREAM result=PASS layer=2 projection=0 tokens=9 ids=10 output_bytes=230400
CORTEX_MOE_STREAM result=PASS layer=2 projection=1 tokens=9 ids=10 output_bytes=230400
CORTEX_MOE_STREAM result=PASS layer=2 projection=2 tokens=9 ids=10 output_bytes=921600
```

The down output is `[2560,10,9,1]` and 921600 bytes. The run read the real
GGUF data through bounded PREAD and retained the 2,150,400-byte projector
scratch measurement. The final projection returned token `248046`, logit
`10.8127`.

## Interpretation

The gate-only result remains consistent with Iteration 125. Enabling up and
down changes the final logit from the old zero-canonical path's `10.9936` to
`10.8127`. This is expected to be a consequence of replacing additional
zero-filled canonical leaves with real expert weights, but it is not by itself
proof of numerical equivalence.

The first unresolved comparison boundary is the up/down output fixture. A
projection-level independent reference must be captured from the same graph
transaction and compared element-by-element before accepting this path as
equivalent. No production arithmetic or quantization kernel was changed.

## Evidence

- Build: `/tmp/cortex-iter125-build-final.log` (rebuilt target)
- Gate run: `/tmp/cortex-iter126-gate.log`
- Gate/up/down run: `/tmp/cortex-iter126-updown.log`
- Existing real inputs: `/tmp/cortex-iter120-l2-gate-activation.bin`,
  `/tmp/cortex-iter120-l2-gate-ids.bin`
- Existing Iteration 125 documentation and gate fixture remain unchanged.

`git diff --check` passes. Existing unrelated worktree modifications remain
uncommitted. This iteration does not claim full end-to-end streamed inference
or readiness for other MoE layers until the up/down numerical comparisons
are completed.

## Next action

Add synchronized test-only captures for the real layer-2 up and down result
tensors, then compare them against independently constructed real-weight
IQ4_XS and IQ4_NL references. If the first mismatch is in activation layout,
expert ordering, or quantization dispatch, correct only that bounded provider
boundary before extending beyond layer 2.
