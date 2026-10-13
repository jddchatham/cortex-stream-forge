# Cortex Iteration 165 — Provider-to-CPU Activation Transfer Forensics

## Scope

This iteration traced a real layer-4 provider invocation through staging,
`ggml_backend_tensor_set`, backend readback, and CPU `MUL_MAT` entry. No
quantization arithmetic or routing rule was changed.

## Preservation and build

- Branch: `cortex-iteration-165`
- Previous worktree changes were preserved; no reset, clean, stash, commit, or push was performed.
- Build command: `cmake --build runtime-flashnext-layer-exp-build --target llama-qwen38-lowres-chat -j4`
- Build result: PASS (exit 0)
- Diagnostic run log: `/tmp/cortex-iter165-run2.log`
- Build log: `/tmp/cortex-iter165-build.log`
- `git diff --check`: PASS (exit 0)

## Diagnostic boundary

Provider tensor names are keyed as
`cortex_moe_l<layer>_p<projection>_t<token>_e<expert>_r<row>_x`.
For the examined layer-4 token-1 gate/up invocation, the trace recorded the
same logical identity at all four available boundaries:

| Boundary | Gate/up token-1 evidence |
|---|---|
| Provider staging | F32 `[2560,1]`, `nb=[4,10240]`, nonzero `2560/2560`, finite `2560/2560`; gate first/last `-0.021405/0.29972` |
| After `ggml_backend_tensor_set` readback | Same shape/stride and same nonzero/finite counts and first/last values |
| Before graph submission readback | Same shape/stride and same nonzero/finite counts and first/last values |
| CPU `MUL_MAT` source | Same tensor name, `ne=[2560,1,1,1]`, `nb=[4,10240,10240,10240]`, nonzero `2560`, finite `2560`, first/last `-0.021405/0.29972` |

The corresponding up invocation showed the same token-1 statistics. The
trace also showed the token-1 down input was nonzero after the correction:
`640/640` nonzero, first `-0.11674`, last `-0.0132519`.

The trace used supported backend readback after `tensor_set` and before graph
submission. CPU source metadata was emitted only when the diagnostic flag was
enabled and was guarded by F32 data validity. No expired graph-local view was
read.

## First verified data-loss boundary

The original provider input layout is token-major with ten selected-expert
slots: `token * KSEL * hidden`. However, the non-down projection loop passed
`activation + t * input_width` to every token. For token 1, that address points
into token 0's unused selected-expert-slot tail. Those slots were zero-filled.

This explains the earlier evidence directly:

- Provider callback input assembly reported token 1 nonzero.
- The first token-1 gate/up provider tile used zero input.
- The CPU boundary consequently reported `src_nonzero=0` for those calls.
- `tensor_set` was not the point of loss; its immediate readback was already
  the expected nonzero row.

The smallest correction was to use `t * KSEL * input_width` for gate/up,
while retaining `(t*KSEL+s)*input_width` for down. This is an indexing/binding
correction, not a change to arithmetic, quantization, routing, or expert data.

## Executed correction

File changed:
`runtime-flashnext-layer-exp/src/flashnext-selected-moe.cpp`

The corrected offset is selected before calling `tiled_matvec`; the provider
now binds each gate/up token to the populated token block. The correction is
test/runtime-path code already controlled by the selected-MoE streaming path;
conventional fallback remains unchanged.

## Real runner

Command used the existing model, index, layer selection, and bounded capture
flags:

```sh
timeout --kill-after=5s 120s bash -lc "printf 'hello\\n/exit\\n' | env \
  CORTEX_FLASHNEXT_TRANSFER_TRACE=1 \
  CORTEX_FLASHNEXT_TOKEN_Q8_CAPTURE=1 \
  CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE=1 \
  CORTEX_MOE_STREAM_LAYERS=2,3,4 \
  CORTEX_MOE_STREAM_GATE=1 \
  CORTEX_MOE_STREAM_UP_DOWN=1 \
  CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter153-layer2-4-index.json \
  runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
  /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

- Runner exit: PASS (0)
- Output token: `248046`
- Reported selected logit: `13.0355`
- Provider trace: layer-4 token 1 gate/up are nonzero at staging, readback,
  pre-submit, and CPU source.
- Full layer-4 numerical equivalence: NOT TESTED in this iteration.
- Per-boundary SHA-256: NOT TESTED; this diagnostic emitted immutable
  identity/statistics but did not persist full snapshots at every boundary.

## Regression

Command:

```sh
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  /tmp/cortex-iter153-layer2-4-index.json 4
```

Result: PASS (exit 0).

- Three-token max absolute error: `3.4757e-06`
- Three-token RMS: `9.24548e-07`
- Scratch remained `2,150,400` bytes in the real runner evidence.

## Status

| Item | Status |
|---|---|
| Matched provider/CPU invocation identity | PASS |
| `tensor_set` offset/length transfer | PASS, by immediate backend readback |
| Token-1 gate/up CPU source restored | PASS |
| Token-1 down input nonzero | PASS |
| Token-1 Q8 capture | NOT TESTED / not claimed from this trace |
| Layer-4 oracle equivalence | NOT TESTED |
| Provider regression | PASS |
| Bounded scratch | PASS |
| Full-model streaming inference | NOT TESTED |

## Next action

Persist full immutable token-0/token-1 F32 and Q8 snapshots keyed by the
provider invocation after this corrected transfer, then run the independent
layer-4 oracle comparison. The transfer defect is proven and corrected; no
further transfer hypothesis should be investigated before that numerical
comparison.
