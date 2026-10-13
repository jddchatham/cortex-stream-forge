# CORTEX Stream Forge — Iteration 136

## Verified execution

The provider callback and provider-owned backend graph execute successfully.
The runner reaches the real layer-2 gate/up/down streamed operations and returns
token `248046`, logit `10.8127`, exit code `0`.

The linked objects are the experimental checkout's own libraries:

```text
libllama.so.0       runtime-flashnext-layer-exp-build/bin/libllama.so.0
libggml.so.0        runtime-flashnext-layer-exp-build/bin/libggml.so.0
libggml-cpu.so.0    runtime-flashnext-layer-exp-build/bin/libggml-cpu.so.0
```

## Dispatch finding

The expected stock path is present in source:

```text
provider tiled_matvec
  -> ggml_backend_graph_compute
  -> CPU GGML_OP_MUL_MAT dispatch
  -> ggml_compute_forward_mul_mat
  -> ggml_compute_forward_mul_mat_one_chunk
  -> type_traits_cpu[type].vec_dot
```

The provider-local diagnostic target is attached to the provider-created CPU
backend and its provider-local match flag is compiled into the CPU library.
Nevertheless the bounded run reports `capture_once=0` for every provider
operation. No trustworthy CPU operand or dot fixture was produced.

This rules out the earlier outer-context backend attachment as the sole issue,
but does not yet prove that the one-chunk path is the path actually used for
the provider graph. The next discriminating checkpoint must be at
`ggml_compute_forward_mul_mat()` entry, before any target or destination test,
with a bounded provider-operation counter.

## Command

```text
timeout --kill-after=5s 90s bash -c "printf 'hello\\n/exit\\n' | env \
 CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE=1 CORTEX_MOE_STREAM_GATE=1 \
 CORTEX_MOE_STREAM_UP_DOWN=1 CORTEX_MOE_BOUNDARY_CAPTURE=1 \
 CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
 runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
 /home/user/models/qwen38-flashnext-abliterated/\
 Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

Exit code: `0`; `git diff --check`: PASS.

## Status

PROVIDER CALLBACK: PASS
CPU COMPUTE PLAN: PASS
CPU MUL_MAT DISPATCH: NOT PROVEN
CPU CHUNK ENTRY: NOT PROVEN
STOCK VEC-DOT REACHED: NOT PROVEN
DIAGNOSTIC CONTEXT PROPAGATION: FAIL
NONZERO GATE CAPTURE: FAIL
NONZERO UP CAPTURE: FAIL
NONZERO DOWN CAPTURE: FAIL
BOUNDED MEMORY: PASS
FULL MODEL STREAMING: NOT TESTED

Earliest actual failure boundary: after provider backend attachment and before
the diagnostic reaches a confirmed CPU `MUL_MAT`/chunk checkpoint. Next action:
instrument the CPU dispatch entry and plan propagation, not another pointer
identity predicate.
