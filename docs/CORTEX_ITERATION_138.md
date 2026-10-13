# Cortex Iteration 138: CPU MUL_MAT boundary

## Scope

This iteration verified the CPU execution path used by the bounded layer-2
provider and executed fresh provider-local operand captures. Production
arithmetic was not changed.

## Verified runtime path

The trace showed, for both provider projection types:

`MAP_CUSTOM3 layer callback -> flashnext_execute_selected_projection_rows_layer() -> tiled_matvec() -> ggml_mul_mat() -> CPU graph plan -> GGML_OP_MUL_MAT -> ggml_compute_forward_mul_mat_one_chunk() -> vec_dot`

The tiled implementation is probed but does not perform this workload: its
batch eligibility check is false for the provider's one-column input. Runtime
counters from `/tmp/cortex-iter138-run5.log` showed `plan=1`, `mul_mat=1`,
`one_chunk=1`, and `vec_dot=12` for IQ4_XS and 40 or 45 for IQ4_NL. The
selected type traits are IQ4_XS/Q8_K and IQ4_NL/Q8_0.

## Diagnostic correction

The previous capture predicate required `ith == 0`, `ir0_start == 0`, and
`ir1_start == 0`. Those coordinates were not a stable provider identity and
caused `capture_once=0` despite the vec-dot loop executing. The test-only
capture now uses the provider-local diagnostic context and selects the first
block whose F32 activation and weight bytes are both nonzero. Fixture writes
are suppressed when no capture occurred, preventing later zero-only calls from
overwriting a valid capture.

Changed diagnostic files:

- `runtime-flashnext-layer-exp/ggml/include/ggml-cpu.h`
- `runtime-flashnext-layer-exp/ggml/src/ggml-cpu/ggml-cpu.c`
- `runtime-flashnext-layer-exp/ggml/src/ggml-cpu/ggml-cpu.cpp`
- `runtime-flashnext-layer-exp/ggml/src/ggml-cpu/tiled/tiled.cpp`
- `runtime-flashnext-layer-exp/src/flashnext-selected-moe.cpp`

## Execution

Build:

```text
cmake --build runtime-flashnext-layer-exp-build --target llama-qwen38-lowres-chat -j4
```

Run:

```text
timeout --kill-after=5s 100s bash -c "printf 'hello\\n/exit\\n' | env CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE=1 CORTEX_FLASHNEXT_CPU_DISPATCH_TRACE=1 CORTEX_MOE_STREAM_GATE=1 CORTEX_MOE_STREAM_UP_DOWN=1 CORTEX_MOE_BOUNDARY_CAPTURE=1 CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

The run exited 0 and produced token 248046 with logit 10.8127. Fresh log:
`/tmp/cortex-iter138-run5.log`.

## Operand evidence

The fresh captures are:

- IQ4_XS weight block: 136 bytes
- Q8_K activation block: 292 bytes
- IQ4_NL weight block: first 18 bytes of the 136-byte diagnostic buffer
- Q8_0 activation block: 34 bytes
- F32 source blocks: 1024 bytes for IQ4_XS and 128 bytes for IQ4_NL

Independent scalar Q8 encoders reproduced the captured Q8_K and Q8_0 bytes
exactly; both had no differing byte. The captured production source metadata
reported valid source and weight storage. The capture lines include
`count=1`, nonzero activation counts, and completed vec-dot counters.

## Numerical validation boundary

The captured production dot value is a complete row dot, while the fixture
currently contains only one quantization block. Therefore a scalar dot made
from that block alone is not a valid comparison to the reported complete-row
result. No IQ4_XS/Q8_K or IQ4_NL/Q8_0 numerical equivalence claim is made in
this iteration. The next required correction is to capture the complete
selected weight row and complete quantized activation row (or capture a
single-block CPU dot explicitly), then compare like-for-like.

## Status

CPU arithmetic path: PASS.

Fresh nonzero gate/up/down captures: PASS for the provider-local CPU boundary
when the corresponding projection call is reached; the captured files are
block-scoped, not full-row fixtures.

Q8_K and Q8_0 byte equivalence: PASS for the captured blocks.

Independent vec-dot, full layer-2 numerical validation, and full-model
streaming: NOT TESTED/NOT PROVEN. The successful token is not used as a
numerical-equivalence claim.

## Hashes

Executable and libraries:

```text
llama-qwen38-lowres-chat 00da662c12f26b3b362186876c82bab7835b42a2f73c688feb0eaf0aa914fbf9
libllama.so              af3c7d99ab7a339172160345ddb8ae364074b136d91612a8ec0fd82accbda3e5
libggml-cpu.so           c8f246ecb0a7d50295e7d2fc4dee2568a60ec1a026e9e8af0567cd4ee312773c
```

`git diff --check` passed after the diagnostic changes.
