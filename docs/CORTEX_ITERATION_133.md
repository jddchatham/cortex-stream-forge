# CORTEX Stream Forge — Iteration 133

## Executed command

```text
timeout --kill-after=5s 120s bash -c "printf 'hello\\n/exit\\n' | env \
 CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE=1 CORTEX_MOE_STREAM_GATE=1 \
 CORTEX_MOE_STREAM_UP_DOWN=1 CORTEX_MOE_BOUNDARY_CAPTURE=1 \
 CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
 runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
 /home/user/models/qwen38-flashnext-abliterated/\
 Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

Build exit code: `0`. Runner exit code: `0`.

Observed real execution:

```text
CORTEX_MOE_STREAM projection=0 PASS
CORTEX_MOE_STREAM projection=1 PASS
CORTEX_MOE_STREAM projection=2 PASS
CORTEX_CHAT TOKEN position=8 id=248046 logit=10.8127
```

## Capture result

The rebuilt provider emitted capture records, but every record reported:

```text
CPU_OPERAND_CAPTURE type=20 count=0 weight_bytes=136 activation_bytes=34 dot=0
```

`capture_once=0` proves the target predicate did not match the executed CPU
destination node. The existing files are therefore not accepted as production
operand fixtures. Earlier files contained a zero-only first down block and are
historical evidence only.

The provider-created CPU backend is now configured with the diagnostic target;
the remaining mismatch is the destination identity/boundary between the
provider graph and the stock CPU `ggml_compute_forward_mul_mat_one_chunk()`.
No byte-equivalence or independent vec-dot claim is made.

## Status

NONZERO CPU CAPTURE: FAIL
Q8_K BYTE EQUIVALENCE: FAIL
Q8_0 BYTE EQUIVALENCE: FAIL
IQ4_XS/Q8_K VEC-DOT: FAIL
IQ4_NL/Q8_0 VEC-DOT: FAIL
GATE PROJECTION: FAIL
UP PROJECTION: FAIL
DOWN PROJECTION: FAIL
FINAL AGGREGATION: FAIL
BOUNDED MEMORY: PASS
FULL MODEL STREAMING: NOT TESTED

Earliest proven divergence: the diagnostic target does not match the actual
stock CPU destination, so no trustworthy production quantized block or dot
result exists yet. Next action: capture by operation/source identity at the
CPU dispatch entry (or pass the diagnostic target through the provider graph)
instead of relying on `target == dst` pointer identity.

`git diff --check`: PASS.
