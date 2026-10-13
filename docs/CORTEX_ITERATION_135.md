# CORTEX Stream Forge — Iteration 135

## Execution

Verified the provider-local match flag is present in the source and rebuilt
library. The bounded real-GGUF runner was executed with:

```text
timeout --kill-after=5s 90s bash -c "printf 'hello\\n/exit\\n' | env \
 CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE=1 CORTEX_MOE_STREAM_GATE=1 \
 CORTEX_MOE_STREAM_UP_DOWN=1 CORTEX_MOE_BOUNDARY_CAPTURE=1 \
 CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
 runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
 /home/user/models/qwen38-flashnext-abliterated/\
 Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

Exit code: `0`. Gate, up and down callbacks completed. The runner returned
token `248046`, logit `10.8127`, and reported 2,150,400-byte scratch.

## Capture result

Every provider diagnostic record still reported `capture_once=0`; therefore the
fresh run did not produce a valid nonzero operand fixture. The files were
timestamped during this run, but their existence is not sufficient evidence:
the capture predicate did not execute.

The remaining boundary is now narrower than the previous cross-graph pointer
issue: the provider-local diagnostic flag reaches provider code, but the CPU
compute parameters either do not carry the target or the selected stock
`MUL_MAT` node does not enter the instrumented chunk predicate. No numerical
comparison was performed from these files.

## Status

RUN COMPLETED: PASS
GATE CAPTURE: FAIL
UP CAPTURE: FAIL
DOWN CAPTURE: FAIL
Q8_K BYTE EQUIVALENCE: NOT TESTED
Q8_0 BYTE EQUIVALENCE: NOT TESTED
INDEPENDENT VEC-DOT: NOT TESTED
FULL LAYER-2 NUMERICAL VALIDATION: NOT TESTED
BOUNDED MEMORY: PASS
FULL MODEL STREAMING: NOT TESTED

Next action: instrument the CPU backend setter/compute-plan propagation and the
entry of `ggml_compute_forward_mul_mat_one_chunk()` to prove whether the
provider target is lost before dispatch or the operation is using another CPU
path. Do not interpret the current fixture files as valid operand captures.

`git diff --check`: PASS.
