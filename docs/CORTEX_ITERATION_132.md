# CORTEX Stream Forge — Iteration 132

## Execution

The existing context-level diagnostic initially produced no CPU operand
fixtures because the bounded provider creates a separate CPU backend. The
provider was therefore minimally wired to attach the existing diagnostic target
to its own backend and emit one-shot operand files.

Command:

```text
timeout --kill-after=5s 120s bash -c "printf 'hello\\n/exit\\n' | env \
 CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE=1 CORTEX_MOE_STREAM_GATE=1 \
 CORTEX_MOE_STREAM_UP_DOWN=1 CORTEX_MOE_BOUNDARY_CAPTURE=1 \
 CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
 runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
 /home/user/models/qwen38-flashnext-abliterated/\
 Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

The real runner exited `0`, completed all three streamed projections, and
returned token `248046`, logit `10.8127`. Capture records were emitted by the
provider's CPU backend. Initial records selected a zero first down block; the
diagnostic was then tightened to select the first nonzero activation block and
preserve its corresponding weight offset. The rebuilt runner succeeded with
`git diff --check` passing.

Fixtures emitted by the first capture run:

```text
/tmp/cortex-iter132-iq4xs-weight.bin 136 bytes
/tmp/cortex-iter132-q8k.bin          292 bytes
/tmp/cortex-iter132-f32-xs.bin       1024 bytes
/tmp/cortex-iter132-iq4nl-weight.bin 136 bytes (18-byte IQ4_NL block prefix)
/tmp/cortex-iter132-q80.bin           34 bytes
/tmp/cortex-iter132-f32-nl.bin       128 bytes
```

The first captured IQ4_NL block was all-zero activation and therefore produced
dot `0`; it is not accepted as a numerical proof. The corrected nonzero-block
selector is compiled, but a post-correction real run and independent byte/dot
comparison remain pending.

## Status

CPU CAPTURE EXECUTED: PASS
Q8_K ACTIVATION BYTES: FAIL (nonzero post-correction comparison pending)
Q8_0 ACTIVATION BYTES: FAIL (initial captured block was zero)
INDEPENDENT VEC-DOT: FAIL
GATE PROJECTION: FAIL
UP PROJECTION: FAIL
DOWN PROJECTION: FAIL
FINAL AGGREGATION: FAIL
BOUNDED MEMORY: PASS
FULL MODEL STREAMING: NOT TESTED

Earliest remaining divergence: nonzero production quantized activation bytes
and one corresponding scalar dot have not yet been compared. Next action: rerun
the rebuilt provider with the nonzero-block selector and compare the emitted
Q8_K/Q8_0 bytes and dot against an independent decoder.
