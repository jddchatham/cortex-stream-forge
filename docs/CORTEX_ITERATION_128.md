# Cortex Stream Forge — Iteration 128

## Status

The standalone scalar decoder executes directly against selected real GGUF
rows without calling the production provider dequantizers. It implements the
official IQ4_XS and IQ4_NL block layouts and a separate Q8_K activation
round-trip.

The decoder build and run both exit `0`, but projection equivalence fails.

## Command

```sh
timeout --kill-after=5s 90s \
  runtime-flashnext-layer-exp-build/bin/llama-flashnext-l2-moe-reference \
  /tmp/cortex-iter120-layer2-index.json \
  /tmp/cortex-iter120-l2-gate-activation.bin \
  /tmp/cortex-iter120-l2-gate-ids.bin \
  /tmp/cortex-iter127-l2-up-output.bin \
  /tmp/cortex-iter127-l2-down-output.bin \
  /tmp/cortex-iter127-l2-out.bin
```

Reference output, tolerance `1e-4` absolute:

```text
gate      n=57600  max_abs=1.94815  rms=0.304924  mismatches=51197  first=819
up        n=57600  max_abs=1.69971  rms=0.269166  mismatches=51189  first=1253
down      n=230400 max_abs=0.764412 rms=0.0117262 mismatches=227108 first=0
aggregate n=23040  max_abs=0.152755 rms=0.00640988 mismatches=22358 first=0
```

All values were finite. The decoder's inspected nonzero IQ4_XS row matched
the official GGML block interpretation exactly; IQ4_NL uses the official
32-value block and nonlinear lookup table. The reference reads one selected
row at a time and does not materialize expert tensors.

## Finding

The earliest divergence is the gate projection. Routing IDs and final
aggregation indexing are valid. The independent scalar block decoding is not
the demonstrated defect. The remaining discrepancy is the exact CPU
IQ4_XS/IQ4_NL vec-dot activation packing/order; the simple Q8_K
round-trip does not reproduce the production kernel closely enough.

The separate down-output times routing-weight calculation still matches the
production aggregation operation itself with max error `1.80e-8`, RMS
`6.64e-10`, and zero elements above `1e-6`. No production arithmetic was
changed.

Evidence:

- `/tmp/cortex-iter128-finalbuild.log`
- `/tmp/cortex-iter128-reference.log`
- `/tmp/cortex-iter127-l2-*.bin`
- `git diff --check`: PASS

INDEPENDENT IQ4_XS DECODING: PASS
INDEPENDENT IQ4_NL DECODING: PASS
GATE PROJECTION: FAIL
UP PROJECTION: FAIL
DOWN PROJECTION: FAIL
FINAL MoE AGGREGATION: FAIL
BOUNDED MEMORY: PASS
FULL MODEL STREAMING: NOT TESTED
NEXT REQUIRED ACTION: Reproduce the exact CPU IQ4_XS/IQ4_NL vec-dot activation packing and ordering in the independent scalar reference, then rerun all projection comparisons.
