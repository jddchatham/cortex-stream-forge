
## Iteration 174 exact reconstruction

A new isolated branch `cortex-iteration-174` preserved the dirty Iteration 173
state. No production code, routing, quantization kernel, or output placement
was changed.

The target coordinate was token 1, selected-expert slot 9, physical expert
257, down output row 0. The oracle resolved descriptor offset
`35603219744` and row length 360 bytes. Its down input began at logical offset
12160 in the `[token, expert-slot, channel]` reference representation.

The corrected independent Q8_0 encoder now uses round-to-nearest-even FP16
scale conversion. The previous truncating conversion produced a scale-byte
mismatch. The exact target comparison is:

```text
oracle bytes = 680
CPU bytes    = 680
byte mismatches = 0
first mismatch = none
oracle/cpu FNV = f449ac15fba07731
oracle scale = 0x164b
CPU scale    = 0x164b
```

Target scalar results:

| Quantity | Value |
|---|---:|
| CPU dot | 0.006443969905376434 |
| Independent scalar from exact CPU bytes | 0.006443967811833318 |
| Oracle from byte-identical Q8_0 reconstruction | 0.006443971768021584 |
| Oracle vs CPU absolute error | 1.86265e-09 |
| Oracle vs scalar absolute error | 3.95619e-09 |

This proves the remaining target discrepancy was oracle FP16 rounding, not a
runtime arithmetic or weight-row defect.

## Fresh full oracle result

Command:

```sh
CORTEX_ORACLE_TRACE_TARGET=1 \
 runtime-flashnext-layer-exp-build/bin/llama-flashnext-l2-moe-reference \
 /tmp/cortex-iter153-layer2-4-index.json \
 /tmp/cortex-iter156-l4-gate-activation.bin \
 /tmp/cortex-iter156-l4-gate-ids.bin \
 /tmp/cortex-iter156-l4-up-output.bin \
 /tmp/cortex-iter156-l4-down-output.bin \
 /tmp/cortex-iter156-l4-out.bin 4
```

Exit code: 0. Results:

```text
gate      n=57600  max_abs=5.24521e-06 rms=4.10268e-07 mismatches=0
up         n=57600  max_abs=4.29153e-06 rms=3.47447e-07 mismatches=0
down      n=230400 max_abs=1.56615e-04 rms=5.06390e-06 mismatches=7 first=105986
aggregate n=23040  max_abs=2.95072e-01 rms=1.88967e-02 mismatches=21963 first=0
```

The target down coordinate is corrected, but seven other down elements still
exceed the current `1e-4` diagnostic mismatch threshold. The aggregate remains
unaccepted; its failure cannot yet be attributed solely to those seven down
entries and requires coordinate-level routing/output comparison.

## Status

| Stage | Status |
|---|---|
| FP16 decoder/scale correction | PASS |
| Target 680-byte Q8_0 identity | PASS |
| Target scalar Path A | PASS |
| Target scalar Path B | PASS |
| Gate | PASS |
| Up | PASS |
| Complete down | FAIL (7 residual coordinates) |
| Routed aggregate | FAIL / not accepted |
| Provider regression | NOT RERUN in this iteration |
| Real inference | NOT RERUN in this iteration |
| Bounded memory | NOT TESTED |
| Full layer-4 equivalence | NOT COMPLETE |

`git diff --check`: PASS.

## Next action

Emit the seven failing down coordinates with token, expert slot, physical
expert, output channel, canonical row offset, CPU Q8 hash, and scalar dot.
Compare each against keyed CPU records. This is the smallest discriminating
experiment before touching routed aggregation.
