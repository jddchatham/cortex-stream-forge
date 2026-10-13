# Cortex Iteration 140: atomic full-row dot records

## Execution

The bounded real-GGUF layer-2 workload built and exited 0. The run log is
`/tmp/cortex-iter140-run.log`; scratch remained 2,150,400 bytes.

## Atomic format

`/tmp/cortex-iter140-projection-{0,1,2}.bin` contains one record with magic,
version, projection ID, GGML weight type, width, complete weight-row bytes,
complete quantized activation-row bytes, F32 count, the completed dot, and the
three complete payloads. A process-local atomic acceptance bit prevents later
calls from replacing an accepted projection record.

Records emitted:

| Projection | Format | Record | CPU dot |
|---|---|---|---:|
| gate (0) | IQ4_XS/Q8_K | present, 14576 bytes | -0.2227051705 |
| up (1) | IQ4_XS/Q8_K | missing | n/a |
| down (2) | IQ4_NL/Q8_0 | present, 3656 bytes | -0.08832758665 |

The up call did not satisfy the diagnostic's nonzero-row acceptance condition;
it is explicitly not reported as a pass.

## Independent scalar comparison

The down record was independently decoded from its complete 360-byte IQ4_NL
row and 680-byte Q8_0 activation row. Scalar result: `-0.08832757411`;
absolute error `1.25e-8`, relative error `1.42e-7`. Full-row PASS.

The gate record was independently decoded from its complete 1360-byte IQ4_XS
row and 2920-byte Q8_K activation row. Scalar result: `0.6508514182` versus
CPU `-0.2227051705`; absolute error `0.873556589`. FAIL. The remaining issue
is in the independent IQ4_XS block layout/scale/order calculation. Production
CPU arithmetic was not modified.

## Status

ATOMIC CAPTURE: FAIL (gate/down present; up missing)
GATE FULL-ROW DOT: FAIL
UP FULL-ROW DOT: NOT TESTED
DOWN FULL-ROW DOT: PASS
FULL GATE PROJECTION: NOT TESTED
FULL UP PROJECTION: NOT TESTED
FULL DOWN PROJECTION: NOT TESTED
FINAL AGGREGATION: NOT TESTED
BOUNDED MEMORY: PASS
FULL MODEL STREAMING: NOT TESTED

`git diff --check` passed. Next: correct the independent IQ4_XS scalar decoder
and make the up acceptance identity explicit.
