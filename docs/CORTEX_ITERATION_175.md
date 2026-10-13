# Cortex Iteration 175 — Seven Down Residuals and Aggregate Identity

## Preservation

- Branch: `cortex-iteration-175`.
- Dirty Iteration 174 state and unrelated changes preserved.
- No reset, clean, stash, merge, commit, or push performed.
- `git diff --check`: PASS.

## Down residual enumeration

Seven elements exceeded the diagnostic `1e-4` threshold after comparing
runtime `[channel,slot,token]` against oracle `[token,slot,channel]`:

| token | slot | expert | channel | runtime | oracle | abs error |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 252 | 1026 | -0.018675484 | -0.018518869 | 0.000156615 |
| 7 | 0 | 252 | 675 | -0.013800934 | -0.013680715 | 0.000120219 |
| 4 | 1 | 252 | 2208 | -0.011231972 | -0.011117005 | 0.000114967 |
| 4 | 1 | 252 | 1298 | 0.040355705 | 0.040468093 | 0.000112389 |
| 7 | 0 | 252 | 537 | 0.017119497 | 0.017009806 | 0.000109691 |
| 4 | 1 | 252 | 2146 | 0.023031734 | 0.022923965 | 0.000107769 |
| 4 | 1 | 252 | 2060 | 0.025090117 | 0.025194423 | 0.000104306 |

All seven share physical expert 252. Exact CPU operand records were not
available for every coordinate, so their first arithmetic boundary is not
proven. No production correction was made.

## Aggregate identity correction

The oracle incorrectly used the layer-2 routing fixture
`/tmp/cortex-iter127-l2-weights.bin` for layer-4 aggregation. It now selects
`/tmp/cortex-iter156-l4-weights.bin` for layer 4. This is an independent
reference correction and does not change runtime aggregation.

Before correction:

```text
max_abs=0.295072 rms=0.0188967 mismatches=21963
```

After correction:

```text
aggregate n=23040 max_abs=3.76585e-05 rms=3.45532e-06 mismatches=0
```

The aggregate is corrected to below the diagnostic `1e-4` mismatch threshold,
but fails the strict `1e-5` maximum bound because the seven down residuals
remain.

## Fresh oracle results

```text
gate      n=57600  max_abs=5.24521e-06 rms=4.10268e-07 mismatches=0
up        n=57600  max_abs=4.29153e-06 rms=3.47447e-07 mismatches=0
down      n=230400 max_abs=1.56615e-04 rms=5.06390e-06 mismatches=7
aggregate n=23040  max_abs=3.76585e-05 rms=3.45532e-06 mismatches=0
```

The oracle command exited 0. Provider regression and deterministic real
inference were not rerun in this iteration because their model/runtime
arguments were not available from the bounded fixture set.

## Status

| Stage | Status |
|---|---|
| Gate | PASS |
| Up | PASS |
| Down | FAIL — 7 expert-252 residuals |
| Aggregate identity | PASS — layer-4 weights selected |
| Aggregate strict numerical bound | FAIL — max abs `3.76585e-05` |
| Provider regression | NOT TESTED |
| Deterministic real inference | NOT TESTED |
| Bounded memory | NOT TESTED |
| Full layer-4 equivalence | FAIL |

## Next action

Capture exact CPU Q8_0 rows and CPU dot results for all seven expert-252
coordinates using keyed row capture. Compare each against canonical IQ4_NL
rows and scalar results before changing arithmetic or tolerances.
