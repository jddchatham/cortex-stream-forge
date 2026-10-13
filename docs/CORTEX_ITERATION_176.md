# Cortex Iteration 176 — Expert 252 Residual Forensics

## Preservation

- Branch: `cortex-iteration-176`.
- Iteration 175 dirty state preserved without reset, clean, stash, merge, or production modification.
- `git diff --check`: PASS.

## Seven residual coordinates

Explicit comparison used runtime index `channel + slot*2560 + token*25600` and oracle index `(token*10 + slot)*2560 + channel`.

| token | slot | expert | channel | runtime | oracle | abs |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 252 | 1026 | -0.018675484 | -0.018518869 | 1.56615e-4 |
| 7 | 0 | 252 | 675 | -0.013800934 | -0.013680715 | 1.20219e-4 |
| 4 | 1 | 252 | 2208 | -0.011231972 | -0.011117005 | 1.14967e-4 |
| 4 | 1 | 252 | 1298 | 0.040355705 | 0.040468093 | 1.12388e-4 |
| 7 | 0 | 252 | 537 | 0.017119497 | 0.017009806 | 1.09691e-4 |
| 4 | 1 | 252 | 2146 | 0.023031734 | 0.022923965 | 1.07769e-4 |
| 4 | 1 | 252 | 2060 | 0.025090117 | 0.025194423 | 1.04306e-4 |

All failures are physical expert 252: five are token 4/slot 1 and two are token 7/slot 0. They span multiple output rows.

## Exact CPU evidence

Existing exact CPU captures cover token 0/1 expert 252 activation rows, but no fresh keyed CPU IQ4_NL rows or CPU dot records exist for the seven token 4/token 7 coordinates. Exact byte and dot identity for all seven is therefore NOT PROVEN. No arithmetic or tolerance change was made.

The previously proven token 1/expert 257 coordinate remains valid: exact Q8_0 bytes and CPU dot agree with the independent scalar within nanoscopic error.

## Fresh full comparison

Using the correct layer-4 routing-weight fixture:

```text
gate      n=57600  max_abs=5.24521e-06 rms=4.10268e-07 mismatches=0
up        n=57600  max_abs=4.29153e-06 rms=3.47447e-07 mismatches=0
down      n=230400 max_abs=1.56615e-04 rms=5.06390e-06 mismatches=7
aggregate n=23040  max_abs=3.76585e-05 rms=3.45532e-06 mismatches=0
```

The aggregate correction reduced the previous `0.295072` maximum error to `3.76585e-05`. No aggregate element exceeds the diagnostic `1e-4` threshold, but strict `max_abs <= 1e-5` acceptance still fails.

## Status

| Check | Status |
|---|---|
| Expert-252 enumeration | PASS |
| Exact CPU operands for all seven | NOT TESTED |
| Scalar-vs-CPU dot for all seven | NOT TESTED |
| Full down strict acceptance | FAIL |
| Layer-4 routing-weight identity | PASS |
| Aggregate strict acceptance | FAIL |
| Provider regression | NOT TESTED |
| Deterministic real inference | NOT TESTED |
| Bounded memory | NOT TESTED |
| Full layer-4 equivalence | FAIL |

## Next action

Run the real layer-4 workload with keyed CPU capture for physical expert 252, tokens 4 and 7, and these seven output rows. Compare exact Q8_0 bytes, IQ4_NL `wdata` bytes, CPU dot, stored destination, and scalar oracle for each coordinate. This is the smallest experiment that distinguishes a remaining oracle issue from a runtime defect.
