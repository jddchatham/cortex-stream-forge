# Cortex Iteration 177 — Targeted Expert-252 CPU Capture

## Preservation

- Branch: `cortex-iteration-177`.
- Iteration 176 and unrelated dirty changes preserved.
- No reset, clean, stash, merge, commit, push, or production arithmetic change.
- `git diff --check`: PASS.

## Exact requested coordinates

The seven failures are:

| token | expert slot | physical expert | output channel | runtime flat index |
|---:|---:|---:|---:|---:|
| 4 | 1 | 252 | 1026 | 112706 |
| 7 | 0 | 252 | 675 | 179875 |
| 4 | 1 | 252 | 2208 | 113888 |
| 4 | 1 | 252 | 1298 | 112978 |
| 7 | 0 | 252 | 537 | 179737 |
| 4 | 1 | 252 | 2146 | 113826 |
| 4 | 1 | 252 | 2060 | 113740 |

Runtime flat indices use the captured physical layout `channel + slot*2560 +
token*25600`. Oracle indices use `(token*10 + slot)*2560 + channel`.

## Capture feasibility result

The current provider-local diagnostic installs a fresh CPU target per tiled
provider call, but `ggml-cpu.c` still has these limitations:

1. `row_capture_once` accepts only one row per target.
2. The CPU diagnostic target carries projection/layer/token/expert metadata
   at the provider boundary but no output-row/channel identity into the CPU
   kernel.
3. The CPU capture predicate records worker-0's first row (`ir1 == 0`,
   `ir0 == 0`) and cannot prove that a later captured row is one of the seven
   requested channels.
4. Existing files cover token 0/1 expert 252, not token 4/7 expert 252 for
   all seven requested channels.

Therefore no trustworthy seven-coordinate CPU Q8_0/IQ4_NL/dot capture was
produced in this iteration. A capture file or a call-order association would
not satisfy the identity requirement.

## Existing numerical evidence

The fresh corrected oracle remains:

```text
gate      n=57600  max_abs=5.24521e-06 rms=4.10268e-07 mismatches=0
up        n=57600  max_abs=4.29153e-06 rms=3.47447e-07 mismatches=0
down      n=230400 max_abs=1.56615e-04 rms=5.06390e-06 mismatches=7
aggregate n=23040  max_abs=3.76585e-05 rms=3.45532e-06 mismatches=0
```

All seven down residuals are physical expert 252. The corrected layer-4
routing-weight fixture removes the previous aggregate error of `0.295072`,
but strict `max_abs <= 1e-5` acceptance remains failed.

## Status

| Check | Status |
|---|---|
| Seven coordinate enumeration | PASS |
| Seven CPU Q8_0 captures | NOT TESTED — row identity unavailable |
| Seven CPU IQ4_NL captures | NOT TESTED — row identity unavailable |
| Seven scalar-vs-CPU dot comparisons | NOT TESTED |
| Down strict acceptance | FAIL |
| Correct layer-4 aggregate weights | PASS |
| Aggregate strict acceptance | FAIL |
| Provider regression | NOT TESTED |
| Deterministic real inference | NOT TESTED |
| Bounded memory | NOT TESTED |
| Full layer-4 equivalence | FAIL |

## Smallest next action

Extend the diagnostic-only target with a bounded requested-row key carrying
physical layer, token, expert slot/ID, projection, output row, and invocation
ID. Propagate that key into the CPU MUL_MAT call and replace the first-row
predicate with fixed-capacity write-once slots. Only then capture the seven
expert-252 rows and compare exact operands and dots.
