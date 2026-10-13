# Cortex Iteration 173 — Down Oracle Reconciliation

## Preservation

- Branch: `cortex-iteration-173`
- Iteration 172 and unrelated dirty changes preserved.
- No reset, clean, stash, merge, commit, or push.
- `git diff --check`: PASS.

## Baseline reproduction

The existing oracle was rebuilt and run against fresh layer-4 fixtures. It
still reports:

```text
gate n=57600 max_abs=5.24521e-06 rms=4.10268e-07 mismatches=0
up   n=57600 max_abs=4.29153e-06 rms=3.47447e-07 mismatches=0
down n=230400 max_abs=3.27526 rms=0.0792954 mismatches=229698 first=0
aggregate n=23040 max_abs=1.05916 rms=0.0629292 mismatches=22918 first=0
```

The identity-matched target remains:

- token 1, slot 9, expert 257, output channel 0
- scalar: `0.006443967811833318`
- CPU dot/runtime destination: `0.006443969905376434`
- oracle value at the same flattened coordinate: `0.0002231794933322817`

## Oracle hypothesis tested and rejected

The first hypothesis was that the captured activation fixture required an
explicit `[channel,1,token]` to `[token,channel]` transpose. A diagnostic
oracle-only transpose was implemented and executed. It caused gate and up to
fail immediately (`max_abs` approximately 5–6), while the untransposed path
continues to pass gate/up. The change was reverted. This proves the fixture as
consumed by the existing oracle is already in the representation that matches
gate/up; retaining the transpose would be an incorrect fix.

## Oracle defect found and corrected

The target trace exposed a concrete independent-reference bug in
`flashnext-l2-moe-reference.cpp`: `fp16()` declared the five-bit exponent as
`bool`. Every decoded FP16 scale therefore used the wrong exponent path. The
fix changes the exponent to an integer type; it affects only the debug oracle,
not GGML, the provider, routing, or inference arithmetic.

The corrected target trace is:

```text
ORACLE_TARGET t=1 slot=9 expert=257 row=0 desc_off=35603219744
row_bytes=360 xoff=12160 raw_fnv=e3ad0873a4a45f91
z=0.0064370818436145782
```

This confirms the descriptor offset and row bytes used by the oracle are the
expected target row. The corrected value is close to, but not equal to, the
identity-matched CPU result `0.006443969905376434`; the remaining difference
is `6.88806e-06`. The exact remaining discrepancy is therefore now inside the
oracle's down-input/quantization reconstruction or a related reference
semantics issue, not the previously suspected FP16 decoder or a proven runtime
placement defect.

With the correction, the fresh full run reports:

```text
gate      n=57600  max_abs=5.24521e-06  rms=4.10268e-07  mismatches=0
up        n=57600  max_abs=4.29153e-06  rms=3.47447e-07  mismatches=0
down      n=230400 max_abs=8.08954e-04  rms=2.59522e-05  mismatches=2184 first=2
aggregate n=23040  max_abs=2.95152e-01  rms=1.89017e-02  mismatches=21963 first=0
```

The aggregate remains invalid because the complete down stage is not yet
equivalent. No runtime arithmetic change was made.

## Current conclusion

Runtime output placement and CPU arithmetic remain proven for the target dot.
The original oracle down path was provably wrong due to the FP16 exponent
type bug, and that oracle-only defect is corrected. The remaining down
discrepancy has not yet been reduced to a proven first failing byte or
operation. No production code was changed.

| Check | Status |
|---|---|
| Target CPU dot/runtime identity | PASS |
| Gate oracle | PASS |
| Up oracle | PASS |
| FP16 oracle decode repair | PASS |
| Down oracle repair | FAIL / residual mismatch |
| Full down equivalence | FAIL |
| Aggregate equivalence | NOT ACCEPTED |
| Provider regression | PASS from Iteration 170 |
| Full-model streaming | NOT TESTED |

## Validation

- Oracle build: PASS.
- Fresh corrected oracle run: exit 0, gate/up pass; down/aggregate remain
  numerically failing as above.
- Real runner baseline remains token `248046`, logit `13.0355`.
- `git diff --check`: PASS.

## Next action

Compare the oracle's generated Q8_0 row byte-for-byte with the existing
680-byte CPU capture for token 1/expert 257, then use the exact captured
runtime down-input as an oracle input. This discriminates reference
down-input reconstruction from IQ4_NL decoding without touching runtime
arithmetic.
