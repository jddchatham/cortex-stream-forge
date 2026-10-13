# Cortex Iteration 170 — Exact CPU IQ4_NL Row and Dot Proof

## Preservation

- Branch: `cortex-iteration-170`
- Iteration 169 work and unrelated dirty changes preserved.
- No reset, clean, stash, merge, commit, or push.
- Build and real runner completed successfully.

## Exact invocation

The keyed record is:

`/tmp/cortex-iter170-l4-p2-t1-e257-r0.bin`

Header evidence:

- Projection: `2` (down)
- Layer: `4`
- Token: `1`
- Physical expert: `257`
- Output row: `0`
- Weight type: `GGML_TYPE_IQ4_NL` (`20`)
- Input width: `640`
- Weight bytes: `360`
- Q8_0 activation bytes: `680`
- CPU dot: `0.006443969905376434`
- Record SHA-256: `68e37c30d051691cde57cd6c1e6a5c3e2401fb6bf04c513b1749a4666f834cc9`

The record was emitted only when the provider identity matched
`layer=4, projection=2, token=1, expert=257, row=0`, so this is not a
call-order inference.

## Activation identity

The exact captured Q8_0 row remains the Iteration 169 record:

- Size: `680` bytes
- CPU SHA-256: `8cbba4da35ef053f0f59fe4a5d0da6417136dbfeaafdc992a034d3c798c37be1`
- Independent Q8_0 encoder SHA-256: identical
- Byte mismatches: `0`

Q8_0 activation identity: PASS.

## Weight identity and diagnostic correction

For `blk.4.ffn_down_exps.weight`, the correct IQ4_NL row size is 360 bytes.
The canonical expert-257 row 0 is at:

`35366368544 + 257 * 921600 = 35603219744`

Canonical row SHA-256:
`f30f0da6b650b2547bffb8050c089958ab3d94228a595c80f76d74a4e12d1aef`

The first atomic record appeared mismatched because the diagnostic copied
`src0->data`, while the CPU kernel uses `wdata`. The captured bytes were eight
zero padding bytes followed by the first 352 canonical bytes. This proves the
capture source—not the production weight binding—was wrong.

The diagnostic was corrected to copy `wdata`, the exact vec-dot row pointer.
This is diagnostic-only and does not alter arithmetic or routing. A fresh
post-correction rerun is still required to produce the corrected row hash.

## Dot comparison

Using the canonical 360-byte IQ4_NL row and the exact captured 680-byte Q8_0
activation, an independent scalar decoder produced:

- Independent scalar: `0.006443967811833318`
- CPU captured dot: `0.006443969905376434`
- Absolute error: `2.09354e-09`
- Relative error: approximately `3.25e-07`

IQ4_NL/Q8_0 dot equivalence: PASS within expected float accumulation error.

The prior provisional graph-output mismatch was therefore not evidence of a
CPU dot defect; it was not identity-matched to the CPU accumulator.

## Validation

- Real runner: PASS, exit 0; token `248046`, logit `13.0355`.
- Provider regression: PASS; max abs `3.4757e-06`, RMS `9.24548e-07`.
- `git diff --check`: PASS.
- Full nine-token down projection equivalence: NOT TESTED after corrected
  diagnostic capture.
- Full aggregate equivalence: NOT TESTED after corrected diagnostic capture.
- Full-model streaming: NOT TESTED.

| Check | Status |
|---|---|
| Exact invocation identity | PASS |
| Exact Q8_0 bytes | PASS |
| Corrected CPU IQ4_NL row capture | NOT TESTED after patch |
| Canonical row versus CPU row | NOT TESTED after patch |
| Exact CPU dot versus scalar | PASS |
| Full down projection | NOT TESTED |
| Full aggregate | NOT TESTED |
| Provider regression | PASS |

## Next action

Run one bounded capture with the corrected `wdata` copy and verify the new
360-byte CPU row hash equals the canonical hash. Then rerun the nine-token
oracle with correct tensor-layout comparison. Do not change the down kernel.
