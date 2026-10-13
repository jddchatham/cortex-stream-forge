# Cortex Iteration 169 — Exact Down-Projection Byte and Dot Validation

## Preservation

- Branch: `cortex-iteration-169`
- Iteration 168 state and unrelated dirty changes preserved.
- No reset, clean, stash, merge, commit, push, or production change.
- `git diff --check`: PASS.

## Exact Q8_0 activation comparison

Target identity from the write-once capture key:

`projection=2, layer=4, token=1, expert=257`

Fixtures:

- F32 source: `/tmp/cortex-iter162-p2-l4-t1-e257-f32.bin`, 640 values / 2560 bytes
- CPU-consumed Q8_0: `/tmp/cortex-iter162-p2-l4-t1-e257-q.bin`, 680 bytes

The independent encoder used 20 blocks of 32 values, FP16 block scales, and
signed Q8 values with the GGML-compatible nearest rounding rule.

| Result | Value |
|---|---:|
| Expected bytes | 680 |
| Actual bytes | 680 |
| Byte mismatches | 0 |
| First mismatch | none |
| CPU capture SHA-256 | `8cbba4da35ef053f0f59fe4a5d0da6417136dbfeaafdc992a034d3c798c37be1` |
| Independent SHA-256 | `8cbba4da35ef053f0f59fe4a5d0da6417136dbfeaafdc992a034d3c798c37be1` |

Q8_0 activation byte identity: PASS.

## IQ4_NL row mapping

The layer-4 down descriptor is:

- GGUF tensor: `blk.4.ffn_down_exps.weight`
- Shape: `[640,2560,512]`
- Type: IQ4_NL
- IQ4_NL row size: `360` bytes (`640 / 32 * 18`)
- Expert: `257`
- Output row: `0`
- Canonical byte offset: `35366368544 + 257 * 921600 = 35603219744`
- Canonical row SHA-256: `f30f0da6b650b2547bffb8050c089958ab3d94228a595c80f76d74a4e12d1aef`

The row was read by bounded pread from the canonical GGUF shard and has the
expected 360-byte length. The current CPU diagnostic did not persist a
keyed IQ4_NL weight-row fixture for expert 257; the older atomic row record is
for token 0/expert 322 and is not substituted. Therefore CPU-versus-GGUF
weight byte identity for this exact invocation is NOT TESTED.

## Scalar dot status

Using the exact captured Q8_0 row and independently decoded canonical IQ4_NL
row, the scalar result for expert 257, token 1, output row 0 is:

- Independent scalar dot: `0.006443967811833318`
- Streamed graph output at logical `[channel=0, expert=0, token=1]`:
  `0.030586689710617065`
- Difference: `0.024142721898783748`

This is a diagnostic discrepancy, not yet a proven CPU vec-dot discrepancy:
the graph output is not a keyed CPU accumulator record, and the exact CPU
weight row/dot invocation has not been persisted. The required next capture is
one atomic record containing layer, token, expert, output row, 360-byte IQ4_NL
weight row, 680-byte Q8_0 row, and completed CPU dot result.

## Relation to full projection failure

The previous independent real-activation comparison remains:

- Gate/up token-0 and token-1: PASS within approximately `5e-6`.
- Down token-0: FAIL, max abs `3.275258`.
- Down token-1: FAIL, max abs `0.512695`.
- Aggregate: FAIL downstream of down.

The exact Q8_0 activation boundary is now exonerated for the tested token-1
expert row. IQ4_NL weight-row identity and CPU dot identity remain unresolved.

## Validation

- Real runner: PASS, exit 0, token `248046`, logit `13.0355`.
- Provider regression: PASS, prior max abs `3.4757e-06`, RMS `9.24548e-07`.
- Bounded scratch: `2,150,400` bytes.
- Full layer-4 numerical equivalence: FAIL / incomplete at down stage.
- Full-model streaming: NOT TESTED.

| Check | Status |
|---|---|
| Exact Q8_0 byte identity | PASS |
| Exact IQ4_NL CPU weight-row identity | NOT TESTED |
| Exact CPU scalar dot identity | NOT TESTED |
| Down projection equivalence | FAIL |
| Aggregate equivalence | FAIL |
| Provider regression | PASS |
| Full-model streaming | NOT TESTED |

## Next action

Extend the existing bounded capture record—not the production arithmetic—to
persist the keyed 360-byte IQ4_NL row and completed CPU dot for token 1,
expert 257, row 0. Then compare that atomic record with the canonical row and
the independently decoded Q8_0 row. Do not classify the `0.02414` difference
as a dot-kernel defect until those exact invocation operands are proven.
