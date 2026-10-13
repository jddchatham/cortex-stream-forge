# Cortex Iteration 161 — token-1 replay and activation forensics

## Preservation

Work continued on isolated branch `cortex-iteration-161`, preserving the dirty
Iteration 160 state. No production arithmetic, routing, quantization, or
streamed-layer selection was changed.

## Exact token extraction

The captured layer-4 activation is F32 `[2560,1,9,1]` with contiguous logical
channel order and token stride `nb[2] = 10240` bytes. Token `t` is the byte
range `[t*10240, (t+1)*10240)`. IDs are I32 `[10,9,1,1]`, with token `t`
stored at `t*10*4` bytes.

Token 0 and token 1 are distinct:

```text
t=0 activation SHA-256 7c46b8eb69cb57f45dfbc4954f6f72bc3c3e225e7e9a95b0024f6b569bf9cf6b
t=1 activation SHA-256 1e0e3f630690f0e6af680be0596ffcc8f94263e34c9f84aaec3eee514c54006d
t=0 IDs        342dc92a4811d1e07046d751680cdb08737a7164214eeeea27147d1d6560e4fc
t=1 IDs        6085770872d633b5ef2ae3e959ab59fddf913ce62e21c460885299f7f39f0b80
```

Both rows contain 2560 finite nonzero values. Token 1 IDs are
`309,252,453,209,106,488,496,59,169,257`.

## Single-token replay

The standalone oracle was changed only to infer `n_tokens` from fixture size,
allowing a one-token run with token 1's exact activation and IDs. It continued
to read bounded selected rows from the canonical layer-4 GGUF descriptors.

Token-1 oracle versus the corresponding original multi-token streamed slices:

| stage | elements | max abs | RMS | first mismatch | status |
|---|---:|---:|---:|---:|---|
| gate | 6400 | 2.59191 | 0.446288 | 0 | FAIL |
| up | 6400 | 1.88591 | 0.362182 | 0 | FAIL |
| down | 25600 | 0.00232945 | 0.000527264 | 1 | FAIL |
| aggregate | 2560 | 0.00119003 | 0.000215704 | 0 | FAIL |

This reproduces the multi-token failure after isolating token 1. It therefore
rejects the hypothesis that flattened index 6400 alone is a comparison-layout
error. The token-0 control still agrees for every expert slot within roughly
`5e-6` for gate and up.

## Quantization/layout assessment

The provider callback source copies ordinary gate/up activation data using
`activation + i*nb[0] + t*nb[2]`, and copies IDs using
`ids + s*nb[0] + t*nb[1]`; the standalone fixture extraction uses the same
logical strides. Output copying uses `f*dst->nb[0] + s*dst->nb[1] +
t*dst->nb[2]`, matching the proved `[channel,slot,token]` layout.

Thus no provider or oracle index correction is justified by the current
evidence. The remaining discriminating boundary is the actual per-token
activation quantization/input bytes inside the provider for token 1 versus the
independently quantized token-1 row. Existing capture infrastructure records
the first accepted atomic row, but not a complete token-indexed quantization
record for this transaction.

## Status

| check | status |
|---|---|
| token-1 stride extraction | PASS |
| token-0/token-1 distinctness | PASS |
| token-1 single-token oracle replay | PASS (executed) |
| token-1 gate equivalence | FAIL |
| token-1 up equivalence | FAIL |
| token-1 down equivalence | FAIL, downstream/activation-sensitive |
| token-1 aggregate equivalence | FAIL |
| full layer-4 equivalence | FAIL |
| provider synthetic regression | PASS |
| full-model streaming | NOT TESTED |
| `git diff --check` | PASS |

## Next action

Add a token- and projection-indexed capture of the actual Q8_K/Q8_0 row passed
to each provider matmul, then compare token 0 and token 1 bytes directly with
independent encodings. This is the smallest test that distinguishes an actual
provider token-stride/quantization defect from a mismatch in the graph capture
transaction. No production correction should be made before that byte-level
comparison.
