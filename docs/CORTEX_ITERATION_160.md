# Cortex Iteration 160 — expert-slice mapping forensics

## Preservation

Work continued on isolated branch `cortex-iteration-160`, preserving the dirty
Iteration 159 state. No production arithmetic, routing, quantization, or
streamed-layer selection was changed.

## Proven layout

Layer-4 gate and up outputs are F32 `[640,10,9,1]`; down is F32
`[2560,10,9,1]`. GGML contiguous indexing is `f + 640*s + 640*10*t` for
gate/up and `h + 2560*s + 2560*10*t` for down. The provider callback explicitly
copies its expert-major buffer into that layout using
`dst + f*nb[0] + s*nb[1] + t*nb[2]`.

Therefore flattened index `6400` maps to:

```text
gate/up: t = 1, selected-expert slot s = 0, channel f = 0
```

It is the first element of token 1's first expert slice, not the second expert
of token 0. The boundary is a token stride (`640*10`), while expert slices are
640 elements.

## Identity table

The captured IDs are stored as ten IDs per token. The first three token rows
were:

```text
t=0: 108,309,252,106,59,453,82,484,373,165
t=1: 309,252,453,209,106,488,496,59,169,257
t=2: 496,309,252,453,106,488,35,169,257,59
```

The oracle and provider both resolve each entry through
`resolve_expert(layer, ids[t*10+s], projection)`, adding the descriptor's
expert stride before reading rows. The provider writes its temporary result at
`output + (t*10+s)*rows`; the oracle uses the same flattened index.

## Per-slice comparison

Using fresh real layer-4 activation/ID fixtures and fresh streamed B outputs,
the independent oracle was dumped per projection and compared by token and
expert slot.

Gate maximum absolute errors by token 0 were all below `5.3e-6`; up errors
were all below `4.3e-6`. For token 1 and later, every expert slot diverged at
approximately unit scale. Representative maxima:

```text
gate t=0: s0..s9 <= 5.25e-6
gate t=1: s0=2.59191, s1=1.46156, s2=1.72053 ...
up   t=0: s0..s9 <= 4.30e-6
up   t=1: s0=1.88591, s1=1.14287, s2=1.39409 ...
```

This proves the first flattened divergence is exactly index 6400 (`t=1,s=0,
f=0`). It is not explained by an expert-slot permutation: all ten token-0
slices agree, and all ten token-1 slices diverge.

Whole-stage oracle metrics:

| stage | elements | max abs | RMS | first mismatch >1e-4 | status |
|---|---:|---:|---:|---:|---|
| gate | 57600 | 2.99211 | 0.444845 | 6400 | FAIL |
| up | 57600 | 2.08128 | 0.367934 | 6400 | FAIL |
| down | 230400 | 3.27526 | 0.0743469 | 0 | FAIL |
| aggregate | 23040 | 1.05916 | 0.0612104 | 0 | FAIL |

## First remaining boundary

The provider's expert/output indexing is consistent with GGML for token 0 and
with the documented strides. The first failure occurs when the oracle consumes
the captured activation for token 1. This narrows the issue to one of:

1. the multi-token activation fixture's token stride or capture transaction;
2. provider versus oracle per-token activation quantization/input semantics;
3. a token-dependent execution/capture mismatch.

No indexing correction is justified yet. The next discriminating test is a
single-token layer-4 run for token 1's exact activation, followed by a direct
provider replay from the captured token-1 2560-F32 row. This separates fixture
identity from quantization/row decoding without changing production behavior.

## Validation

| check | status |
|---|---|
| index-6400 semantic mapping | PASS |
| expert IDs and descriptor resolution | PASS for both paths by source trace |
| per-slice token-0 gate/up agreement | PASS |
| full gate/up equivalence | FAIL |
| down equivalence | FAIL, downstream of earlier divergence |
| aggregate equivalence | FAIL, downstream of earlier divergence |
| full layer-4 residual | NOT TESTED |
| full-model streaming | NOT TESTED |
| provider synthetic regression | PASS |
| `git diff --check` | PASS |
