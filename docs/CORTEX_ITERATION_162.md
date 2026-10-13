# Cortex Iteration 162 — token-indexed Q8 activation capture

## Preservation

Work continued on isolated branch `cortex-iteration-162`, preserving the dirty
Iteration 161 state. No production arithmetic, routing, or streamed-layer
selection was changed.

## Provider input evidence

The provider-local trace was enabled for physical layer 4. Immediately before
the streamed projection loop it reported:

```text
projection=0 gate token=0 nonzero=2560 first=-0.422326 last=0.446902 source_nb2=10240
projection=0 gate token=1 nonzero=2560 first=0.251130 last=0.263710 source_nb2=10240
projection=1 up   token=0 nonzero=2560 first=-0.422326 last=0.446902 source_nb2=10240
projection=1 up   token=1 nonzero=2560 first=0.251130 last=0.263710 source_nb2=10240
projection=2 down token=0 nonzero=640  first=-0.0570162 last=0.213478 source_nb2=25600
projection=2 down token=1 nonzero=0    first=0 last=0 source_nb2=25600
```

This proves the gate/up callback receives a distinct, nonzero token-1 row with
the expected `[2560,1,9,1]` token stride. The down zero is downstream of the
unresolved gate/up divergence.

## Diagnostic correction

The prior CPU row-capture predicate accepted a record when either the source
activation or the weight row was nonzero. Since weights are nonzero, that could
accept a zero activation and overwrite a token-indexed fixture. The predicate
now requires nonzero source activation for F32 inputs (or nonzero quantized
source bytes for already-quantized inputs). Token/projection/expert records are
write-once to prevent later provider calls from overwriting them.

## Capture result

The run completed exit 0 and emitted provider-input records. The CPU diagnostic
still only accepted the first row per graph/diagnostic target; consequently no
trustworthy token-1 Q8_K/Q8_0 fixture was emitted after the predicate change.
The existing token-keyed files from earlier runs were rejected as stale because
they contained zero F32 token-1 data despite the provider-entry trace proving
nonzero gate/up input.

Therefore byte-level token-1 Q8 identity is **NOT TESTED**, not PASS or FAIL.

## Classification

| check | status | evidence |
|---|---|---|
| token-1 provider input binding | PASS | 2560/2560 nonzero gate/up values and `source_nb2=10240` |
| token-1 down input | FAIL | zero, downstream of unresolved gate/up result |
| token-indexed Q8_K bytes | NOT TESTED | CPU diagnostic accepts only first row |
| token-indexed Q8_0 bytes | NOT TESTED | downstream input is zero |
| weight-row identity | NOT TESTED in this run | descriptor identity remains from Iteration 159 |
| gate/up numerical equivalence | FAIL | prior token-1 divergence remains |
| provider regression | PASS | unchanged synthetic regression |
| bounded memory | PASS | only bounded diagnostic rows added |
| full-model streaming | NOT TESTED | |
| `git diff --check` | PASS | completed |

## Smallest next action

Replace the one-global-row capture gate with an explicit per-invocation
token/expert record slot at the CPU boundary, or capture the quantized row
directly from each `tiled_matvec` invocation before installing the next backend
diagnostic target. The record must include the actual `src1` row and quantized
`wdata` bytes for token 0 and token 1. Only then can the Q8 byte comparison
distinguish provider quantization from oracle quantization.
