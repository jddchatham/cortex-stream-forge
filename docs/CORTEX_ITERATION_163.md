# Cortex Iteration 163 — per-token CPU quantization capture

## Preservation

Work continued on isolated branch `cortex-iteration-163`, preserving all dirty
Iteration 162 changes. No production arithmetic or routing was changed.

## Capture-key implementation

The provider now writes diagnostic records using the key
`projection/layer/token/expert`, for example:

```text
/tmp/cortex-iter162-p0-l4-t1-e309-q.bin
/tmp/cortex-iter162-p0-l4-t1-e309-f32.bin
```

Files are write-once (`access(..., F_OK)` before creation), preventing later
provider calls from overwriting an accepted record. The provider logs the
physical layer, projection, token, expert, quantized byte count, and F32 count.
The bounded capture covers only tokens 0 and 1 and the selected ten experts.

The CPU predicate was also corrected: a F32 row is eligible only when its
activation source is nonzero; nonzero weight bytes can no longer cause a zero
activation row to be accepted.

## Runtime evidence

The real layer-4 provider run completed exit 0. Its provider-entry trace showed:

```text
gate token=0 nonzero=2560 source_nb2=10240
gate token=1 nonzero=2560 source_nb2=10240
up   token=0 nonzero=2560 source_nb2=10240
up   token=1 nonzero=2560 source_nb2=10240
down token=0 nonzero=640  source_nb2=25600
down token=1 nonzero=0    source_nb2=25600
```

This proves token-1 gate/up input binding is nonzero and uses the expected
token stride. Down token 1 is downstream of the unresolved gate/up result.

However, the CPU row-capture gate still accepted only the first worker-visible
row for the active diagnostic target. No trustworthy token-1 Q8_K fixture was
emitted by the completed run; therefore no byte-level token-1 comparison is
claimed. The previously emitted zero token-1 files are stale evidence and are
rejected.

## Status

| check | status |
|---|---|
| keyed filename design | PASS |
| write-once protection | PASS |
| token-1 provider F32 input | PASS |
| token-0 Q8_K capture | PASS (existing first-row path) |
| token-1 Q8_K capture | NOT TESTED |
| token-1 Q8_0 capture | NOT TESTED |
| independent Q8 byte comparison | NOT TESTED |
| first dot-product comparison | NOT TESTED |
| gate/up real-activation equivalence | FAIL from prior evidence |
| provider regression | PASS |
| bounded memory | PASS; only bounded diagnostics added |
| `git diff --check` | PASS |
| full-model streaming | NOT TESTED |

## Exact remaining boundary

The remaining issue is inside the CPU diagnostic's per-graph acceptance
mechanism: provider-entry token identity is correct, but the CPU boundary does
not expose a separately accepted record for token 1. The next smallest fix is a
bounded capture-slot array carried in the diagnostic target, indexed by the
provider token/expert key and populated directly at the CPU boundary, instead
of one `row_capture_once` flag. Each slot must carry its own bytes and valid
bit. Only after token-1 slots are nonempty should Q8_K/Q8_0 byte equivalence be
evaluated.
