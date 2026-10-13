# Cortex Iteration 178 — Bounded Requested-Row CPU Capture

## Preservation

- Branch: `cortex-iteration-178`.
- Iteration 177 dirty state and unrelated changes preserved.
- No reset, clean, stash, merge, commit, push, or production modification.
- `git diff --check`: PASS.

## Requested keys

| token | slot | expert | channel | runtime flat index |
|---:|---:|---:|---:|---:|
| 4 | 1 | 252 | 1026 | 112706 |
| 7 | 0 | 252 | 675 | 179875 |
| 4 | 1 | 252 | 2208 | 113888 |
| 4 | 1 | 252 | 1298 | 112978 |
| 7 | 0 | 252 | 537 | 179737 |
| 4 | 1 | 252 | 2146 | 113826 |
| 4 | 1 | 252 | 2060 | 113740 |

## Implementation-level result

The CPU implementation exposes local output row `i1`, tile coordinate `ir1`,
destination strides, and actual `wdata`/IQ4_NL row pointers. The diagnostic
contract still has only one target object per provider graph and one
`row_capture_once` flag. The provider submits the complete multi-row graph,
while the target has no requested-row set or invocation-scoped row key.

Adding one requested-row integer would still be unsafe: it would not establish
all seven token/expert/invocation associations or prevent a worker from
capturing a different graph row. No partial diagnostic change was applied.

The precise blocker is the missing bounded registry containing the complete
key: layer, graph invocation, projection, token, expert slot/ID, output row,
and quantization type, with write-once slots.

## Numerical baseline

The corrected layer-4 oracle reports gate max `5.24521e-06`, up max
`4.29153e-06`, down max `1.56615e-04` with 7 mismatches, and aggregate max
`3.76585e-05` with 0 elements beyond the diagnostic `1e-4` threshold. Strict
down and aggregate acceptance remain failed.

## Status

| Check | Status |
|---|---|
| Seven target keys | PASS |
| Requested-row registry | NOT IMPLEMENTED |
| CPU capture coverage | FAIL / 0 of 7 new valid records |
| Operand byte comparisons | NOT TESTED |
| Scalar/CPU/runtime/oracle comparisons | NOT TESTED |
| Down strict acceptance | FAIL |
| Aggregate strict acceptance | FAIL |
| Provider regression | NOT TESTED |
| Deterministic real inference | NOT TESTED |
| Bounded memory | NOT TESTED |

## Next action

Add a fixed-capacity diagnostic registry to `ggml_cpu_diagnostic_target`,
populate it with the seven complete keys, and capture only when the CPU `i1`
and stride-derived logical row match a requested slot. Store exact `wdata`
IQ4_NL bytes, Q8_0 bytes, dot, destination offset, and validity per slot.
