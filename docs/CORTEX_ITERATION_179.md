# Cortex Iteration 179 — Seven-Row Diagnostic Capture

## Preservation and implementation

- Branch: `cortex-iteration-179`.
- Existing dirty work preserved; no reset, clean, stash, merge, commit, or
  push.
- Diagnostic-only fields were added to `ggml_cpu_diagnostic_target`:
  `requested_row`, `captured_dest_offset`, and `captured_output`.
- The CPU capture now matches the requested output row against the actual
  `ir0` output-channel coordinate, copies the exact `wdata` activation and
  IQ4_NL row, records the dot and destination offset, and uses atomic
  write-once state.
- The provider runs bounded diagnostic repeats for only the seven known rows;
  normal execution is unchanged when the flag is absent.

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

## Execution evidence

Build command:

```sh
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-qwen38-lowres-chat -j2
```

Build: PASS, exit 0.

The first execution after the initial implementation proved that comparing
against `i1` was wrong: the graph has `ne1=1`, so `i1` is the input-column
coordinate. The implementation was corrected to use `ir0`, the output-channel
row. A rebuilt run then reached every requested provider key but reported
`status=MISSING` for all seven rows. The CPU trace showed the requested key at
the outer graph boundary, while no matching `ir0` dot record was emitted.

This establishes that the remaining blocker is the CPU kernel's worker/tile
partition mapping for `ir0`, not the original first-row predicate. The current
diagnostic run did not produce valid operand files and therefore no byte or
dot comparison is claimed.

## Status

| Check | Status |
|---|---|
| Requested-row registry fields | PASS |
| Actual `ir0` mapping correction | PASS (source/build) |
| Seven-row execution attempt | PASS |
| Capture coverage | FAIL — 0/7 valid records |
| Q8_0 byte comparisons | NOT TESTED |
| IQ4_NL byte comparisons | NOT TESTED |
| Scalar/CPU/runtime/oracle dots | NOT TESTED |
| Down strict acceptance | FAIL |
| Aggregate strict acceptance | FAIL |
| Provider regression | NOT TESTED |
| Deterministic real inference | NOT TESTED |
| `git diff --check` | PASS |

## Next action

Capture the actual `ir0_start`/`ir0_end` partition and tile-local row mapping
for the down graph, then translate the requested global output channel into
the worker's local `ir0` coordinate before the vec-dot call. The next run
must emit one immutable record per requested key; no arithmetic change is
justified yet.
