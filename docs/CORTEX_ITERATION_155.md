# Cortex Iteration 155 — Post-Layer-4 Graph Execution Failure

Date: 2026-10-13
Branch: `cortex-iteration-155`
Base: preserved dirty Iteration 154 state; no commit or push.

## Reproduction

The previously reported layer-4 post-callback failure was rerun with
`CORTEX_MOE_STREAM_LAYERS=2,3,4`, the same GGUF, prompt, backend, and runtime
configuration. The guarded command completed with exit 0, reached layers 5–47,
serialized state boundaries, and produced final logits. No assertion, signal,
allocation error, deadlock, or dangling-pointer evidence was observed.

Evidence: `/tmp/cortex-iter155-b.log`.

The earlier incomplete run was not reproducible; it was an incomplete or
terminated experiment rather than a proven post-layer-4 graph defect.

## A/B results

Control A (`CORTEX_MOE_STREAM_LAYERS=2,3`) from
`/tmp/cortex-iter154-2_3.log`:

```text
exit 0
token 248046
top-1 logit 10.7801
top-2 token 14556 logit 9.20749
```

Run B (`CORTEX_MOE_STREAM_LAYERS=2,3,4`) from this iteration:

```text
exit 0
token 248046
top-1 logit 11.2143
top-2 token 248045 logit 9.69807
top-3 token 14556 logit 9.66976
```

The token remains equal, but the logit and ranking change materially. Full
logit vectors and synchronized A/B intermediate fixtures were not captured, so
end-to-end numerical equivalence is not claimed.

## Callback and ownership evidence

The B run emitted layer-4 gate/up/down PASS records and continued through later
contexts. Layer-4 callback output sizes were gate `230400`, up `230400`, and
down `921600` bytes. State serialization reached layer 47 and the final head
completed. No output-lifetime failure was demonstrated.

Layer-4 provider-only regressions still pass:

```text
one token:   max_abs=3.19351e-6, RMS=7.9655e-7
three token: max_abs=3.86965e-6, RMS=9.53667e-7
```

## Memory

The B run retained approximately `1301.89–1308.27 MiB` conventional-context
reservations and a `0.08 MiB` generation context. No memory optimization was
attempted.

## Status

| Check | Status | Evidence |
|---|---|---|
| Preserve Iteration 154 | PASS | isolated branch, dirty state retained |
| Reproduce crash/failure | NOT REPRODUCED | B completed exit 0 |
| Layer-4 callback completion | PASS | gate/up/down records |
| Downstream graph execution | PASS | layers 5–47 and final head completed |
| A/B successful completion | PASS | both exit 0 |
| Full-logit equivalence | NOT TESTED | full vectors not captured |
| Output equivalence | FAIL / unresolved | logit changed `10.7801` → `11.2143` |
| Provider regressions | PASS | layer-0 and layer-4 exit 0 |
| Memory improvement | NOT TESTED | no optimization attempted |
| `git diff --check` | PASS | clean |
| Full-model streaming | NOT TESTED | only layers 2–4 streamed |

## Next action

Capture identical layer-4 input, routing IDs/weights, post-MoE output, and
residual output for A and B, then compare the first divergent tensor. Keep layer
4 test-gated and do not enable more layers until the changed logit is explained.
