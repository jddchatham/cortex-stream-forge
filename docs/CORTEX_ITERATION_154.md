# Cortex Iteration 154 — First Divergent Layer / Logit Equivalence

Date: 2026-10-13
Branch: `cortex-iteration-154`
Base: preserved dirty Iteration 153 state; no commit or push.

## Preservation

The Iteration 153 worktree was preserved on a new branch. No reset, clean,
stash, merge, force operation, or unrelated-file overwrite was performed.

## Controlled A/B configurations

Both runs used the same built executable, GGUF shard, prompt (`hello` followed
by `/exit`), backend, thread/runtime configuration, and descriptor registry
`/tmp/cortex-iter153-layer2-4-index.json`.

### A — layer 4 conventional

```text
CORTEX_MOE_STREAM_LAYERS=2,3
CORTEX_MOE_STREAM_GATE=1
CORTEX_MOE_STREAM_UP_DOWN=1
```

Evidence: `/tmp/cortex-iter154-2_3.log`.

Result: exit 0, token `248046`, selected logit `10.7801`. Layers 2 and 3 each
reported gate/up/down streaming callbacks. This is the completed control run.

### B — layer 4 streamed

```text
CORTEX_MOE_STREAM_LAYERS=2,3,4
CORTEX_MOE_STREAM_GATE=1
CORTEX_MOE_STREAM_UP_DOWN=1
```

Evidence: `/tmp/cortex-iter154-2_3_4.log`.

Result: layers 2, 3, and 4 all reached their three bounded callbacks, but the
run did not complete the later context sequence or emit final token/logit
records within the guarded execution. It ended while progressing through later
layer contexts. Therefore B is an execution failure/inconclusive comparison,
not a demonstrated numerical divergence at a specific tensor.

## First proven boundary

The earliest proven difference is not a value comparison. It is the execution
boundary after successful layer-4 gate/up/down callback completion. No captured
layer-4 input, routing vector, post-MoE vector, residual output, or final logits
from B are available for a valid A/B numerical comparison.

The layer-4 provider itself remains independently validated from Iteration 153:
one-token max abs `3.19351e-6`, RMS `7.9655e-7`; three-token max abs
`3.86965e-6`, RMS `9.53667e-7`. This does not prove graph-level equivalence.

## Memory and reservation evidence

The completed A run retained the previously observed 1300 MiB-class conventional
contexts. The incomplete B run reached later contexts with the same class of
reservations before termination. No safe whole-process memory comparison was
accepted. No optimization or scheduler change was made.

## Validation

The layer-aware provider regression for layers 2–4 remains passing under the
established limits. `git diff --check` passes. No production arithmetic was
changed. The descriptor-driven layer selection remains test-gated; without the
environment flag, conventional fallback remains active.

## Status

| Check | Status | Evidence |
|---|---|---|
| Preserve Iteration 153 | PASS | isolated branch and dirty state retained |
| A control, layers 2–3 | PASS | exit 0, token/logit `248046/10.7801` |
| B streamed layer 4 | FAIL | callbacks pass, run incomplete before final logits |
| Layer-4 provider numerical regression | PASS | prior bounded one/three-token results |
| First numerical divergent element | NOT TESTED | B produced no comparable downstream fixtures |
| Full-logit equivalence | NOT TESTED | no completed B logits |
| Whole-process memory comparison | NOT TESTED | invalid because B did not complete |
| Conventional layer-4 fallback | PASS / retained | default when stream layer flag is absent |
| `git diff --check` | PASS | clean |
| Full-model streaming | NOT TESTED | not demonstrated |

## Next action

Keep layer 4 on conventional fallback for end-to-end runs. Add a bounded
layer-4 transaction capture immediately before and after the MoE result, plus a
resource/termination diagnostic around the first post-callback context. Repeat
with a single layer-4 streaming transaction before attempting the full
multi-layer run. Do not enable additional streamed layers until that transaction
completes and its output matches the conventional boundary.
