# Cortex Iteration 148 — targeted scheduler allocation investigation

Branch: `cortex-iteration-148`
Base: pushed Iteration 145 checkpoint `7174649`; no commit or push for this iteration.

## Preservation

The branch was created from the dirty Iteration 147 state with
`git switch -c cortex-iteration-148`. No reset, clean, stash, merge, or
overwrite was performed. Existing unrelated changes remain unstaged.

## Exact allocation mechanism

The path is:

```text
llama_context::sched_reserve()
  -> graph_reserve(n_tokens=9, ...)
  -> ggml_backend_sched_reserve()
  -> ggml_gallocr_alloc_graph_impl()
  -> ggml_gallocr_allocate_node() for graph leaves
```

In `ggml-alloc.c`, `ggml_gallocr_is_allocated()` returns false for the expert
leaves because their canonical GGML objects enter the reservation graph with
`data == NULL` and `buffer == NULL`. `ggml_gallocr_allocate_node()` therefore
requests scheduler storage for the complete declared tensor size. The relevant
code does not distinguish the fact that the production bounded provider later
reads selected rows through PREAD; that provider is separate from the
canonical graph leaf dependency.

The allocator profile and expert trace show the first three full leaves:

```text
blk.0.ffn_down_exps.weight  type=20  bytes=471859200
blk.0.ffn_gate_exps.weight  type=23  bytes=445644800
blk.0.ffn_up_exps.weight    type=23  bytes=445644800
combined=1363148800 bytes
```

These are actual scheduler allocations, not merely shape metadata. Their
intervals are adjacent in the allocator chunk, and the scheduler reports
`1308.03 MiB`, consistent with the combined payload plus alignment and other
small allocations. Equivalent expert leaves for later layers are allocated or
reused as streamed contexts are constructed.

## Safety conclusion

The allocation is unnecessary only if the canonical expert leaf is removed
from the executed graph dependency and every consumer is replaced with a
validated bounded provider output. That graph-dependency proof is not present
in this iteration. Attaching null or undersized selected-row storage would be
invalid because the canonical tensor declares hundreds of MiB and stock GGML
may consume it during reservation or execution.

Therefore no tensor or scheduler allocation was modified. A one-tensor removal
experiment would not be safe yet and could change the actual graph dependency,
not merely reduce unused storage.

## Regression

```text
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-flashnext-iq4xs-provider-test -j4
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  flashnext-tensor-index.json
```

Build exit: 0. Iteration 145 one-token and three-token routed aggregate
regression: PASS. The existing bounds remain `max_abs <= 1e-5` and
`RMS <= 2e-6`.

## Status

- Iteration 145 numerical regression: **PASS**
- Expert leaves connected to scheduler allocations: **PASS**
- Scheduler reservation mechanism identified: **PASS**
- One expert allocation proven removable without graph change: **FAIL**
- Safe allocation experiment: **NOT TESTED**
- Before/after memory reduction: **NOT TESTED**
- Arithmetic changes: **NONE**
- `git diff --check`: **PASS**

## Recommended next experiment

Construct a test-gated layer-2 graph in which one selected projection’s
canonical full expert leaf is absent and the bounded provider output is an
explicit graph input. Enumerate graph leaves and execute that graph before
comparing scheduler reservation and RSS. Only after numerical equivalence and
dependency removal are proven should the same ownership change be considered
for production streaming.
