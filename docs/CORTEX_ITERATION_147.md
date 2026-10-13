# Cortex Iteration 147 — allocation lifetime and RSS attribution

Branch: `cortex-iteration-147`
Base: pushed Iteration 145 checkpoint `7174649`; Iteration 146 changes were
preserved without reset or stash.

## Commands

```text
git switch -c cortex-iteration-147
env CORTEX_ALLOC_PROFILE=1 CORTEX_MOE_STREAM_GATE=1 \
    CORTEX_MOE_STREAM_UP_DOWN=1 \
    CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
    llama-qwen38-lowres-chat <model> 1
```

The real runner was bounded by a 100-second timeout and fed `hello` and
`/exit`. It exited 0 in approximately 27 seconds. Full allocator output was
written to `/tmp/cortex-iter147-alloc.log` (278,923 lines, 20,197,616 bytes).

## Scheduler reservation

`llama_context::sched_reserve()` is the confirmed call site. It calls
`graph_reserve()` for the worst-case prompt graph (`n_tokens=9`), token graph,
and another prompt graph, then calls scheduler reservation and records the CPU
compute buffer. The run reported:

```text
CPU compute buffer size = 1308.03 MiB
```

The profile showed the reservation graph had 134 nodes and one split. This is a
planned scheduler compute buffer, not proof that all bytes are resident.

## Allocation interval evidence

The existing bounded `CORTEX_ALLOC_PROFILE` instrumentation emitted allocation
and free events with tensor name, type, shape, byte count, chunk, and offset.
The largest concrete model leaves at the first layer context were:

| tensor | bytes | type | observed role |
|---|---:|---:|---|
| `blk.0.ffn_down_exps.weight` | 471,859,200 | IQ4_NL | expert model leaf |
| `blk.0.ffn_gate_exps.weight` | 445,644,800 | IQ4_XS | expert model leaf |
| `blk.0.ffn_up_exps.weight` | 445,644,800 | IQ4_XS | expert model leaf |

Combined: `1,363,148,800` bytes. They were emitted at scheduler allocator
offsets adjacent to the initial model input/embedding allocations, proving that
these canonical leaves are scheduler-accounted in the profiled path. Equivalent
expert-sized leaves for other physical layers were observed at reused or
different offsets during later layer contexts. The log also contains explicit
free events and reuse messages, so repeated event count is not equivalent to
simultaneous residency.

Representative activation allocations included:

```text
hc_init       368,640 bytes
hc_gate-0     368,640 bytes
conv_input    491,520 bytes
node_32     3,145,728 bytes
```

These are orders of magnitude smaller than the three expert leaves. The
selected-MoE scratch arena remains `2,150,400` bytes and is not responsible for
the multi-gigabyte peak.

## RSS/residency monitor

A separate `/proc` monitor sampled status and `smaps_rollup` every 200 ms during
the same workload. Results:

```text
VmRSS peak:          5,622,252 kB
VmHWM:               5,647,288 kB
smaps Rss peak:      5,624,804 kB
smaps Pss peak:      5,620,299 kB
Private_Dirty peak:  5,614,616 kB
VmSwap peak:           364,436 kB
smaps Swap peak:       364,436 kB
final VmRSS:         1,759,436 kB
final smaps Rss:     1,694,056 kB
final VmSwap:           16,236 kB
```

The process-specific RSS high-water is therefore real and mostly temporary or
reclaimable by teardown. The scheduler reservation accounts for approximately
1.36 GiB of the planned/owned expert storage, but the trace does not reconcile
the remaining roughly 4.2 GiB of peak private dirty RSS to a single owner.
Potential contributors include additional layer expert leaves, model loading
copies, scheduler/graph arenas, and page touching during streamed execution.
The current profile does not safely distinguish all of these at physical-page
granularity.

## Top confirmed peak contributors

The top confirmed live-size class is the three layer-0 expert leaves above
(1,363,148,800 bytes combined). The next confirmed individual allocations are
temporary graph tensors in the low-megabyte range; no single traced temporary
approaches the RSS peak. Because allocator offsets are reused and allocation
events are repeated across contexts, a complete top-five *simultaneously live*
table cannot be proven from the current event stream alone.

## Optimization decision

No memory optimization was implemented. Removing or externally rebinding the
three expert leaves is not yet safe: the graph currently presents canonical
full-size GGML leaves to scheduler planning, while bounded PREAD execution is a
separate provider path. A smaller reservation or altered ownership could
invalidate tensor binding and numerical correctness. The evidence-supported
candidate is to eliminate scheduler ownership of canonical expert leaves only
after an executed graph-dependency proof demonstrates that every consuming MoE
operation uses bounded provider outputs instead.

## Validation

- Allocator profile execution: **PASS**, exit 0
- Scheduler reservation attribution: **PARTIAL**; call site and expert leaves
  proven, full RSS reconciliation incomplete
- RSS/HWM/swap sampling: **PASS**
- Top-five simultaneous live allocation proof: **NOT TESTED**
- Safe memory reduction: **NOT TESTED**
- Arithmetic changes: **NONE**
- Iteration 145 numerical regression: preserved from checkpoint; no production
  code changed in Iteration 147
- `git diff --check`: **PASS**

## Smallest next experiment

Extend the allocator trace with a stable allocation ID and release/reuse table
for each backend buffer, then sample RSS at the exact scheduler-reserve and
MoE execution boundaries. Correlate model-leaf buffer IDs with physical layer
activation and prove whether the additional expert leaves are simultaneously
resident before changing ownership or scheduler policy.
