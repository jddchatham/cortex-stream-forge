# QIES-085 — Redundant staged-state snapshot investigation

Date: 2026-11-07. **GO for a separate, narrowly scoped implementation experiment removing the unused staged copy. No production change or inference run was made.** The copy is genuinely duplicated anonymous memory, with a measured 113.37 MiB incremental RSS/PSS cost at the representative state size. Overall inference peak reduction remains unmeasured and may be zero because a later save-time allocation overlaps the retained rollback snapshot.

## Scope and evidence standard

Read QIES-082, QIES-083 and QIES-084. Audited the published `flashnext-chat.cpp` ownership paths first, the related owned-layer diagnostic/driver, serializer entrypoints and the existing optimized executable. Published baseline commit remains `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`.

Measurements below come from a bounded **standalone C++ ownership diagnostic**, not QUORTEX inference or a live heap profile of the old run. It uses the default C++ vector allocator with the installed compiler and -O3/-DNDEBUG, no replacement allocator, trimming, tuning or cache-eviction calls. One original 118,870,628-byte state file is mapped read-only as the representative byte source. The diagnostic creates ordinary vectors of that size, performs the same empty-vector copy assignment, tests independence and reference aliasing, then separately models save-time overlap. This does not establish every real pre-decode snapshot's size, the original inference peak or any numerical correctness result.

## Ownership and lifetime map

All chat source paths below are under `cortex-stream-forge-qies074/runtime-flashnext-layer-exp/examples/flashnext-chat.cpp`.

| Object / creator / owner | Allocation and copied source | Readers/writers and lifetime | Classification and removal consequence |
|---|---|---|---|
| Active hybrid state, llama context/memory subsystem | Context RS/KV and associated state; QIES-082 logged 112.57 MiB RS plus 6/1.5 MiB KV capacities, not an additive snapshot size | Restored before decode; graph A updates context; serializer reads it; freed with context after layer | Authoritative mutable computational state. Cannot substitute an old snapshot or discard backing without continuity proof |
| `capture_seq_state()` local vector, lines 125–129 | Queries serialized length n, value-initializes n bytes, serializer fills the buffer | Returned to caller; return-value elision/move normally transfers vector storage, not another n-byte payload copy. Failed data capture calls clear(), which does not itself guarantee capacity release; destruction eventually releases allocation | Serialization owner. Live anonymous copy of context state, not a reference to context tensors or a file mapping |
| `result.state_before`, created in `run_layer_decode`, line 349 | Takes ownership of capture made **before** llama_decode/graph A | Read by rollback calls at 550/558. Travels with transaction result to loop variable `res` at 767. Survives save_layer_state and context teardown; destroyed at end of loop iteration | Required pre-transaction rollback snapshot under current contract. It becomes old-state data after graph A mutates context. Removing it breaks rollback; capturing it after graph A is too late |
| `txc.staged_state`, declaration 375, copy assignment 536 | Fresh empty vector receives a deep copy of `result.state_before`; n additional bytes | Only assignment and implicit destruction found. Helpers receive txc, but read selected fields, never staged_state or its pointer. Scope ends on every return/unwind from run_layer_transaction, before caller saves state | **Redundant** owned duplicate. No checkpoint, rollback, disk-store or graph reader. Removing this allocation does not remove the rollback owner |
| Returned `result` / caller `res` | Ownership transfer of result and its vector; named-return elision or implicit move expected | Caller tests flags/output, calls save, then destroys result at loop end | Not a new staged payload copy. Do not multiply n by each C++ variable name or return boundary. No historical allocation trace proves every compiler path, but no explicit lvalue result-copy exists on this chat success path |
| `save_layer_state` local `const state`, line 325 | New serialization of **post-transaction** context; separate n_after allocation | Read by store.store -> memcpy, then destroyed as save helper returns. Overlaps caller res.state_before; txc.staged_state has already died | Required current save transport under existing API. Old rollback bytes are not equivalent to newly committed bytes. Reuse/defer ideas need separate commit/failure audit |
| State-store slot / 48 persistent files, lines 258–331 | mmap MAP_SHARED, O_RDWR/O_EXCL, ftruncate on growth; memcpy from save vector | Future positions restore via data()/size() and llama_state_seq_set_data. Slot mappings survive layers/tokens until growth/destruction | Authoritative persistent per-layer handoff after successful save. Not redundant anonymous staged allocation; dropping it would lose multi-token state |
| HC/output/PLE token-history handoff | Result arrays, txc arrays and loop-owned output/token history | Used for subsequent layers/tokens; independently managed from serialized-state vector | Required live inputs/outputs. Their sizes and semantics must not be conflated with unused staged_state |

### Order and overlap

For streamed layers 1–47: restore persistent state -> capture pre-state A -> graph A changes context -> allocate staged duplicate B=A -> router/MoE/graph B -> destroy B on transaction return -> retain A in caller result -> allocate save serialization C of current context -> copy C into persistent slot -> destroy C -> free context/model -> destroy A at loop end.

Thus A+B overlap during transaction helpers, and A+C overlap during save. **B and C do not overlap on this chat path.** Layer 0 uses its separate direct decode/save path and does not create this txc staged duplicate. There are not 48 staged copies retained simultaneously. The accepted 27-step workload implies up to 47×27=1,269 staged-copy executions on successful streamed layers, not 1,269 simultaneously resident copies. Payload sizes are dynamic; repeated logical copy traffic is not SSD traffic or peak RAM.

### Proof that staged_state is unused

Repository-wide search finds the chat member declaration and one assignment only. The struct is translation-unit-local; no address/pointer to staged_state is exported. Router capture, selected MoE and graph B access tensor_index, layer_desc, values and numeric arrays. Both rollback sites read result.state_before. save_layer_state independently serializes the context. Destruction of the unused byte-vector has no state-persistence role.

This is more than source-level dead storage: disassembly of the **existing published executable** shows the nonempty result-state branch at 0xd94d to 0xdc90, `operator new` at 0xdc9a, and `memcpy` at 0xdcac, returning to transaction setup at 0xd953. The unused copy was not optimized out in that build. Supporting excerpts are saved under qies075-data/logs/qies085. No executable was rebuilt or replaced.

### Related diagnostics, not additional chat allocations

`flashnext-owned-layer.cpp` has the same member and transaction assignment (633/994), plus older diagnostic main-path assignments at 1834 and 1856. Search found no reads of that member there either. The second assignment to an already-sized vector copies bytes again but does not imply a second simultaneously owned allocation; capacity reuse is permitted. `state_before_transaction` at 1684 is a const reference to transaction.state_before, not another vector payload.

Other owned-layer snapshots have actual readers: state0/state1 around 1218/1245 are exported; layer2_before/after around 1461/1472 support rollback comparisons; committed_state and state_before_pos1 around 1771/1777 support handoff equality, with further position/chain captures around 1923–1932, 2073/2121 and 2331/2471/2550 used by diagnostic/generation paths. These captures are not additional owners in the quortex chat executable. Preserve their test/reference functions rather than deleting captures by name. The separate layer driver includes owned-layer source and exposes state-copy APIs; it is not the chat target. No diagnostic-only snapshot memory is attributed to the production run here.

## Bounded measurement

Diagnostic source/binary: `qies085-diagnostics/snapshot_probe.cpp` / `snapshot_probe`. No engine/model linkage. It reads one original file via O_RDONLY/O_NOFOLLOW and a read-only shared mapping. Three trials test distinct vector addresses, full vector equality, mutation of the copy without mutation of source, a const-reference alias control, and save-buffer overlap with unchanged pre-state. Original bytes remain unchanged. The modeled save buffer is deliberately modified to represent a distinct generation; it is not a valid inferred QUORTEX state and is never restored into an engine.

Entry requirements: 3 GiB MemAvailable, 4 GiB swap free, 30 GiB SSD free. Runtime host reserve 2 GiB, swap free 4 GiB, disk free 20 GiB, full memory PSI avg10 below 5%, no cgroup OOM, 60-second deadline. A transient user service independently enforced MemoryMax=512M, MemorySwapMax=0, TasksMax=16, RuntimeMaxSec=70, TimeoutStopSec=2s. Existing inference guards, allocators and system settings were untouched. The source compiled with warnings enabled; final compile was clean. Service result: success, 2.067 seconds, peak cgroup 343.2 MiB, zero swap.

Per-stage rollup values below are MiB; anonymous pages are a subset of RSS/PSS, not an addition. Shared library PSS explains the RSS/PSS difference. The read-only file source remains mapped across vector trials to distinguish it from anonymous duplication.

| Stage (trial 1 unless stated) | Process RSS | Process PSS | Anonymous | Private dirty | Cgroup current | MemAvailable |
|---|---:|---:|---:|---:|---:|---:|
| 00-start | 4.090 | 0.498 | 0.324 | 0.375 | 0.664 | 4067.250 |
| 01-file-hot | 117.465 | 113.873 | 0.332 | 0.383 | 114.367 | 4081.051 |
| t1-capture | 230.832 | 227.240 | 113.699 | 113.750 | 228.113 | 3948.367 |
| t1-staged | 344.199 | 340.607 | 227.066 | 227.117 | 341.863 | 3847.469 |
| t1-staged-destroyed | 230.832 | 227.240 | 113.699 | 113.750 | 228.242 | 3948.590 |
| t1-const-reference | 230.836 | 227.244 | 113.703 | 113.754 | 228.492 | 3948.762 |
| t1-save-overlap | 344.203 | 340.611 | 227.070 | 227.121 | 341.992 | 3846.516 |
| t1-save-destroyed | 230.836 | 227.244 | 113.703 | 113.754 | 228.621 | 3950.070 |
| 99-all-vectors-destroyed | 117.469 | 113.877 | 0.336 | 0.387 | 116.270 | 4088.109 |
| zz-unmapped | 4.102 | 0.510 | 0.336 | 0.387 | 116.270 | 4087.055 |

Incremental staged-copy cost, capture -> staged:

| Trial | RSS increase | PSS increase | Anonymous increase | Cgroup increase | MemAvailable change |
|---|---:|---:|---:|---:|---:|
| 1 | 113.367 | 113.367 | 113.367 | 113.750 | -100.898 |
| 2 | 113.367 | 113.367 | 113.367 | 113.500 | -114.000 |
| 3 | 113.367 | 113.367 | 113.367 | 113.750 | -112.160 |

Both vector size and capacity were exactly 118,870,628 bytes in this fixture. Each copy touched about **113.367 MiB of independent anonymous/private-dirty resident pages**, not shared file-cache pages. RSS/PSS fell by the same amount after its normal destructor in all trials. No manual free/trim or allocator intervention was used. A const reference added zero RSS/PSS at sampled precision and retained the same address; it is a lifetime demonstration, not a production patch. File unmapping at the end removed about 113.367 MiB of file-backed RSS but left the cgroup cache charge, consistent with QIES-083.

At this isolated size, MemAvailable fell about 101–114 MiB when allocating the extra staged buffer. Background desktop activity makes that a noisy host observation, not exact per-object attribution; the matching anonymous/PSS and cgroup increments provide stronger evidence of real duplication. Zero probe swap and unchanged host SwapFree throughout rule out shifting the cost into swap during this test. Baseline host availability was approximately 3.97 GiB and stayed above reserves. File reads/cache effects precede the copy measurement; anonymous increments are not inferred by subtracting virtual mapped size.

The modeled save overlap added the same approximately 113.367 MiB RSS/PSS as the staged copy. Therefore this two-buffer model's maximum is essentially unchanged if only the earlier unused copy is removed. In real inference, helper/weight/graph lifetimes differ by phase and may make the earlier transaction peak dominant; that requires instrumentation before any peak-saving claim. The separate probe does not measure context RS/KV allocation or byte size at earlier token positions, and historical QIES-081 PSS/heap attribution cannot be recovered.

## Estimated reclaimable memory and alternatives

- **Proven removable ownership:** one unused staged byte-vector on successful streamed-layer transactions. At this representative size, 113.367 MiB additional resident anonymous memory during that phase, plus small allocation overhead.
- **Whole-run peak reduction estimate:** 0 to approximately 113 MiB, conditional on which phase dominates. No GiB-scale saving is supported. This alone will not bridge the approximately 2 GiB desktop admission shortfall or justify changing the gate.
- **Best initial implementation candidate:** remove only the unused member and assignment in an isolated local experiment. No reference is needed for a value with no reader. A reference would introduce lifetime obligations without a present benefit.
- **Do not defer the required pre-state capture past graph A** or alias mutable context/state-store memory as rollback. The context mutates; persistent slots may be replaced. A borrowed alias would cease to be an immutable pre-state when its backing changes or unmaps.
- **Do not remove save serialization by substituting pre-state.** It describes a different generation. Reducing the later A+C overlap might be useful, but explicitly narrowing A's lifetime after all rollback readers end changes failure/ownership contracts and needs a separate experiment. Calling vector.clear() alone is not proof memory returned.

## Risks, acceptance and recommendation

**GO for a separate minimal removal experiment; NO-GO for claiming accepted numerical correctness or desktop readiness now.** Source and optimized-binary evidence establish unused duplicated storage. Removing an unused allocation still changes allocation timing, addresses and possible allocation-failure behavior, so bit-exactness must be tested rather than inferred solely from dead-data analysis. No inference arithmetic change is needed or authorized here.

The experiment should preserve result.state_before and persistent state semantics, instrument transaction/save boundaries, compare baseline/candidate anonymous PSS and cgroup peaks, and require exact pre-state, committed serialization, rollback restoration and retry output comparisons on small layer tests. Exercise sparse-MoE and graph-B failure paths, descriptor/decode failures and repeated-token transitions. The baseline currently does not uniformly restore pre-state on every error return, and rollback return values are not fully checked; document existing behavior rather than silently changing it with this memory experiment.

Preserve attention KV, DeltaNet recurrence, indexer, PLE and HC/token-history handoffs. Progress to separately authorized guarded numerical tests only after small checks pass; token agreement alone is insufficient. Retain reference selection arrays and ultimately verify the relevant 48-layer/full-logit bit-exact standard. No full model was run in this investigation.

Next strongest candidate if peak improvement is negligible: the overlap of retained rollback capture with save-time serialization, followed by the 112.57 MiB active recurrent-state allocation/inactive backing audit from QIES-082. First measure those actual anonymous allocations and lifetimes. Do not weaken guards, evict mandatory state or replace serialization with guessed padding to pursue savings.

## Preserved evidence

- `qies075-data/logs/qies085/trial-20261107-ownership/`: raw per-stage smaps, rollups, host memory, cgroup memory/stat/events/swap, process I/O, allocation addresses/capacities and page-fault counters; `PASS`, `summary.json`, `baseline-verification.json`.
- `qies075-data/logs/qies085/chat-snapshot-references.txt`, `staged-copy-branch.asm`, `staged-copy-allocation.asm`.
- `qies085-diagnostics/snapshot_probe.cpp`, standalone executable and analysis script.

All 12 protected baseline hashes and all 48 state sizes/mtimes passed verification; the original layer-0 SHA256 also matched. Both native source trees have no tracked differences. No source/guard/library/executable replacement, model modification, allocator setting, system configuration change, GPU work, commit or push occurred. New diagnostic artifacts are outside the production inference source tree.
