# QIES111 — Ripple 80-layer streaming orchestration

**GO for isolated QIES112 synthetic development. NO-GO for real-weight or production execution.** All 80 real-width synthetic layers completed two-token continuation and matched the independent stock-GGML reference byte-for-byte.

## Isolation and source identity

New detached worktree: `<workspace>/qies111-worktree`.
HEAD: `14b3f7902864b02e1a84e5e6168a3cf99087e670`; verified engine ancestor `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. QIES106–110 experimental sources were copied with provenance, without altering their original worktrees. There is no new commit. The exact uncommitted implementation is identified by the final source SHA-256 manifest and `qies111/implementation.diff`.

Stock reference source: llama.cpp b12167 `240d68fcfe0c2a1f298c8e18ca181368b1914259`; Ricursive source hash `427da0ad63c070b69fb2e78f6876893d4fa47ebad3f8711bef54d57d521e1f7b`. GCC 16.2.1, C++17 `-O2 -Wall -Wextra`, one CPU compute thread, i5-7200U AVX2/FMA/F16C. Preserved native Release/OpenMP SDK libraries are candidate GGML 0.25.3 and independent stock GGML 0.29.0. Build manifests contain complete commands, source hashes and binary/library hashes; the stock source check covers 4,084 files.

The independent oracle is a separately compiled conventional **synthetic** stock-GGML full-layer graph. It does not call the streamed projector or read its fixture files. It remains distinct from a fully qualified real-model llama executable. Neither this task nor the oracle establishes real-weight numerical equivalence.

## Orchestration and dataflow

`qies111/run.py` executes every requested layer in exact ascending order. It supplies the preceding layer's F32 output to the next layer and keeps the original synthetic scaled embedding unchanged for value-residual mixing. The stock source explicitly retains `inp_vres = inpL` before its layer loop. The first layer receives the same embedding at both destinations.

`common.h` validates the layer, fixed capacity, absolute position, input length/finiteness, state identity/checksum and CLI contract. The copied QIES109 state format binds each checkpoint to its exact layer and capacity. Both controller and worker reject cross-layer state and token-position drift.

`stream.cpp` retains QIES110's kernels, graph ordering, bounded row source and 128-row quantized projections. It reads the evolving activation and original embedding separately. `reference.cpp` independently consumes the same inputs through its conventional graph and chooses the matching Q4_K/Q5_K FFN profile. No kernel or accumulation order was changed.

Each layer runs in a fresh sealed synthetic-only process. Its fixed 1 MiB scratch arena is reused across the layer's sub-operations, then released. Scratch addresses are not retained across subprocesses. This isolates allocator retention and crashes at the cost of process and context startup; it is not a persistent-context redesign or a production performance claim.

Only two boundedly generated synthetic tensor containers are needed, shared read-only across layer-bound plans. Weights intentionally repeat within each quantization profile; they are not eighty independently sampled layers or real model weights. All tensor/activation dimensions are actual Ripple widths. The real GGUF is never mounted in numerical namespaces.

## Verified model schedule

`layer-schedule.json` is checked against the actual GGUF metadata: 64 sliding-window layers and 16 global layers. Layer IDs with `id % 5 == 4` are global; each intervening group of four uses a 1,024-position sliding window. Q/K positional handling follows the preserved stock graph for the respective variant.

FFN layers 0–14 and 65–79 use Q5_K; 15–64 use Q4_K. Q/gating projections use Q4_K, K/V Q8_0, attention output Q6_K, and scale/lambda parameters F32. Per-layer schema, quantized row widths and file identities remain enforced by the unchanged QIES110 reader.

## Whole-token state and supported recovery

Only the active layer's KV payload, pending copy, GGML buffers and bounded serialization are loaded in its worker. Independent layer checkpoints persist on SSD. Previous whole-token state is mounted read-only. Required state therefore grows on disk with layer count; active numerical process residency need not grow with the stack.

A token stages all its layers under `.UNCOMMITTED`. After all succeed, the controller writes and synchronizes `COMMIT.json`, synchronizes nested capture directories and atomically publishes `.COMMITTED`, followed by parent-directory synchronization. The commit binds layer IDs, token position, capacity, state hashes and output-manifest hashes. Previous committed token directories remain intact. Unpublished state cannot become a prior token.

Restart checks source/controller/build identities, input hashes, output hashes, layer identities and checkpoint positions before reusing completed work. Verified layer prefixes are not reevaluated; repeated calls for an already committed token do not advance it again. Partial outputs and failed attempts remain preserved separately.

This is a synthetic controller prototype, not a production durable-generation protocol. Publication/synchronization failures can have an uncertain durability outcome; a restart verifies the published checkpoint rather than claiming storage rollback. It provides no exactly-once external text delivery. Standalone checkpoints still need stronger model/configuration/session identity before production integration.

## Sliding-window scope

For sliding layers, entries older than the allowed window are cleared in **pending** KV after attention evaluation, before publication; the authoritative input remains untouched. Global layers retain that history. The five-layer boundary case checks positions 1023 and 1024, containing both variants. Seeded position-1023 history is synthetic fixture history, not 1,023 evaluated tokens.

This executor uses fixed 256/1,280-slot capacities. Clearing expired entries does not shrink the allocation and is not a reusable KV ring. Execution beyond capacity fails closed. Arbitrary-length rolling generation remains future work.

## Baseline and failure validation

Fresh QIES109 replay: 312 exact captures and 78 faults (`qies109/runs/20261111-172821-771dbd8a`). Fresh QIES110 replay: 312 exact captures and 78 faults (`qies110/runs/20261111-173154-3788816b`). The new worktree also reproduced 30 independent real-tensor storage hashes, 38 QIES101 extraction faults, seven malformed-plan refusals and 20 ASan/UBSan adapter cases. Real-byte tests performed storage operations only; the model identity remains bound to the existing verified whole-file receipt, not a newly claimed whole-40-GB hash.

New controller tests: 31 cases, including interruptions before/after each layer, prior-state integrity, invalid-state rejection, session/role/layer/position rejection, publication boundaries and duplicate-source/distinct-destination bindings. Numerical/entry tests: 13 cases, including KV-update, serialization, capture and publication faults, real/no-marker refusals for both actors, cross-layer KV and mismatched weight-plan layer.

New numerical caller ASan/UBSan tests: four cases, covering exact continuation plus KV-update, serialization and publication failures, with leak detection enabled. The adapter/caller/state code is instrumented; the prebuilt GGML libraries are not. Two actual synthetic restart cases confirm byte-exact continuation after an interrupted prefix or pre-publication failure, without reevaluating verified prefixes or modifying prior checkpoints. Tests do not simulate every real power-loss/filesystem failure.

## Crash evidence

The historical reported crash remains **unresolved**: no identified target executable/command/PID/signal and no matching available core or kernel OOM evidence. Current-day diagnostics are saved in `cores.log`, `kernel.log`, `crash-evidence.json`. Absence of evidence does not establish a cause.

One reproducible QIES111 startup error was isolated and fixed before extending to 80 layers. A dictionary of bind mounts collapsed the identical first-layer input/original source into one destination, leaving `/state/input.f32` missing. The worker rejected the input with ordinary exit 1 before weight execution; it did not segfault or advance state. The failed campaign `20261111-173515-232345df` is preserved. A list of source/destination pairs and a dedicated regression fixed it. This is not identified as the historical crash's cause.

## Safety and remaining integration limits

Real arithmetic still enters unchanged QIES104 admission: the original 6 GiB MemAvailable gate followed by an unconditional unqualified-runtime lock. No new option unlocks it. Production FlashNext, QIES086, frozen refs and global policies are untouched.

Synthetic workers retain the existing independent model-free monitoring: 1.5/1.75 GiB candidate/reference entry, 1 GiB running MemAvailable floor, 2 GiB swap and 20 GiB disk reserves, 256/640 MiB worker RSS ceilings, 120-second per-worker timeout and ancestor OOM-event monitoring. Existing cgroup and address-space settings are observed without reconfiguration; shared desktop memory peaks are not isolated worker accounting.

This milestone stops at transformer-layer outputs. Token embedding/output-head execution, tokenizer/sampling, arbitrary context length, production checkpoint identity and a qualified real-weight reference/guard path remain outside its validation. Synthetic weights repeat, so timings and filesystem-cache behavior cannot be extrapolated to eighty distinct real-weight layers.

## Completed comparisons and memory

| Consecutive layers | Positions | Byte-exact captures | Independent weight hashes |
|---:|---|---:|---:|
| 2 | 0, 1 | 104 | 32 |
| 4 | 0, 1 | 208 | 64 |
| 5 | 1023, 1024 | 260 | 80 |
| 8 | 0, 1 | 416 | 128 |
| 80 | 0, 1 | 4160 | 1280 |

Total: **5,148 captures and 1,584 independently matching projection-weight hashes**. The 80-layer/two-token stretch contributes **4,160 captures**. Every layer boundary, all captured intermediates and per-layer KV checkpoint matched byte-for-byte. No tolerance was introduced. The final independent audit also checked actual activation routing, fixed original embedding, prior-checkpoint identities, masks, zero masked probabilities, RoPE variant behavior, quantization profiles and cleanup.

Evidence: `qies111/verified-results.json`; small campaign `qies111/runs/20261111-173551-4fd516a5`; full stack `qies111/runs/20261111-174023-e367d945`. Per-job results retain exit statuses, input/output identities, resource monitoring and cgroup events. All accepted numerical jobs exited zero; deliberate refusals/faults exited one. No recorded signal exit, OOM-event increase or monitor safety violation occurred in these accepted campaigns.

| Layers / context | Stream RSS / PSS (MiB) | Reference RSS / PSS (MiB) | Controller RSS maximum (MiB) | Required stack KV payload (MiB) |
|---|---:|---:|---:|---:|
| 2 / 256 | 24.33 / 15.19 | 541.64 / 532.56 | 38.99 | 2.00 |
| 4 / 256 | 24.49 / 15.36 | 540.60 / 531.09 | 38.99 | 4.00 |
| 5 / 1280 | 35.84 / 26.70 | 560.21 / 551.30 | 37.73 | 25.00 |
| 8 / 256 | 38.99 / 31.40 | 541.74 / 532.68 | 38.99 | 8.00 |
| 80 / 256 | 38.96 / 31.37 | 541.72 / 532.65 | 38.96 | 80.00 |

Full-stack streaming anonymous/file proportional residency peaks: **28.19/3.46 MiB**. Controller peak PSS: **31.37 MiB**. Its RSS across the 80-layer candidate passes was 37.98–38.96 MiB; the first/last samples were 38.96/37.98 MiB. No layer-count-proportional controller RSS growth was observed. Worker teardown returns tracked tensor/weight and source-file ownership to zero. Allocator-retained pages are still visible in cleanup RSS; zero ownership does not mean zero RSS.

The maximum live streamed weight tile is 3,604,480 bytes; full-stack peak GGML tensor-buffer allocation is 3,768,832 bytes, maximum planned compute workspace 571,136 bytes, and the reusable source cache 1,048,320 bytes. The arena itself reserves 1 MiB even when the plan needs less. Graph metadata and activation vectors remain bounded per layer. Boundary-test attention buffers are larger because context capacity is 1,280 rather than 256.

Accounting correction: raw `METRICS.json` controller keys named `*_kib` actually contain **bytes**, as returned by the inherited monitor. Raw evidence is unchanged; `verify_results.py` converts these explicitly. Worker phase telemetry KiB and process-tree monitor byte fields are distinct and interpreted separately. RSS may double-count shared pages; PSS apportions them. Process-tree samples include namespace/loader and possibly pre-exec inherited pages; they are not solely the numerical worker's private allocation. Peak components need not occur simultaneously. Measurements cover deterministic campaigns rather than a statistical performance study.

KV formula: F16 K+V is `2 × 1024 × 2 × capacity` bytes per layer: 1 MiB at 256 slots, 5 MiB at 1,280. Eighty layers therefore need **80/400 MiB per complete checkpoint payload**, plus headers/metadata. These payloads persist on SSD; the active worker loads one layer, pending copies and bounded serialization, not all eighty layers. Previous token snapshots and diagnostic captures are intentionally retained, so disk usage grows per token. Production retention/garbage collection is not implemented.

The streamed/reference process measurements exclude the separate controller; their peaks must not be treated as total system RAM. The numerical path does not map the GGUF or retain full matrices. Linux page cache for synthetic files, checkpoints and diagnostics can still affect system/cgroup memory outside process RSS. Shared desktop cgroup history is not isolated worker attribution. There is no claim that RSS reduction equals reclaimed system RAM.

These results support bounded-memory **synthetic** orchestration on this laptop. They do not validate a 4 GiB real-model system: desktop/kernel/cache headroom, longer global KV history, embedding/output-head execution and actual weight I/O remain unqualified. The unchanged 6 GiB gate itself forbids real execution on a 4 GiB host. Repeated synthetic profiles also make timing/cache behavior unsuitable for extrapolation to the full 40.426 GB model.

## Decision and QIES112

**GO** to extend the isolated synthetic framework, with all 80-layer/two-token baseline and fault tests retained. A focused next step is bounded multi-token KV wrap/retention and stronger standalone checkpoint/model/configuration identity, requiring the same independent byte-exact checks at boundaries and after restart. Correct accumulation/mask ordering must survive any future ring implementation.

**NO-GO** for production integration or real-weight arithmetic. Those require a separately qualified independent real-weight reference and live guard path, matched model/tensor/input identities, CPU configuration qualification and the unchanged minimum 6 GiB MemAvailable. Neither synthetic success nor this prototype authorizes unlocking the real path. No commits, merges, pushes, resource-policy changes, production FlashNext changes or frozen-milestone changes were made.

Reproduction: `qies111/README.md`, `pandacode-run.sh`. Exact source hashes: `qies111/SOURCE-SHA256.json`; baseline/ref/guard integrity: `qies111/final-integrity.json`.
