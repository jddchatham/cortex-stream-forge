# QIES106 — Ripple real-width streaming architecture

**Scope:** isolated procedural-synthetic real-width operations and a complete-layer streaming design. No real GGUF tensor bytes were read, no production code was changed, and no real execution was unlocked. The frozen QIES105 archive remains intact. Runtime results and the final recommendation appear below after the architecture/accounting sections.

## Source, input identities and implementation boundary

Worktree: `<workspace>/qies106-worktree`, detached at QIES105 commit `14b3f7902864b02e1a84e5e6168a3cf99087e670`. Its verified engine ancestor remains `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. All implementation additions are under `qies106/`; only this isolated worktree's `pandacode-run.sh` dispatch changes. No commits, merges or pushes are authorized or performed.

Accounting uses the actual previously verified Ripple descriptor index `qies098-checks/ripple-tensor-index.json`, cross-checked against all 1,203 independent stock descriptors in `qies104/reference-1203-descriptors.tsv`. The matching model identity is SHA256 `9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc`; QIES106 does not rehash or open the 40,426,079,712-byte GGUF. Exact descriptor hashes/calculations are in `qies106/memory-accounting.json`.

Stock source is independent llama.cpp b12167 `240d68fcfe0c2a1f298c8e18ca181368b1914259`; all 4,084 pinned source hashes were rechecked. Relevant source: `src/models/ricursive.cpp` and `llama-graph.cpp::build_lora_mm` (ordinary `ggml_mul_mat` when no LoRA/scales are present). The experiment reuses QIES105's source-built CPU SDKs, copying and hashing three GGML libraries into its own build directory. It does not modify their originals or enable the retired `libllama` C ABI. Candidate GGML is 0.25.3; stock GGML is 0.29.0. These are different source revisions, not identical kernels by declaration.

New files:

- `account.py`: independent descriptor consistency, exact row/tile/activation/KV calculations, two separately derived case catalogs.
- `stream.cpp`: bounded output-row streaming matvec, explicit resource ownership and partial-output cleanup.
- `reference.cpp`: independent upstream whole-matrix CPU observer. No candidate numerical/scheduling source is compiled into it.
- `fixture.h`: shared deterministic **input definition only**; dyadic activations/weights, no model readers. Each worker independently quantizes its own fixture and hashes all quantized bytes.
- `telemetry.h`: `/proc/self/smaps_rollup` and kernel peak/fault diagnostics.
- `build.py`, `run.py`: reproducible CPU compilation, sealed execution, process-tree/system/cgroup observation, exact comparison and fault tests.
- `plan.cpp`, `plan_only.py`: metadata-only GGML graph-buffer and workspace plans, including large contexts without allocating their data buffers.
- `README.md`: exact runnable instructions and supported scope. No production Ricursive or FlashNext implementation is replaced.

## Descriptor-derived weight and activation budget

Layer-0 GGUF dimensions below are `[input columns, output rows]`. MiB means 1,048,576 bytes. These are exact **calculated byte counts**, not measured RSS.

| Tensor/group | Type | Dimensions | Whole tensor MiB | 128-output-row weight tile MiB |
|---|---|---|---:|---:|
| Q projection | Q4_K | 5120 × 10240 | 28.125 | 0.3515625 |
| K projection | Q8_0 | 5120 × 1024 | 5.3125 | 0.6640625 |
| V projection | Q8_0 | 5120 × 1024 | 5.3125 | 0.6640625 |
| Attention sigmoid-gate projection | Q4_K | 5120 × 10240 | 28.125 | 0.3515625 |
| Attention output projection | Q6_K | 10240 × 5120 | 41.015625 | 1.025390625 |
| FFN gate | Q5_K | 5120 × 40960 | 137.5 | 0.4296875 |
| FFN up | Q5_K | 5120 × 40960 | 137.5 | 0.4296875 |
| FFN down | Q5_K | 40960 × 5120 | 137.5 | 3.4375 |
| Four hidden-width RMS scale vectors | F32 | 5120 each | 0.078125 combined | load individually |
| Q/K RMS scale vectors | F32 | 128 each | 0.0009765625 combined | 512 bytes each |
| Value-residual lambda | F32 | scalar | 4 bytes | 4 bytes |

The complete layer-0 weights total **545,752,068 bytes (520.46973 MiB)**. This AD-Q4_K_M model does not use Q4 for every matrix: each FFN matrix uses Q5_K in 30 layers and Q4_K in 50. Treating all three layer-0 FFNs as Q4 would undercount. All tensor types/sizes were checked against quantization block lengths; no format conversion is proposed for real weights.

F32 activation sizes per token: hidden/x0/residual each 20 KiB; Q and attention/gate each 40 KiB; K or V each 4 KiB; each full FFN gate/up/activation 160 KiB. Keeping all three FFN vectors simultaneously is **480 KiB**, small enough to preserve operation boundaries rather than changing the reduction order. Q/K RMS reduces complete 128-element heads; hidden RMS reduces all 5,120 elements. Value residual preserves `x0 + (a - x0) * sigmoid(lambda)` in that GGML order.

Clamping is the existing upstream `ggml_swiglu_clamp`, with limit 20 from metadata: the gate has an upper clamp, the up branch has a symmetric clamp, and then the existing SiLU/multiplication sequence runs. Do not replace it with a superficially equivalent clamp placement, custom activation or fused approximation.

## Complete-layer streaming dataflow and last-consumer schedule

The proposed outer token/layer order stays unchanged; within a layer the existing operator dependencies and precision are retained. Output-row tiling never splits the dot-product reduction axis. GGML computes each output using its full 5,120/10,240/40,960-element input; there are no partial-sum accumulators or newly reordered reductions.

| Stage | Required live values | Weight/storage lifetime and release boundary |
|---|---|---|
| Pre-attention RMS | layer input x, original x0, normalized a | One 20 KiB scale vector; retain x for the attention residual and **retain a through attention gating**, not merely Q/K/V |
| Q then K projection | a, Q40 KiB, K4 KiB | One quantized row tile at a time; overwrite only after synchronous compute and output copy finish |
| Value residual and V | x0, a, lambda, V-input20 KiB, V4 KiB | Scalar sigmoid then sub/mul/add; release V-input after V projection; x0 persists across layers |
| Per-head RMS and SWA RoPE | Q/K, tiny gamma vectors, position | Preserve full head reduction; RoPE only for SWA layers; global layers retain NoPE behavior |
| Tentative KV update/attention | Q, selected layer's persistent KV, mask, tentative new K/V | Keep authoritative prior state valid until successful layer completion; do not publish a streamed projection as a state commit |
| Attention gating | attention result, **a**, gate40 KiB | Compute projected gate, sigmoid, elementwise multiply; release a and ungated/gate buffers after final consumer |
| O projection/post-RMS/residual | gated attention40 KiB, original x, post-RMS scale | One Q6_K tile; create residual A20 KiB; release original x only after addition |
| FFN RMS, gate, up | A, FFN input20 KiB, gate160 KiB, up160 KiB | Reuse the same bounded weight storage across the two projections; retain activation outputs only |
| SwiGLU/clamp | gate/up, active160 KiB | Existing GGML operator; free gate/up after it returns; no unvalidated in-place assumption |
| Down, post-RMS, final residual | **full active160 KiB**, A20 KiB, output20 KiB | Stream output rows with full 40,960-column reductions; release active after down, A after final add |
| Existing state/output boundary | successful layer output and next KV | Preserve serialized-state ownership and existing publication boundaries; any failed stage leaves no new committed state |

A future real source can use bounded `pread` ranges at `tensor.offset + row * row_bytes`, copy original quantized bytes into the one active GGML tile, and validate identity/ranges. No entire layer/model mapping, requantization, asynchronous use-after-reuse or graph retention is needed. That real reader is deliberately **not connected** in this prototype.

The implemented reusable allocation is largest for FFN-down: **3,604,480 weight bytes** at 128 rows, plus its full F32 input, small output, bounded source row, metadata and measured GGML scratch. This tile size is a chosen bound, **not a mathematical minimum**. A 137.5 MiB whole FFN matrix is not unavoidable. To preserve the tested full-inner-product approach, the 160 KiB FFN input remains live with at least one complete quantized row and kernel scratch. At long context the KV and attention allocations, not these weight tiles, dominate.

## GGML graph planning: calculated sizes, not resident allocations

`plan.cpp` uses `ggml_init(no_alloc=true)`, `ggml_gallocr_reserve_n_size` (the upstream implementation explicitly uses `no_alloc=true`) and `ggml_graph_plan`. It never calls tensor-buffer reserve/allocation, work-buffer allocation or graph compute. Inputs are pinned through each isolated stage so the planner does not silently reuse a leaf needed outside that stage. These are planner-derived requirements, not measurements of physical RAM or a completed streaming layer.

| Isolated stage | Planned tensor buffer bytes | Planned CPU scratch bytes |
|---|---:|---:|
| Hidden RMS + gamma | 61,440 | 0 |
| Q and K per-head RMS + gamma | 91,136 | 0 |
| Value residual mixing | 61,504 | 0 |
| Sigmoid attention gating, excluding its projection | 122,880 | 0 |
| SwiGLU clamp, gate/up/output | 491,520 | 0 |
| Attention, C=256 | 1,737,728 | 41,024 |
| Attention, C=1024 | 6,705,152 | 163,904 |
| Attention, C=4096 | 26,574,848 | 655,424 |
| Attention, C=32768 | 212,025,344 | 5,242,944 |
| Attention, C=2,097,152 advertised maximum | 13,564,461,056 | 335,544,384 |

Attention includes F16 K/V, Q, masking, score/softmax and the existing contiguous transposed-V path; no attention arithmetic was executed here. Different future lifetime/external-buffer pinning changes these plans. Prior serialized-state copies, other layers, process libraries and the OS are additional costs.

The initial 256 KiB scratch assumption rejected the first reference before arithmetic. Source audit found GGML's tiled CPU workspace reserves a 512 KiB slot per task plus alignment and input quantization. The new experimental allocation precondition is 1 MiB for these fixed single-threaded matvecs, derived from that formula and checked against actual plans. This is a new prototype's allocation bound, not a change to an existing inference guard. Large-context attention would need separately budgeted scratch; the prototype does not grant it execution.

## KV scaling and the 4 GiB objective

F16 K+V requires `2 × 128 × 8 × C × 2 = 4096 C` bytes per layer. Keeping all 80 layers at full context requires `327680 C` bytes: 80 MiB at C256, 320 MiB at C1024 and 1.25 GiB at C4096. A selected-layer serialized-state architecture need not hold all those layers resident simultaneously, but state storage/cache residency still matters.

A future correctly implemented 64-SWA/16-global scheme retaining only a 1024-token SWA window would require `4096 × (64 min(C,1024) + 16 C)` bytes for all KV: 512 MiB at C4096, 1.25 GiB at C16384, 2.25 GiB at C32768, and 4.25 GiB at C65536. This is a **calculation**, not a claim that the current state format/windowing already implements this optimization. Indexing, position and eviction correctness would require separate tests.

At the advertised 2,097,152 context, **one global layer's K+V alone is 8 GiB**, the conventional contiguous transposed V adds 4 GiB, and the planner requires more than 12 GiB for the attention stage. Output-row weight streaming cannot make that unmodified attention path fit a 4 GiB machine. Changing softmax/reduction scheduling or dropping history is outside this experiment and cannot be assumed byte-exact.

For a short bounded C256 test, the calculated one-layer KV plus attention, <1 MiB of FFN activations and <8 MiB of explicit matvec allocations suggest a modest process budget is plausible; libraries/loader/serialization/allocator/OS remain additional and the complete layer must be measured. An illustrative 4 GiB allocation of 1 GiB OS/services + 2 GiB engine/state/cache + 1 GiB reserve is a planning scenario, **not measured hardware qualification**. The current laptop has about 7.63 GiB physical RAM and a busy desktop; its process results cannot establish the same desktop fits in 4 GiB.

Most decisively, **4 GiB total RAM cannot pass the unchanged 6 GiB MemAvailable real-execution gate**. No policy is lowered or reinterpreted here. The success claim is real-width **synthetic operation** feasibility, not permission or readiness to run real Ripple on 4 GiB.

## Working implementation and independent numerical results

Successful final campaign: `qies106/runs/20261111-110200-6ff0dc68/summary.json`. The source-built binary identities, copied CPU library hashes, exact compiler commands, source hashes, quantized fixture hashes and output hashes are recorded there and in `qies106/build/build-manifest.json`. Final recompilation using the documented SDK-state environment variable produced **identical worker binary SHA256s** to that successful campaign.

The candidate creates one fixed-tile graph and one CPU tensor buffer, supplies the full activation input, generates a bounded synthetic weight row, and fills the active quantized tile. Each synchronous `ggml_graph_compute` completes before its output is copied and its weight storage reused. The workspace, activation and source-row buffers are reused throughout the operation. The private CPU buffer is destroyed before graph metadata/backend teardown. No array grows with the number of tiles, no complete weight tensor is kept, and SHA256 is updated incrementally rather than retaining a weight history.

The independent reference separately creates the full **single-matrix** tensor, fills it row-by-row from the deterministic input contract, then executes one unpartitioned upstream `ggml_mul_mat` graph. It does not compile or invoke `stream.cpp`, the tiled scheduler or a candidate result. Sharing the input definition and telemetry is not sharing expected numerical outputs. Both workers independently quantize and hash the complete weight sequence; every pair must have the same dimensions, type-derived byte count, weight SHA256 and output bytes. No new quantization type or dequantized approximation is used. Synthetic quantization is fixture construction only, not a proposed processing step for actual GGUF weights.

CPU execution is single-threaded, ordinary CPU tensor buffers, full reduction dimensions, existing upstream numerical kernels and planner. No special allocator settings, reduced precision, GPU backend, stock `llama_context` or whole-model loader is introduced. The unchanged QIES104 small-fixture `r99::execute` geometry cap remains 256; QIES106 does **not** loosen or call that interface. It provides a new, explicitly procedural-only matvec diagnostic with fixed real-width cases and no model reader.

| Completed sub-operation | Quantization | Input → output width | Candidate row tiles tested | Result |
|---|---|---|---|---|
| Attention Q projection | Q4_K | 5120 → 10240 | 128, 64, 16 | 3/3 byte-exact |
| Attention K projection | Q8_0 | 5120 → 1024 | 128 | byte-exact |
| Attention O projection | Q6_K | 10240 → 5120 | 128 | byte-exact |
| FFN gate projection | Q5_K | 5120 → 40960 | 128 | byte-exact |
| FFN down projection | Q5_K | 40960 → 5120 | 128 | byte-exact |

**Seven comparisons, 82,944 F32 outputs / 331,776 bytes, all identical; zero tolerance and no mismatches.** Full-shape matrices are evaluated, not a sampled or reduced-width substitute. Five independent full-matrix stock captures underpin those seven comparisons. Repeating the Q projection at different tile counts gives the same output hash `1ea03c71ac74957ae11a939c742ff4f94b4d8a07872d3797db8e8aedbcd717c1`. FFN gate output hash: `b69a523b7a6a254b054d3ba4de4dfc61708e98e44a4729bb0b48c9f5de1cb5fb`; FFN down: `55942b5192f334e5706ce716bad915dd5dcb40f73495c5b89429f9239c8bcb77`.

These establish the tested synthetic **sub-operations**, not full Ricursive layer equivalence, real-model inference, all quantization/dimension combinations or multi-token KV correctness. Norm, RoPE, attention, sigmoid, value mixing, SwiGLU/clamp and residual composition are designed/planned but not yet integrated and numerically executed by QIES106.

## Measured physical residency and cleanup

`qies106/measurements.json` consolidates the final campaign's actual observations. RSS below is the maximum of the kernel-reported HWM and observed worker `smaps_rollup` RSS. PSS/anonymous values are phase-sampled maxima, not guaranteed instantaneous high-water marks. A separate 50 ms host monitor observes the whole synthetic process tree; it sums PSS to avoid double-counting shared pages and also records RSS, system availability, swap and inherited cgroup statistics.

| Sub-operation (128 rows) | Streamed peak RSS MiB | Streamed sampled PSS MiB | Streamed sampled anonymous MiB | Whole-matrix stock peak RSS MiB | Stock sampled PSS MiB |
|---|---:|---:|---:|---:|---:|
| Q, Q4_K | 11.020 | 5.372 | 2.504 | 38.535 | 32.970 |
| K, Q8_0 | 10.656 | 5.053 | 2.316 | 15.230 | 9.618 |
| O, Q6_K | 11.605 | 5.958 | 3.258 | 51.543 | 45.932 |
| FFN gate, Q5_K | 11.094 | 5.479 | 2.578 | 148.168 | 142.520 |
| FFN down, Q5_K | **14.668** | **9.036** | **6.211** | **148.559** | **142.946** |

The largest streamed operation has 3,604,480 weight bytes, a **3,768,832-byte total CPU tensor buffer**, **571,136 bytes of explicitly allocated GGML workspace**, 1 MiB reserved graph metadata, 355,840 bytes of synthetic source-row buffers and 164,352 host activation/output bytes. This is **5,908,736 bytes (~5.635 MiB) of explicit bounded allocations**, before backend/library/allocator/process overhead. It is not a measured full-layer peak. The actual RSS includes those additional costs.

Other measured CPU scratch plans: Q and FFN gate 530,304 bytes, O 536,128 bytes, K 5,504 bytes. No complete-layer 128 MiB graph allocation or 520 MiB weight bundle is used by these sub-operation workers.

FFN-down's observed whole-matrix-reference minus streamed-worker RSS difference is ~133.89 MiB for this synthetic operation. This is not a measurement of QUORTEX engine-wide savings or guaranteed system MemAvailable improvement. The reference and candidate link different pinned GGML revisions; output equivalence was checked directly rather than inferred.

Physical accounting details:

- Quantized synthetic weights are genuinely anonymous CPU allocations, not virtual mappings or shared file-backed model pages. Every weight row is generated and touched. No tensor weight file/mapping is present.
- FFN-down's sampled file-backed PSS maximum was 2,915 KiB, from executable/shared libraries; its sampled worker swap was zero. This is not 3.44 MiB of hidden model-file mapping. Worker anonymous residency is reported separately from file-backed/shared pages.
- FFN gate's anonymous live footprint was 2,636 KiB at its first tile and 2,640 KiB through tile319; FFN down was 6,356 KiB at tile0 and 6,360 KiB at tile39. This shows bounded residency within these operations, not accumulation of a whole matrix.
- After FFN down cleanup: RSS 10,344 KiB, PSS 4,535 KiB, anonymous 1,620 KiB, zero owned GGML/backend/output/digest resources. Residual library/allocator residency is not called leaked weight storage or assumed returned RAM. Repeated complete-layer execution in one long-lived process still needs testing.
- The Q candidate recorded two major faults; all other final jobs recorded zero. No model was mapped, and faulting addresses were not traced, so an exact cause is not assigned. Minor faults and kernel HWM remain in phase logs; freeing a buffer does not reset HWM. No `malloc_trim`, cache drop, `madvise`, swap/zram change or allocator-environment tuning was used.
- The inherited desktop cgroup is shared, not a dedicated worker scope. For FFN down, `memory.current` was 2,290,491,392 before and 2,292,736,000 after, sampled maximum 2,311,917,568; its inherited `memory.peak` was 2,505,158,656 and was **not reset**. These values cannot be attributed solely to the worker or used as its peak.
- Actual machine MemTotal was 8,190,992,384 bytes (~7.63 GiB). Final-campaign system MemAvailable never sampled below ~2.18 GiB. These global values include unrelated desktop activity; no application was closed. No 4 GiB physical-machine/whole-system test occurred.

Elapsed job times included fresh procedural weight generation, quantization, hashing, output I/O and monitoring: Q pair ~12 s each; K ~0.12 s; O ~3.2 s; FFN gate ~53 s; FFN down ~52–54 s. Fixture quantization dominates these timings. They are **not** measured real-weight I/O/inference speedups or token-throughput estimates.

## Failure tests, interruption handling and preservation

The final campaign comprised 12 successful numerical jobs (five reference, seven candidate), four injected-failure jobs and six interface-rejection jobs. Failures after input staging, after the third tile load, after its compute, and before output publication all returned failure with zero owned resources and no final or partial output left by the handled exception. Unknown/file-valued cases and real/no-marker requests were rejected; both real requests used the unchanged existing real admission. No request bypassed it and no real model data were supplied.

Before successful publication the prototype writes only an exclusive `output.partial`; a handled failure removes that owned partial. A completed output is linked without replacing an existing final name. This diagnostic-file protocol does not claim generation checkpoint crash consistency or state rollback.

The first scratch-budget refusal, the earlier successful Q smoke campaign and the interrupted broad campaign are retained. The interruption stopped the latter during its candidate FFN gate; its missing result/summary and any unpublished partial remain **incomplete**, not accepted or resumed as a successful test. The final campaign used a completely new directory and completed. No evidence was overwritten or hidden.

The additional monitor applies only to this small synthetic experiment: it checks available RAM/swap/disk, OOM events, wall time and conservative process-tree RSS bounds; terminates its own tree on a violation; and preserves logs. It never overrides original `Policy()` or changes cgroups, rlimits, swap, kernel settings or resource thresholds. Its successfully passing tests are not positive live-guard qualification for real inference. The unchanged 6 GiB gate and unconditional QIES104 real-execution lock remain in place.

## Reproduction

```sh
cd '<workspace>/qies106-worktree'
./pandacode-run.sh
python3 -I -B qies106/plan_only.py
```

Requirements and an optional alternate verified SDK-state path are documented in `qies106/README.md`. `build.py` verifies the pinned stock source, recompiles only the two procedural-synthetic workers, and records hashes. Numerical workers are host-non-executable and require the sealed synthetic namespace; only fixed case/tile/fault names are accepted, not input filenames. `/home`, real model directories and other worktrees are absent inside that namespace. Synthetic output directories are new per run. No model download, NumPy installation or global package change is performed.

## QIES107 implementation task and decision

**GO: extend this executor to the complete real-width synthetic FFN branch.** Implement the existing sequence `FFN RMS → gate matvec → up matvec → ggml_swiglu_clamp(20) → down matvec → post-FFN RMS → residual add`, preserving the full 40,960-element reduction input and exact operator boundaries. Retain only the small activation vectors and one reusable weight tile, with explicit last-consumer release. Do not replace the upstream clamp/SiLU expression or introduce reassociated partial sums.

Build an independent stock whole-FFN graph, capture every intermediate, and require exact bytes for multiple deterministic activations and the actual Q4_K/Q5_K FFN variants. Add failure injection between phases and repeated operations in one process to prove cleanup/steady residency, rather than relying solely on fresh-process peaks. A provisional synthetic process budget can be informed by the measured <15 MiB matvec peaks and <1 MiB FFN activation estimate; it must be measured for the composed graph, not advertised as achieved.

That is an implementation step toward a complete layer, **not another guard-qualification investigation**. Subsequent attention integration must cover Q/K normalization, SWA/global masks and RoPE/NoPE, value mixing/gating, two or more KV positions, unchanged state serialization and exact independent outputs at a bounded context before any complete-layer claim.

Remaining exclusions: no integrated complete layer, no real-weight reads/arithmetic, no live real reference/guard unlock, no long-context attention memory solution, no persistent engine/loader/tokenizer accounting, and no verified 4 GiB total-system operation. **NO-GO for real-model execution or claiming the full engine runs on 4 GiB.** Existing guards, frozen milestones and production code remain untouched; no commit, merge or push occurred.
