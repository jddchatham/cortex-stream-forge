# QIES107 — Complete synthetic Ricursive FFN streaming

This experiment implements a complete, full-width Ripple FFN branch using only deterministic synthetic inputs and the existing upstream CPU numerical kernels. It does not load a model or implement the attention path. Final runtime measurements, fault results and decision follow the source/dataflow description.

## Preservation, revisions and reproduced baseline

Worktree: `<workspace>/qies107-worktree`, isolated and detached at QIES105 archive commit `14b3f7902864b02e1a84e5e6168a3cf99087e670`, whose verified engine ancestor is `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. All QIES107 implementation is uncommitted. The frozen archive branch/tag, original QIES106 worktree and production FlashNext remain untouched.

Before implementation, QIES106's 17-file source/report SHA256 manifest was checked. Its source, results, build identities and private workers were preserved as `qies107/QIES106-preserved.tar.gz`: 263 files with a per-file SHA256 manifest, archive hash and receipt beside it. Original files remain in place. This archive contains only the synthetic milestone, not model weights; it is a local preservation artifact, not a Git change or public release.

All **seven QIES106 byte-exact comparisons were reproduced** in this worktree at `qies106/runs/20261111-111540-4c8b4643/summary.json`, with output hashes equal to the preserved original. Its original 4 cleanup faults and 6 interface rejections passed as part of that replay. New arithmetic is gated on the presence of that passing reproduction and matching hashes.

Independent upstream source: llama.cpp b12167 `240d68fcfe0c2a1f298c8e18ca181368b1914259`; all 4,084 pinned source-file hashes checked. Candidate uses baseline GGML 0.25.3, reference uses independent stock GGML 0.29.0. Workers compile with GCC 16.2.1, `-std=c++17 -O2 -Wall -Wextra`; CPU SDK libraries use Release `-O3 -DNDEBUG -march=native -fopenmp`, ordinary CPU buffer allocation. Repack is built but extra/repacked buffers are not selected. There is no GPU or `libllama` dependency. Exact SDK flags, compiler, CPU features, libraries, binary hashes and source hashes are recorded in `sdk-compile-flags.json`, `cpu-config.json` and each build manifest.

Hardware: Intel Core i5-7200U, x86_64, two cores/four hardware threads. Each numerical graph uses **one thread**, matching CPU buffer/precision configuration. No affinity, allocator environment, operating-system setting, swap, cgroup or address-space limit is modified. Source reproducibility and kernel-version identity are distinguished: the two GGML versions differ, so numerical agreement is demonstrated rather than assumed.

## Verified stock contract and complete implementation

The ordering is taken from the pinned `src/models/ricursive.cpp`, FFN section after attention residual construction, and its `llama-graph.cpp::build_lora_mm` ordinary no-LoRA matmul path. `reference_ffn.cpp` independently constructs that whole FFN graph using stock GGML. It does not call or compile the candidate's streaming scheduler or numerical graph builder.

Let R be the stock **ffn_inp** residual after attention, rather than original embedding x0 or pre-attention x:

1. A = `RMSNorm(R, epsilon=1e-6) * gamma_ffn`.
2. G = `W_gate * A`, U = `W_up * A`.
3. S = `ggml_swiglu_clamp(G, U, 20)`.
4. D = `W_down * S`.
5. P = `RMSNorm(D, epsilon=1e-6) * gamma_post_ffn`.
6. F = `P + R`.

The existing clamped SwiGLU operator includes both SiLU and elementwise multiplication: `g=min(G,20)`, `u=clamp(U,-20,20)`, output `g/(1+exp(-g))*u`, as implemented by upstream. There is **no extra multiplication** outside that operator and no lower clamp on the gate. No custom approximation, changed quantization, reassociated reduction or altered epsilon is introduced. Both learned gamma vectors are represented by nontrivial deterministic F32 test values.

Actual dimensions are H=5120, FFN=40960: gate/up matrices `[5120,40960]`, down `[40960,5120]`, gamma vectors 5120. Both actual quantization profiles were verified against the saved GGUF descriptors: layer 0 uses Q5_K throughout the three FFN projections; layer 15 uses Q4_K throughout. The two profiles occur in 30 and 50 real layers respectively. No GGUF tensor bytes are read or requantized in QIES107.

Files:

- `qies107/stream_ffn.cpp`: complete streamed FFN and residual; last-consumer release; repeated execution.
- `qies107/reference_ffn.cpp`: independently assembled, unpartitioned stock FFN graph with all required captures.
- `qies107/fixture.h`: input definition only, full real dimensions, distinct gate/up/down seeds, nonzero and zero activations.
- `qies107/support.h`: resource ownership, bounded fixture construction, telemetry and exclusive capture publication. No candidate numerical graph is shared with the reference.
- `qies107/faults.inc`: 100 named allocation/compute/capture fault boundaries.
- `qies107/build.py`, `run.py`: provenance, compilation, exact comparisons, process-tree/system monitoring and deterministic tests.
- `qies107/README.md`, root `pandacode-run.sh`: tested model-free execution instructions.

## Bounded ownership and tensor lifetimes

Each projection streams **128 output rows**, retaining the entire dot-product input dimension. Its graph/context and CPU tile buffer are scoped to that projection; the same tile is overwritten only after synchronous compute and output copy complete. The graph is reused across all output tiles. No complete gate/up/down matrix is resident in the candidate, and no split-reduction partial sums are introduced.

A single 1 MiB workspace is reused across all matvecs and elementwise graphs and all three repeated branches. Host activation storage scales with H/F, rather than three H×F weight matrices. Input gamma is released after input normalization; normalized A after up projection; G/U after SwiGLU; S after down; D and post-gamma after post-RMS; P/R/final vector after final add/capture. R remains authoritative until its final residual consumer. No attention/KV state exists in this synthetic FFN, so repeated tests are stateless branch reuse, not multi-token state-continuity evidence.

The full reference deliberately holds all three matrices for its single whole-FFN graph. For Q5_K they total **432,537,600 bytes (412.5 MiB)**; for Q4_K,353,894,400 bytes (337.5 MiB). The candidate's largest Q5_K weight tile is **3,604,480 bytes (3.4375 MiB)** for down; gate/up tiles are450,560 bytes each. The active tile storage is reused and destroyed before the next projection's graph is allocated.

### Dense synthetic fixture definition

The new fixture factory quantizes 257 small 256-element block templates per projection independently through each worker's upstream library, then uses a deterministic row/block hash to fill every quantized block in the dense full-width matrix. Distinct projection seeds prevent a gate/up alias from accidentally passing. Gate/up amplitudes make positive/negative clamp branches observable; down uses a separate amplitude/seed. All quantized matrix bytes are generated, written into actual CPU storage, hashed and multiplied by GGML. No rows or reductions are skipped.

This bounded template cache is **input construction**, not a proposed compression of model weights or a new inference kernel. It avoids spending minutes quantizing unrelated random fixtures for each fault test. A real reader would supply original verified quantized bytes verbatim; it is intentionally absent here. Candidate and reference must independently produce matching SHA256 for all three weight sequences on every repetition. Shared input definitions and non-numerical ownership/telemetry do not supply expected output data to the candidate.

## Independent comparison and diagnostic publication

Required taps are G, U, S, D and final F. Input-normalized A and post-normalized P are also compared, giving **seven captures per branch**. Every dimension, F32 byte count, complete weight identity and output byte must match. Tolerance is zero; mismatch details are saved and the campaign stops. The stock graph computes all intermediates before capturing them, without candidate-supplied expected outputs.

The default campaign covers Q5/Q4 with two distinct nonzero vectors plus a zero vector. Nonzero cases must actually exercise gate>20, gate<-20, up>20 and up<-20; the zero case must produce a zero final result through the epsilon path. Three repeated Q5 branches in one worker additionally compare every tap to the same independent reference, checking bounded reuse and determinism.

Capture files are exclusively staged, synchronized, and published only as a complete directory using `renameat2(NOREPLACE)` after successful branch completion. Handled exceptions remove only that invocation's provisional files. Numerical success requires a final capture directory **and** successful job status. No production checkpoint, recovery or stdout exactly-once claim is made.

## Final numerical and fault results

**PASS: complete real-width synthetic FFN, both verified quantization profiles.** Final evidence is `qies107/runs/20261111-112705-03caed05/summary.json`; per-job stdout, monitor, error logs, captures and result records remain in that directory or its separate completion directory. `qies107/measurements.json` consolidates memory observations.

| Test | Result |
|---|---|
| Preserved QIES106 comparisons | 7/7 reproduced, original output hashes identical |
| Q5_K: two nonzero vectors + zero vector | All 7 taps byte-exact for each vector |
| Q4_K: two nonzero vectors + zero vector | All 7 taps byte-exact for each vector |
| Three repeated Q5 branches in one process | All 7 taps exact on each repetition; every repeated weight identity checked |
| Combined new comparisons | **63/63 captures,1,290,240 F32 values /5,160,960 bytes, zero tolerance** |
| Allocation/compute/capture fault boundaries | **100/100 expected failures**, correct fault reached, zero modeled resources, no published/provisional capture directories remaining |
| Real dispatch / missing namespace / file-valued input | 6/6 rejected; no weight allocation |
| Resource violations / changed rlimits | 0 /0 |

Clamp coverage was verified from the independent captures, not assumed from constants. For Q5 vector0:16,885 gate values exceeded+20;16,097 were below−20 (not lower-clamped);15,796 up values exceeded+20 and 17,240 were below−20. Q5 vector1 and both nonzero Q4 vectors also exercised every branch. Zero vectors produced zero final outputs. Distinct gate/up hashes prevent accidentally treating one projection as the other. No mismatch or tolerance relaxation occurred.

The fault list covers before/after backend/workspace and each fixture allocation, host activation setup, capture-owner creation, each graph context and CPU tensor buffer, host output allocation, projection loading/compute/output/phase boundaries, elementwise compute boundaries, each tap capture and final publication. Projection load/compute hooks use the first tile; after-output/after-phase hooks cover full projection completion. This is not exhaustive injection at every row/byte or a forced allocator-OOM/sanitizer/power-loss test. Exceptions follow the actual C++ unwinding path; partial captures are removed and all tracked owners reach zero.

### Interruption and completion provenance

The final sweep was interrupted after 103 complete jobs, with `fault-up.before-output-allocation` lacking a completed result. No numerical source or worker binary had changed. `finish.py` verified every recorded source SHA256 and both worker hashes, preserved that incomplete attempt, and ran the remaining 16 jobs in `qies107/runs/20261111-112705-03caed05-completion/`. These comprise10 outstanding fault cases, including a fresh retry of the interrupted one, and 6 interface rejections.

All 63 numerical comparisons, all repeated weight identities and clamp/zero-vector assertions were rechecked before final acceptance. The final summary accounts for 119 accepted jobs:13 numerical jobs (6 reference,6 single-branch candidate,1 three-branch candidate),100 fault jobs and 6 rejections. The 103 existing results were accepted only with valid status/resource/monitor records; an incomplete/missing result was never counted. Old files and intermediate campaigns remain intact. A resume receipt records the source/binary check, prior incomplete directory and all retried labels. This is test-orchestration recovery, not production checkpoint semantics.

## Measured memory and cleanup

Each numerical worker reports `/proc/self/smaps_rollup`, kernel peak RSS/fault counts, actual CPU tensor-buffer sizes, graph workspace requirements, resource counts and weight lifetimes. The host samples the whole worker tree at 50 ms, summing PSS to avoid double-counting shared pages. RSS below is the maximum of kernel HWM and observed worker RSS; PSS/anonymous/file-backed values are phase-sampled maxima and may miss shorter transients. Full records, rather than rounded table values, are authoritative.

Profile rows below are maxima across the three vectors; independent peaks need not coincide in time. MiB means1,048,576 bytes.

| Execution | Observed peak RSS MiB | Sampled PSS MiB | Sampled anonymous MiB | Sampled file-backed PSS MiB |
|---|---:|---:|---:|---:|
| Q5 streamed FFN | **16.359** | 10.594 | 7.441 | 3.266 |
| Q5 whole-branch stock graph | **424.813** | 418.953 | 415.898 | 3.083 |
| Q4 streamed FFN | **15.602** | 9.832 | 6.707 | 3.250 |
| Q4 whole-branch stock graph | **349.809** | 343.924 | 340.871 | 3.076 |
| Q5 three branches in one process | **16.461** | 10.700 | 7.438 | 3.267 |

The Q5 synthetic comparison shows about408.45 MiB lower observed worker peak RSS for the streamed branch than the complete-matrix reference. This is a prototype branch measurement with matched numerical results across two pinned GGML revisions, **not** measured QUORTEX engine-wide savings or an exact improvement in system MemAvailable.

| Owned allocation | Streamed Q5 | Whole-branch Q5 reference |
|---|---:|---:|
| Maximum simultaneous quantized weights | 3,604,480 bytes | 432,537,600 bytes |
| Peak actual CPU tensor buffers | 3,768,832 bytes | 433,213,440 bytes |
| Allocated reusable workspace | 1,048,576 bytes | 1,048,576 bytes |
| Largest GGML plan scratch requirement | 571,136 bytes | 571,136 bytes |
| Metadata context reservation | 1 MiB per active graph | 1 MiB |
| All three bounded Q5 fixture caches | 135,696 bytes | 135,696 bytes |

Actual graph plans request zero scratch for the elementwise operators; the matvecs share the same workspace reservation. The 1 MiB allocated capacity is distinguished from the smaller571,136-byte maximum plan requirement. Input/activation vectors, a bounded row buffer, backend/library/allocator overhead and diagnostic infrastructure are additional. No complete weight matrices are hidden in a mapping or in the fixture caches.

Weights are touched anonymous CPU storage, not virtual-only allocations or file-backed model mappings. File-backed PSS here is executable/library residency; no real model/weight file is mounted or read. Private/shared/clean/dirty and swap observations remain in worker phase logs. Cgroup memory.current/stat/peak and host MemAvailable/swap/disk are recorded by the monitor. The inherited desktop cgroup is shared and its peak is historical, **not reset or attributable solely to these jobs**. No global RAM-recovery claim is made from it.

Every accepted worker ended with zero modeled owners and zero active weight/tensor buffers. In the three-branch candidate, branch-complete anonymous residency was3,932 KiB,7,612 KiB,7,612 KiB; the second and third plateaued. After all owners were destroyed, RSS remained15,828 KiB and anonymous6,584 KiB. That residual residency is reported rather than called reclaimed RAM: ordinary allocator/library retention can outlive logical frees. Three iterations and explicit lifetimes are useful bounded-use evidence, not a general long-duration leak proof.

The final accepted campaign sampled system MemAvailable no lower than approximately 2.55 GiB. No application was terminated. Additional synthetic monitoring stops its own process tree on available RAM/swap/disk exhaustion, OOM events, timeout or its fixed process RSS bound. Reference memory is intentionally larger because it holds all three full matrices: QIES107 defines a640 MiB synthetic reference ceiling and 256 MiB candidate ceiling from these known shapes; it does **not** lower or modify QIES106's limits or any real-inference guard. Inherited cgroup and address-space limits remain unchanged.

No real-weight inference, whole-model loading, disk weight-I/O performance, complete-layer memory or4 GiB total-system qualification is established. A4 GiB machine still cannot satisfy the unchanged6 GiB MemAvailable real-execution gate. Synthetic fixture generation/hash/capture/monitoring dominates these short test timings, so they are not token throughput or real-weight speedup measurements.

## Preservation and reproduction

```sh
cd '<workspace>/qies107-worktree'
./pandacode-run.sh --smoke
./pandacode-run.sh
```

The first command verifies one complete Q5 branch; the second reproduces the full campaign. Both require the passing QIES106 replay already present and the source-built CPU SDK identity recorded in the README. New runs use new directories. `finish.py` is specifically an audited one-time completion of the interrupted final campaign; after successful completion it refuses to overwrite its existing completion directory/summary.

All numerical workers are host-non-executable, with namespace-only RPATHs. The approved launcher exposes only fixed procedural inputs, private CPU libraries and a new output directory inside a sealed synthetic filesystem. No model/home directories or arbitrary file inputs are exposed. Both workers compile unchanged QIES104 real admission: the 6 GiB threshold and unconditional real-execution lock persist. The small-fixture QIES104 geometry cap is unchanged; this separate procedural FFN does not weaken or call that capped layer interface.

Original QIES106 source/results, frozen QIES105 commit/tag, original real guard and retired canonical stock-library permissions are checked again in `final-integrity.json`. `SOURCE-SHA256.json` identifies the uncommitted source/report snapshot; no new Git commit ID is invented for it. No commit, merge or push occurred.

## Remaining attention work and QIES108 recommendation

**GO for QIES108: implement and independently validate the complete real-width synthetic attention branch, then connect it to this FFN.** This should be implementation and numerical comparison work, not another speculative guard-readiness investigation.

Required attention components remain:

1. Hidden RMS and bounded Q/K/V projections with the actual5120/10240/1024 widths and verified quantization types.
2. Original embedding x0 ownership and exact sigmoid-lambda value residual mixing; x0 must not be confused with the FFN residual R.
3. Full128-element per-head Q/K RMS; SWA RoPE versus global NoPE and matched positional/frequency configuration.
4. Grouped attention:80 query heads,8 KV heads, head dimension128, exact causal/SWA masks, existing softmax scaling/accumulation and F16 KV semantics.
5. Attention sigmoid gate, elementwise gating, O projection, post-attention RMS and residual addition to produce this branch's `ffn_inp`.
6. Bounded synthetic KV at a short initial context, at least two consecutive positions, independent intermediate captures and exact state continuity/serialization tests before a complete-layer claim.

Keep the full reduction axes and existing CPU numerical kernels; do not assume inner-context softmax tiling or long-context state eviction is bit-exact. Add composed-layer lifetime/fault tests and longer repeated execution. Real source wiring, live qualification and long-context/4 GiB system feasibility remain separate blocked work; QIES107 does not unlock them.

**Decision: GO for the bounded complete synthetic FFN and QIES108 attention integration. NO-GO for claiming a complete Ricursive layer, real-weight correctness, production acceptance or4 GiB model operation.**
