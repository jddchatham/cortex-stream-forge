# QIES110 — Ripple real-GGUF streaming adapter

**GO for QIES111 experimental integration. NO-GO for real-weight execution.** Final-source storage, synthetic numerical and fault validations passed; real execution remains locked.

## Scope and provenance

Work is isolated in `<workspace>/qies110-worktree`, detached at `14b3f7902864b02e1a84e5e6168a3cf99087e670`, with inherited QIES106–109 sources explicitly preserved as uncommitted experimental material. The verified engine ancestor is `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. No commit, merge, push, branch overwrite, or frozen-archive edit was performed.

The independent stock source is llama.cpp b12167, `240d68fcfe0c2a1f298c8e18ca181368b1914259`; Ricursive source SHA-256 is `427da0ad63c070b69fb2e78f6876893d4fa47ebad3f8711bef54d57d521e1f7b`. The pinned SDK source manifest verifies 4,084 source files. The independent numerical oracle is the preserved QIES109 stock-GGML synthetic full-layer graph, not a qualified full-model stock executable. Candidate GGML is 0.25.3; reference GGML is 0.29.0. Matching outputs demonstrate equivalence for these tested configurations, not general backend equivalence.

Compiler: GCC 16.2.1; workers: `-std=c++17 -O2 -Wall -Wextra`; source-built SDKs: Release `-O3 -DNDEBUG`, native CPU/OpenMP configuration. CPU: i5-7200U, AVX2/FMA/F16C, one compute thread. Full compiler commands, library and executable hashes are in each campaign's `build-manifest.json`. Sanitizer caller flags add `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`.

## Adapter and tensor lifetimes

`qies110/range_source.h` supplies a read-only `File` and `RowSource`. `adapter.py` reuses the unchanged QIES101 1,203-descriptor validator. A complete layer plan identifies all 15 tensors, file identity, offsets, shapes, quantization and expected digests. It is bounded to 32 KiB. `File` validates the model-specific schema, quantized row sizes, 32-byte offsets, non-overlap, overflow and file bounds before reading tensor payloads.

The reader opens one regular file with `O_RDONLY|O_CLOEXEC|O_NOFOLLOW`, checks descriptor/path stat identities before and after reads, retries EINTR/short reads and rejects premature EOF. There is no whole-model mapping. A single shared row cache is capped at 1 MiB, with whole quantized blocks preserved. A failed refill invalidates the cache before any retry. A 128-row GGML weight tile is populated from this source; the existing kernels, graph operations, accumulation order and 1 MiB compute workspace remain unchanged. F32 scale vectors are bounded to 20 KiB. Projection buffers are destroyed at their final consumer; the shared cache and file are released before capture publication.

`stream.cpp` changes source access only: the eight QIES109 projection sources and seven scalar/vector parameters now come from the adapter. It retains QIES109's separate authoritative/pending KV ownership and immutable input checkpoint. `common.h`, `faults.inc` and independent `reference.cpp` are byte-identical to QIES109.

The same reader accepts real GGUF plans for **data-only** validation and synthetic plans for numerical testing. Every numerical projection rejects a REAL source through unchanged `q104::real` before weight allocation. The synthetic actor also requires a SYNTHETIC plan, synthetic file magic/size, and a sealed namespace with no GGUF mounted. `storage.cpp` and independent `reference_hash.cpp` contain no numerical execution path; the latter uses the stock GGUF parser, not the QUORTEX descriptor parser. Private numerical workers remain non-executable on the host and run through the sealed loader namespace.

## Exact real-tensor mapping

Each suffix below resolves to `blk.<layer>.<suffix>`; dimensions are GGUF `[columns, rows]`.

| Suffix | Type | Shape | Layer-0 bytes |
|---|---|---|---:|
| `attn_q.weight`, `attn_gate.weight` | Q4_K | `[5120,10240]` | 29,491,200 each |
| `attn_k.weight`, `attn_v.weight` | Q8_0 | `[5120,1024]` | 5,570,560 each |
| `attn_output.weight` | Q6_K | `[10240,5120]` | 43,008,000 |
| `attn_norm.weight`, `post_attention_norm.weight`, `ffn_norm.weight`, `post_ffw_norm.weight` | F32 | `[5120]` | 20,480 each |
| `attn_q_norm.weight`, `attn_k_norm.weight` | F32 | `[128]` | 512 each |
| `attn_vres_lambda.weight` | F32 | `[1]` | 4 |
| `ffn_gate.weight`, `ffn_up.weight` | Q5_K | `[5120,40960]` | 144,179,200 each |
| `ffn_down.weight` | Q5_K | `[40960,5120]` | 144,179,200 |

Layers 0–14 and 65–79 use Q5_K FFN; layers 15–64 use Q4_K FFN. All 80 layer catalogs resolve 15 tensors each: 1,200 layer mappings plus three global tensors. Model metadata describes 64 sliding-window and 16 global layers. Complete numerical tests exercise representative layers 0 and 4; remaining layer mappings/profiles are descriptor coverage, not 80-layer arithmetic coverage.

## Real-byte storage validation and identity

External file: `qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf`, 40,426,079,712 bytes. HF revision: `df550a7178a87bbfb6ee2bdbd37603c553d536ee`.

Previously verified whole-model SHA-256:

`9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc`

This task binds to the unchanged device/inode/size/mtime/ctime in QIES102's verified receipt. It does **not** claim a fresh whole-40-GB hash pass. On a different machine or changed identity, rehash the complete file before trusting a new receipt.

Fresh independent SHA-256 comparisons cover all 15 layer-0 and all 15 layer-4 tensors. Each path independently resolves names, offsets, byte lengths and types; the adapter also traverses every row through its bounded scheduling interface. All 30 digests match. Inventory/digests are in `qies110/storage-results-final/summary.json`. Original QIES101 fault tests pass in a separate scratch directory; the original test artifacts are not overwritten. Seven new corrupted-plan cases reject missing/count/mode/shape/duplicate/offset/digest errors with zero open source files after failure.

Expected hashes in plans are not automatically recomputed on every row read. The storage validator performs complete selected-tensor hashing. Trustworthy publication of model/descriptor receipts remains a prerequisite for any future real arithmetic.

## Crash investigation

The user's reported interruption has no identified QUORTEX executable, command, PID, exit status or signal. Available QIES109 results record successful exits or intentional fault/refusal exit 1, with no negative signal statuses or recorded monitor violations. The accessible kernel journal query has no matching OOM/kill/segfault entry. The only listed coredump is a November 7 Thorium `SIGTRAP`, predating this layer work; it is unrelated and was not extracted.

**Reported crash: unresolved.** Missing logs are not evidence of a particular allocation, lifetime, KV or out-of-bounds failure. See `qies110/coredumps.log`, `kernel-oom.log`, and `prior-exit-evidence.json`. No real-weight reproduction was attempted.

A separate final source audit found and fixed a new adapter retry hazard: failed refills must invalidate the cached row range before a subsequent read. This is not established as the cause of the historical interruption. The dedicated regression requires a new source read after an injected failed refill. Twenty isolated sparse-synthetic cases pass under ASan/UBSan with leak detection: normal reads, EINTR/short-read retry, zero-read rejection, overflow/bounds, missing tensors, invalid rows/types/shapes, overlap/alignment, source truncation/replacement/symlinks, mode mismatch and failed-refill recovery. All close file ownership. Instrumentation covers the adapter/caller, **not** the prebuilt GGML shared libraries or a complete real-model runtime.

## Persistent state, safety and limitations

The 78 inherited deterministic faults cover allocation/execution, KV mutation, residual/FFN, captures and serialization boundaries. They simulate exceptions; they do not prove allocator behavior under a genuine system OOM. The suite checks input-checkpoint hashes, rejects corrupted/truncated copies, requires cleanup and no published partial capture after failure, and retries a verified checkpoint to the independent expected output. It does not establish general power-loss recovery or a durable production transaction protocol.

The unchanged real guard requires 6 GiB MemAvailable and then still rejects unqualified execution unconditionally. No receipt or flag unlocks it. Synthetic monitoring retains its separately established thresholds: candidate/reference entry 1.5/1.75 GiB, minimum 1 GiB while running, 2 GiB swap reserve, 20 GiB disk reserve, candidate/reference sampled RSS ceilings 256/640 MiB, 120-second worker timeout and ancestor OOM-event monitoring. These are not replacements for real-model policy. Inherited cgroup/rlimit settings are observed without reconfiguration; this machine's available scopes/limits are not a new isolated enforcement guarantee.

## Complete-model budget: estimates, not validation

Measured per-layer KV at 256/1,280 slots is 1/5 MiB. Formula for F16 K+V: `2 × 1024 × 2 × slots` bytes. Holding that state for all 80 layers gives **80/400 MiB**, excluding active-layer pending copies, serialization, allocator and compute storage. Full 80-layer execution is not implemented or measured here. The descriptor catalog is only 1,203 records; the shared staging cache/workspace can be reused per layer rather than multiplied by 80.

A bounded 128-row tile needs at most 3,604,480 weight bytes for this profile; the shared row cache adds at most 1 MiB, the compute workspace 1 MiB, and GGML graph metadata is bounded per context. Attention tensor buffers and authoritative/pending/checkpoint KV copies must also be counted. Current synthetic checkpoint code intentionally copies only bounded test state; it is not an optimized all-layer checkpoint manager.

The source does not yet implement arbitrary-context ring-buffer KV scheduling. Naively storing all 80 layers at 32,768 positions alone needs 10 GiB. A future validated sliding-window ring design could reduce that, but is not a QIES110 result. Embedding/output projections, all-layer activation/value-residual plumbing, long-context scheduling, process overhead and durable checkpoints remain outside the measured single-layer budget.

Low process RSS makes a small-context 4 GiB design plausible as an engineering hypothesis. It does not demonstrate system headroom: `pread` populates filesystem cache outside process RSS, and desktop/services, cgroup file accounting, swap and storage stalls remain material. No model mapping does not mean no file-cache memory. **A 4 GiB machine cannot satisfy the unchanged 6 GiB MemAvailable policy, so real execution remains NO-GO there.** No policy change is proposed here.

## Reproduction and next milestone

See `qies110/README.md` and `pandacode-run.sh`. Pinned SDKs are required; all commands are CPU-only. Results are retained by campaign directory. `--resume` refuses changed source/build identities. Source hashes and protected-baseline checks are in `qies110/SOURCE-SHA256.json` and `final-integrity.json`.

QIES111 should retain the same kernels and storage interface, qualify the independent **real-weight** layer reference and its live guard on suitable hardware, verify whole-model and input identities, then perform exactly one matched layer comparison only when every original safeguard passes and real execution is separately authorized. Until then, extend storage/fault or synthetic integration tests only. No real-weight output, throughput or bit-exactness claim is made here.

## Final validation and measured memory

The fresh, unchanged QIES109 replay passed **312 byte-exact comparisons and 78 faults** (`qies109/runs/20261111-164550-6c139818/summary.json`). The final adapter-backed campaign also passed **312 byte-exact comparisons and 78 faults**, plus six invalid-state/interface refusals and 36 quantized fixture row checks. Its evidence is [summary.json](qies110/runs/20261111-165958-790fe5f0/summary.json).

Coverage includes two-token continuation from immutable checkpoints, positions 1023–1025 with synthetic boundary history, sliding/global masks and RoPE behavior, repeated continuation, and successful retry after all injected failures. The boundary history is not a claim that 1,023 real tokens were evaluated. All successful outputs/checkpoints match the independent graph oracle byte-for-byte. No comparison tolerance was relaxed.

Final storage validation: **30 independent tensor hashes, 1,200 layer mappings, seven malformed-plan refusals and 38 unchanged QIES101 fault tests**. Final sanitizer validation: **20/20 adapter cases**, leak detection enabled, exit 0, no sanitizer diagnostics. The earlier campaigns are preserved separately. No final job had a signal exit or monitor violation; fault/refusal exit 1 is intentional failure, not successful execution.

| Measurement (MiB) | Adapter-backed synthetic layer | Whole-matrix synthetic reference |
|---|---:|---:|
| Sampled process-tree RSS | 35.68 | 560.53 |
| Sampled process-tree PSS | 26.62 | 551.46 |
| Sampled anonymous residency | 23.50 | 548.13 |
| Sampled file PSS | 3.33 | 3.43 |
| Worker high-water RSS | 32.19 | 556.93 |
| Maximum live projection weight bytes | 3.438 | 520.391 |
| Maximum live GGML tensor buffer | 8.411 | 529.928 |
| Maximum planned compute workspace | 0.545 | 0.545 |

Real storage-only adapter peak sampled RSS/PSS/anonymous/file-PSS: **10.27/4.53/3.04/1.50 MiB**. Independent stock parser storage RSS peaked at 18.21 MiB. Adapter maximum read/cache was 1,048,320 bytes, below 1 MiB. Complete synthetic fixture size is 545,752,160 bytes on disk; it is never loaded as one process allocation.

Process-tree sampling includes namespace/loader processes and can capture peaks absent from worker phase samples; RSS can count shared pages more than once, whereas PSS apportions them. Maxima across components need not coincide. These are one deterministic campaign, not three-trial statistical estimates. RSS is not total system memory, and these ratios are not measured real-model speedups. Shared desktop cgroup statistics and historical peaks are retained in raw job evidence, not attributed as isolated worker memory.

On every tested success/failure path, tracked GGML/weight ownership and adapter source descriptors return to zero. Allocator-retained process pages remain visible in post-cleanup telemetry; zero tracked ownership does not mean RSS falls to zero. Model page cache is not quantified as private process residency. Approximately 55 GiB SSD space remains, above the unchanged 20 GiB experiment reserve.

**Decision:** the bounded adapter/dataflow objective is met, with storage-only real bytes and independently verified synthetic arithmetic kept separate. Continue isolated QIES111 work; do not unlock real arithmetic or declare real-weight equivalence. Outstanding prerequisites are qualified independent real execution, matched inputs/configuration, verified model/tensor identities, qualified live guard coverage and at least 6 GiB MemAvailable. A complete 80-layer runtime and 4 GiB system operation remain unvalidated.
