# QIES109 — Complete synthetic Ripple layer streaming

This milestone joins the verified attention and FFN branches into one real-width synthetic Ricursive layer, with one workspace and a full-layer KV publication boundary. Final acceptance and measured results are recorded below. No real-weight inference, production FlashNext changes, guard changes, commits, merges or pushes are authorized or performed.

## Preserved lineage and reference configuration

Worktree: `<workspace>/qies109-worktree`, isolated and detached at `14b3f7902864b02e1a84e5e6168a3cf99087e670`. Verified engine ancestor: `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. The uncommitted additions have a source SHA256 manifest rather than an invented new commit ID.

QIES107's 63 saved FFN captures and QIES108's 228 saved attention captures were checked against their independent reference files and recorded hashes. Their 100/105 fault records, cleanup and resource receipts were verified. Implementations, complete prior suites and results are copied into this lineage; originals remain untouched. `qies109/preservation.json` records per-file hashes. This is verification of saved baselines, not a claim that both old suites were newly executed here.

The independently assembled full-layer observer uses stock llama.cpp b12167 source `240d68fcfe0c2a1f298c8e18ca181368b1914259`. All 4,084 pinned source hashes are checked. Its operator graph follows `src/models/ricursive.cpp` and ordinary CPU graph helpers. It does not compile the candidate scheduler or stage functions. Shared code defines inputs, bounded state IO, ownership and telemetry only.

Candidate GGML is 0.25.3, stock GGML is 0.29.0: kernel versions differ, and equivalence is tested directly. GCC 16.2.1 compiles workers with C++17, `-O2 -Wall -Wextra`; source-built SDKs use the recorded Release/native CPU profile. Both paths use one CPU thread, ordinary buffers, F16 KV, F32 score accumulation and nonflash attention. Exact source, compiler command, libraries and binary hashes are in each campaign's build manifest. No GPU or libllama dependency is introduced.

## Complete layer implementation

`qies109/stream.cpp` implements the following dependency order, derived from stock:

1. Normalize layer input x and apply attention gamma.
2. Stream Q/K projections; compute value residual `x0 + (normalized_x-x0)*sigmoid(lambda)` and stream V.
3. Normalize complete 128-element Q/K heads and apply SWA RoPE or global NoPE.
4. Tentatively update private F16 K/V; compute grouped scores, exact causal/window mask, scaled softmax and attention values.
5. Stream sigmoid attention-gate projection; gate values; stream O; post-attention RMS/gamma; add x to produce attention residual R.
6. RMS-normalize R with FFN gamma; stream gate/up; execute upstream `ggml_swiglu_clamp(...,20)`; stream down.
7. Post-FFN RMS/gamma and add R for the final layer output.
8. Serialize the tentative next KV only after all previous steps succeed; publish complete captures/checkpoint only after backend cleanup.

The clamp operator contains SiLU and elementwise multiplication; no second multiplication or approximate activation is added. Dot-product/softmax reduction axes remain complete. No kernel, quantization, epsilon or precision is changed to obtain lower memory.

Actual dimensions are H5120, FFN40960, Q/gate10240, K/V1024, 80 query heads, 8 KV heads and head dimension128. Descriptor-verified layers0 and4 use Q4_K Q/gate, Q8_0 K/V, Q6_K O and Q5_K FFN. Both SWA and global variants are implemented. Q4_K FFN remains verified in QIES107 but is not claimed as a joined full-layer variant here.

Files: `stream.cpp`, independently constructed `reference.cpp`, fixed input/state `common.h`, type-aware `fixture_check.cpp`, `build.py`, resumable `run.py`, `faults.inc`, and `README.md`. The local root Run Script executes only this model-free procedure.

## Unified ownership and full-layer commit boundary

One backend and one 1 MiB scratch reservation serve both branches. Projection weights are streamed in128-row tiles; no complete projection matrix is kept by the candidate. A scope releases all attention-only fixtures/activations before FFN allocation. Runtime assertions require zero active weight/tensor buffers at both `attention-released.before-ffn` and `ffn-released.before-checkpoint` boundaries.

Only R and tentative KV cross the branch boundary. FFN normalization, gate/up, activation and down vectors are released after their last consumers. State copies used for rollback are explicit and bounded; serialization uses another bounded payload. Allocator/library retention and filesystem cache are not misrepresented as absent merely because logical owners reach zero.

The input checkpoint is mounted read-only and its in-memory state remains authoritative and unchanged. Attention increments only a private candidate position. A failure in residual integration, FFN, output capture or serialization discards that candidate. No standalone attention checkpoint is published between branches. The new full-layer checkpoint magic/fixture identity109 rejects standalone QIES108 checkpoints.

Successful publication uses an exclusive capture directory and non-overwriting rename. Acceptance requires successful status, checksummed state, expected position, matching source/worker/input identities and exact independent outputs. The host then selects that next checkpoint. This is a synthetic diagnostic transaction, not a claim of production crash consistency, power-loss recovery or a general multi-model checkpoint ABI.

## Validation coverage

The reference is one conventional full-layer graph with all eight projection matrices, not two separately accepted branches spliced at an unchecked boundary. Captures include all earlier attention taps, attention residual, FFN normalization/gate/up/activation/down/post-normalization, final output, mask and complete KV.

Cases start empty at positions0/1 for SWA and global layers. Separate boundary cases execute1023/1024/1025 using explicitly synthetic seeded history, then persisted computed continuation. They do not claim 1023 evaluated prefix tokens. Every masked probability must be exactly zero; SWA uses1024 visible keys at the boundary while global counts grow to1025/1026. NoPE identity and nonzero-position RoPE are checked.

Reference and candidate use fresh processes for each token and load their own verified previous checkpoints. Repeating the same second-token continuation before and after injected failures must reproduce all bytes. Per-job receipts permit verified resumption after interruption: complete jobs are rechecked, incomplete attempts are preserved and retried, and no incomplete result counts as success.

Faults cover procedural weight extraction/staging, projection execution, KV copies/updates, attention compute, attention residual, the inter-branch boundary, FFN operators, final writeback, serialization and cleanup/publication. Inputs remain unchanged on every accepted failed attempt. These are deterministic exception hooks, not forced malloc-OOM, sanitizer or exhaustive power-loss tests.

## Final numerical and state results

**PASS: complete real-width synthetic layer, both tested attention variants.** Evidence: `qies109/runs/20261111-145928-b3603f27/summary.json` and per-job capture/monitor/receipt files. The final campaign contains107 accepted jobs:23 numerical/fixture jobs,78 injected failures and6 malformed-state/interface refusals.

- **312/312 captured files matched exactly**, totalling54,941,184 compared bytes:24 F32 taps, mask and complete checkpoint for each of 12 candidate/reference pairings.
- All eight projection-weight identities matched. Shape/length/byte comparisons use zero tolerance; no mismatch or tolerance relaxation occurred.
- Positions0/1 persisted KV through fresh processes for both SWA and global cases. Boundary cases covered1023/1024/1025; SWA retained1024 visible keys, global retained1024/1025/1026. Masked probabilities were exactly zero and RoPE/NoPE assertions passed.
- Fourteen resumed numerical executions independently preserved every KV byte outside the current2048-byte row in each cache. Each accepted checkpoint advanced position exactly once, never once per branch.
- Repeated second-token execution before and after the fault sweep reproduced every full-layer output and checkpoint byte. This is fresh-process repetition, not long-lived in-process stress testing.
- All 78 fault hooks returned the expected failure with zero modeled resource owners, no accepted or provisional output directory and an unchanged input checkpoint. In particular, faults between branches and throughout FFN did not publish the attention-only candidate state.
- Truncated/corrupt checkpoint copies and real/missing-namespace dispatch were rejected. All inherited rlimits remained unchanged; no resource violation occurred.

The interruption stopped an unfinished fault attempt, not a completed campaign. Resumption checked original source and worker hashes, command/input/output identities and receipts; preserved incomplete attempts under new names; and reran only missing jobs. It rechecked all numerical comparisons and masks before writing final success. No partial attempt was silently accepted.

## Measured complete-layer residency

Worker RSS below is max(kernel HWM, observed worker RSS). PSS, anonymous and file-backed values are phase-sampled maxima, not guaranteed instantaneous peaks. Rows aggregate equivalent tests; separate maxima may occur at different times. Full observations are in `qies109/measurements.json`.

| Execution / KV slots | Peak RSS MiB | Sampled PSS MiB | Sampled anonymous MiB | Sampled file-backed PSS MiB |
|---|---:|---:|---:|---:|
| Streamed layer /256 | **19.598** | 13.637 | 10.285 | 3.363 |
| Conventional reference /256 | **538.141** | 528.012 | 525.207 | 3.361 |
| Streamed layer /1280 | **31.477** | 22.393 | 19.102 | 3.299 |
| Conventional reference /1280 | **556.996** | 536.016 | 533.215 | 3.332 |

These are measured whole synthetic layer peaks, replacing the earlier estimates from separate branches. At1280 slots the observed worker RSS difference is about 525.52 MiB. It is not an engine-wide or system MemAvailable saving, and the small difference from QIES108's attention-only peak is ordinary run-to-run variation, not proof that adding FFN lowers memory.

| Bounded allocation | Streamed | Conventional reference |
|---|---:|---:|
| Peak quantized projection weights | 3,604,480 bytes | 545,669,120 bytes |
| Peak CPU tensor buffers,256 slots | 3,768,832 bytes | 548,718,688 bytes |
| Peak CPU tensor buffers,1280 slots | 8,819,712 bytes | 555,669,600 bytes |
| One allocated reusable workspace | 1,048,576 bytes | 1,048,576 bytes |
| Largest requested graph scratch | 571,136 bytes | 571,136 bytes |
| Logical authoritative KV | 1 MiB /5 MiB | 1 MiB /5 MiB |

The projection-weight counter excludes small F32 gamma/lambda tensors; all model layer weights including those total545,752,068 bytes from descriptors. Normalization scales are included in actual CPU buffer observations. Context metadata, host activations, bounded synthetic fixture caches, tentative KV and checkpoint serialization payloads are additional owned memory represented by measured RSS.

Runtime boundary records explicitly show zero active weight/tensor buffers before FFN and before checkpoint serialization. Scratch is allocated once and reused. The authoritative KV remains live outside backend counters; temporary backend and tentative copies are distinct. Serialization adds a bounded payload rather than hiding a copy. After cleanup, zero modeled owners does not imply that allocator/library pages or the authoritative KV have vanished from RSS. No allocator tuning, cache drop or system-memory reconfiguration is used.

Actual weights in these tests are touched anonymous CPU storage, not virtual-only mappings or shared file-backed model pages. File-backed PSS principally reflects code/libraries. Checkpoint and diagnostic page cache is additional system memory. The monitor records process-tree PSS, available RAM/swap/disk and inherited shared cgroup statistics. That cgroup's historical peak is not reset or treated as the worker peak. The campaign's sampled system MemAvailable stayed above approximately 3.31 GiB.

## 4 GiB system feasibility: extrapolation, not acceptance

Measured active-layer RSS below32 MiB makes a bounded short-context layer numerically plausible within4 GiB, but does not qualify a complete engine or this desktop configuration on that hardware. The current machine has about 7.63 GiB physical RAM; no4 GiB total-system experiment occurred. Real file IO/cache, tokenizer/loader, scheduling, all 80 layers and engine metadata are absent from this prototype.

For illustration only, a4,096 MiB budget with 1,024 MiB OS/services,1,024 MiB safety reserve,512 MiB unmeasured loader/engine overhead and64 MiB active-layer allowance leaves1,472 MiB for KV/cache/other overhead. Full F16 KV for 80 layers is80 MiB at 256 slots or400 MiB at 1280 slots, leaving approximately 1,392/1,072 MiB respectively under those assumptions. These are conservative planning allocations, not measured requirements. A larger desktop, real IO cache or allocator retention can consume that headroom; global-attention KV continues growing with context.

Most decisively, a4 GiB machine cannot meet the unchanged6 GiB MemAvailable real-execution gate. This milestone does not lower, reinterpret or unlock it. **No real-weight or4 GiB system GO is granted.**

## Remaining real-weight integration and QIES110

**GO for QIES110 implementation of a bounded real-tensor source adapter and synthetic/data-only verification, retaining the real-execution lock.** The concrete next task is to replace procedural projection/input factories with typed read-only sources, not to repeat a speculative readiness audit.

Required work includes:

1. Supply exact GGUF quantized row ranges without requantization, bounded staging/IO and whole-model mapping; verify dimensions, types, offsets and tensor/model hashes through the already established independent extraction paths.
2. Supply actual F32 normalization weights and learned value-residual lambda, not the synthetic constants; parameterize layer identity/quantization, including the Q4_K FFNs already verified standalone.
3. Bind checkpoint identity to real model/version/hyperparameters and reject synthetic checkpoints. Preserve full-layer publication and failure semantics.
4. Provide an independently loaded real-weight stock full-layer observer with matched input/state/CPU configuration. Synthetic full-layer equality is not proof for actual GGUF bytes.
5. Require the unchanged RAM/ancestor/cgroup/address-space conditions and positive live guard/reference qualification before any future real arithmetic. They remain prerequisites, not satisfied by the lower synthetic RSS.

No whole-model logits, tokenizer, generation loop or full prompt inference is implemented here. Joined Q4-FFN layers, longer in-process repetition, larger contexts and multi-layer scheduling also require direct tests. The success claim is one complete synthetic layer at real dimensions, for the two descriptor-verified layer variants tested.

## Reproduction and decision

```sh
cd '<workspace>/qies109-worktree'
./pandacode-run.sh --smoke
./pandacode-run.sh
# Only if a campaign is unfinished:
./pandacode-run.sh --resume qies109/runs/CAMPAIGN
```

Completed campaigns refuse overwrite. The approved interface is synthetic-only, with private CPU libraries, fixed cases and at most a read-only bounded KV input inside a sealed filesystem. The existing6 GiB gate and unconditional real-execution lock remain compiled unchanged. Frozen branch/tag and original source/results are verified in `final-integrity.json`; `SOURCE-SHA256.json` records the final uncommitted snapshot.

**GO: complete synthetic layer and QIES110 bounded adapter development. NO-GO: real-weight numerical execution, production acceptance, whole-model inference or a4 GiB system claim.** No production FlashNext or frozen milestone was modified; no commit, merge or push occurred.
