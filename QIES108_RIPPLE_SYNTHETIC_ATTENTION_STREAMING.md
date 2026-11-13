# QIES108 — Real-width synthetic Ricursive attention streaming

**GO for the complete bounded synthetic attention branch and QIES109 full-layer integration.** Real-weight execution, production acceptance and complete-layer correctness remain unqualified. No real model weights, FlashNext changes, guard changes, commits, merges or pushes were used.

## Lineage and preserved baseline

Worktree: `<workspace>/qies108-worktree`, isolated from the QIES107 lineage and detached at frozen source ancestor `14b3f7902864b02e1a84e5e6168a3cf99087e670`. Engine ancestor remains `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. New uncommitted source is under `qies108/`; the local Run Script dispatches only synthetic attention.

Before work, QIES107's source/report manifest was verified and its source, results and inherited QIES106 preservation archive were copied into `qies108/QIES107-preserved.tar.gz`, with SHA256/per-file manifest in `preservation.json`. Original QIES106, QIES107 and the QIES105 archive remain intact.

A fresh QIES107 replay completed at `qies107/runs/20261111-130001-e209c0f6/summary.json`: **63 byte-exact comparisons, 100 fault tests and six refusals**, with output hashes equal to the preserved original. An interrupted earlier replay is retained but was not counted. The new attention driver requires this fresh passing replay before arithmetic.

Independent stock source: llama.cpp b12167 `240d68fcfe0c2a1f298c8e18ca181368b1914259`; all 4,084 pinned source hashes verified. Operator order follows `src/models/ricursive.cpp`, `build_lora_mm`, RoPE and ordinary nonflash attention construction in `llama-graph.cpp`. The independent observer builds a conventional stock-GGML graph; it does not compile the candidate's streaming or stage implementation. This is an independently assembled stock-GGML branch reference, not a claim that a full stock model object was run.

Candidate GGML 0.25.3 and stock GGML 0.29.0 differ; exact agreement is demonstrated rather than inferred from identical versions. Workers use GCC 16.2.1, C++17, `-O2 -Wall -Wextra`, single-threaded ordinary CPU buffers, F16 KV and F32 score accumulation. Source-built SDKs use the recorded Release/native CPU profile. Source, compiler command, library and executable hashes are in the per-campaign build manifest. CPU remains the Intel i5-7200U; no GPU, libllama, system install, allocator tuning or system configuration changes occur.

## Implemented computation and streaming

Actual dimensions: hidden 5120; query/gate10240; K/V1024; 80 query heads,8 KV heads,128 dimensions per head. Projection types match actual GGUF descriptors for layers 0 and4: Q/gate Q4_K, K/V Q8_0 and O Q6_K. Saved model identity remains SHA256 `9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc`; QIES108 reads descriptors, not the model file.

Implemented sequence:

1. RMS-normalize layer input x and apply its gamma.
2. Stream Q and K projections.
3. Compute `x0 + (normalized_x - x0) * sigmoid(lambda)` using distinct original-embedding x0 and layer-input x; stream V from that value-residual input.
4. Apply complete 128-element per-head Q/K RMS and gamma.
5. For layer 0/SWA, apply NEOX RoPE with position,128 rotary dimensions, base10000 and matched scale/context settings. Layer4/global uses NoPE.
6. Convert current K/V to F16 through existing GGML copy kernels into private cache slots; retain the authoritative prior cache separately.
7. Compute grouped QK scores with F32 accumulation, exact causal/sliding masks, scaled softmax, then V multiplication and layout conversion through existing CPU kernels.
8. Stream the attention-gate projection from normalized input; sigmoid and multiply attention values.
9. Stream O projection, apply post-attention RMS/gamma, and add original layer input x. This output is the FFN input for later integration.

All dot-product inner dimensions and attention reductions remain intact. Weight staging uses128 output rows and synchronous GGML execution; no partial-sum reassociation, approximate softmax, reduced precision or new quantization is introduced. The largest weight tile is O:1,075,200 bytes. Projection buffers are released before the next stage; small activations are released after their final consumers. One1 MiB scratch reservation is reused, with each plan checked to fit.

`stream.cpp` contains the bounded stage executor. `reference.cpp` constructs one whole conventional attention graph and holds all five projection matrices. `common.h` defines synthetic inputs, type-aware fixture construction, ownership and the bounded checkpoint/capture ABI; no candidate numerical graph is shared with the observer. `fixture_check.cpp` validates packing independently, `build.py` records provenance and `run.py` implements comparisons, continuation and resumable test orchestration.

The first smoke attempt rejected a Q8_0 fixture-size assumption before attention arithmetic: the old FFN factory assumes256-value K blocks, whereas Q8_0 has32-value blocks. QIES108 now uses a separate type-aware factory based on `ggml_row_size(type,256)`, leaving QIES107 unchanged. Eighteen block-expanded versus whole-row quantization checks across Q4_K/Q8_0/Q6_K, two row widths and three row identities passed with buffer canaries intact. This is synthetic fixture construction only; no real weight is requantized.

## Numerical comparisons, masks and continuation

Final evidence: `qies108/runs/20261111-130947-6dfa5b92/summary.json`. **228/228 captured files matched byte-for-byte, totalling48,059,904 compared bytes.** Each paired step checks five complete quantized weight hashes,17 F32 intermediates/final outputs, the mask and the entire serialized checkpoint. Tolerance is zero; a mismatch saves details and stops. No mismatch/tolerance relaxation occurred.

| Case | Executed positions | Correct visible keys |
|---|---|---|
| Empty SWA cache, fresh-process continuation | 0,1 | 1,2 |
| Empty global cache, fresh-process continuation | 0,1 | 1,2 |
| SWA window boundary | 1023,1024,1025 | 1024,1024,1024 |
| Global layer at equivalent positions | 1023,1024,1025 | 1024,1025,1026 |

For every head, all masked probabilities are exactly zero. Every mask element is checked against the causal/window rule. Global Q/K current bytes equal their normalized bytes (NoPE); SWA nonzero positions produce rotated Q bytes. Boundary tests start from explicitly synthetic nonzero history at 1023, not from an asserted1,023-token generated prefix. Positions1024/1025 do continue from the previous computed checkpoint.

First and second tokens start from empty state and use independently computed, persisted KV. Each reference/candidate token is a fresh process loading its own preceding checkpoint, not an in-memory shortcut. Full state files match across paths, and position advances exactly once. Fourteen resumed numerical executions were additionally checked to leave every K/V byte outside the newly written2048-byte row of each cache unchanged.

Repeating the same candidate continuation before faults, then again after all faults, reproduced every intermediate and checkpoint byte against the independent second-token reference. This is repeated fresh-process execution; long-lived in-process attention reuse is not claimed.

## Persistent-state and failure boundaries

KV is F16, separately owned from temporary weights:1 MiB at 256 slots,5 MiB at 1280 slots. Headers encode fixed magic/version, layer, slot count, position, lengths and synthetic fixture ID; SHA256 covers metadata and payload. Reads reject bad length, wrong identity/dimensions/position and checksum mismatches. The format is a fixed synthetic-test ABI, not a production multi-model checkpoint or authentication format.

The input checkpoint is mounted read-only and never overwritten. The authoritative in-memory State is const; K/V copies and tentative updates are private. The branch serializes a new checkpoint into an exclusive provisional capture directory only after its numerical steps succeed. Publication requires completed computation and uses non-overwriting directory rename. The host promotes that result for the next token only after successful job status, integrity checks and numerical comparison. Build/worker identities are bound by campaign and per-job receipts.

**105/105 injected fault cases passed**, covering loading, projection allocations/compute, Q/K normalization/RoPE, shadow KV allocation, each K/V write, attention compute, state download, O projection, captures, serialization and the cleanup-before-publication boundary. Every handled failure returned failure, left zero modeled owners and no accepted/provisional output directory, kept the input file hash unchanged and reported unchanged authoritative state. Failures after K or V writes affect only temporary backend buffers.

Two malformed copies—truncation and payload corruption—were rejected. Real dispatch and absent-namespace requests were rejected for both workers. These are six additional refusals, separate from the six baseline QIES107 refusals. Checkpoints and models were not damaged for testing. No claim is made for forced allocator-OOM, ASan, process power loss, atomic production crash recovery or all possible fault combinations.

## Measured memory

Numbers below are actual worker observations, not virtual map sizes. RSS is max(kernel HWM, observed worker RSS); PSS/anonymous/file-backed figures are phase-sampled maxima and may miss shorter transients. Context rows aggregate equivalent tests. Full raw records and separate process-tree samples are retained in `measurements.json` and job logs.

| Execution / cache slots | Peak RSS MiB | Sampled PSS MiB | Sampled anonymous MiB | Sampled file-backed PSS MiB |
|---|---:|---:|---:|---:|
| Streaming /256 | **16.492** | 9.796 | 6.441 | 3.354 |
| Conventional reference /256 | **125.020** | 115.400 | 112.633 | 3.315 |
| Streaming /1280 | **31.652** | 23.805 | 20.375 | 3.434 |
| Conventional reference /1280 | **143.859** | 123.401 | 120.637 | 3.315 |

The candidate's maximum actual CPU tensor allocation is1,868,800 bytes at 256 slots and8,819,712 at 1280. The reference peaks at 115,525,728 and122,476,640 bytes respectively. Five complete projection weights account for 113,131,520 bytes in the reference, versus a maximum1,075,200-byte streamed weight tile. Largest plan scratch is536,128 bytes (the allocated reusable reservation is1,048,576). Attention's own buffers include KV, scores/probabilities and the reference-style transposed-V copy; they are bounded by the fixed slot count.

State safety deliberately has a cost: authoritative KV, a tentative copy and backend K/V coexist; serialization adds a bounded temporary payload. State-file page cache and diagnostic writes are additional system memory beyond worker RSS. After backend cleanup, the authoritative1/5 MiB state remains live until process exit; zero backend-resource counters do not mean zero persistent KV or zero allocator/library residency. File-backed PSS here is primarily code/libraries, not mapped model weights.

All runs respected inherited rlimits and model-free monitoring; no resource violation occurred. The inherited desktop cgroup is shared, with historical memory.peak, so it is not interpreted as an isolated worker peak or reset. The full campaign sampled MemAvailable no lower than about 3.64 GiB. Existing real inference still requires6 GiB and remains locked; these synthetic measurements are not permission to run it or proof of 4 GiB total-system operation.

## Estimated complete-layer budget and remaining work

A joined layer is **not implemented or measured here**. QIES107 measured a three-branch FFN peak of 16.46 MiB. QIES108 attention measured16.49 MiB at 256 slots and31.65 MiB at 1280. These separate peaks cannot simply be advertised as the complete-layer peak.

A conservative planning envelope adds both observed subsystem peaks plus retained old/tentative KV: roughly 35–40 MiB at 256 slots and55–65 MiB at 1280, before integration-specific allocator retention, checkpoint copies, process state, tokenizer/loader and filesystem cache. With strict last-consumer release, a maximum rather than a sum may be achievable, but that is an **estimate requiring measurement**. Longer global context grows KV and attention storage; weight streaming does not eliminate it. No long-context eviction/reduction change is proposed or numerically qualified.

**QIES109 should join the independently validated attention and FFN into one synthetic real-width layer**, pass the attention residual as FFN input, preserve x0 correctly, compare every intermediate and final layer output, and test at least two positions for SWA and global layers. Crucially, attention's next KV must remain provisional until the FFN also succeeds; standalone attention checkpoints are not completed full-layer transactions. Add faults between branches and repeated execution in one long-lived process, then measure the composed memory peak. Preserve the exact upstream numerical kernels, serialized ownership and current guards.

## Recovery of interrupted testing and preservation

Completed jobs were reused only after checking source/binary hashes, command, input-checkpoint hash, status, output-state hash and resource receipts. Incomplete attempts were renamed and preserved, not accepted or overwritten. The final resume rechecked all numerical comparisons/masks before completing remaining tests. The initial fixture refusal and interrupted runs remain on disk.

Run the synthetic procedure with `./pandacode-run.sh`, a short check with `--smoke`, or an unfinished campaign with `--resume qies108/runs/CAMPAIGN`. Completed campaigns refuse overwrite. `README.md` describes dependencies and scope. `final-integrity.json` and `SOURCE-SHA256.json` identify preservation and the final uncommitted snapshot.

**Decision: GO for the complete synthetic attention branch and QIES109 synthetic full-layer integration. NO-GO for a complete-layer, real-weight, production or4 GiB model-operation claim.** QIES105–107, the 6 GiB guard, production FlashNext and retired real execution paths remain preserved. No commits, merges or pushes occurred.
