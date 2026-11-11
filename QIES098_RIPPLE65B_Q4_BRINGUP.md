# QIES098 - Ripple-65B AD-Q4_K_M Bring-Up

Date: 2026-11-10. CPU-only acquisition and metadata/source investigation. No inference or production FlashNext changes.

**Status: DOWNLOAD COMPLETE AND SHA-256 VERIFIED.** Final metadata/index inspection was repeated on the accepted complete file and matched the provisional tensor directory. **Conditional GO for further isolated Ricursive adaptation; NO-GO for inference on the current desktop.** The final unchanged guard check found only 3,985,264,640 bytes (3.71 GiB) MemAvailable versus 6,442,450,944 required. A native Ricursive single-layer adapter and qualified reference executable are also not available.

## 1. Protected baseline and isolated work

Worktree: `/home/user/PandaCode Bots/meatball-84ccea00/qies098-worktree`.
Branch: `experimental/ripple-65b-q4`.
Base/HEAD: `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`, the verified continuous-generation baseline.

Fetched all GitHub branch refs and tags without pruning, force-pushing or overwriting existing milestone refs. The new branch was absent before creation. Existing FlashNext, QIES089 and QIES095 recovery branches/tags and their worktrees remain intact. The recovery commit is not an ancestor selected for this experiment; QIES086 was not applied. No tracked production source changed. New code and evidence are confined to this worktree and its sibling `qies098-data` directory.

QIES096 recommended a tensor-index identity-cache experiment; no local QIES097 implementation/results were found, and this worktree has not applied a caching patch. Existing generic metadata/vocabulary caches are source-supported reuse candidates, not proof that this baseline recognizes Ricursive.

## 2. Exact acquisition and storage

Authoritative quantized repository: [AtomicChat/Ripple-65B-Preview-GGUF](https://huggingface.co/AtomicChat/Ripple-65B-Preview-GGUF). Public and ungated; the pinned card declares Apache-2.0, and the original model's Apache license was retained separately. No authentication or license acceptance action was required.

| Identity | Value |
| --- | --- |
| Requested file | `Ripple-65B-Preview-AD-Q4_K_M.gguf` |
| Repository revision | `df550a7178a87bbfb6ee2bdbd37603c553d536ee` |
| Exact repository size | 40,426,079,712 bytes (~40.426 GB / 37.650 GiB) |
| Published LFS SHA-256 | `9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc` |
| Observed complete-file SHA-256 | `9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc` (MATCH) |
| Final model path, outside Git | `/home/user/PandaCode Bots/meatball-84ccea00/qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf` |
| CLI | huggingface_hub 2.4.0, isolated `qies098-data/hf-venv` |

The LFS SHA-256 identifies the complete weights file; it is not the repository Git revision or pointer blob ID. Only this quantization/file was requested. AD-Q4_K_M is distinct from the repository's plain Q4_K_M preset.

Initial filesystem free capacity was about 112.8 GB. The download uses local-dir incomplete storage followed by atomic rename, with no second full-sized model copy. Required incremental budget was one 40.426 GB file plus a 30 GiB reserve (~72.638 GB combined), plus small CLI/metadata artifacts. Two simultaneous full copies would not safely fit that same reserve, so duplication was avoided.

`download_model.py` uses one HF file worker, pinned revision, HTTP resume, implicit authentication disabled and Xet disabled to keep storage behavior straightforward. It checks remaining bytes versus free space and a 30 GiB reserve throughout. It preserves interrupted partials. Once HF reports completion, it quarantines the file by rename, verifies size and streams SHA-256 with a 4 MiB buffer, then publishes the accepted filename without copying. Existing completed targets are never overwritten. Private download logs are not reproduced because service errors can include signed URLs.

Final receipt: `qies098-checks/download-verification.json` (also retained outside Git in `qies098-data/metadata/download-result.json`). Download plus full SHA verification took 2633.8 seconds (~43.90 minutes). Remaining SSD capacity: **72,300,974,080 bytes (67.34 GiB)**. The verified file and containing directory were synchronized; final file identity was recorded. Only one completed GGUF is present, with no incomplete/quarantine model copies remaining.

## 3. Metadata and tensor directory

A bounded reader inspected only the GGUF header/directory while the resumable file was downloading. It reads at most 64 MiB, omits large tokenizer arrays from its JSON output, checks counts, dimensions, alignment, quant block sizes, complete-file bounds and overlapping tensor ranges. It never reads tensor payloads for inspection. Final inspection was repeated against the fully hash-verified file; all 1,203 tensor directory entries matched the provisional parse exactly.

Observed header:

| Field | Value |
| --- | --- |
| Architecture / GGUF | `ricursive`, version 3 |
| Parameters from tensor shapes | 64,782,689,360 |
| Transformer blocks | 80 |
| Hidden / FFN dimensions | 5,120 / 40,960 |
| Query / KV heads | 80 / 8; head width 128 |
| Attention pattern | 64 sliding-window blocks, 16 global blocks; every fifth block global |
| Sliding window / advertised context | 1,024 / 2,097,152 tokens |
| RMS epsilon / RoPE base | Stored F32 ~1e-6 / 10,000 |
| Embedding scale / SwiGLU clamp | Stored F32 71.5541763305664 / 20.0 |
| Vocabulary | 100,352; tokenizer `gpt2`, pretokenizer `dbrx` |
| Tensor count / directory end | 1,203 / tensor-data offset 3,615,712 |
| Quantized weights per block | 467,108,868..545,752,068 bytes (~445.5..520.5 MiB) |
| Embedding / output weight bytes | 272,957,440 / 421,478,400 |

Actual tensor quantization counts, not inferred from the filename:

| Type | Tensors |
| --- | ---: |
| F32 | 561 |
| Q4_K | 310 |
| Q5_K | 90 |
| Q6_K | 81 |
| Q8_0 | 160 |
| IQ4_XS | 1 |

Q/K/V, gate, output and FFN tensors therefore require type-aware dispatch. For example the first block's K/V weights are Q8_0, attention output is Q6_K, and its FFN projections are Q5_K. Norms and value-residual lambda are F32; embedding is IQ4_XS.

The read-only index records exact name, shape, quant type, byte length and absolute offset. All **1,203 named descriptors** matched the unchanged baseline index parser's offset/size/type/first-two-dimension output in a metadata-only compatibility probe. This validates descriptor plumbing only. FlashNext's expert stride `/512`, its descriptor block_size field (logical width rather than a native quant block size), MoE resolver and hardcoded 2560/640 dimensions are not Ricursive dense-layer semantics and must not be used.

## 4. Pinned architecture reference

[llama.cpp b12167 Ricursive implementation](https://github.com/ggml-org/llama.cpp/blob/240d68fcfe0c2a1f298c8e18ca181368b1914259/src/models/ricursive.cpp), resolved commit `240d68fcfe0c2a1f298c8e18ca181368b1914259`, is the reference source. Relevant model registry, graph, tensor-name and SWA-cache files and MIT license were retained under `qies098-checks/upstream-b12167/`.

Original model configuration/source was inspected as data at base-model revision `b172323ffd63e62de9ce1ba3830dfffe493c7671`. It was not executed with trust_remote_code. That separately pinned current base-code revision is not a claim about the exact BF16 revision used to create the quantization.

Reference operation order: per-block pre-attention RMSNorm; Q/K projections and per-head Q/K RMSNorm; RoPE on sliding layers only, NoPE on global layers; value input mixing of the original scaled token embedding with normalized current hidden state using sigmoid(lambda); attention; sigmoid attention-output gating; output projection; post-attention RMSNorm and residual. FFN uses pre-RMSNorm, gate/up projections, clamped SwiGLU, down projection, post-RMSNorm and residual. Preserve these graph operations and their order. [Pinned Ricursive graph](https://github.com/ggml-org/llama.cpp/blob/240d68fcfe0c2a1f298c8e18ca181368b1914259/src/models/ricursive.cpp).

The base-model reference upper-clamps the gate projection at 20 and clamps the up projection to [-20,20] before the SiLU product; do not substitute post-activation or output clamping. The original embedding basis is the same per-token x0 across all blocks; it is not the immediately previous layer output or a previous token. GQA ratio is ten query heads per KV head. Preserve the GGUF F32 embedding-scale value, not a newly computed approximation. The baseline already exposes a clamped SwiGLU kernel, but matching CPU kernels/reduction paths across versions still require direct validation.

QIES098 has not built or runtime-qualified a b12167 reference executable. Source inspection is not an oracle qualification or bit-exactness result.

## 5. Reuse versus required adaptation

| Component | Reusable basis | Required Ricursive work |
| --- | --- | --- |
| GGUF/file-backed access | Header parsing, exact offsets, pread and generic quant block APIs | Pin model/index identity, use the actual mixed tensor types and 5120/40960 widths |
| Metadata cache | Existing loader/vocab caches avoid repeated GGUF parsing | Baseline lacks `ricursive` registry/factory; cache cannot add architecture support |
| Bounded weights | Existing one-layer residency and tiled CPU matvec concepts | New dense-layer selectors/graph, no sparse top-k or FlashNext expert layouts |
| CPU GGML | RMSNorm, sigmoid, matmul, RoPE, attention and SwiGLU primitives exist | Match b12167 graph/kernel settings and clamping; API presence is not numerical proof |
| Layer scheduling | Sequential input/position/history and scoped model/context ownership | 80 layers, 5120 hidden vector and separate x0; no 48-layer/HC/DeltaNet assumptions |
| State persistence | File-backed storage and explicit ownership principles | Ricursive global/SWA attention KV, positions, masks and eviction; never import FlashNext recurrent blobs |
| Tensor index | Named descriptor resolution worked for all tensors | A future cache must invalidate by identity; no QIES097 cache result was applied here |
| Token/text handling | Generic tokenizer/EOG/output safety concepts | Use embedded Ripple template/tokenizer metadata; QIES072's Qwen prompt/template is not a Ripple oracle |

The verified baseline has no Ricursive architecture enum/registration/model class. An isolated port needs the architecture registry, metadata keys/hparams, tensor construction and graph builder, plus explicitly bounded KV allocation and state serialization. Copying a new graph file into the old tree alone is insufficient. Preserve FlashNext sources/binaries and keep any future Ricursive changes architecture-scoped.

The older helper's 10,240-value HC stream and Ripple's 10,240-wide Q/attention representation happen to have the same element count but are unrelated layouts. Rebinding one as the other would be wrong.

## 6. Memory feasibility and state risks

One block's quantized weights are below 0.55 GB, so a bounded one-block or tiled dense executor is plausible without retaining all 40.4 GB in RAM. This is a source/size-based feasibility observation, not a measured working implementation. Full-F32 dequantization of a 5120 x 40960 projection alone needs ~0.839 GB; doing several at once defeats the intended bound. Keep quantized/tiled kernels and bounded scratch.

The FFN gate/up/activation vectors are 163,840 bytes each at one token, while x0 is 20,480 bytes. Graph buffers, state capture/restore, backend allocations and allocator retention can be much larger and remain unmeasured.

For F16 K and V, each layer/token's ideal payload is 4,096 bytes. An ideal window-trimmed aggregate is:

`4096 * (64 * min(T,1024) + 16 * T)`

| Context T | Ideal aggregate KV payload |
| --- | --- |
| 256 | 80 MiB |
| 1,024 | 320 MiB |
| 4,096 | 512 MiB |
| 32,768 | 2.25 GiB |
| 2,097,152 | 128.25 GiB |

These are not native allocation limits: SWA cache padding, ubatch headroom, sequence metadata, full-size-SWA mode and implementation bookkeeping add overhead. Upstream's SWA cache explicitly pads cells; do not size safety guards from this ideal formula.

A nominal one-layer graph may still allocate or serialize caches for all 80 layers unless the native memory filter is scoped correctly. Repeating a full-model cache per layer can create large redundant serialized/resident state, as prior FlashNext work demonstrated. Verify actual per-layer allocation and serialization scope before claiming the ideal 1 MiB per-layer payload at context 256.

This is attention-only persistence, not FlashNext Gated DeltaNet constant-size recurrent state. Global layers grow with context; SWA eviction and position handling need exact tests around window boundaries. The advertised two-million-token context is not feasible under this laptop's RAM budget with ordinary F16 KV.

Hardware inspected: Intel i5-7200U, 2 physical cores/4 threads, ~7.63 GiB RAM; Micron 1100 SATA SSD with encrypted btrfs. CPU SIMD features were recorded. Dense inference potentially touches nearly every weight per token; low memory does not imply low I/O or interactive latency. No token/s throughput is predicted.

## 7. First bounded single-layer test

Developed `layer_contract.py`, a **model-free test contract/admission helper**, with layer 0 (SWA) and layer 4 (global) plans, 15 required tensor descriptors per block, exact dimensions/types/offsets, context <=256, batch 1, F16 KV estimates and comparison points. It never executes a decoder. A native single-layer test executable is **not implemented**, because registry/graph/KV adapters and a qualified reference still need development. Do not mistake a prepared contract for executed model validation.

Tests completed: three tiny header-only parser groups (including malformed/partial headers, bounds/alignment/type rejection), three contract/admission groups, and exact comparison of all 1,203 named descriptors. Parser error paths close their files. These are model-free checks, not layer/logit comparisons.

Future bounded execution procedure:

1. Verify the complete model SHA-256, pinned source/library/compiler/ISA identities, exact quantized reference and adapted index.
2. Build an isolated CPU-only Ricursive adapter with one active layer and bounded weight/graph/state buffers. Demonstrate cache filtering/serialization scope and no full-model weight allocation before any decode.
3. Require fresh unchanged guard admission: >=6 GiB MemAvailable and every existing swap/disk/cgroup/PSI/deadline rule. A separately reviewed command adapter is necessary because the existing guard launch path pins FlashNext; do not overwrite its executable or lower policy to run Ripple.
4. Use a qualified matching b12167+ quantized reference on a suitable CPU host. Capture deterministic x/x0, Q/K/V, norm/gate, attention output, FFN and final layer outputs plus KV data/positions. Reference instrumentation must not change clean numerical results.
5. Compare layer 0, then layer 4; repeat for a second token with preserved KV. Later test SWA boundaries at positions 1023/1024/1025, global continuity, masks, save/read and failure handling. Preserve exact graph order, quantization and comparison tolerances.
6. Stop on any guard violation or mismatch. Do not advance to a full-model chain until per-layer/state evidence passes.

Current initial resource check: MemAvailable 3.63 GiB, swap free 16,062,517,248 bytes, no measured PSI full pressure; resources_ready=false. No inference was attempted. Final resource check: **MemAvailable 3,985,264,640 bytes (3.71 GiB), resources_ready=false, exit status 1**; swap free 16,062,521,344 bytes, disk free 72,300,974,080 bytes, PSI full avg10 0.0. Inference remained blocked.

## 8. Artifacts and next decision

- `qies098-data/metadata/`: pinned API/card/license, isolated CLI installation record, private transfer log and final download receipt.
- `qies098-checks/upstream-b12167/`: pinned reference and base-model sources/config/licenses, read as data only.
- `qies098-tools/inspect_gguf.py`: bounded metadata/directory inspector.
- `check_index_compat.py` and `index_compat.cpp`: named-descriptor comparison against unchanged baseline code.
- `layer_contract.py`, `test_contract.py`, `test_metadata.py`: safe admission/metadata tests.
- `finalize_metadata.py`: re-inspects only verified final metadata and writes the accepted tensor index.
- `pandacode-run.sh`: analysis-only reproduction; never downloads weights or launches inference.

**Conditional GO** for architecture-scoped Ricursive registration/graph/KV adapter development and independent reference qualification. **NO-GO** for current-desktop inference or any bit-exactness/performance claim. The model artifact is accepted after full size/SHA verification; native execution remains unimplemented and unqualified.

No FlashNext production code, resource guards, system settings, QIES086 patch, recovery merge or GPU dependency changed. No inference, commit, merge or push occurred.
