# QIES105 — Ripple preservation and validation handoff

**GO for source preservation and portable model-free replay. NO-GO for real-weight execution.**

QIES098–104 source, tests, reports, historical evidence and the unchanged resource guard are preserved on a separate archive branch. A relocatable source-only replay package rebuilds both CPU SDKs and reproduces the matched-profile synthetic results. No real-weight arithmetic or model download occurred in QIES105. No FlashNext source, guard policy, operating-system setting, QIES086 patch, or recovery prototype was changed/applied.

## Source provenance and Git refs

| Item | Identity |
|---|---|
| Repository | `https://github.com/jddchatham/cortex-stream-forge.git` |
| Verified engine ancestor | `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388` |
| Active original Ripple branch | `experimental/ripple-65b-q4`, unchanged at that ancestor |
| Separate publication branch | `archive/ripple65b-qies104` |
| Annotated release tag | `ripple65b-qies104-frozen` |
| Independent stock source | llama.cpp b12167, `240d68fcfe0c2a1f298c8e18ca181368b1914259` |
| Stock source archive SHA256 | `b7c50c28d1c25942fdbca447326a692657238cd87beeea45797264b02bc64b4f` |
| Original QIES077 guard SHA256 | `ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2` |

Original worktree: `/home/user/PandaCode Bots/meatball-84ccea00/qies098-worktree`.
Publication worktree: `/home/user/PandaCode Bots/meatball-84ccea00/qies105-publication`.
The original experimental artifacts were untracked at 8fac93; that HEAD alone does not identify their contents. `qies105/evidence/original-status.txt` records them. The frozen-source manifest hashes 327 selected historical source/report/evidence files (4,497,990 bytes) without rewriting their contents. Historical source excerpts and required licensing are included; the independent stock source is fetched at the pinned hash or supplied offline and all 4,084 source files checked.

Before publication the requested branch and tag were absent locally and remotely. No existing ref is replaced. The archive inherits the verified baseline, adds isolated Ripple material and the new handoff tools, and does not merge into the continuous-generation engine. The complete tracked `runtime-flashnext-layer-exp` tree remains identical to 8fac93. Existing FlashNext, QIES089 and QIES095 milestones remain separate.

The commit ID, annotated tag object ID, peeled commit ID, package SHA256 and remote verification are bound in the final publication receipt appended to the delivered copy of this report. The committed copy uses ref names because embedding its own final commit/package hash would be circular. The annotated tag message binds the source package SHA256. Do not confuse the tag object ID with its peeled commit.

## Portable package and changes

Instructions: `qies105/README.md`. Entry: `qies105/replay.py`; portable test orchestration: `qies105/portable_tests.py`; read-only default-policy diagnostics: `qies105/resources.py`; exact guard: `qies105/guards/qies077-guard.py`. `pandacode-run.sh` reports locked status only. These are handoff tools, not production inference changes.

The source package contains the complete inherited baseline source, QIES098–104 experimental source/tests/reports and selected evidence, integrity manifests, exact model/revision/download instructions, build profile, original guard, independent reference procedure, layer-0 contract, comparison requirements and rollback instructions. It excludes the 40.426 GB model, other weights, extracted real tensors, numerical captures, ELF workers, SDK binary archives, source/build caches and credentials. Historical binary hashes are provenance records, not included executables or promised rebuild hashes.

The portable driver uses fresh compiler products, a pinned independent upstream source, newly generated bounded synthetic fixtures and a bubblewrap namespace without home/model directories. The stock reference SDK/observer is instantiated only in that synthetic namespace. Fresh host numerical workers have namespace-only library paths and no executable permission; the reference ELF is compressed data outside the namespace. The generated canonical stock `libllama` is disabled after its synthetic SDK is packed. No production raw executable is restored. The five retired builder/dispatch scripts still reject host use. Frozen historical scripts are retained byte-identically; their laptop paths are not silently reinterpreted.

## Verification actually performed

Source builds used GCC 16.2.1, Release CPU-only CMake, CPU repack enabled, native CPU profile, CUDA/Vulkan/OpenCL/BLAS disabled, examples/tools/server/app/tests disabled, at most two build jobs. Exact cache-derived flags/compiler and worker commands/hashes are in `qies105/evidence/source-build-configuration.json` and `portable-results.json`. Candidate GGML 0.25.3 differs from stock GGML 0.29.0; this is matched-profile comparison, not a claim that their kernels are identical.

The source-only tree was relocated without Git metadata, original fixture directories or prebuilt workers. Both SDKs were freshly built from the recorded sources; that new external compiler-output directory was then supplied to the relocated replay, which generated all fixtures and workers itself. NumPy 2.3.4 came from the existing isolated CPython 3.14 environment; no global installation. An initial driver attempt resolved the venv symlink to system Python and failed before fixture creation; that path handling was corrected, its failure log retained, and the complete replay then passed.

| Verification | Observed result |
|---|---|
| Fresh baseline CPU GGML / independent stock CPU llama builds | PASS |
| Original guarded QIES099 layer body | 14 groups PASS |
| Original guarded QIES102 adapter body | 16 checks PASS |
| Original QIES101 bounded-reader faults | 38 PASS |
| Independent stock raw-extractor original faults | 4 PASS |
| Fresh five-layer candidate vs independent stock, F32 | 360/360 byte-exact |
| Fresh five-layer candidate vs independent stock, quantized ordinary-buffer profile | 360/360 byte-exact |
| Independent one-layer stock vs fresh five-layer stock, F32/quantized | 72 + 72 byte-exact |
| Allocation-detector positive control | exit 97, expected trap confirmed |
| Active real CLI/dispatch/direct-API rejection cases | 10 rejected before GGML allocation, trap not triggered |
| Host/outside-domain worker attempts | 5 nonzero refusals (namespace-only library/admission) |
| Retired Python builders/dispatchers | 5 BLOCKED; unique final evidence filenames |
| Synthetic namespace / inherited AS+RSS limits | Home/model directories hidden; limits unchanged |
| Layer-0 input/schema regeneration and corruption checks | 9 contract invariant groups PASS; no fake golden hashes |
| Archived stock descriptors vs saved model index | 1,203 matched (metadata evidence replay, no new real GGUF parse) |
| Historical QIES098–104 selected source/evidence hashes | PASS |
| Unmodified engine source vs baseline | PASS |

Fresh results are in `qies105/evidence/portable-results.json` and associated logs. The final driver differs from the successfully replayed version only in giving five retired-builder log files unique names; those unchanged rejection assertions were rerun separately against the publication source. No numerical assertion or tolerance was altered.

Historical evidence is retained separately: QIES104's 40 rejection cases, the previous synthetic 720/144 comparisons, six historical regression suites including QIES103's 27 readiness characterizations, eight QIES101 real-byte comparisons and the later 15 layer-0 byte comparisons. These are **historical results**, not claims that every path/inode/binary-hash assertion was rerun unchanged in the portable environment. Old QIES103 tests intentionally characterize old gaps and include a below-threshold refusal; they are not a valid positive qualification on a larger receiving host. New replay counts above are explicit.

## Resource requirements and current blocking conditions

Read-only snapshot: MemAvailable 4,028,395,520 bytes (~3.75 GiB), SwapFree 15,336,349,696 bytes (~14.28 GiB), SSD free 63,377,215,488 bytes (~59.02 GiB), memory full PSI avg10 0.0. These are a snapshot, not guaranteed future capacity. The unchanged 6 GiB MemAvailable entry gate fails. No inference was attempted.

Original policy: entry >=6 GiB available RAM, >=4 GiB free swap, >=30 GiB disk, full PSI avg10 <1. Finite cgroup ancestor memory.max/high need >=7 GiB headroom, finite swap.max >=2 GiB. Preserve and inspect existing address-space/RSS limits; historical unlimited limits do not constitute a newly added hard cap. Default runtime stops at available RAM <1 GiB, swap <2 GiB, disk <20 GiB, process-tree RSS >=7 GiB, OOM/monitor errors, sustained pressure or four hours; whole-cgroup TERM/KILL cleanup. No policy overrides.

Layer-0 weights total 545,752,068 bytes (~520.47 MiB). Tensor staging, CPU weight buffers and compute workspace may coexist; real peaks and runtime remain unmeasured. Do not infer a real-layer memory bound from tiny synthetic results or invoke a full stock 40.4 GB model loader as a shortcut. Reserve model capacity plus paired extracted weights, captures/builds/logs and the original entry disk reserve.

**Hardware alone does not unlock the release.** QIES104 deliberately rejects all real-weight roles even above the RAM gate. No environment flag, marker, CLI alternative or receipt enables it. The independent stock observer supports only the synthetic 256-hidden profile; a bounded real 5120-hidden observer/canonical KV export and positive live original-guard qualification remain absent. The archive must preserve this fact rather than manufacture a runnable real-layer command.

## Receiving-host execution sequence

1. Verify package SHA256 against the annotated tag/final receipt, source manifests and source commit/tag identities. Follow `qies105/README.md` to build source and replay model-free suites in the synthetic-only namespace.
2. Obtain only the exact model outside Git, with enough disk and a resumable temporary name. Model revision `df550a7178a87bbfb6ee2bdbd37603c553d536ee`, file `Ripple-65B-Preview-AD-Q4_K_M.gguf`, 40,426,079,712 bytes, SHA256 `9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc`. Verify full hash and all required independent tensor bytes/descriptors; regenerate local stat/path identities instead of reusing laptop inode receipts.
3. Run the read-only resource command from the README against the actual persistent SSD. Verify user systemd/cgroup-v2 access, every relevant ancestor limit, inherited AS/RSS limits, CPU-only backend, original guard hash and natural entry readiness. Do not lower thresholds or edit system settings to pass.
4. **Stop until a separately reviewed bounded real stock observer, canonical KV export and real admission protocol exist.** Qualify harmless normal/failure/descendant/interrupt/monitor-error children under unchanged `run_guard` defaults; bind evidence to exact worker/adapter/guard hashes. A passing resource snapshot is not a live-guard receipt.
5. Only after all qualification and guards pass, run the independent real reference and candidate sequentially under the original guard. Use the exact layer-0 contract in `qies103/inputs/contract.json`: identical external F32 input/x0, position0 ->1, sequence0, empty F16 KV, 256-slot sliding mask/window1024, plain CPU one-thread nonflash profile and the 15 verified tensors. No altered decode scheduling or full-model run.
6. Capture and compare every specified intermediate/final tensor and written KV range (21 schemas), exact bytes/shapes/dtypes, zero tolerance; reject missing/nonfinite results, state-position errors, identity mismatches or guard failures. Hash and retain independently generated reference artifacts separately. Current expected output hashes remain null.

The README specifies the independent upstream path, build/capture procedure requirements and comparison/rollback semantics. A working real observer and positive live guard cannot be supplied by relabeling synthetic tests; they remain explicit implementation/qualification prerequisites.

## Rollback and decision

Do not merge this archive into FlashNext or continuous-generation production. Keep the original worktrees and milestone refs; return to a separate verified 8fac93 checkout if abandoning the prototype. Do not reset/clean experiments or restore retired executable permissions. Preserve all logs, model files and historical results.

**Preservation/portable synthetic handoff: GO. Real-weight numerical qualification and production integration: NO-GO pending the prerequisites above.**
