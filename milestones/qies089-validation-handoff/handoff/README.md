# QIES-089 portable validation handoff

**Status: SOURCE/BUILD QUALIFICATION ONLY. MODEL INFERENCE AND OPTIMIZATION ACCEPTANCE BLOCKED.** Nothing in this package automatically launches inference or approves its oracle. No model weights are included. The two files in anchors/ are historical baseline selection arrays, not a complete independent golden set.

Start by verifying `sha256sum -c SHA256SUMS` from this directory. The manifest records hashes of the shipped files and evidence; transfer/transport integrity does not establish scientific authority. Preserve this package and all original results read-only in practice; create fresh working/output directories, without changing existing artifact permissions or overwriting them.

## 1. Recreate exact source; build only

Dependencies: Linux CPU host, Git, CMake, compatible C/C++ compiler, Python 3.11+ for tooling. Original host used GCC 16.2.1 20260810, CMake 4.4.4. Different hosts/toolchains may produce different executable bytes and must record new identities. No global dependency changes are required by this handoff.

Set PACKAGE to this extracted package and RUNROOT to a new persistent SSD directory. These commands create a new baseline checkout and an isolated **uncommitted** candidate; they do not run a model:

```sh
git clone --branch feature/flashnext-first-text-generation "$PACKAGE/source/repository.bundle" "$RUNROOT/cortex-stream-forge-qies074"
git -C "$RUNROOT/cortex-stream-forge-qies074" rev-parse HEAD
# Must be 8fac93bb2bfc4b2d7069c4044f2f71cd25f80388.
git -C "$RUNROOT/cortex-stream-forge-qies074" worktree add -b experiment/qies086-validation "$RUNROOT/candidate" 8fac93bb2bfc4b2d7069c4044f2f71cd25f80388
git -C "$RUNROOT/candidate" apply --check "$PACKAGE/source/qies086.patch"
git -C "$RUNROOT/candidate" apply "$PACKAGE/source/qies086.patch"
cmake -S "$RUNROOT/cortex-stream-forge-qies074/runtime-flashnext-layer-exp" -B "$RUNROOT/cortex-stream-forge-qies074/build-quortex" -DGGML_CUDA=OFF -DGGML_CPU_REPACK=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$RUNROOT/cortex-stream-forge-qies074/build-quortex" --target llama-flashnext-chat -j2
python3 -I -B "$PACKAGE/tools/build_pair.py" --baseline "$RUNROOT/cortex-stream-forge-qies074" --candidate "$RUNROOT/candidate" --output "$RUNROOT/controlled-build"
```

Use lower build parallelism if compilation resources require it. The helper recompiles only the baseline chat, two-line candidate chat and stock runner against **one common baseline library set**, preserving the recorded compile flags. It writes executable/library hashes and commands to controlled-build/identity.json and never executes them. All source files/headers are from the specified revision; only the candidate member/copy deletion differs. No binaries are shipped because existing RPATH, CPU ISA and host ABI assumptions are not portable. Rebuild after moving RUNROOT. The baseline repository contains its own contributor guidance; read it before any further edits.

Local-hash expectations, not mandatory cross-toolchain byte identities:

- Baseline CLI: 329ebd647b04a2d6464461ddd94fc07a2691a948b2918d320ac1eb3703f9c101
- Controlled candidate: 952eb2e92a657fba693b27dc9bd9fd55a86c0003848a0cd2453adc25d92f70e2
- Published-stock-source build: 6a29bf69f5976e9f4bd0e6df8e351336e4b52ca93d492d7e66bd82ee8cb22963
- Historical stock executable: eda8ceefc94f296a66d217b79d0aba64317b5f24295b822e086e7d3736169fb2 (not present)

The last two identities differ. A build is NOT an oracle qualification. Record every linked library, compiler, CPU/ISA, build flag and actual context/backend thread setting. Original historical stock dependent-library/configuration identity and source-overlay digest remain incomplete; see evidence/QIES072_NONTHINKING_FIRST_ANSWER.md. The older overlay is identified only by a 4faa9bd2... prefix and was not recovered.

## 2. Place model and guards without changing them

Use the exact three shards and embedded GGUF tokenizer specified in protocol/inputs.json. Verify complete shard SHA256 before a new host's first run; filenames alone do not establish identity. Do not download/copy weights if a verified accessible local copy exists. The tokenizer is pinned by complete GGUF identities, not a substitute tokenizer revision. Inspect architecture as QWEN4EXP: 48 layers, 2560 hidden, four HC streams, 512 experts/top-10, full attention at 3,7,...47, recurrent elsewhere, PLE as described in the historical state inventory. Static source support is not a successful load test.

Deploy unmodified guards/qies077-guard.py and qies079-measure-memory.py to RUNROOT (not under a nested guards/ directory). They derive ROOT from their own location; ROOT/qies075-data/{models,state,outputs,logs} must be persistent SSD paths. Use new/exclusive file creation; do not overwrite a prior deployment. Baseline guard CLI expects RUNROOT/cortex-stream-forge-qies074. Clone's published tensor index stays immutable; create a separate local index whose only changed field is model_dir, verify every other field equal, and record its new hash. Do not use the source laptop's absolute model_dir or rebuild RPATH as a portable path.

The guard policy is unchanged: before inference MemAvailable >=6,442,450,944 bytes, SwapFree >=4 GiB, SSD free >=30 GiB, memory full PSI avg10 <1%. Ancestor memory.max/high must have >=7 GiB headroom and swap limit >=2 GiB headroom where finite. Runtime floors: MemAvailable 1 GiB, swap free 2 GiB, disk 20 GiB; max tree RSS 7 GiB; four-hour deadline; existing OOM/PSI/cleanup/failure rules. Passing the screen is not proof a stock loader fits.

No current-desktop gate checks are part of package preparation. On the intended suitable host, only when ready to establish admission, use the **unchanged** read-only measurement and full guard checks. No policy overrides, threshold changes, global cache drops, service/swap/zram/cgroup/kernel changes are part of this package. Prefer the original oracle machine with its verified environment or a dedicated CPU host with more headroom; a fresh user-controlled headless session on the laptop is a conditional option, never an inferred pass.

**BLOCKED adapter:** the guard CLI pins the baseline executable. A separately reviewed adapter must call the existing run_guard with default Policy and the selected stock/baseline/candidate command, preserving all checks and exit/failure semantics. Do not overwrite baseline binaries to choose a candidate. Validate the adapter using harmless children and the unchanged admission policy in a passing environment. No inference launcher is provided here, and these build instructions are not a launch authorization.

## 3. Oracle qualification and new golden generation

Read protocol/inputs.json, protocol/formatted-prompt.txt and all provenance evidence first. Recover original stock tensors/binary/dependency bundle if possible, hash against a trusted original manifest and label RECOVERED_ORIGINAL. Otherwise label every generated file NEW_STOCK_REFERENCE with a new run ID. Never use candidate outputs as reference data.

For the new stock path, freeze its source/executable/library identities and model/configuration before launch. This runner runs normal full-graph CPU execution, clearing streamed cuts; it does not use the staged-state candidate coordinator. It is not automatically bounded by QUORTEX's 1 MiB expert arena. Audit normal loader/mmap/repack allocation and available headroom before a guarded run. Use a clean environment; inherited CORTEX flags can change graph/capture behavior. Do not assume OMP_NUM_THREADS=1 determines every context thread default.

Clean prompt-reference settings: STOCK_NO_DIAG=1; STOCK_PROMPT exactly `What is 2 + 2? Answer with only the number.`; STOCK_PROMPT_MAX_TOKENS=16 (runner default is only one); STOCK_PROMPT_LOGITS_DIR a new persistent directory; OMP_NUM_THREADS=1. No streamed CORTEX flags or diagnostic dumping for this pass. Exact n_ctx=256, n_batch=1, seq_id=0, 26 input tokens and native non-thinking rendering are in inputs.json. Verify those tokens before accepting output.

Expected historical answer ID19, then EOG248046. Require all 248,320-element F32 vectors, not just argmax: stock logits-select-25.bin must match anchors/logits-select-0.bin; stock logits-generated-1.bin must match anchors/logits-select-1.bin. Their hashes are in the manifest. Stop on disagreement. Matching these two anchors is necessary but insufficient to qualify full layer/state coverage.

Run a separate audited stock diagnostic pass for all 48 layer outputs at each relevant position and prove its logits are byte-identical to the clean pass. Capture must not modify arithmetic or use streamed output as its oracle. Require deterministic repeat runs. The older two-token protocol is inputs [248045,846], with historical argmax [846,198]; do not mix it with the 26-token prompt fixtures.

For the changed chat path, --capture-dir enables prefix logits and per-position captures. It also performs extra output projections at prompt prefixes, so correctness-capture runs and performance runs are distinct workloads and must not be mixed. With the accepted one-answer/EOG outcome there are positions 0..26. Capture layer handoffs for each position, all 26 prompt-prefix vectors plus answer/EOG selections. Audit the raw capture filename/graph-name mapping against stock hc_combine-L and record it in the comparison manifest. Historical owned-layer diagnostics alone do NOT exercise the changed chat source.

Hash completed files only after clean successful process exit and successful guard cleanup. Reject partial/nonfinite arrays, wrong shape/token sequence or guard failure even if some hashes match. Golden manifest must record model/index/tokenizer identity, exact formatted input bytes, explicit token IDs/positions/sequence IDs, context/thread/backend/build configuration, source/executable/dependent-library/guard hashes, raw capture mapping, full lengths/dtypes, resource result and every file SHA256. New references require an independently reviewed qualification receipt; no automated build marks them QUALIFIED.

## 4. Exact comparisons (no NumPy dependency for supplied checker)

The included comparator requires a qualified reference manifest and complete coverage. It verifies reference hashes and exact file lengths, then every byte. It rejects equal-prefix-only matches, signed-zero representation changes, missing/extra/duplicate IDs and reference/actual aliases of the same file. It does not certify that a caller's provenance assertions are true; independently review the qualification receipt, generation logs and executable/model identities first.

- `--case two-token`: 96 layer files (layer-00-p00 through layer-47-p01) plus logits-p00/logits-p01 =98 files.
- `--case prompt`: 48 layer outputs at every position 0..26 plus prefix-00..25, select-answer and select-eog =1,324 files. Each layer file is 40,960 bytes; each logits file 993,280 bytes.

```sh
python3 -I -B "$PACKAGE/tools/compare_exact.py" --self-test
python3 -I -B "$PACKAGE/tools/compare_exact.py" --case prompt --manifest "$MANIFEST" --reference "$ORACLE_ROOT" --actual "$CANDIDATE_CAPTURE_ROOT"
```

Apply the same checks to baseline and candidate separately. protocol/comparison-manifest-template.json is deliberately BLOCKED with empty pairs; it cannot pass. After real qualification, each pair entry has id, reference and actual relative paths, bytes and trusted reference sha256. Use `--case two-token` for that separate input protocol. Path relocation is recorded in a new mapping manifest, not edits to golden bytes. The original NumPy comparison scripts are retained in source history; their float-equality/minimum-length behaviors alone do not meet this stronger byte-complete contract. NumPy environment provenance is in evidence; the cp314 Linux wheel pin is only for that platform, not an arbitrary Python version.

## 5. Checkpoint/resume and recovery - explicitly BLOCKED pending harness

Neither current chat CLI nor the stock runner provides a cross-process resume switch. Do not invent --resume flags or treat a synthetic file hash test as model resume correctness. A separately audited harness must call existing state APIs and exercise the changed chat transaction implementation without changing inference math:

1. On identical explicit inputs, compare uninterrupted versus serialize/restore into an identically configured fresh context at a nonzero boundary. Validate returned byte counts, all hybrid state including recurrent/conv, attention KV/cell metadata, indexer and PLE predecessors. Store token history, sequence/position, pending-token identity and any layer-boundary HC handoff needed for the exact checkpoint boundary; a state blob alone need not be sufficient coordinator state.
2. Compare every subsequent layer and full logits byte-for-byte, across multiple further tokens, not just the next argmax. Re-capture serialized state and compare its defined layout/metadata. Do not omit allegedly unused state to reduce memory.
3. Exercise sparse-MoE and graph-B pre-commit failure, rollback, retry, descriptor/decode error behavior, serialization failure and interrupted checkpoint writes; require the baseline's documented behavior and no corrupt checkpoint accepted. Preserve result.state_before. Detect invalid access/free in a separately instrumented bounded test; account for sanitizer overhead without weakening the original guard.
4. For cross-process restart, require complete checkpoint commit/metadata validity before loading and fail closed on incomplete state. Audit writeback/durability separately; the existing store is not thereby certified crash-atomic.

Harness and outputs are missing. No acceptance status may be changed until they exist and pass.

## 6. Memory/performance trials and decision

Only after all correctness/continuity/recovery gates pass, run at least three full trials per build, with fresh persistent state/output/log directories. Order: baseline-1, candidate-1, candidate-2, baseline-2, baseline-3, candidate-3. Use the controlled common-library pair, identical model/index/prompt/max16/cache/arena/threads/CPU constraints and guard defaults. Do not use QIES069_STOP_AFTER_FIRST_PREFIX or reuse old state. Performance runs omit unnecessary full-prefix/layer captures for both builds; correctness capture runs remain separate.

Record process-tree anonymous RSS/PSS, kernel/tree peak RSS, cgroup memory.current/peak/stat/anon/file/dirty/swap/events, host MemAvailable/PSI/swap, page faults and device/process I/O. Separate transaction-phase from later save-serialization and full multi-token peaks with a reviewed measurement-only observer. Existing guard alone does not provide all PSS/phase attribution; observer remains BLOCKED pending validation. Record monotonic wall time, generated token count and decode-phase versus full-run throughput; one generated token after a long prompt is not a stable steady-state benchmark.

Report all trials and paired deltas, median/range, sample resolution and background/cache variability. Clean file RSS is not extra unusable RAM. A phase-specific 113 MiB vector-copy saving is not a measured full-run peak saving. Do not claim a performance non-regression or smaller model footprint until actual matched trials establish it. Budget at least 31.9 GiB logical for six state directories plus reference/checkpoint copies, captures/logs and >=30 GiB entry reserve (20 GiB runtime floor); recompute target-filesystem capacity before a series.

Any missing identity, unqualified oracle, incomplete coverage, nonfinite/mismatched byte, checkpoint/rollback error, invalid memory access or guard violation is a blocking failure: stop, retain logs/artifacts, do not merge.

## 7. Reversibility

Production baseline and libraries remain untouched. Run baseline by selecting its distinct executable through the validated adapter. To discard the optimization, stop using the candidate worktree; after saving its outputs and checking its exact diff, `git -C "$RUNROOT/candidate" apply -R --check "$PACKAGE/source/qies086.patch"` verifies reversible patch applicability. Applying that reverse patch is an explicit later local action, not automated here. Do not delete model files, artifacts or worktrees as part of rollback. No commit, merge, push or automatic desktop shutdown is part of this handoff.

See protocol/acceptance-gates.json: missing runtime gates remain BLOCKED. Source/build reproduction and synthetic tooling tests are the only newly passed stages.
