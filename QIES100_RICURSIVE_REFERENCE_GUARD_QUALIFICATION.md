# QIES100 — Ricursive Reference and Guard Qualification

## Decision

**CONDITIONAL GO for QIES101 qualification work; NO-GO for real-model execution now.** Independent stock-source provenance, real-GGUF structural recognition and matched-path synthetic numerical comparisons pass. Live passing-gate guard/adapter qualification and bounded real-layer reference extraction remain incomplete. No real-model tensor evaluation, token generation or inference occurred. Only small synthetic models were evaluated.

Work remained on `experimental/ripple-65b-q4` in `/home/user/PandaCode Bots/meatball-84ccea00/qies098-worktree`, HEAD `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. No production source, FlashNext, QIES086, guard thresholds, system settings, other branches or Git history changed. No commits, merges or pushes.

## Independent reference provenance

Upstream llama.cpp **b12167 = `240d68fcfe0c2a1f298c8e18ca181368b1914259`**, independently confirmed using remote tag lookup (`qies100/upstream-tag.txt`). [Pinned upstream Ricursive source](https://github.com/ggml-org/llama.cpp/blob/240d68fcfe0c2a1f298c8e18ca181368b1914259/src/models/ricursive.cpp).

The upstream archive was downloaded into this isolated worktree. Archive SHA-256: `b7c50c28d1c25942fdbca447326a692657238cd87beeea45797264b02bc64b4f`. `qies100/stock-source-sha256.json` records 4,084 original source-file hashes, verified again after building. Its Ricursive implementation matches the previously preserved QIES098 source exactly. No upstream numerical source was modified.

Compiler: GCC C++ 16.2.1 (20260810); CMake 4.4.4. Release build, two compiler jobs, CPU backend only, native CPU targeting, OpenMP ON, BLAS OFF, CPU repacking support ON, CUDA/Vulkan/OpenCL OFF; dynamic backend loading OFF. Complete options are in `stock-build/CMakeCache.txt`, and actual compiler flags in its generated build files. The host is the existing i5-7200U AVX2/FMA laptop. `build-provenance.json` records compiler, CPU and source identities.

Build commands, from the Ripple worktree (unset PandaCode's incompatible library environment):

```sh
env -u LD_LIBRARY_PATH cmake   -S qies100/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259   -B qies100/stock-build   -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF   -DGGML_CPU_REPACK=ON -DLLAMA_BUILD_COMMON=ON   -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON   -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF   -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release   -DLLAMA_BUILD_NUMBER=12167 -DLLAMA_BUILD_COMMIT=240d68fc
env -u LD_LIBRARY_PATH cmake --build qies100/stock-build --target llama-simple -j2
```

The initial tools/common configuration did not expose a CLI target; explicitly enabling common utilities and building the verified `llama-simple` target succeeded. The source archive has no `.git`: upstream GGML's automatic informational commit stamp discovered the enclosing QUORTEX Git repository and printed `8fac93b`. **That stamp is not source provenance.** The verified archive, remote tag and full source hashes establish the source identity. Do not use the misleading GGML informational stamp as a qualification check.

`stock_probe.cpp` links only the independently built stock libllama/GGML libraries. It calls stock `llama_model_load_from_file`, `llama_init_from_model` and `llama_decode` with a capture callback; it contains no replacement attention or FFN implementation. `candidate_probe.cpp` separately links QIES099 and the pre-existing GGML 0.25.3 CPU libraries. Stock uses GGML 0.29.0. Separate processes and RPATHs prevent shared-library symbol interposition. `probe-build-commands.json` and `*-ldd.log` document this separation. Shared fixture bytes are inputs, not experimental golden outputs.

Final stock synthetic probe SHA-256: `3bce6c61af8a1ca42723745fa8be56d637ebb7dcdd8b57ddc84b6fbd2f1c8c89`.
Stock `llama-simple` SHA-256: `1dea9a07cc9d8b54749206c43fec8c36e8c5eb900c5903fc24f9df18a482ec09`.
All reference/candidate executable and library identities: `executable-library-sha256.json`.

## Downloaded GGUF and structural qualification

Model remains outside Git: `../qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf`, 40,426,079,712 bytes. SHA-256 identity remains `9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc`. QIES099's complete-file hash is reused after verifying device, inode, size, mtime and ctime against the recorded hashed-file identity; this milestone did not reread all 40 GB. See `model-identity-check.json`.

Two independent stock checks passed:

1. `stock-metadata` uses stock GGUF parsing without tensor payload allocation: architecture `ricursive`, 1,203 tensors, with stock conversion traits for every encountered type. Counts: 561 F32, 160 Q8_0, 310 Q4_K, 90 Q5_K, 81 Q6_K, one IQ4_XS (`stock-model-metadata.log`).
2. `stock-probe --structure MODEL` uses `no_alloc=true`, `load_mode=NONE`, CPU and ordinary buffers. Source audit verifies the dummy-buffer path returns before loading tensor data. Stock validates architecture hyperparameters and tensor creation/shapes and reports **Ricursive 65B, 80 layers, hidden 5120**, zero-byte model buffer, no context or decode (`stock-real-structure.log`).

An earlier vocabulary-only check deliberately failed the architecture-dimension assertion: stock's vocab-only path skips architecture initialization and reports zero layer/embedding dimensions. Its log is preserved; it is not counted as successful structural qualification. The later no-allocation structural check resolves that limitation.

The 15 per-layer tensor names and shapes agree with QIES099: attention RMSNorm; Q/K/V; head Q/K RMSNorm; sigmoid value-residual lambda; attention gate/output; post-attention norm; FFN norm/gate/up/down/post-norm. Actual dimensions are hidden 5120, FFN 40960, 80 query heads, eight KV heads and head size 128. Upstream order agrees with QIES099: norm → Q/K and mixed-value projection → Q/K norm → SWA-only RoPE → masked attention → sigmoid gating/output projection → post-norm/residual → pre-norm/clamped SwiGLU/down projection → post-norm/residual. Global layers are NoPE. Structural recognition does not establish real-weight numerical equivalence.

## Synthetic differential comparison

`make_fixture.py` creates bounded five-layer synthetic GGUFs: hidden 256, FFN 512, four query heads, two KV heads, head size 64, context 256, sliding window two, global layer four. Four consecutive positions cross the short SWA boundary. External F32 embeddings and original `x0` are deterministic. Stock and candidate use one thread, non-flash attention, F16 KV and identical weights/inputs. No downloaded model weights enter these tests.

The generator uses the existing isolated QIES087 NumPy 2.3.4 environment. Direct imports of upstream GGUF writer/constants avoid its optional YAML metadata dependency; no global packages were installed. Quantized fixture bytes are encoded once, shared with both processes, and include all five model quantization types. IQ4_XS embedding and Q6_K output-head formats receive stock loading/dequantization coverage; the layer arithmetic exercises Q4_K/Q5_K/Q6_K/Q8_0 matrices. External embeddings mean the fixture does not test IQ4_XS embedding lookup arithmetic.

| Case | Intermediate comparisons | Byte-exact | Verdict |
|---|---:|---:|---|
| F32 | 360 | 360 | PASS |
| Quantized, stock extra/repacking buffers enabled | 360 | 274 | Diagnostic mismatch; not accepted |
| Quantized, stock ordinary CPU buffers matched to candidate | 360 | 360 | PASS |
| Direct independent dequantization | five types | five types | PASS |

Eighteen captures × five layers × four positions produce 360 checks. Eight checks per case explicitly resolve stock's global NoPE Q/K pointer aliases: its later callback renames the unchanged normed tensor. These are documented identity aliases, not eight additional independent computations. No other missing captures occur. Captures cover Q/K normalization and RoPE, mixed V, masks/attention probabilities, attention/gating, FFN activations and residual outputs.

The initial quantized mismatch was isolated by changing **only stock's `use_extra_bufts` setting**, leaving production and experimental layer code unchanged. Maximum initial absolute discrepancy was approximately `2.2649765e-6`; 19/20 layer outputs matched, with the final token/global-layer output differing by up to `1.4305115e-6`. Setting stock `use_extra_bufts=false` selects ordinary CPU buffers like QIES099 and restores exact agreement for all 360 checks, including all 20 layer outputs. This demonstrates dispatch/buffer-path sensitivity in these fixtures; it does not prove all possible repacking or kernel-version differences are understood.

**No numerical tolerance was accepted or relaxed.** Accepted comparisons use exact bytes. Default/repacked stock output must not silently replace the matched-path reference. `initial-results/`, `stock-probe-repack-original` and `comparison-quantized.json` preserve the failing evidence; `comparison-quantized-plain.json` records the accepted matched-path result.

`error-tests.json`: missing stock fixture and missing candidate weight files return expected nonzero statuses; a 17 MiB sparse synthetic-file test is rejected before model loading, validating the 16 MiB harness budget. Existing QIES099 deterministic tests already cover invalid schemas, finite values, state ownership and native resource cleanup; those source files were preserved. QIES100 does not claim live real-model cleanup qualification from these synthetic cases.

Reproduce the model-free comparisons using `./pandacode-run.sh`; this compiles only small probes against the already built libraries and runs the synthetic fixtures plus saved guard-rejection checks. `final-run-script.log` records success. Regeneration commands are `.../qies087-validation/venv/bin/python -I qies100/make_fixture.py` and the same with `--quantized`; preserve existing evidence before regenerating. All fixture/capture files are small, separate from downloaded models and original state artifacts. Stock's synthetic execution mode rejects files above 16 MiB; its real-file structure mode cannot decode.

## Guard enforcement audit and adapter status

The unchanged original guard is `../qies077-guard.py`, SHA-256 `ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2`. The QIES099 runner/adapter was inspected and preserved; no qualification receipt was fabricated.

Original policy and enforcement points:

- Admission in `run_guard`: MemAvailable ≥6 GiB, SwapFree ≥4 GiB, free target-filesystem space ≥30 GiB, full-memory PSI avg10 <1.
- Ancestor accounting: reject inadequate finite memory.max/memory.high headroom below 7 GiB or swap headroom below 2 GiB. Monitor target and ancestor OOM events.
- Runtime: stop below 1 GiB available RAM, 2 GiB swap or 20 GiB disk; tree RSS ≥7 GiB; three sustained PSI samples ≥10; four-hour ceiling; monitoring failure or operator interruption.
- Child service: CPU harness command, separate cgroup, `KillMode=control-group`, two-second TERM grace then KILL, unit-status checks and no surviving descendants before success.
- The original guard checks/monitors cgroups but does **not** install a hard per-job MemoryMax/MemoryHigh/MemorySwapMax or RLIMIT_AS. QIES075 explicitly documented that design. This host's inspected ancestor memory ceilings and soft/hard address-space limits are unlimited. Existing limits remain unchanged; do not describe the guard as a hard address-space cap or invent a new threshold.

`qies100/guard_adapter.py` adds a fail-closed qualification/readiness interlock and actual reference-file/hash/configuration checks without changing the original guard. Its real launch entry intentionally refuses until remaining integration qualification is completed. It is **not a qualified runnable real-model adapter**. The native QIES099 path retains its separate 6 GiB check, CPU backend selection, active-layer tensor budget and graph/storage ceilings; inherited resource limits are not reset. These internal bounds do not substitute for external monitoring.

`guard-tests.json` and `guard-tests.log`: 14 model-free contract checks pass (exact admission boundaries, fixed policy values, unqualified references and blocked launch paths). One actual call of the original `run_guard(['/usr/bin/true'], ...)` with default policy returned failure, `ok=false`, reason `resource preflight failed`, cleanup true. It launched no target process. This real refusal was performed once; later checks read its preserved evidence rather than repeat failed gate attempts. No policy overrides, simulated low thresholds or real-model launches were used.

A zero child/wrapper exit alone is insufficient acceptance. Successful live launch, complete process-tree monitoring, cgroup cleanup and post-run comparison evidence remain to be qualified under naturally passing conditions. Failure-only preflight evidence cannot certify those paths.

## Current runtime readiness

Final snapshot: 3.795 GiB MemAvailable; 14.243 GiB free swap; 62.285 GiB SSD free; PSI 0. Cgroup v2 is present. Exact byte counts and inherited limits are in `final-readiness.json`. The unchanged 6 GiB gate fails. Original baseline executable remains SHA-256 `329ebd647b04a2d6464461ddd94fc07a2691a948b2918d320ac1eb3703f9c101`.

## Smallest real-layer plan and QIES101 prerequisites

1. Keep the independently built stock graph and matching ordinary CPU-buffer configuration. Preserve source/executable/library hashes and use one thread, non-flash attention, F16 KV, context 256, and the same model identity. Synthetic qualification is a prerequisite, not real-model qualification.
2. Implement/review a bounded stock single-layer extraction harness, starting with **layer 0, position 0 only** and fixed identical F32 input/scaled `x0`. Reuse original stock graph construction and capture exact intermediates. The existing synthetic-only probe is deliberately not a real-weight runner. Do not assume full stock model loading or graph warm-up has bounded residency merely because mmap is available.
3. Verify real tensor selection, byte ranges and bounded active-layer allocations before any tensor arithmetic. Reconfirm the model's SHA-256 identity; compute a new complete hash if identity changed or the provenance chain cannot be trusted. Avoid duplicate 40 GB files.
4. Qualify the original-guard integration with harmless children after naturally satisfying its unchanged admission and ancestor checks. Test successful exit, nonzero exit, descendants, interruption, monitor failure and cleanup, with no policy override. Record actual inherited limits and verify the monitored filesystem. Any requested new finite hard limit needs a separately reviewed design; this milestone installs none.
5. Require a fresh ≥6 GiB MemAvailable check immediately before each real job, sufficient swap/disk and unchanged guard enforcement. Use a suitable larger-memory host or independently verified headless session; do not close this desktop automatically.
6. Produce stock reference artifacts under those guards, including inputs, model/library/binary identities, all intermediate shapes/dtypes, final output and canonical written KV. Then run candidate against identical artifacts in a separate guarded job. Compare exact bytes; stop on any mismatch or guard failure. Preserve partial/failure logs. Do not merely label a manifest qualified or trust an output-free zero exit.
7. After layer 0 passes, add global layer 4 and nonempty prior-state consecutive positions. Establish direct canonical KV/state continuity comparisons before accepting generation integration. Long-window behavior, full model/logit equivalence, throughput and real peak memory remain unproved.

The practical handoff is **GO for completing these bounded qualification tools; NO-GO for QIES101 real-model execution until all prerequisites pass**. `qualification-status.json` keeps each partial qualification explicit. All current production behavior and resource safeguards remain preserved.
