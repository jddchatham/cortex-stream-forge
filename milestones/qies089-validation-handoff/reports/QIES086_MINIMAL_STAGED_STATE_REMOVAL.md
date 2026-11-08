# QIES-086 - Minimal staged-state removal

**Decision: NO-GO for acceptance or production integration. The minimal experimental patch is prepared; required numerical, continuity, peak-memory and performance validation remains blocked.** No model inference was launched. A build or ownership proof is not a substitute for the requested golden regressions and three comparable trials per build.

## Baseline and isolation

Published baseline commit: `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`.

| Checkout | Branch / commit | Initial tracked changes |
|---|---|---|
| `cortex-stream-forge-qies074` | `feature/flashnext-first-text-generation`, published commit | None; existing untracked `build-quortex/` preserved |
| `qies082-memory-worktree` | Detached at published commit | None; existing untracked qies082/qies083/qies084 diagnostics preserved |
| New `qies086-worktree` | `experiment/qies086-minimal-staged-state-removal`, published commit | Clean when created; experimental patch remains uncommitted |

No new commit was created, no branch was merged, and no push occurred. "Exact commit" here identifies the base commit; candidate identity is the base plus the patch digest. Initial status is saved in `qies075-data/logs/qies086/initial-git.json`.

The verified baseline executable remains `cortex-stream-forge-qies074/build-quortex/bin/quortex`, SHA256 `329ebd647b04a2d6464461ddd94fc07a2691a948b2918d320ac1eb3703f9c101`. It and the protected libraries, scripts, indexes and reference selection arrays were checked against the QIES-082 manifest before editing. No baseline path was replaced or rebuilt.

## Ownership and exact change

QIES-085's ownership audit was rechecked against the chat source and all txc helper accesses. Before the patch there were exactly two `staged_state` occurrences: its vector declaration and copy assignment. No error, rollback, checkpoint, diagnostic or helper path reads that member or exports its buffer. The empty vector's copy assignment allocates n bytes and copies `result.state_before`; its implicit destructor releases that storage on transaction return or exception unwinding. The compiled baseline allocation/copy evidence is preserved in QIES-085.

`result.state_before` remains the pre-decode rollback owner. Both rollback restoration sites continue reading it. Context state remains mutable authoritative hybrid state, and the state-store files/mappings remain the persistent per-layer handoff. The later save helper still independently serializes current context state. This resolves the staged-copy ownership question, while leaving runtime correctness subject to regression testing.

Only `runtime-flashnext-layer-exp/examples/flashnext-chat.cpp` changed:

```diff
@@ struct flashnext_transaction_context
-    std::vector<uint8_t> staged_state;
@@ run_layer_transaction
-    txc.staged_state = result.state_before;
```

The saved exact unified patch is `qies075-data/logs/qies086/minimal-removal.patch`.

- Patch SHA256: `50c0475c7cfd731b4655e8de44651c997b4f69a15fe023902325c7ea0a3737cd`
- Baseline source SHA256: `3e78f699b92a21f199b01274adaf50f5e88b69729359eab002424e988339e8d0`
- Candidate source SHA256: `2b8438e925919770baee49bd0e1fb44e639079c7e818c6d6b01fbb0961801d03`

An exact text comparison confirmed no other source changes. `git diff --check` passed. No aliases, deferred captures, state-buffer reuse, allocator policy, state_write/state_read, persistent context behavior, tensor arithmetic, numerical operation order or scheduling policy was changed. Removing an unused allocation inherently changes allocation timing/failure opportunities and the internal struct layout; those effects are not claimed bit-exact-safe merely from static review.

The repository-required code-review skill was absent locally and was read from its [upstream source](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/skills/code-review/SKILL.md). Scope, security/lifetime and general checks were applied. The patch introduces no references or new input parsing. Private review found missing runtime validation as the acceptance blocker, not an unresolved reader of the deleted member. No upstream contribution or review message was sent.

## Build and harmless validation

**Build succeeded** (exit 0). Experimental executable: `qies086-worktree/build-quortex/bin/quortex`, SHA256 `4446bebc201af2720f8b951195fbb6d2104c0fdce59f3942b2e7003565e35b2d`. Both verified baseline and candidate returned zero for `--help`; source inspection confirms this branch exits before any model initialization. That is a startup/linkage check, not inference validation. The protected baseline hashes and all 48 state sizes/mtimes also passed after the build. Build settings matched the recorded baseline CPU-only options; compiler was GCC 16.2.1 20260810, CMake 4.4.4. Existing warnings in unchanged runtime sources are preserved in the build log. Fresh library hashes are recorded in `build-results.json`; only source identity/settings, not rebuilt-library binary identity, are asserted.

Build script at the experimental worktree root: `qies086-worktree/pandacode-run.sh`. It only configures and builds; it does not launch inference. It uses the documented Release CPU-only CMake configuration, GGML_CPU_REPACK=ON, examples ON, tools/server/app/tests OFF, and builds llama-flashnext-chat at -j2. Build output is `qies075-data/logs/qies086/build.log`. No dependencies or GPU packages were installed.

A second **controlled comparison pair** also built successfully, using the original recorded CLI compile flags and each checkout's matching header roots, with both linked to the same hash-verified baseline shared libraries. Paths: `qies086-worktree/build-controlled/bin/quortex-baseline` and `qies086-worktree/build-controlled/bin/quortex-candidate`. SHA256 baseline: `329ebd647b04a2d6464461ddd94fc07a2691a948b2918d320ac1eb3703f9c101`; candidate: `952eb2e92a657fba693b27dc9bd9fd55a86c0003848a0cd2453adc25d92f70e2`. Both passed isolated-environment `--help`, and ldd resolved all four engine libraries to the verified baseline directory. The controlled baseline executable is byte-for-byte identical to the previously verified executable (matching SHA256). This reproduces the baseline build, not a new baseline inference measurement. These are the binaries named in the comparison plan, avoiding library rebuild differences as a trial variable. The first control compile mixed checkout header roots and failed on duplicate definitions; the corrected retry succeeded without source changes. Both logs are preserved. This is compilation/linkage evidence only; neither binary evaluated model layers.

The source-level rollback calls, capture function, save function and state store remain byte-for-byte unchanged after accounting for the two deletions. This is a static preservation check, not error-injection, checkpoint-resume or sanitizer evidence.

## Safety and regression blockers

The unchanged `qies077-guard.py --check` returned 1, `resources_ready=false`: recorded MemAvailable **4,020,715,520 bytes (3.745 GiB)** versus required **6,442,450,944 bytes (6 GiB)**. SwapFree was 16,386,322,432 bytes and disk free 114,474,164,224 bytes; memory admission is the failed condition. This was a read-only admission refusal before any model launch, not a runtime guard breach. The earlier check also failed. The post-build check likewise reported resources_ready=false with 4.022 GiB available. Compilation has its own transient memory demand, so a favorable later snapshot must not be assumed.

The existing golden comparators were invoked unchanged:

```text
python3 -I -B qies086-worktree/milestone-flashnext-48layer/compare_chain48.py qies086 48
python3 -I -B qies086-worktree/milestone-flashnext-multitoken/compare_multitoken.py qies086 48
```

Both exited 1 at import with `ModuleNotFoundError: No module named 'numpy'`; no comparisons executed. Independently, their expected root `/home/user/quortex-diag-20261103` does not exist on this laptop. The source scripts are published, but their stock oracle tensors and original chain fixtures were not transferred. Existing local QIES-081 selection arrays do not supply all 48 layer outputs, all prompt-prefix stock logits or a checkpoint-resume oracle. No dependency was installed merely to reach another known missing-data blocker.

No bit-exact mismatch, state corruption or runtime safety breach was observed because candidate inference was not launched. "Not observed" is not a passing result. Comparisons were not weakened, truncated, replaced with token-only agreement or populated using the candidate's own output as its oracle.

| Required validation | Result / evidence |
|---|---|
| All 48 layer outputs against established stock reference | BLOCKED: missing NumPy, reference fixtures and fresh candidate outputs |
| Consecutive complete logits, including prompt prefixes and answer/EOG | NOT RUN; baseline selection files remain protected |
| Recurrent / attention KV / indexer / PLE / HC multi-token continuity | NOT RUN; serialization and handoff code unchanged statically |
| Checkpoint write/read/resume identical output | NOT RUN; prior synthetic fixture tests are not an engine resume test |
| Error paths and rollback/retry | Static owner preservation only; dynamic failure injection NOT RUN |
| UAF / double-free / uninitialized access | No new pointers/aliases in patch; sanitizer/model workload NOT RUN |
| Three comparable baseline and three candidate inference trials | 0 / 0; unchanged entry gate refuses current desktop |
| Performance / token throughput | NOT MEASURED; absence of a meaningful regression is not established |

The chat CLI also has no cross-process checkpoint/resume switch: it creates a new state directory and restores per-layer state within the running generation. A checkpoint/resume acceptance test needs an audited harness using the existing serializer/restorer, not a claim that the CLI already resumes old directories. That harness was not added in this minimal patch.

The old diagnostic tools exercise a related owned-layer implementation, while the patch changes only chat. Passing an unchanged owned-layer test would not prove execution of the modified chat path; the eventual harness must actually cover the changed translation unit.

## Baseline versus candidate measurements

| Metric | Baseline, new comparable trials | Candidate trials | Measured improvement |
|---|---|---|---|
| Anonymous RSS / PSS | Not run | Not run | Unknown |
| Peak process/tree RSS | Not run | Not run | Unknown |
| Cgroup current / peak, anon / file / swap | Not run | Not run | Unknown |
| Host MemAvailable / PSI | Admission snapshot only | No model snapshot | Unknown |
| Save-time serialization peak | Not run | Not run | Unknown |
| Multi-token generation peak | Not run | Not run | Unknown |
| Wall-clock / token throughput | Not run | Not run | Unknown |

QIES-085's standalone vector-copy trials measured 113.367 MiB of additional anonymous RSS/PSS per staged copy at one representative serialized size. Those were not three baseline/candidate QUORTEX runs and are not carried into this table as engine savings. The rollback snapshot remains alive during later save serialization, so whole-run peak reduction could be zero even if transaction-phase memory falls. No 113 MiB peak improvement or performance benefit is claimed.

## Reproducible comparison specification

`qies075-data/logs/qies086/comparison-plan.json` records the fixed workload and alternating six-trial order: baseline-1, candidate-1, candidate-2, baseline-2, baseline-3, candidate-3. This is a plan, not an executable benchmark runner or a completed baseline measurement.

The controlled comparison pair must use the installed, previously verified three IQ4_XS shards and matching local tensor index, CPU only, loader cache enabled, 1,048,576-byte expert arena, layer-range mode and OMP_NUM_THREADS=1. Prompt: `What is 2 + 2? Answer with only the number.` Native non-thinking formatting and maximum 16 generated tokens remain the existing binary behavior. Use the identical prompt/history and fresh persistent SSD state/output/log directories per trial; never reuse baseline artifacts or mutate model weights. Record CPU/compiler/library identity, cache warmth, swap/pressure and concurrency for every trial; do not globally reset caches.

Use the unchanged guard with its default Policy: entry 6 GiB available / 4 GiB swap free / 30 GiB disk; runtime 1 GiB available / 2 GiB swap free / 20 GiB disk, 7 GiB tree-RSS ceiling, 4-hour deadline, existing PSI/OOM and process-tree cleanup behavior. The guard CLI pins the baseline executable; **do not overwrite it to test the candidate**. A comparison adapter must call the existing run_guard with the candidate command and default policy, preserving its checks and failure semantics. That adapter has not been implemented or validated in this blocked stage.

For actual measurements, capture anonymous RSS/PSS and kernel peak RSS in addition to the existing guard's cgroup/host/tree-RSS telemetry. Attribute transaction and save boundaries separately with a reviewed measurement-only mechanism; the two-line candidate patch currently adds no phase instrumentation. Measure complete model generation peaks and wall-clock/throughput rather than CLI startup. Report paired deltas and min/median/max across three trials per build; desktop/background variability, cache ownership, sampling interval and instrumentation overhead are uncertainty sources.

Before trials, restore or regenerate independently trusted golden fixtures and provide NumPy in an isolated environment. Existing comparator output should be supplemented by full file-length and byte comparisons where needed for the stated bit-exact standard; a script that compares only the minimum common logits length cannot establish full-vector identity. Do not alter tolerances or use a partial-vector match as acceptance.

Six retained state directories alone require about 31.9 GiB logical space, plus logs, captures, binaries and filesystem overhead. Recalculate actual free space and reserve before the series. Given the roughly 106.6 GiB pre-build free-space snapshot, that basic state budget fits, but checkpoint copies and oracle captures need explicit additional budgeting. Stop on any mismatch, corruption or guard failure; preserve failed-run evidence and do not merge automatically.

## Decision and remaining work

**NO-GO for acceptance; validation incomplete.** The reviewed two-line change is isolated and reversible, but the mission's smaller-memory/same-results objective has not been demonstrated. The safe next execution point is a user-controlled headless session that satisfies the unchanged gate, with golden fixtures and comparison dependencies restored, followed by changed-path rollback/checkpoint tests and the six-trial guarded series. No desktop shutdown or system reconfiguration was performed automatically.

The published baseline and original guard are preserved. Keep this patch experimental until all mandatory correctness, continuity, peak-memory and performance checks pass. See the logs directory for the exact patch, ownership audit, source digests, initial Git inventory, admission result, comparator errors, build output and final protected-artifact verification.
