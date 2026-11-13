> Archived diagnostic report. Machine paths/process identifiers are redacted; raw machine logs and binaries are not distributed. For public evidence and relocatable reproduction, use [qies112a/README.md](qies112a/README.md). Historical commands/paths below describe the original local run.

# QIES112A — QUORTEX crash and stability diagnostic

Updated: 2026-11-13T10:50:26.385729-06:00

**Historical inference crash: INCONCLUSIVE / not attributed.**
**QIES111 synthetic stability: PASS for tested synthetic configuration.** Completed fresh trials: 3/3.

## What is and is not identified

No available record identifies the executable, command, PID, timestamp or signal of the alleged historical **inference-program** crash. It must not be assumed to be the QIES111 synthetic layer worker.

The broader core inventory does identify `/usr/bin/pandacode` [historical process], [historical thread] (`tokio-rt-worker`), on November 6 at 21:14:45 CST: signal 7 SIGBUS, `BUS_ADRERR`, command `pandacode`. WebKit network/web processes also have SIGBUS dumps shortly afterward. The Thorium browser has a separate November 7 SIGTRAP dump. These are desktop/browser programs, not FlashNext or Ripple inference executables, and predate QIES111. There is no established correlation to the later user-reported interruption. Their underlying cause is not diagnosed here. No unrelated core memory was extracted.

The known real FlashNext executable remains `cortex-stream-forge-qies074/build-quortex/bin/quortex`, SHA256 `329ebd647b04a2d6464461ddd94fc07a2691a948b2918d320ac1eb3703f9c101`, source commit `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. QIES081 records its QIES080 run returning 0 after answer `4`, EOG 248046 and1,326.023 seconds. That is evidence of successful completion, not a recorded inference crash. The current executable hash still matches. It was not launched for this diagnostic.

A separate, identified QIES111 prototype startup failure (`20261111-173515-232345df`) returned ordinary exit 1 with `activation input length`. The published QIES111 report attributes it to a missing input bind, fixed before the currently preserved artifact was accepted. Its stdout/stderr/result are copied read-only under `known-startup-failure/`. This does not identify a signal crash or justify changing the currently tested executable.

The unfinished QIES112 generation experiment is preserved and suspended. Its one accepted synthetic prediction, interrupted controller sessions and partial stacks are not automatically classified as numerical-program crashes. No QIES112 feature development is part of this mission.

Evidence preserved before execution: `qies112a-diagnostic/tested-provenance.json`, `historical-candidates.json`, `qies112-snapshot.json`, `cores-before.log`, `kernel-before.log` and `desktop-core-metadata.local`. The kernel query reports no matching accessible OOM/kill/segfault entries; absence of logs does not exclude an unrecorded historical event.

## Exact binaries under test

| Role | Existing executable | SHA256 |
|---|---|---|
| candidate | `<workspace>/qies111-worktree/qies111/build/candidate` | `a3513830e5a5bf820cd586530954d15d68e005aba8555ebb0e4c0ba634fb440b` |
| reference | `<workspace>/qies111-worktree/qies111/build/reference` | `aa2bc00dd5f19a64fb9452c902fdca191b3fed9b368bba98e1928b893f23d81f` |

These are the **already-preserved QIES111** binaries, not rebuilt replacements. Their three GGML library hashes, sources and synthetic input hashes were checked before testing. Host mode600 and sealed-namespace dispatch are retained. Native workers have no real GGUF mounted.

Source HEAD: `14b3f7902864b02e1a84e5e6168a3cf99087e670` plus preserved, uncommitted QIES106–111 experimental sources identified by the original source manifest. HEAD alone is not a claim that these workers are in the archive commit. Stock source: `240d68fcfe0c2a1f298c8e18ca181368b1914259`. Compiler: `c++ (GCC) 16.2.1 20260810`. Worker commands retain C++17 `-O2 -Wall -Wextra`; SDK libraries retain their original native CPU configuration. One numerical CPU thread, no GPU. Full build commands/library hashes are in `tested-provenance.json` and original `qies111/build/manifest.json`.

## Reproduction and safeguards

```sh
env -u LD_LIBRARY_PATH python3 -I -B qies112a-diagnostic/run.py --trials 3 --four-tokens
```

The diagnostic driver calls the unchanged QIES111 `campaign`, `transaction` and `launch` APIs. Each new trial runs all 80 layers for positions 0/1 in both candidate and independent reference. Every trial must produce4160 byte-exact captures matching the original QIES111 SHA256 map, preserve separate layer KV identities and complete320 numerical subprocesses successfully.

The actual commands are preserved per job in `result.json`, with stdout/stderr and process-tree/cgroup monitor logs. A representative native invocation is `/lib64/ld-linux-x86-64.so.2 /workers/candidate --synthetic 0 256 0 seed contract-v1 none` inside the unchanged bwrap namespace, marker, CPU environment, read-only synthetic weights and prior-state mounts. The reference uses `/workers/reference`. The host binary is copied into neither a new file nor a new build: its original path is bound read-only.

No real-weight entry is enabled. The existing real 6 GiB MemAvailable gate and unconditional unqualified-runtime lock remain unchanged. The separately established synthetic policies remain1.5/1.75 GiB entry,1 GiB running floor,2 GiB swap reserve,20 GiB disk reserve,256/640 MiB worker RSS ceilings,120-second per-worker timeout and ancestor OOM-event monitoring. No cgroup, address-space, swap, zram or OS policy was modified. Shared desktop cgroup peaks are not isolated worker accounting.

## Fresh trial results

| Trial | Exact captures | Native subprocesses | Exit | Signal | Streaming peak RSS/PSS MiB | Reference peak RSS/PSS MiB |
|---:|---:|---:|---:|---|---:|---:|
| 1 | 4160 | 320 | 0 | none | 39.24/32.26 | 541.69/532.66 |
| 2 | 4160 | 320 | 0 | none | 36.84/29.85 | 541.70/532.65 |
| 3 | 4160 | 320 | 0 | none | 24.52/16.39 | 541.66/532.78 |

RSS/PSS are sampled subprocess-tree peaks, including namespace/loader and potentially pre-exec inherited pages; RSS may double-count shared mappings. Native high-water RSS is also retained separately. Components and unrelated peaks are not summed into a total-system claim. Per-layer completion, KV state/header/checksum, capture hashes, input identities, zero tracked ownership and zero open adapter files are checked. Allocator-retained pages can remain after tracked cleanup; zero modeled ownership is not zero RSS.

## Stress, faults, sanitizers and interruption
- Original bounded faults: completed; see `qies112a-diagnostic/runs/20261113-095319-4c13aa/bounded-faults/summary.json`.
- Preserved sanitizer build: completed; see `qies112a-diagnostic/runs/20261113-095319-4c13aa/sanitizer/summary.json`.
- Interrupted execution and retry: completed; see `qies112a-diagnostic/runs/20261113-095319-4c13aa/interruption/summary.json`.
- Sliding/global boundary: completed; see `qies112a-diagnostic/runs/20261113-095319-4c13aa/boundary/layers-5-start-1023/summary.json`.
- Four-token extension: completed; see `qies112a-diagnostic/runs/20261113-095319-4c13aa/trial-3/layers-80-start-0/four-token-summary.json`.

Fault injection uses existing exception hooks at allocation/compute/serialization boundaries. It does not force actual system OOM or change allocator behavior. Malformed/truncated/cross-layer/wrong-position state is tested only with diagnostic copies or mismatched requests. Expected refusal/fault exit 1 must not be reported as successful inference.

Sanitizer testing explicitly uses the existing separate `qies111/sanitizer-build/candidate` identified by its preserved hash, never replaces the ordinary candidate. ASan/UBSan/leak detection cover instrumented caller/state/adapter code; prebuilt GGML libraries are not fully instrumented. Passing these cases is not proof that all library paths are bug-free.

The initial diagnostic wrapper was deliberately interrupted with SIGINT after a static audit found that its invalid-layer case would bind a nonexistent fixture directory before reaching native bounds validation. The corrected wrapper instead binds valid layer 0 fixtures and passes invalid layer80 to the native CLI. This change affected only diagnostic routing, not the executable, source, guards or math being diagnosed. The first partial run, source snapshot, exit 130 and correction reason are preserved in `harness-interruption.json`, `run-initial.py` and the initial series directory; it is not counted as a completed trial or an unexplained inference crash.

## Attribution and recommended action

No QUORTEX root cause is established from historical evidence. Do not assign OOM, segmentation fault, invalid tensor access, KV corruption, resource-limit enforcement or supervision failure merely from an interrupted assistant session. Recorded desktop SIGBUS signals are distinct evidence, with no demonstrated inference correlation.

No inference-engine fix is justified yet, and none has been applied. For the next **actual** failure, preserve the exact executable SHA256, full guarded command, timestamp/PID, exit/signal, stdout/stderr and matching target/ancestor OOM counters. If a matching core exists, inspect it using the matching executable/debug identity. If only PandaCode disappears, diagnose its app/core separately; do not weaken QUORTEX's guards or modify its arithmetic as a workaround.

Synthetic stability results apply to this full-width, fixed 256-slot, repeated-synthetic-profile configuration. They do not qualify real Ripple/FlashNext inference, arbitrary context lengths, full-state power-loss recovery or a long-lived persistent process. Four-token extension and a seeded1023/1024 sliding/global boundary are reported separately from the repeated two-token trials. No real-weight arithmetic, production changes, frozen-ref changes, commits, merges or pushes are authorized or performed.

## Final diagnostic result

**PASS — preserved QIES111 synthetic configuration.** Three fresh 80-layer/two-token trials completed:960 successful native subprocesses,12,480 captures matching the original SHA256 map, no recorded signal exits, no safety violations and no KV-integrity mismatch. These are three repeated deterministic workloads, not evidence of every possible input or long-lived process state.

**PASS — bounded stress and recovery cases.** All 78 original injected exception hooks and 5 invalid-state/dimension refusals returned the expected failure 1 while preserving authoritative input. Four selected layer 0 cases passed in the preserved ASan/UBSan executable, without sanitizer/leak diagnostics in the instrumented scope. The 80-layer repetitions themselves were ordinary builds; no full-stack/full-library sanitizer qualification is claimed. Deliberate termination left no published partial capture or live descendants; retry from the same checkpoint matched all 26 independent reference captures. The 5-layer seeded sliding/global boundary matched 260 captures.

The four-position extension reused trial 3's verified positions 0/1, then evaluated positions 2/3 through all 80 layers in both branches:4,160 additional exact captures,8,320 across the four positions. It is not an independent fourth fresh trial or QIES112 autoregressive language generation. All layer checkpoint identities, positions, checksums and candidate/reference projection-weight digests passed final audit.

**INCONCLUSIVE — historical inference crash.** No matching inference executable/failure record was recovered. The separately recorded PandaCode/WebKit SIGBUS dumps establish desktop failures on November 6, not a QUORTEX root cause. That journal's coredump-service memory peaks belong to crash processing, not to the inference workload. The tested ordinary executables, source, three GGML libraries, synthetic input files and guards retain their before-test hashes; frozen refs and disabled legacy libllama permissions are unchanged. No program fix was applied.

Final machine-readable evidence: `qies112a-diagnostic/final-results.json`, `final-integrity.json`, `system-evidence-after.json`. Per-process raw stdout/stderr, commands, monitor samples and completion records remain under the active series in `active.json`. Accessible kernel/core queries after testing are retained separately; their absence cannot establish what happened in an unlogged historical event.

Recommended next action: preserve the working QIES111 artifact and obtain the exact historical failing command/time/executable hash before any engine change. A separately reviewed **logging-only** supervision improvement should record parent cancellation reason, child exit/signal, last completed layer, stderr and OOM deltas even when the supervising session is interrupted. This closes an attribution gap; it is not a claim that supervision caused the historical crash. Do not relax resource gates or replace the verified engine based on the current evidence.

Four-position stress measured peaks: streaming subprocess-tree RSS/PSS **24.52/16.39 MiB**, reference **541.78/532.95 MiB**. These are sampled tree values, not total-system RAM requirements. See native high-water RSS and host MemAvailable minima separately in `final-results.json`.

| Possible historical cause | Evidence-based classification |
|---|---|
| OOM / resource-limit enforcement | Not established historically; no recorded guard/OOM stop in accepted synthetic trials. |
| Segmentation fault / invalid tensor access | No matching inference signal/core recovered; handled bounds refusals passed. |
| KV corruption | No unexplained mismatch in tested checkpoints; deliberately corrupted copies were rejected. |
| Process supervision failure | Historical attribution unproven; controlled termination/restart passed. |
| Another cause | Separate desktop SIGBUS events are confirmed; they are not identified inference crashes. |

The historical scan covered **1,799 available result/status files** and found no negative exit-status or recorded safety-stop entry. A separate scoped historical error-log scan found no matching SIGSEGV/ASan/UBSan/GGML-assert/core/signal text. This is an artifact search, not proof that an unlogged failure never occurred. Search scope and results are retained in `historical-status-scan.json` and `historical-error-log-scan.*`.
