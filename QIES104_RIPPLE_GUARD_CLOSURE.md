# QIES104 — Ripple Guard Closure and Runtime Qualification

## Decision

**Approved-harness fail-closed closure: PASS. Real-weight runtime qualification: NOT ACHIEVED. NO-GO for QIES105 real-weight execution.**

The new QIES104 release rejects every supported real-weight numerical entry before model allocation. Real execution is deliberately locked even on a host that passes the unchanged 6 GiB entry gate; no environment variable or certificate enables it. Explicit synthetic interfaces run in a sealed filesystem without the real models/raw weight bundles, with pinned fixture bytes and bounded geometry. Numerical regressions pass.

This is a negative-path/control qualification, not a successful live real-weight run. Positive original-guard lifecycle qualification and a bounded real-weight reference execution remain pending. No real-weight arithmetic occurred. No production FlashNext, QIES086, global resource policy, source mathematics, system configuration, commit, merge or push changes occurred.

Worktree: `/home/user/PandaCode Bots/meatball-84ccea00/qies098-worktree`; branch `experimental/ripple-65b-q4`; HEAD `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. Tracked/staged diffs remain empty because the experimental artifacts are untracked. Exact local changes, including permission changes, are recorded separately; HEAD alone does not identify this release.

## 1. Entry-point inventory and closure scope

Machine-readable inventory: `qies104/entrypoints.json`.

| Path | Classification and enforcement |
|---|---|
| `qies104/worker` real/storage/admission/unknown dispatch | Shared native admission; real locked before model parsing/allocation |
| QIES104 copy of `r99::execute` | Direct-call admission before schema/backend/weight allocation; real source/config cannot bypass CLI checks |
| QIES104 `ModelFile` constructor | Shared real admission before opening the model |
| QIES104 `LayerSource` / `execute_layer` | Shared real admission; synthetic sources must pass pinned-file checks |
| `qies104/reference-worker` | Guarded frontend with **no host stock-SDK dependency**; real role rejected before private backend |
| Private stock reference engine | Compressed non-executable host data; instantiated only in sealed namespace; has its own shared admission check |
| Synthetic layer/candidate/adapter tests | Explicit namespace admission, geometry/file restrictions; guarded wrappers before original test bodies |
| 13 former numerical worker ELFs | **Not executable** on the host; bytes preserved unchanged |
| Five former build/dispatch scripts | Host execution rejected; historical execution permitted only in sealed synthetic namespace |
| Legacy stock `libllama.so.0.9.0` C ABI | **Not loadable through its canonical host path**, mode 000; exact bytes preserved compressed |
| QIES101 raw extractor, stock metadata reader and index parser | Data-only utilities, not numerical execution paths; metadata/raw bytes only |
| Preserved source/archives and generic math dependencies | Not approved host model-execution interfaces; deliberate rebuilding/restoration is outside this closure |

Retired numerical workers are the QIES099 CLI/test/sanitized test; QIES100 candidate, stock observers, quantize/dequantize helpers and stock `llama-simple`; QIES102 worker/tests/sanitized tests; and QIES103 stock observer. The complete 13-path list is `retire-list.json`.

This scope addresses normal executable dispatch and accidental direct calls in the approved harness. It is not an adversarial security boundary against the workspace owner deliberately restoring permissions, extracting archives, replacing source or compiling a new caller. Such actions invalidate the closed release. Frozen source/objects remain recoverable rather than being deleted or misrepresented as guarded code.

## 2. Exact, reversible changes

### Shared boundary

`admission.h` / `admission.cpp` provide the shared native boundary for candidate CLI, direct layer API, model-file construction, bundle source, reference frontend and private reference backend:

1. Real entry checks Linux **MemAvailable ≥6 GiB**.
2. Regardless of host memory, this release then rejects real execution because positive reference/live-guard qualification is missing.
3. Rejection precedes GGML contexts, backends and model-weight buffers in the reviewed real paths.
4. Synthetic numerical execution requires the namespace marker/environment established by the launcher. Marker-only host forgery fails.
5. Synthetic geometry is restricted to the existing small test profiles (hidden ≤256, FFN ≤512, query/KV heads ≤4/2, head size ≤64), preventing full real-model geometry from being relabeled synthetic.
6. File-backed synthetic sources and reference GGUFs must match the compiled fixture catalog. Unknown paths/changed bytes fail before model loading.

`gate.py` blocks retired host builder/dispatch scripts. It is an additional host-entry check, not an alternative numerical approval path. No successful real approval exists in the native release.

The guard overlays are under QIES104, not in production source: `ricursive-layer.cpp`, `bundle.cpp`, `candidate.cpp`, `synthetic_candidate.cpp`, `reference.cpp`, `reference_front.cpp`. `patches/` contains admission-only diffs against their earlier sources. Tensor arithmetic, quantization routines, decode/operator scheduling and state ownership are unchanged.

### Separate synthetic domain

`sandbox.py` and `bootstrap.py` use existing Bubblewrap without installing services or changing host configuration. The domain has:

- A minimal read-only system/toolchain view, private PID/network namespaces and private temporary files.
- Read-only, hash-pinned synthetic fixtures; real home/model/state directories and raw tensor bundles are absent.
- No inherited model descriptors; worker stdin is closed. The approved launch does not expose the real GGUF through another path.
- A bounded historical test snapshot copied into ephemeral tmpfs. Old workers may be rebuilt there, but no real model bytes are present.
- Evidence-only export; rebuilt historical ELFs are never exported. Namespace teardown removes them.

The namespace is a data-access restriction, not a reduction of the inference memory gate. Synthetic/model-free tests remain explicitly separate from real execution.

### Legacy retirement and stock SDK

Thirteen old worker files retain their exact bytes but have execute bits removed. Five builder/dispatch scripts receive a shared namespace-only entry check; their original assertions and bodies are preserved in `preserved/` and the regression snapshot.

The legacy stock library's canonical host path is mode 000, closing ordinary direct `ctypes`/C-ABI loading. Its exact SDK bytes are preserved in **compressed** `stock-sdk.tar.gz`. The reference engine is also compressed (`reference-engine.gz`). Neither is a directly loadable host ELF. Bootstrap verifies archive/source hashes, decompresses the SDK/backend only into private namespace tmpfs, and supplies them to the guarded reference frontend. The frontend loads only ordinary system/Crypto libraries on the host; it does not load stock model code before admission.

`preservation.json`, `stock-sdk-manifest.json`, `engine.json`, `builds.json` and `patches/` identify original hashes/modes, replacement controls and compiled artifacts. Original SDK bytes were verified through an already-open descriptor after its permission change and again through the archive. These changes are reversible, but do not automatically restore old entry points: restoring them would reopen the gaps QIES103 identified.

The published FlashNext executable outside this Ripple worktree remains SHA-256 `329ebd647b04a2d6464461ddd94fc07a2691a948b2918d320ac1eb3703f9c101`. The downloaded models and existing persistent inference states were not modified.

## 3. Guard enforcement evidence

**40 fail-closed checks pass** (`closure-tests.json`): candidate real/admission/storage/unknown/empty dispatch; direct numerical and file-source APIs; synthetic interfaces outside their domain; forged old/new environment flags; real plans relabeled through synthetic dispatch; unregistered reference files; direct private-reference dispatch; all 13 retired workers; five retired builders; and the retired stock C ABI.

An allocation detector intercepts GGML context/backend allocations during rejection tests. A separate, model-free positive control deliberately attempts a tiny context and **must exit 97**. That proves the detector is loaded. All actual guard rejection tests avoid that trap; direct API resource counters remain zero, no model reads/outputs are attempted, and no stale child processes or output directories are accepted.

An initial detector path contained a space and was split by the dynamic loader. Its allocation-specific evidence was discarded. Corrected host tests preload through an inherited `/proc/self/fd/N` descriptor, and namespace tests use the space-free `/active/` path. The positive control and all rejection checks were rerun. `pre-correction-run.log` is retained but is not qualification evidence; **`final-validation.log` is the corrected canonical result**.

The original guard, SHA-256 `ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2`, remains byte-identical. One harmless default-policy `/usr/bin/true` request was naturally refused at preflight: nonzero status, `ok=false`, `cleanup=true`, and no target execution log. Reruns inspect that preserved result rather than repeat the failed preflight.

**No policy or threshold was overridden.** Entry remains RAM/swap/disk 6/4/30 GiB and PSI below 1; original ancestor headroom, 1/2/20 GiB runtime minima, 7 GiB tree-RSS ceiling, pressure/OOM/timeout/interrupt handling and cgroup cleanup policy are unchanged. Existing graph/weight limits remain unchanged. Synthetic geometry validation is an input-domain restriction, not a weaker inference policy.

AS/RSS rlimits are inherited unchanged (currently unlimited); cgroup policy files are recorded in `status.json` and not written. Bubblewrap virtualizes visibility, not the inherited memory ceilings. No new hard MemoryMax/RLIMIT_AS guarantee is claimed. Current real calls receive **no allocation approval at all**.

Positive live monitoring, post-launch failure/descendant/OOM handling and successful cleanup cannot be qualified below the unchanged gate. They remain explicitly pending. Negative admission evidence must not be called complete live runtime qualification.

## 4. Independent reference non-inference qualification

Upstream remains stock llama.cpp **b12167 / `240d68fcfe0c2a1f298c8e18ca181368b1914259`**. All 4,084 upstream source hashes remain intact. Stock numerical source and SDK bytes are unchanged; admission/frontend code is external. The SDK manifest binds the exact previously verified libraries. Private backend SHA-256: `a80182804c75874e6099e4a2792cae9f69c7a56c2576e283bc8f3f19b29906db`; its source/admission/catalog/SDK fingerprints are bound in `engine.json`.

Reference tensor location uses stock GGUF parsing, independently of the QUORTEX index/bundle parser. This task verified:

- **1,203 independent stock descriptors**, under the allocation detector, without creating a GGML context.
- **15 complete actual layer-0 tensor byte comparisons**, preserving types/shapes/offsets/lengths and hashes.
- The fixed QIES103 F32 activation/x0, position-zero mask, initial KV contract and 21 output schemas; no golden output hashes were fabricated.
- Reference guard enforcement both at frontend and direct private-backend real dispatch.
- **144 exact synthetic reference comparisons** using the admitted one-block stock observer against the preserved stock layer-0 evidence.

The matched profile is CPU, one thread, non-flash attention, F16 KV and **ordinary buffers**, with extra/repacking buffers explicitly disabled. Candidate GGML 0.25.3 and stock GGML 0.29.0 are different kernel versions. Exact synthetic agreement is supporting evidence, not proof of real high-dimensional equivalence.

`reference-qualification.json` distinguishes these passing non-inference/data/control checks from **real reference execution: disabled/unqualified**. The synthetic observer currently supports its small profile; a certified bounded real-capsule worker, actual canonical written KV export and real-contract execution still require implementation/qualification. The 5120-dimensional contract is verified as data/schema, not claimed executed by that observer.

## 5. Regression results

No existing assertion or numerical tolerance was weakened.

| Suite | Result |
|---|---|
| QIES099 original synthetic layer suite | 14/14 groups pass |
| QIES100 original numerical comparisons | 360/360 F32 and 360/360 ordinary-buffer quantized exact |
| QIES101 extraction faults | 38 index-path and four stock-path checks pass |
| QIES101 preserved actual extraction evidence | Eight complete tensors, 757,370,880 bytes per path, reverified |
| QIES102 original adapter suite | 16/16 pass |
| QIES103 original readiness suite | 27/27 checks pass, including its 16-check cleanup sub-suite |
| New guarded core/adapter builds | Original 14/16 suites pass against the new admission overlays |
| New guarded candidate comparisons | 720/720 exact synthetic intermediate comparisons |
| New guarded independent reference | 144/144 exact synthetic comparisons |
| New closure tests | 40 rejection checks pass; detector positive control confirmed |

Historical suites run unchanged in a preserved, sealed, ephemeral view. In particular, QIES103 contains assertions characterizing the **old** source gaps; these assertions are kept intact, and their historical diagnostic wording is not presented as a result about the new QIES104 boundary. Current closure is tested separately against the new workers/direct APIs. The new candidate/reference comparisons also exercise the new admitted builds, not merely archived executables.

QIES100's extra/repacking-buffer diagnostic still differs as before and is not accepted. All accepted comparisons use exact bytes. Real-weight arithmetic, full logits and real numerical equivalence were not tested.

See `final-validation.log`, `historical-regressions.json`, individual regression logs, `active-comparisons.json`, `closure-tests.json` and `reference-qualification.json`.

## 6. Reproduction, remaining blockers and QIES105

Approved validation command:

```sh
cd '/home/user/PandaCode Bots/meatball-84ccea00/qies098-worktree'
./pandacode-run.sh
```

It builds/validates the closed release, executes only synthetic numerical tests, performs data-only verification and records fail-closed readiness. It cannot enable real execution. Old QIES099–103 numerical workers/launch recipes are intentionally no longer usable as host execution paths. The model files and the separate published FlashNext baseline remain available and unchanged.

Latest recorded host values: **3.834 GiB MemAvailable**, 14.023 GiB free swap, 60.137 GiB free disk, full-memory PSI avg10 **3.98**. RAM remains below 6 GiB; the recorded PSI snapshot also exceeds the original entry threshold. Fresh resource checks are required before any future live qualification. No application was terminated and no swap/zram/kernel/cgroup configuration was changed.

Required before QIES105 real execution:

1. A naturally passing host/session satisfying all unchanged entry/ancestor checks.
2. Successful original-policy live guard qualification with harmless children, including success, failures, descendants, interruptions, monitoring failures and cleanup.
3. A reviewed positive real-admission mechanism tied to actual original-guard supervision. Do not remove the hard lock or replace it with an environment-variable/cgroup-name assertion merely to permit a run.
4. A bounded real-weight stock worker/capsule, verified target activation contract and canonical KV export, qualified independently under that same guard.
5. Matched model/tensor/binary/SDK/input identities and strict intermediate/final/KV comparison, followed by actual real RSS/PSS/cgroup/swap/I/O measurements.

**NO-GO for real-weight execution.** QIES105 may continue model-free qualification engineering. Guarded negative paths and preserved numerical regressions are established; successful real runtime qualification is still pending.
