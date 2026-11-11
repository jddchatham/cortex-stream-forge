# QIES103 — Ripple Real-Weight Execution Readiness

## Decision

**NO-GO for QIES104 real-weight execution.** Source provenance, byte extraction and synthetic equivalence have supporting evidence. A bounded real-weight stock worker and its live guarded execution are not qualified. The requirement that every numerical entry point be unable to bypass the guard is also not satisfied by the current tools/APIs. These are blockers in addition to the failing 6 GiB RAM gate.

QIES104 may perform model-free engineering to resolve these gaps. No real-weight arithmetic ran in QIES103. No production FlashNext code, numerical kernels, guards, resource limits, system settings or other branches changed. No QIES086, GPU work, commits, merges or pushes.

Worktree: `/home/user/PandaCode Bots/meatball-84ccea00/qies098-worktree`; branch `experimental/ripple-65b-q4`; HEAD `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. Tracked and staged diffs remain empty. Prior experimental sources/reports are untracked; the new files are identified by `qies103/SHA256-MANIFEST.json`, not by claiming a new commit. `git-status.txt` records the starting tree.

## 1. Reference qualification

Pinned stock source is llama.cpp **b12167 / `240d68fcfe0c2a1f298c8e18ca181368b1914259`**. Relevant source: [Ricursive graph](https://github.com/ggml-org/llama.cpp/blob/240d68fcfe0c2a1f298c8e18ca181368b1914259/src/models/ricursive.cpp), plus the retained `llama-graph.cpp`, `llama-model.cpp`, KV-cache implementation and loader. The audit reverified **4,084 stock source hashes** and QIES100 executable/library hashes. The original stock numerical source remains unchanged.

Candidate execution uses preserved GGML **0.25.3** CPU libraries; stock uses independently built GGML **0.29.0**. Both target CPU execution with one thread, non-flash attention, F16 KV and ordinary CPU weight buffers. **The kernel binaries/versions are not identical.** QIES100/QIES102's 360/360 F32 and 360/360 matched ordinary-buffer quantized comparisons support the selected synthetic configurations, not universal real-weight equivalence. Stock's extra/repacking-buffer profile is a known nonmatching diagnostic and must stay disabled for the proposed reference.

The reference observer includes stock llama/GGML headers and links only the pinned stock libraries. It does not link the QUORTEX layer/adapter. `capsule-build.json`, the compile command and prior stock library hashes establish independence. Compiler remains GCC 16.2.1; new observer flags are C++17, `-O2 -Wall -Wextra`. Existing stock libraries retain the QIES100 Release/OpenMP/native CPU configuration, with GPU backends disabled.

### New synthetic execution evidence

A bounded reference design was exercised with **synthetic weights only**. `capsule_fixture.py` creates one-block versions of QIES100's small F32 and quantized five-block fixtures. `stock_oneblock_probe.cpp` uses the original stock model loader, Ricursive graph and `llama_decode`, with an observer callback. It runs four consecutive positions, preserving stock KV continuity.

The one-block stock model's layer-0 outputs/intermediates match layer 0 of the original five-block stock model in **144/144 direct byte comparisons**: 18 captures × four positions × two weight profiles. No alias substitution or tolerance is used in this comparison. This specifically exercises effects of declaring layer 0 the last layer, including stock's output-row selection, on the small fixtures. Evidence: `capsule-comparison.json`, `capsule-run.log`, profile reference logs. Peak process RSS was **16,872 KiB F32 / 17,000 KiB quantized**. These are synthetic-profile measurements.

This is stronger than successful compilation, but it does **not** qualify execution using the real 5120/40960-dimensional layer weights. The current observer rejects ordinary numerical inputs above 16 MiB and requires hidden size 256/one block; its real-file modes remain metadata-only. It is intentionally not a real-weight worker.

### Bounded real-reference proposal

`bounded-reference-plan.json` describes a future **one-block stock GGUF capsule** containing the exact fifteen verified real layer-0 tensors (545,752,068 bytes) and small synthetic unused global tensors. A 32-entry F32 dummy embedding/head plus output norm would add 1,331,200 bytes; total payload would be approximately 547 MB, avoiding the original 40.4 GB tensor set.

The unmodified stock Ricursive graph would receive an externally supplied activation, with a one-element SWA/RoPE pattern and one block. Context/head/FFN/norm/clamp/RoPE parameters remain those of the real layer. Vocabulary is synthetic/none and embedding scaling is set to one because the supplied activation is already scaled. No token lookup is part of this test; dummy output-head results are excluded from acceptance.

**This is a transformed test artifact, not the original complete GGUF.** Before treating it as an oracle, review/prove every relevant consequence of changed layer count, last-layer row selection, external embedding scaling/x0 identity, cache layout and dummy globals for the single-token contract. The 144 synthetic comparisons support this approach but do not finish that proof for real weights. Hash the new artifact and preserve its transformation manifest. No real capsule was created or executed here.

Extraction has measured bounded evidence (QIES101/QIES102); real reference execution has only component estimates. Its allocation/load/warm-up/capture behavior must be qualified under the unchanged guard. Ordinary full-model stock loading must not be assumed bounded from mmap's virtual size alone.

## 2. Guard audit and live qualification status

Original guard remains `../qies077-guard.py`, SHA-256 `ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2`.

The managed candidate path is:

1. `qies102/runtime.py` checks host resources, reference/guard qualification evidence, model/tensor/plan/binary identities.
2. It calls the original `run_guard(command, logdir)` **without policy overrides**.
3. The original guard repeats admission and ancestor-headroom checks before starting the user service.
4. `probe --real` parses metadata and small input/state files, then constructs `LayerSource` before calling `r99::execute`.
5. `LayerSource` checks ≥6 GiB MemAvailable, the launcher marker and a guard-service cgroup name before the layer backend/weight allocation inside `r99::execute`.

Thus, the reviewed managed candidate path places admission before model tensor allocation. Small input/manifest allocations are earlier; they are not model weight allocations. Admission is not reinterpreted as a new 6 GiB threshold during every weight chunk; the original runtime thresholds remain unchanged.

One actual, harmless `/usr/bin/true` job was submitted to the original guard with default policy while host admission naturally failed. It returned nonzero, `ok=false`, reason `resource preflight failed`, `cleanup=true`, with no child execution log. No target service/process was launched. Later test reruns inspect that preserved receipt rather than repeat the failed admission test. See `guard-preflight-rejection/`.

**Live success, monitoring-after-launch, descendant handling and cleanup-under-load remain unqualified in this environment.** No lowered admission policy, telemetry override or fabricated receipt was used. Static ordering and pure boundary tests are not live enforcement evidence.

Original policy remains:

| Check | Unchanged value |
|---|---|
| Entry MemAvailable / SwapFree / disk free | 6 / 4 / 30 GiB |
| Entry memory full-PSI avg10 | below 1 |
| Finite ancestor memory/high and swap headroom | 7 / 2 GiB |
| Runtime minimum MemAvailable / swap / disk | 1 / 2 / 20 GiB |
| Tree RSS ceiling | 7 GiB |
| Runtime pressure / time | PSI ≥10 for three samples / four hours |
| Termination | whole cgroup, TERM then two-second grace and KILL |

OOM events, monitoring failure and interruption remain abort conditions. Snapshot `RLIMIT_AS` and `RLIMIT_RSS` are unlimited; inspected ancestor memory ceilings are recorded in `audit.json`. The guard checks inherited limits and monitors usage; it does not install a hard per-job MemoryMax or address-space cap. No stronger hard-limit guarantee is claimed, and none was added.

### Alternate-entrypoint and reference coverage gaps

`entrypoint-audit.json` documents why the blanket requirement is currently **not verified**:

- The managed QIES102 path is guarded, but the native marker/cgroup-name check alone does not attest to a live monitor or independently qualify reference evidence.
- Legacy `qies099/layer-cli.cpp --run` checks 6 GiB and a cooperative environment marker, not the entire current monitor/qualification contract.
- Public `r99::execute` is a reusable numerical API without intrinsic host admission; callers supply the source. It must not be presented as an independently guarded API.
- Ordinary stock `llama-simple` has no QUORTEX admission gate. It is not approved for real-weight execution.
- The stock capture tools used here support only synthetic numerical execution or metadata inspection. There is **no corresponding qualified guarded real-reference entrypoint** yet.
- Raw extraction, metadata inspection and storage roundtrip utilities do not execute model arithmetic and are outside the numerical-entrypoint claim.

No bypass was exercised with real tensors. Frozen legacy artifacts were not deleted, rewritten or silently declared safe. A local user's arbitrary binaries/library callers cannot be turned into a security boundary by an environment variable. QIES104 must define the supported execution boundary and ensure both approved workers use the original policy before any model allocation; if the requirement literally includes every legacy executable/arbitrary caller, it remains unmet without a separately authorized enforcement design.

There is also a qualification bootstrapping gap: QIES102's candidate launcher requires already qualified reference outputs, while no guarded reference-qualification worker exists. A reference-only qualification stage must be designed explicitly under the same original guard. It may generate provisional stock evidence after source/synthetic qualification; candidate acceptance must stay blocked until that evidence is independently reviewed and qualified. Never fill the gap by manufacturing a qualified status label.

## 3. Exact layer-0 execution contract

`qies103/inputs/contract.json` and accompanying files define a deterministic contract. They contain **no golden output hashes**; all expected output hash fields remain null.

| Property | Exact value |
|---|---|
| Input/x0 | F32 little-endian, [5120,1], 20,480 bytes each |
| Activation values | `x[i] = ((i % 17) - 8) / 16`, exactly representable; x0 bytes equal x |
| Meaning | external, already-scaled synthetic activation; actual quantized layer weights |
| Token position / sequence / tokens | 0 / 0 / 1 |
| Layer / context / window | 0 / 256 / 1024, sliding causal |
| Q heads / KV heads / head dimension | 80 / 8 / 128 |
| Hidden / FFN | 5120 / 40960 |
| RMS epsilon / clamp / RoPE base | F32 1e-6 / 20 / 10000, NeoX at position 0 |
| Attention scale | F32 `1 / sqrtf(128)` = 0.0883883461356163 |
| Attention mask | 256 F32 values: index 0 zero; indexes 1–255 negative infinity |
| Initial KV | logical length zero; zero F16 key/value buffers [128,8,256] each |
| Native previous-state input | `-` |
| Success position | exactly 1 |
| CPU profile | one thread, non-flash attention, F16 KV, ordinary buffers/no extra repacking |

Input SHA-256: `edc12d25f89ab142a15cb9dbed09787e07121946f5abfaa71269ec4086531207`; x0 has the same hash. Both paths must read these exact bytes and prove that their graph actually receives them without an additional embedding-scale transformation. This is a real-weight layer test, not tokenization or text generation.

The contract embeds exact hashes/types/shapes/lengths of all fifteen QIES102 layer tensors, plus the tensor-plan and manifest hashes. The original GGUF identity still matches QIES102's complete SHA-256 receipt, model SHA-256 `9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc`. All fifteen extracted file hashes were recomputed this task. A new complete-model hash was not needed because the verified identity chain remains unchanged.

Expected captures are the eighteen named intermediates: attention norm, mixed value input, Vcur, normalized Q/K, transformed Q/K, softmax probabilities, attention output/gate/gated result/post-norm, residual FFN input, FFN norm/SwiGLU/down output/post-norm, and `l_out`. Add `layer-output.f32` and canonical `key-written.f16`/`value-written.f16`: **21 output artifacts, 651,264 bytes** for the listed comparison payloads. The contract specifies every shape/dtype/length, including softmax [256,1,80,1], final output [5120,1], and 2,048 written bytes per KV component.

Reference capture must use stock observer callbacks after tensor computation, obtain contiguous logical F32 values from the backend, validate every write/close/length, and record tensor names/dimensions/types and source/configuration identities. Capture actual stock-written F16 cache rows after its cache update, not values reconstructed by candidate arithmetic. Stock's non-flash V-cache layout may be transposed: canonicalization must be independently verified rather than taking a raw first-2,048-byte prefix. Compare written logical rows only; unwritten/padded slots are excluded. This canonical export is not yet implemented/qualified.

Acceptance requires exact shapes, dtypes, byte lengths, finite computed values and **every byte equal**, with zero tolerance. `layer-output` must equal `l_out`; next position must be one. Missing/corrupt outputs, guard violations, nonzero exits or cleanup failures invalidate the run. A zero wrapper exit or matching final output without required intermediate/KV evidence does not qualify it.

## 4. Resource feasibility

| Item | Measurement or estimate |
|---|---|
| Actual layer-0 quantized payload | **545,752,068 bytes (520.47 MiB)**, verified |
| Largest single storage-test tensor | 144,179,200 bytes, QIES102 measurement |
| QIES102 storage-only peak RSS/PSS | 177,880 / 173,601 KiB observed; one tensor at a time |
| QIES103 small stock prototype RSS | 16,872 / 17,000 KiB, F32/quantized measured |
| Full real-layer GGML workspace | **Unmeasured**; existing graph buffer ceiling 128 MiB |
| Native metadata arenas | 18 MiB source-supported reservation |
| One layer F16 KV | 1 MiB payload; roughly 2–3 MiB including host/backend/result lifetimes, estimate |
| Chunk/capture storage | 1 MiB loading chunk; bounded sub-MiB listed comparison payloads |
| Planning real-job peak RSS/PSS | roughly **700–900 MiB**, low-confidence estimate, not a guarantee |
| Planning wall time | roughly **10–180 seconds per serial job**, low-confidence estimate; real timing unknown |

The RSS/PSS planning range combines known active weights, metadata, existing graph ceiling, KV and allowance for backend/allocator/library overhead. Real workspace, allocator retention and reference loader behavior need measurement. The storage-only peak cannot be substituted for a full layer's peak. File cache/writeback can additionally affect cgroup memory and MemAvailable without appearing in process RSS. No system-wide or cgroup upper bound follows from this table.

The time estimate is only an operational allowance. A prior integrity scan moved 40.4 GB in about 162 seconds including hashing, suggesting a few seconds of sequential I/O for one layer under similar conditions; cache state, load mode, quantized CPU arithmetic, capture I/O and swap can dominate differently. No actual-weight arithmetic timing exists. Keep the original four-hour guard; do not replace it with a guessed short limit.

Current snapshot: MemAvailable **3.687 GiB**, SwapFree **14.234 GiB**, disk free **60.149 GiB**, PSI 0.0. A future ~547 MB reference capsule, bounded captures and logs fit while preserving the existing 30 GiB entry reserve. It has not been built. The smaller estimated job does **not** authorize lowering or bypassing the unchanged 6 GiB entry gate.

## 5. Deterministic validation results

`final-run-script.log` records success for:

- Source/library/model-identity/fifteen-tensor hash audit and unchanged tracked/staged tree.
- Reproducible deterministic input/mask/zero-KV files and 21 exact output schemas, with no fabricated golden data.
- **144/144** one-block versus five-block stock layer-0 synthetic comparisons.
- **27 QIES103 checks**, including policy values/boundaries, input identity, dimensions, corrupt-input rejection, admission ordering/source audit, no policy overrides, missing qualification refusal, actual default-policy preflight refusal, reference linkage independence and explicit detection of alternate-entry gaps.
- The **16 existing QIES102 adapter checks** execute as a cleanup sub-suite using synthetic fixtures and admission-only real metadata. Copies of temporary plans live in QIES103; earlier artifacts are preserved.

Pure boundary inputs are model-free decision tests, not substituted live telemetry. Static checks are labeled source audits. The actual guard refusal only proves its naturally failing preflight path. Positive service lifecycle, OOM/descendant behavior and real execution remain unqualified. Detected gaps are passing audit observations, not passing qualification requirements.

New sources: `contract.py`, `audit.py`, `capsule_fixture.py`, `compare_capsule.py`, `stock_oneblock_probe.cpp`, `tests.py`. `entrypoint-audit.json`, `bounded-reference-plan.json`, `audit.json`, test results and source/binary hashes provide the handoff. The root Run Script runs these model-free/synthetic checks and never real-weight arithmetic; its prior version is preserved as `qies103/qies102-pandacode-run.sh`.

## 6. Required QIES104 work

1. Resolve the supported-entrypoint policy and shared admission boundary for both workers; retain all frozen artifacts. Do not claim all entry points guarded while known alternatives remain.
2. Implement a bounded stock real-reference worker/capsule with reviewed metadata/graph equivalence, fail-closed captures and actual canonical KV export. Keep the numerical source independent and unchanged.
3. Qualify the exact reference and candidate launch paths with harmless live jobs on a naturally ≥6 GiB session/host, original policy intact. Demonstrate success, failures, descendants, interruption, monitor failure and cleanup before tensor allocation tests.
4. Stage a separately named, guarded reference-only qualification run, then independently review its outputs and identity evidence. Only qualified reference artifacts can unlock the candidate run.
5. Run the single-position candidate under the same guard and contract, stop on any mismatch, and measure real RSS/PSS, cgroup memory/peak/swap, faults, I/O and elapsed time.

**NO-GO for real-weight execution until these blockers are closed.** Model-free development toward that qualification can proceed. No real inference or safeguard modification is required to act on this handoff.
