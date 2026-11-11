# QIES102 — Real-Tensor Ricursive Layer Integration

## Decision and scope

**CONDITIONAL GO for QIES103 reference/guard qualification and a single-layer comparison after all gates pass. NO-GO for real-weight numerical execution now.** The tensor-to-layer adapter is implemented, its synthetic numerical results match stock exactly, and real quantized bytes round-trip through GGML CPU storage unchanged. Actual Ripple-weight arithmetic and real-layer workspace remain untested.

All work remained in `/home/user/PandaCode Bots/meatball-84ccea00/qies098-worktree`, branch `experimental/ripple-65b-q4`, HEAD `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. Tracked/staged diffs remain empty; the experimental source and reports are untracked. This HEAD identifies the published baseline, not the new uncommitted adapter; use the source/evidence SHA manifest for the latter. Working-tree status is recorded in `qies102/git-status-before.txt` and `git-status-after.txt`.

No production engine, FlashNext, QIES086, numerical kernel, decode schedule, hybrid-state ownership, guard threshold or system setting was changed. No commits, merges, pushes or GPU execution occurred. No real-weight arithmetic or full real-model inference was attempted.

## Baseline verification

A fresh complete streaming pass verified all 40,426,079,712 GGUF bytes against pinned published Hugging Face LFS metadata. Expected and actual SHA-256:

`9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc`

Model revision remains `df550a7178a87bbfb6ee2bdbd37603c553d536ee`. Evidence: `model-hash-result.json`, `published-model-metadata.json`, `model-sha256.txt`, `hash-run.log`. The new receipt is in QIES102; QIES101 receipt contents remain intact. This pass took 161.70 seconds with a 1 MiB read chunk. Original model identity is checked again when preparing the bundle.

QIES100's matched stock profiles are reconfirmed by the new adapter:

- **360/360 F32 intermediate comparisons byte-exact**.
- **360/360 quantized ordinary-CPU-buffer comparisons byte-exact**.

Each case uses five synthetic layers and four consecutive positions, including SWA window crossing and global NoPE. The eight documented global Q/K identity aliases per case are preserved; they are not extra independent computations. Reference outputs remain those produced by the independent pinned stock graph, not newly manufactured candidate golden data. Stock's extra/repacking buffer profile remains a known nonmatching diagnostic and is not accepted as the oracle configuration.

QIES101's **eight complete real-tensor comparisons, 757,370,880 bytes per path**, were reverified by its unchanged saved-artifact verifier (`qies101-reconfirmation.log`). The existing 1,203-descriptor index is revalidated by the unchanged QIES101 reader before bundle construction. All pinned stock source-file hashes and prior QIES099–101 source/evidence manifests were rechecked, apart from the intentionally maintained root Run Script. Earlier Run Scripts are preserved through the existing provenance copies, including `qies102/provenance/qies101-pandacode-run.sh`.

Independent stock source remains b12167, commit `240d68fcfe0c2a1f298c8e18ca181368b1914259`. Candidate kernels remain the unchanged GGML 0.25.3 CPU binaries used by QIES099; stock evidence uses its independently built GGML 0.29.0. QIES100 established the matched ordinary-buffer synthetic profile. Full commands, library hashes and executable identity appear in `build-identity.json`. Current QIES102 probe SHA-256: `99412e5ebfe8b09c0c466761f9441a18a591f32da42fc659a468a43dacd0861b`.

## Adapter implementation

New files are confined to `qies102/`, plus this report and the maintained root Run Script:

- `bundle.h`, `bundle.cpp`: a bounded manifest, raw-file `Reader`, `LayerSource : r99::Source` and `execute_layer()` calling the unchanged QIES099 `r99::execute()`.
- `prepare.py`: imports QIES101's unchanged `Extractor` and descriptor validator, builds a full real layer-0 tensor bundle and ten synthetic bundle plans, and obtains independent raw stock comparisons for newly required tensors.
- `probe.cpp`: synthetic kernel integration, real CPU-storage-only roundtrip, metadata/admission checks, and a gated real single-layer CLI.
- `tests.cpp`: schema/range/identity/admission and cleanup fault tests.
- `build.py`, `validate.py`: pinned CPU builds and deterministic validation.
- `runtime.py`: prepared original-policy real-run launcher with qualification/identity checks and strict output comparison.
- `hash_model.py`: task-local copy of the bounded full-file verifier, producing QIES102 evidence.

The bundle format records mode, layer/configuration, exact tensor name, GGML type, columns/rows, original GGUF offset, byte length, raw-file path, SHA-256 and device/inode/size/mtime/ctime. Manifest input is capped at 128 KiB, rows at 8 KiB and tensor count at 15. QIES099's schema checker requires exactly the layer's 15 tensors with expected shapes, permitted quantization and F32 norm/lambda fields. It rejects missing, duplicate, mismatched or excessive allocations. Fixture mode has a separate small-shape/size namespace.

QIES101 supplies raw quantized files without conversion. Two previously verified layer-0 files are reused in place; thirteen missing tensors were extracted using QIES101 and independently located/extracted using its pinned stock GGUF path. All fifteen complete files match their stock counterparts byte-for-byte. Shared selection names do not share descriptor parsing or offsets with the stock extractor. `bundle-manifest.json` records every hash and identity, reference path and exact comparison length; the runtime launcher verifies those hashes before execution.

`Reader::read()` resolves the exact manifest entry and checks the caller's name/type/shape/original-offset/length against it. It reads **at most 1 MiB** into the caller's buffer, handles partial reads, rejects out-of-range reads and changed files, and opens read-only descriptors with no symlink following. Each descriptor is closed through RAII before the call returns. There is no reader-owned tensor-sized buffer, conversion, model-wide mmap or weight cache. QIES099's existing layer loader copies those chunks directly into its GGML CPU tensors.

`LayerSource` reports the real/synthetic namespace correctly. For a real source, construction requires the unchanged ≥6 GiB MemAvailable entry gate, a qualified-launch marker and the original guard's service cgroup. These checks prevent accidental direct execution; the marker alone is not independent qualification evidence. Admission is checked once at single-layer entry, not imposed as a new 6 GiB runtime threshold on every chunk. The original external guard retains its existing runtime thresholds.

## Exact real tensor mapping

Layer 0 uses hidden 5120, FFN 40960, 80 query heads, eight KV heads and head size 128, with context 256/window 1024. Original scaled `x0`, activation input and prior KV are separate execution inputs; the embedding/output heads are not layer weights.

| Tensor | GGML dimensions | Type | Bytes |
|---|---|---|---:|
| `blk.0.attn_gate.weight` | [5120, 10240] | q4_K | 29,491,200 |
| `blk.0.attn_k.weight` | [5120, 1024] | q8_0 | 5,570,560 |
| `blk.0.attn_k_norm.weight` | [128] | f32 | 512 |
| `blk.0.attn_norm.weight` | [5120] | f32 | 20,480 |
| `blk.0.attn_output.weight` | [10240, 5120] | q6_K | 43,008,000 |
| `blk.0.attn_q.weight` | [5120, 10240] | q4_K | 29,491,200 |
| `blk.0.attn_q_norm.weight` | [128] | f32 | 512 |
| `blk.0.attn_v.weight` | [5120, 1024] | q8_0 | 5,570,560 |
| `blk.0.attn_vres_lambda.weight` | [1] | f32 | 4 |
| `blk.0.ffn_down.weight` | [40960, 5120] | q5_K | 144,179,200 |
| `blk.0.ffn_gate.weight` | [5120, 40960] | q5_K | 144,179,200 |
| `blk.0.ffn_norm.weight` | [5120] | f32 | 20,480 |
| `blk.0.ffn_up.weight` | [5120, 40960] | q5_K | 144,179,200 |
| `blk.0.post_attention_norm.weight` | [5120] | f32 | 20,480 |
| `blk.0.post_ffw_norm.weight` | [5120] | f32 | 20,480 |

Total: **545,752,068 bytes**. All fifteen names map directly to the existing attention normalization, Q/K/V, value-residual lambda, gate/output, post-attention normalization and FFN norm/gate/up/down/post-norm operations. No tensor arithmetic or order was changed. Raw files are in `qies102/bundle/` or reused from QIES101; they remain generated artifacts excluded from any future source preservation commit.

## Tests and comparisons

`final-run-script.log` records a successful complete run:

1. CPU-only adapter and test builds, with empty compiler warning logs.
2. Unchanged QIES101 saved-descriptor/real-byte verification.
3. New-source synthetic native execution: **720/720 exact intermediate comparisons** against preserved independent stock captures. No tolerance was relaxed.
4. **16 adapter tests pass**: valid reads; missing names; shape/type mismatches; read bounds; oversized requests; identity mismatch; missing files; real admission refusal; failed-load state preservation; cleanup after load failure; malformed/version/size/truncated/missing-name manifests.
5. The same **16 tests pass under AddressSanitizer and UndefinedBehaviorSanitizer**, with no diagnostics (`sanitizer-tests.log`). Existing shared GGML libraries are not sanitizer-instrumented.
6. Real quantized CPU-storage roundtrip: every byte of all fifteen tensors is uploaded via `ggml_backend_tensor_set`, read back through `ggml_backend_tensor_get` and compared exactly. This uses **no computation graph, matmul, normalization, dequantization, activation, attention, synthetic activation evaluation or real-weight arithmetic**. It proves storage/ownership compatibility, not actual-weight execution correctness.
7. Runtime readiness rejects missing real-reference/live-guard qualifications. The native real-source admission test fails before real weight allocation or graph execution. No guard override or fabricated qualification receipt was used.

New synthetic executions use actual existing Ricursive kernels and repeated KV continuation; real tests use actual downloaded weight bytes but only storage transfer. Those two evidence classes must not be conflated. Actual-weight numerical comparisons are blocked.

## Memory ownership and measurements

`real-storage.tsv` and `memory-analysis.json` record baseline, each live CPU tensor allocation, RSS/PSS with that tensor resident, and post-free RSS/PSS. All fifteen tensors are processed serially and freed individually.

| Measurement | Result |
|---|---:|
| Model-wide mappings in adapter | None |
| Native layer-loader chunk | Existing 1 MiB |
| Storage-test chunks | Two × 1 MiB, upload/readback |
| Storage-test metadata arena | 1 MiB |
| Largest simultaneously live real tensor buffer | 144,179,200 bytes |
| Maximum measured tensor alignment overhead | 28 bytes |
| Peak observed storage RSS | 177,880 KiB (173.71 MiB) |
| Peak observed storage PSS | 173,601 KiB (169.53 MiB) |
| Final post-tensor-free RSS/PSS | 37,076/32,797 KiB |
| Real-layer compute workspace measured | **No: execution blocked** |
| Synthetic layer graph buffer | 241,216 bytes |
| Synthetic layer KV payload | 131,072 bytes |

The storage-test baseline was 5,104 KiB RSS/1,135 KiB PSS. Final RSS/PSS did not return to that baseline: small allocations/backend initialization and allocator retention remain, and the two storage-test chunks/backend exist during these post-tensor snapshots. Large buffers demonstrably drop from peak when freed; the residual is not claimed to be zero memory cost. No allocator trimming, kernel setting change or speculative eviction was used.

Every tested source read returns with zero open source descriptors. Synthetic execution returns with zero tracked QIES099 native resources; fault paths preserve the caller's previous state and clean up allocations. The adapter stores no pointers into temporary file buffers in returned results. Manifest ownership outlives synchronous source reads; returned state/output contains owned values rather than graph pointers.

The storage test reads exactly 545,752,068 real tensor bytes through maximum 1 MiB requests. Its peak is for **one tensor at a time**, not the entire real layer. The eventual QIES099 graph retains all active-layer weights, plus graph storage/metadata and KV, until computation completes; that larger real-layer peak remains unmeasured. Source-supported existing ceilings (768 MiB weights plus alignment allowance, 128 MiB graph buffer and existing metadata arenas) are not a measured whole-process RSS bound. Synthetic memory/workspace cannot be scaled into a trustworthy real-layer estimate. Filesystem cache and output writeback also lie outside process RSS/PSS, so no system-wide RAM savings are claimed.

## Complete guarded single-layer procedure — prepared, not executed

The native entry is `probe --real PLAN INPUT.f32 X0.f32 PREVIOUS.kv|- NEW_OUTPUT_DIR`. It is reached through `runtime.py`, not directly. First-run scope is layer 0/position 0/empty prior KV, with full intermediate captures and canonical written KV comparison.

Before use, independently qualify a stock **real-layer execution** reference with the exact model, context 256, one CPU thread, non-flash attention, F16 KV and ordinary CPU buffers (`use_extra_bufts=false`). An extractor and synthetic comparisons do not qualify this oracle. The reference manifest must provide verified source/executable/library identities, input/scaled-embedding hashes, expected position and all intermediate/output/KV hashes. `runtime.py` applies both the QIES100 reference-file contract and QIES099 capture contract.

Then qualify this exact launcher/binary with harmless live jobs under unchanged policy. Its qualification receipt is tied to launcher SHA, candidate SHA and original guard SHA, with hashed evidence for successful exit, failure, descendants, interruption and monitoring failure/cleanup. No receipt is supplied here. Those evidence files require independent review of the actual exercised paths; a status label or JSON field alone is not proof.

After all qualifications and a **fresh naturally passing resource check**, a reviewed launch is:

```sh
cd '/home/user/PandaCode Bots/meatball-84ccea00/qies098-worktree'
systemd-run --user --scope --slice=app.slice \
  env -u LD_LIBRARY_PATH python3 -I -B qies102/runtime.py \
  --reference /absolute/qualified-stock-layer0.json \
  --guard-qualification /absolute/qualified-qies102-guard.json \
  --run-root /absolute/new-persistent-ssd-attempt
```

This is not authorized to run now because prerequisites fail. The monitor scope keeps app.slice populated as required by the original guard. The run root must preexist on the verified persistent filesystem; output/log leaf directories must be new. Native execution is explicit CPU initialization, with no GPU backend discovery. The launcher calls the original `run_guard()` with default policy, never threshold overrides, and requires successful cleanup plus exact output/KV hashes before returning success. A zero child exit alone is insufficient.

Original safeguards remain: admission RAM 6 GiB/swap 4 GiB/disk 30 GiB/PSI below 1; existing ancestor headroom checks; runtime minimum available RAM 1 GiB/swap 2 GiB/disk 20 GiB; tree RSS ceiling 7 GiB; OOM/pressure/timeout/monitor-failure handling and whole-cgroup termination. No new hard cgroup or address-space limit is claimed; inherited limits remain unchanged. Original guard SHA-256 remains `ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2`.

The real positive launch/comparison path has not been exercised. Process crashes can leave partial attempt outputs; they are not accepted checkpoints. Persistent state semantics remain QIES099's existing serialized KV test format, not a new production recovery mechanism.

## Current blockers and QIES103 recommendation

Latest readiness snapshot: **3.582 GiB MemAvailable**, 14.228 GiB free swap, 60.155 GiB free disk, PSI 0.0. The 6 GiB entry gate fails. A fully qualified bounded real-layer stock execution reference and live guard adapter are both absent. `runtime-readiness.json` explicitly records these blockers and no real numerical execution.

Reproduce the current safe validation using `./pandacode-run.sh`. It rebuilds only the adapter/tests, evaluates synthetic fixtures, rechecks preserved byte evidence, performs actual-weight storage transfers and reports readiness. It does not call the real numerical entry. `prepare.py` is one-shot and refuses existing bundle directories; do not rerun it to overwrite evidence.

QIES103 should first finish the independent bounded real-layer oracle and exact live adapter qualification on a suitable naturally passing host/session. Only then execute one guarded real layer/position, compare every intermediate/final output and canonical KV byte, and measure actual active-layer RSS/PSS/workspace. Stop on mismatch or any guard violation. Full generation, production acceptance and a desktop-mode memory claim remain NO-GO.
