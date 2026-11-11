# QIES099 — Native Ricursive Layer Execution

## Decision

**GO for QIES100 reference qualification and controlled single-layer comparison preparation; NO-GO for real-model execution in the present desktop session.** This is experimental native CPU layer support, not accepted Ripple inference. No real-model layer or full inference ran. No commits, merges, pushes, resource-guard edits, FlashNext source edits, or system changes were made.

## Provenance and model identity

Worktree: `/home/user/PandaCode Bots/meatball-84ccea00/qies098-worktree`, branch `experimental/ripple-65b-q4`, HEAD `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. QIES098 report, metadata inspection and tensor index were reviewed. Existing milestones and baseline binaries remain intact. Tracked Git diff remains empty; this implementation consists of new experimental files and an updated, previously untracked model-free Run Script. Its former contents are preserved in `qies099/provenance/qies098-run.sh`.

Model: `../qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf`, 40,426,079,712 bytes. A fresh complete-file SHA-256 read matched `9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc`. Evidence: `qies099/checks/model-sha256.txt`. Quantization repository revision: `df550a7178a87bbfb6ee2bdbd37603c553d536ee`; licensing evidence remains in QIES098.

Numerical graph specification: upstream llama.cpp b12167, commit `240d68fcfe0c2a1f298c8e18ca181368b1914259`, especially `qies098-checks/upstream-b12167/src/models/ricursive.cpp` and attention construction in `src/llama-graph.cpp`. This identifies the source algorithm, not a runtime-qualified comparison oracle.

## Implemented files and integration boundary

- `qies099/ricursive-layer.{h,cpp}`: native one-token, one-layer graph, bounded weight reader, private candidate KV, schema checking, state-file checks and scheduler contract.
- `qies099/layer-cli.cpp`: descriptor inspection and gated experimental layer runner.
- `qies099/test-layer.cpp`: deterministic synthetic native-kernel tests.
- `qies099/build.py`: reproducible CPU-only compilation, pinned existing library checksums and test execution; commands/binary identities in `checks/build-identity.json`.
- `qies099/prepare_plans.py`: 80 descriptor plans derived from the existing GGUF index and verified file identity.
- `qies099/launch_real.py`: prepared original-guard adapter and byte-comparison checker. Its runtime qualification is explicitly incomplete.
- `qies099/reference/BLOCKED.json`: explicit missing-reference marker.
- `pandacode-run.sh`: model-free build, synthetic tests and descriptor checks only.

The path directly reuses the verified baseline GGML CPU backend, tensor storage and quantized matrix kernels. It does not link the full llama model loader or load all weights. No substitute scalar approximations were introduced. GGML kernel binaries are from the existing engine, not newly rebuilt b12167 libraries; their hashes and build commands are recorded. Therefore source-level graph correspondence does not establish bit-exactness with b12167.

## Architecture and operator sequencing

Actual metadata: 80 blocks; hidden 5120; FFN 40960; 80 query heads, eight KV heads, head size 128; RMS epsilon approximately 1e-6; RoPE base 10000; clamp 20. Every fifth block (4,9,...79) is global NoPE; the remaining 64 use sliding-window attention and NeoX RoPE, window 1024.

Each active block resolves exactly these 15 `blk.N.*` tensors: `attn_norm.weight`, `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_q_norm.weight`, `attn_k_norm.weight`, `attn_vres_lambda.weight`, `attn_gate.weight`, `attn_output.weight`, `post_attention_norm.weight`, `ffn_norm.weight`, `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight`, `post_ffw_norm.weight`. The plans contain authoritative names, dimensions, offsets, types and byte lengths; schema validation rejects omissions, duplicates, wrong shapes and unsupported types.

Execution follows upstream structure: learned attention RMSNorm; Q/K projection; sigmoid value-residual mixing of original scaled embedding `x0` and normalized layer input; V projection; per-head Q/K RMSNorm; SWA-only RoPE; candidate F16 KV update; masked scaled attention; sigmoid attention gate; output projection; post-attention norm and residual; FFN pre-norm; clamped SwiGLU gate/up activation; down projection; post-FFN norm and residual. Native GGML operations include `ggml_rms_norm`, `ggml_mul_mat`, `ggml_rope_ext`, `ggml_soft_max_ext`, `ggml_sigmoid`, `ggml_swiglu_clamp`, copies, views and elementwise residual operations. KV writes are scheduled before attention reads.

The active-layer model index uses F32 norms/lambda and Q4_K, Q5_K, Q6_K and Q8_0 matrix types. IQ4_XS is also tested and supported by the native schema/kernel path; the actual model's IQ4_XS embedding is outside this layer implementation. Model-wide totals remain 561 F32, 310 Q4_K, 90 Q5_K, 81 Q6_K, 160 Q8_0 and one IQ4_XS tensor.

## Ownership and bounded residency

Only the selected layer's 15 tensors are loaded through a read-only file descriptor, using 1 MiB chunks and direct backend tensor writes. Descriptor-only checks read no model payload. Actual indexed weight sums range from 467,108,868 to 545,752,068 bytes. The weight ceiling is 768 MiB plus alignment allowance; graph storage is capped at 128 MiB, with separate 2 MiB weight metadata and 16 MiB graph metadata arenas. No model-wide tensor allocation or full float dequantization buffer is created.

One layer owns its F16 key/value state and position. With context 256, its logical host KV payload is 1 MiB; prior state, candidate state and backend state may coexist. The old state is immutable to execution. Layer/configuration ownership, shapes, finite values and positions are checked; only successful graph completion returns a candidate advanced by one. Failure hooks leave input state unchanged. RAII releases graph, weight, backend and file resources, including exception paths. No tensor pointers escape the call. The scheduler requires all 80 layer completions before advancing its token position; it is a tested contract, not an implemented full-model driver.

These bounds do not certify a whole-process RSS ceiling: backend workspace, allocator retention and runtime costs still require guarded real measurements. Optional 18-intermediate captures are bounded to this one-layer test and are not a full-logit capture. Full-model KV storage is not allocated here.

The prototype KV serialization records model/configuration identity, layer, position, F16 payload and integrity fields. It rejects altered, truncated, mismatched and overwritten files. Its FNV integrity checks detect accidental corruption; they are not cryptographic authentication. The exclusive-create/sync file is not an atomic multi-file checkpoint/recovery protocol and is not the production llama serialized-state format. Original state files are not opened for writing.

## Validation results

`checks/tests.log`: **14 synthetic groups PASS**, peak RSS 18,244 KiB, zero live native resources at completion. Coverage includes name/shape/type failures; causal/SWA/global masks; exact residual identity for zero projections; deterministic nonzero execution and operator trace; consecutive SWA/global tokens; RoPE behavior; injected allocation/load/compute/publication failures; nonfinite data; scheduler ordering; checkpoint reload continuation and corruption rejection; all five quantization kernels; cross-layer/configuration state rejection; clamp and value-residual behavior.

`checks/sanitizer-tests.log`: **14 groups PASS** under AddressSanitizer and UndefinedBehaviorSanitizer, peak RSS 110,804 KiB; no sanitizer diagnostics. Instrumentation covers newly compiled code; the existing shared GGML libraries are not sanitizer-instrumented. Build warning logs are empty.

`checks/all-layer-descriptors.log`: **80/80 actual model layer plans validate**, without tensor execution. Python and shell syntax pass. `checks/adapter-tests.log`: original guard hash/import checks pass and unqualified references are rejected for both proposed layer classes. This is not runtime qualification of the adapter.

These are synthetic and structural results, not stock-reference numerical comparisons, real-model memory measurements or full-model correctness evidence.

## Guarded runtime prerequisites and first comparison

Latest preparation snapshot (`checks/runtime-readiness.json`): MemAvailable 3,946,516,480 bytes (3.68 GiB), swap free 15,656,566,784 bytes, SSD free 71,193,411,584 bytes (66.30 GiB), memory PSI 0. The unchanged 6 GiB MemAvailable admission gate fails. No real runner was invoked. Original guard SHA-256 remains `ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2`; baseline executable remains `329ebd647b04a2d6464461ddd94fc07a2691a948b2918d320ac1eb3703f9c101`.

For QIES100:

1. Obtain and independently qualify a stock Ricursive reference from the exact pinned source and model. Record executable/library hashes, compiler/configuration, CPU kernel settings, one thread, non-flash attention and F16 KV. Preserve exact source modifications needed to export reference intermediates. Merely building is insufficient.
2. Capture deterministic scaled `x0`, layer input, prior logical KV and 18 intermediate/output buffers from that reference. Begin with layer 0 SWA and layer 4 global; include consecutive positions and nonempty prior state. Hash every artifact and document tensor layouts. Never generate golden outputs from this candidate.
3. Compare normalized Q/K, transformed Q/K, mixed V, probabilities, attention/gate, FFN intermediates, final output and canonical written KV rows byte-for-byte. Do not compare unwritten padding or confuse stock transposed V storage with logical token-major layout. Any mismatch blocks acceptance; do not relax tolerances.
4. Independently qualify `launch_real.py` with harmless children under conditions satisfying the unchanged guard, covering normal exit, failure, descendants, termination and actual monitored filesystem. Only then create a verifiable adapter qualification receipt; none exists now.
5. Recheck host resources immediately before execution. Use the original QIES077 policy unchanged: entry RAM 6 GiB, free swap 4 GiB, free disk 30 GiB and entry PSI below 1, plus all original runtime thresholds. Inspect the original guard for the complete runtime policy. No weakening or policy overrides are supplied by the adapter.
6. Create a new persistent SSD attempt directory. Review reference manifest schema in `validate_reference()`. After prerequisites, the prepared command is `env -u LD_LIBRARY_PATH python3 -I -B qies099/launch_real.py --layer 0 --reference /absolute/qualified-layer-0.json --run-root /absolute/new-persistent-attempt-directory`; repeat separately for layer 4. The adapter calls the original guard, checks child status, verifies produced evidence and compares hashes. A zero wrapper exit alone is not acceptance. Do not invoke the native `--run` entry directly.

`--check` is a non-inference readiness diagnostic, not permission to bypass any prerequisite. Qualification status fields are evidence pointers requiring human/source review, not proof by themselves.

## Remaining gaps and risks

No embedding/output head/tokenizer/full-generation integration exists; no production Ricursive architecture registry was changed. Context is deliberately limited to 256: synthetic short-window masking is tested, but real 1024-window rollover and global long-context execution are not. There is no production KV eviction, multi-layer persistent-state integration, crash-consistent publication, or independently qualified stock oracle yet. Exact kernel-version compatibility, quantization repacking, attention precision/layout and CPU dispatch require direct comparison. The layer reader reuses the existing index metadata but does not adopt FlashNext recurrent-state semantics for Ripple.

QIES100 should qualify the reference and adapter, then perform narrowly guarded real-layer differential tests only on a suitable host/session. Full Ripple inference and correctness claims remain NO-GO until those comparisons, continuity checks and actual resource measurements pass.
