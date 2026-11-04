# QUORTEX HC Scalar Divergence — Root Cause and Repair
Date: 2026-11-03 (late session). Snapshot: /home/user/quortex-diag-20261103
Reference: THREE_WAY_REPORT.md (the "HC streams-combine scalar divergence")

## 1. Exact root cause

The divergence was NOT in graph numerics. The ggml graph was bit-exact all along.
Root cause: a transaction-plumbing bug in the host-side segment-B bridge
(examples/flashnext-owned-layer.cpp, capture_transaction_router()).

- The bridge builds its FFN-combine (residual + complete_moe * w, w = 2*sigmoid(inject/4))
  from tensors looked up BY NAME in the global diag-capture map
  (flashnext_diag_tensors, llama-context.cpp:3440, LAST-wins per name).
- In the STOCK full graph, the names `diag_hc_residual-N` / `diag_hc_inject-N` are
  registered TWICE per layer (attention combine, then FFN combine); last-wins makes
  them refer to the FFN combine — the correct transaction values.
- In the CHAIN segment-A/pre-MoE graph, the layer returns at the segment-A boundary
  BEFORE the FFN combine, so those names refer to the ATTENTION combine. The bridge
  therefore combined with:
  - residual = layer INPUT (attention-phase residual) instead of post-attention residual
  - inject   = attention inject [-41.63...] instead of FFN inject
- Evidence: chain handoff was exactly reconstructible as
  layer_input_res + w(attn inject) * MoE (stream-consistency 4.8e-7), while the
  segment-A boundary artifact already contained stock-bit-exact post-attn residual
  and FFN inject. Post-attn state matched stock bit-for-bit; only the bridge's
  combine inputs were wrong. The bridge's own reference self-checks
  (FINAL_HC_REFERENCE_PASS) could not catch it because they consume the same
  corrupt transaction values.

Secondary ULP-level gap (found while validating): the sparse-MoE bridge computed
softmax weights in double, silu with host libm exp, and accumulated experts in
double — while the stock graph uses ggml float32 ops (incl. the vectorized
ggml_v_expf approximation in ggml_swiglu_split). This left ~1e-8 residual error
even with correct inputs.

## 2. Minimal correction (implemented, snapshot only)

Diff: HC_TX_FIX_DIFF.diff (sha256 e7ebfb89f5cfe81a2fd2c15e4aa037c8d314e2b0081c378eefd09fcfc7726103)

1. src/models/qwen4exp.cpp — in the segment-A boundary branch, register dedicated
   transaction captures cb(res_hc, "diag_tx_residual", il) and cb(inject, "diag_tx_inject", il)
   (FFN-phase values, computed at lines ~811-816 before the boundary return).
2. src/llama-graph.cpp + src/llama-context.cpp — whitelist the `diag_tx_` prefix
   (both funnel sites).
3. examples/flashnext-owned-layer.cpp — capture_transaction_router() prefers
   diag_tx_residual-N / diag_tx_inject-N, falls back to the old names (full graphs).
4. src/flashnext-selected-moe.cpp — bit-exact MoE bridge:
   - weights now computed in a ggml graph mirroring stock build_moe_ffn
     (soft_max over the 512 captured logits -> get_rows by selected ids ->
     sum_rows -> clamp(6.103515625e-5) -> div; expert_weights_scale is 0 for this
     model so the scale step is absent, matching stock);
   - per-expert activation computed with ggml_swiglu_split (same fused kernel as
     stock) instead of the host libm formula;
   - expert aggregation in float32, product-then-sequential-add in top-k order,
     mirroring ggml_mul + the sequential ggml_add chain.
   - legacy fallback: callers that never populate router_logits keep the previous
     routing_weights behavior.
5. src/flashnext-layer-transaction.h — added router_logits[512]; example copies
   captured router logits into the transaction.

Stock graph mathematics untouched. No changes to attention, convolution, PLE,
quantized kernels, or the allocator. The pre-existing ggml-based segment-B combine
(flashnext-segment-b.cpp) already mirrored stock build_hc_combine op-for-op and
needed no change.

## 3. Before/after numerical comparison (numpy element-wise, f32)

                                    BEFORE                          AFTER
layer-0 out vs stock   max_abs 0.6215 rms 0.01242      EXACT (0 / 0)
layer-1 out vs stock   max_abs 0.3466 rms 0.04522      EXACT (0 / 0)
tx scalars L0          wrong (attn-phase values)       EXACT vs stock FFN combine
chain ON vs OFF        EXACT                          EXACT
intermediate fix state (correct tx values, host MoE): layer-0 max_abs 7.45e-9,
layer-1 5.96e-8 — ULPs from host-side MoE math, eliminated by the ggml mirror.

Validation runs: chain-on4/chain-off4, exit 0 (~4.3-4.5 s, RSS ~2.3 MB),
GENUINE_CHAIN_0_1=PASS both configs. Stock reference: dump-stock-off2 (REPACK=OFF,
deterministic, exit 0).

## 4. Layer equivalence verdicts

- HC scalars: the eight transaction scalars (FFN inject + implied w, both layers)
  match the stock reference bit-for-bit (tx_inject-0 == stock hc_inject-0 EXACT;
  tx_residual-0 == stock diag_hc_residual-0 EXACT; layer-1 likewise).
- Layer-0 output: chain == stock BIT-FOR-BIT (10240 f32).
- Layer-1 output: chain == stock BIT-FOR-BIT (10240 f32).
- REPACK=ON == REPACK=OFF: bit-exact on every compared tensor, both layers.
- Low-residency streaming: operational (per-layer reload + pread provider, exit 0).

## 5. Remaining blockers to full 48-layer inference

- Chain currently validated for layers 0-1 only (per mission scope). Extending to
  48 layers requires the established per-layer reload pattern generalized
  (chain_1_2_3-style) — no new numerical risk expected since layer-1 already
  exercises the full handoff path, but each layer boundary needs the same
  diag_tx_* capture discipline (now in place for any segment-A/pre-MoE graph).
- Stock REPACK=ON full-graph run remains environment-blocked (9.8 GB RAM vs 44.7 GB
  shard with full-model repack buffers); irrelevant to the equivalence result,
  which is carried by the byte-exact ON/OFF chain comparison.
- The old diag_hc_* names still mean "FFN combine" in full graphs and "attention
  combine" in pre-MoE graphs. The bridge no longer depends on them, but any other
  consumer of those names must respect the same ambiguity.
- Multi-token / multi-position streaming not yet exercised (single token 248045,
  pos 0). Conv/SSM state carryover across tokens is implemented in the seam but
  was out of scope here.

## Provenance

Binaries rebuilt in place: build-repack-{on,off}/bin/llama-flashnext-owned-layer
(plus llama-flashnext-stock-two-layer relink). Model shard sha256
28a99b125cb905fc3bdc06baf6266a56b0cbd1b27e833b9142acc2017b673704.
Dumps: dump-chain-{on,off} (run4), captures-chain-{on,off}; logs chain-{on,off}4.log;
stock reference dump-stock-off2. Note: run-bounded.sh writes captures to
captures-<TAG>; tags used here were chain-on/chain-off.
