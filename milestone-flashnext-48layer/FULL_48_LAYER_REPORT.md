# QUORTEX Full 48-Layer Streaming Expansion — Final Report
Date: 2026-11-03. Snapshot: /home/user/quortex-diag-20261103, git branch `quortex-48layer-expansion`.
Prerequisites: HC_DIVERGENCE_REPORT.md (bit-exact 2-layer engine), THREE_WAY_REPORT.md.

## Headline result

Full 48-layer, single-token (248045 @ pos 0), low-residency FlashNext inference
is COMPLETE and BIT-EXACT against the stock llama.cpp reference:

- All 48 layer outputs: EXACT (element-wise, 10240 f32 each) vs the stock oracle.
- Final logits (248,320 f32): EXACT vs the stock reference logits.
- Predicted token 846, logit 21.0668354 — identical to stock
  (STOCK_REF_LOGITS argmax=846 logit=21.0668354).
- REPACK=ON and REPACK=OFF builds produce bit-identical results at 48 layers.
- Golden 2-layer regression (CHAIN_1_2) still passes after all changes.
- Peak process RSS ~2.2 MB (dense /proc poll, 0.05 s); wall ~1.9 s/layer;
  no swap growth, no OOM.

## Source and binary provenance

- Source: src/runtime-flashnext-layer-exp @ fc4456e + accumulated snapshot diffs;
  this expansion adds CHAIN_48LAYER_DIFF.diff (sha256 0938e1f6…c43dd, 1304 lines):
  - examples/flashnext-owned-layer.cpp: generalized CHAIN_0_48 driver
    (configurable CORTEX_FLASHNEXT_CHAIN_0_N checkpoints, PLE token metadata for
    PLE layers, per-layer output exports, host-side final head + tiled output
    projection, flashnext_final_head_mix + flashnext_stream_output_matvec);
    layer-0 complete-chain flow now feeds captured router_logits into the
    transaction (was hitting the legacy unnormalized-weight fallback, 1.5e-8 ULPs);
    CHAIN_0_48 sets CORTEX_FLASHNEXT_LAYER_RANGE=1 from process start (fixes
    build-time null-deref of gated-out per_layer_token_embd in build_inp_ple).
  - examples/flashnext-stock-two-layer.cpp: exports stock reference logits
    (STOCK_LOGITS_OUT, argmax print). Stock math untouched.
  - src/flashnext-selected-moe.cpp, flashnext-layer-transaction.h,
    qwen4exp.cpp, llama-graph.cpp, llama-context.cpp: the previously validated
    HC-transaction + bit-exact MoE bridge fixes (HC_TX_FIX_DIFF).
- Binaries (rebuilt in place): build-repack-{on,off}/bin/llama-flashnext-owned-layer
  and llama-flashnext-stock-two-layer. Golden 2-layer binaries preserved at
  golden-2layer/ with SHA256SUMS.
- Model: Qwen3.8-Flash-Next-Uncensored IQ4_XS shard 1/3,
  sha256 28a99b12…673704. Layer geometry: 48 layers; recurrent (linear-attention)
  everywhere except full attention at layers 3,7,…,47 (interval 4); PLE at layer 1
  only; n_expert 512, top-10; expert_weights_scale 0 (absent); f_norm_rms_eps 1e-6.
- Stock oracle: dump-stock-off2 (48-layer full-graph captures, deterministic,
  exit 0) plus dump-stockref + stock_logits_p0.bin (bit-identical re-run with
  logits export). Single token, position 0 => zero initial recurrent and KV
  state in both pipelines, so no cross-layer state carryover is required.

## Checkpoint-by-checkpoint results (REPACK=OFF unless noted)

Every layer output compared element-wise (numpy float32, np.array_equal)
against stock hc_combine-<L>.bin. RESULT=PASS means every layer EXACT.

| Checkpoint | Wall time | Peak RSS (wrapper) | Result |
|---|---|---|---|
| N=2 (regression) | 4.1 s | 2.3 MB* | PASS (layers 0-1 EXACT) |
| N=4 | 8.2 s | 2.3 MB* | PASS |
| N=8 | 15.4 s | 2.3 MB* | PASS (incl. full-attn layer 3, 7) |
| N=16 | 30.2 s | 2.3 MB* | PASS (incl. layer 15) |
| N=32 | 59.8 s | 2.3 MB* | PASS |
| N=48 | 91.1 s | 2.3 MB* | PASS, logits EXACT (248320), token 846 = stock |
| N=48 REPACK=ON | 93.8 s | 2.3 MB* | PASS, logits ON==OFF EXACT |

*Wrapper-measured (VmHWM of the timeout wrapper); independent dense poll of the
real binary gave peak VmRSS ~2.2 MB per run. Weight streaming uses the OS page
cache over the 44.7 GB shard (host: 9.8 GB RAM, ~5 GB cache steady-state).

Golden 2-layer regression after all rebuilds: regression_check.py RESULT PASS.

## Architecture of the 48-layer pipeline (per layer L = 0..47)

1. Release previous ctx/model; set LOW_RESIDENCY_LAYER=L, LAYER_RANGE=1,
   LAYER_START=L, LAYER_END=L+1, PRE_MOE=1, INTERNAL_TENSOR_DIAGNOSTICS=1.
2. llama_model_load_from_file + llama_init_from_model: one resident layer,
   pread-arena streaming, no dummy-weight dereference (established reload pattern).
3. Caller-owned 10,240-float HC-wide handoff fed via embd batch; PLE layers
   receive token 248045 metadata (only layer 1 in this model).
4. Graph executes through the segment-A/pre-MoE boundary; the corrected host
   bridge (FFN-phase diag_tx captures + bit-exact ggml MoE mirror + ggml
   segment-B combine) completes the layer. Layer output exported as
   cortex_layer<L>_output_p0.bin.
5. Layer 0 runs the stock token-embedding path; layers 1-47 run the range path.

Final head (after layer 47): host-side ggml mirror of the stock final HC mixer
(build_hc_mix unfused, il=-1: per-stream RMSNorm(1e-6) -> output_hc_norm ->
output_hc_down -> silu(scale 1/4) -> output_hc_up -> sigmoid gate -> per-stream
product -> mean over 4 streams) followed by a tiled ggml mul_mat output
projection streamed over output.weight (IQ4_XS, 248,320 rows). Bit-exact by
construction; verified EXACT vs stock logits.

## Memory and residency

- One layer resident at a time by construction (low-residency probe binds
  blk.<L>.* only); all weight bytes stream through pread tiles.
- Measured peak binary RSS ~2.2 MB; page cache carries the streaming load.
- No unbounded growth across 48 reloads (timings scale linearly, RSS flat).

## Remaining limitations

- Single token at position 0 only. Multi-token generation is NOT yet validated:
  it requires recurrent-state and KV-state carryover across per-layer reloads
  (llama_state_seq_get/set_data is the established mechanism) and correct
  predecessor-token metadata for PLE n-grams. Explicitly out of scope here.
- Per-layer reload costs ~0.7 s of the ~1.9 s/layer; a persistent single-load
  multi-graph variant could cut this but was not needed for validation.
- The chain driver depends on CORTEX_* env discipline; the diag_hc_* names
  remain ambiguous between full and pre-MoE graphs (bridge uses diag_tx_*).
- Logits equivalence is proven for the IQ4_XS output weight on this model;
  the tiled projection handles arbitrary ggml row types but is validated only
  for this configuration.
- 2-layer CHAIN_1_2 mode remains the golden reference path; CHAIN_0_48 is the
  generalization validated by this report.

## Reproduction

- Checkpoints: ./run-chain48.sh <tag> <binary> <N> <log> (env: CHAIN_0_48=1,
  CHAIN_0_N, DIAG_DUMP, FLASHNEXT_COMPARE_DIR).
- Compare: python3 compare_chain48.py <tag> <N>  (layers vs stock, logits if N=48).
- Golden regression: ./run-bounded.sh chain chain-{on,off} … && python3 regression_check.py.
- Captures: captures-chain48-{2,4,8,16,32,48,48on}/, dumps dump-chain48-*/,
  logs chain48-*.log/.stdout/.meta, stock_logits_p0.bin, dump-stockref/.
