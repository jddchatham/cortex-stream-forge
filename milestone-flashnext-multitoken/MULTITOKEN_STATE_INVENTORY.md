# MULTITOKEN_STATE_INVENTORY.md
QIES-049: multi-token state continuity for the 48-layer FlashNext streaming engine.
Model: Qwen3.8-Flash-Next (QWEN4EXP), 48 layers, dsv4 HC streams (hc=4, n_embd=2560).
Date: 2026-11-04. Branch: feature/flashnext-multitoken-state @ 4e8a5aa (golden).

## 1. Stock state ownership (traced from source)

Memory class: `llama_memory_hybrid_idx` (llama-memory-hybrid-idx.{h,cpp}), created by
`llama_model::create_memory` for LLM_ARCH_QWEN4EXP with GLOBAL layer filters
(llama-model.cpp:3066-3091): filter_attn = all non-recurrent layers (12 full-attn:
3,7,...,47), filter_recr = all recurrent layers (36), filter_idx = all full-attn
(QSA indexer). The graph range does NOT filter memory: every [L,L+1) context
allocates memory for ALL layers (observed: llama_kv_cache "12 layers", 1.5 MiB;
recurrent rows for 36 layers; this is why a per-layer context state blob is
~118 MB).

Components and owners:

| Component | Owner | Contents | Dimensions | Lifetime | Update mechanism |
|---|---|---|---|---|---|
| Full-attention KV | llama_kv_cache (mem_attn) | K cells (f16) per full-attn layer | 256 cells x 12 layers, 1.5 MiB total | ctx | graph writes K/V at decode; cells carry pos, seq bitmap, and `llama_kv_cell_ext` (ext.tok for PLE predecessors) |
| QSA indexer cache | llama_kv_cache (mem_idx) | indexer keys + pooled cells | mirrors attn layout | ctx | graph-side, with mem_idx_stale tracking |
| Gated-DeltaNet SSM state | llama_memory_recurrent (mem_recr) | per recurrent layer: SSM state row (s) | F32, ~3.2 MB/layer x 36 | ctx | in-place graph update (state rows are graph outputs) |
| Convolution history | llama_memory_recurrent | per recurrent layer: recurrent row (r) incl. conv tail | part of the same row | ctx | in-place graph update via build_conv_state_at |
| PLE conv history | llama_memory_recurrent | "The PLE conv history is a row of the recurrent cache" (qwen4exp.cpp:163) | row of layer 1 | ctx | in-place graph update |
| PLE predecessor metadata | llama_kv_cell_ext inside mem_attn cells | token ids of previous positions | per cell | ctx | `apply_ubatch` stores ubatch.token into ext.tok (llama-kv-cache.cpp:1143) |
| HC stream activation handoff | NOT state - caller-owned | 10,240 F32 between layers | per token | transaction | segment-B output (do not persist between tokens) |
| Rollback snapshots | mem_recr rs_idx / n_rs_seq | per-ubatch recurrent snapshots | cparams.n_rs_seq | transient | used by transaction rollback; re-established on restore |

## 2. Serialization coverage (llama_state_seq_get/set_data)

`llama_context::state_seq_write_data` -> `llama_memory_hybrid_idx::state_write`
(hybrid-idx.cpp:259) writes, in order: hybrid state (KV via
`llama_kv_cache::state_write` + recurrent via `llama_memory_recurrent::state_write`),
then mem_idx. `state_read` mirrors the order ([TAG_HYBRID_IDX_STATE]).

Verified inclusion of every component required for token-to-token continuity:

- KV data + cell metadata: `state_write_meta` writes pos, n_seq_id, seq id, and
  the full `llama_kv_cell_ext` per cell (llama-kv-cache.cpp:2212+). `state_read_meta`
  restores them through a synthetic ubatch + `apply_ubatch` and then puts the saved
  ext back via `cells.ext_set` (llama-kv-cache.cpp:2389-2450). ext.tok (PLE
  predecessor tokens) and M-RoPE positions survive the round trip.
- Recurrent rows: `llama_memory_recurrent::state_write` serializes cell metadata
  plus the data rows (SSM s + recurrent r) for the cells of the sequence
  (llama-memory-recurrent.cpp:784-862). rs_idx mapping re-established on read.
- Indexer: mem_idx state_write/read included (hybrid-idx.cpp:266,291).

Conclusion: `llama_state_seq_get_data()` / `llama_state_seq_set_data()` DO capture
all state needed for two-token continuity. The same-ctx rollback path is already
exercised by the transaction tests (LAYER2_FAILURE_ROLLBACK=PASS). Cross-ctx use is
safe when the target context has an identical memory configuration (same model,
n_ctx, n_seq_max, no MTP context): blob size and layout then match, and only the
restored sequence's cells are touched (state_read_meta begins with seq_rm of the
destination sequence). Empirical validation: capture after position 0 per layer,
restore before that layer's position-1 decode, re-capture after (Section 4).

Known serialization caveat (acceptable here): the blob carries ALL layers' memory
(global filters), so each per-layer blob is ~118 MB and 48 of them (~5.7 GB)
reside in caller storage between positions. This is a memory cost, not a
correctness risk. Whole-state restore touches only seq 0 cells of this ctx.

## 3. Proposed persistence method

Per layer L in 0..47, per position transition:
1. After layer L's position-0 transaction (graph synchronized, state committed),
   `llama_state_seq_get_data(ctx, blob_L, size, 0)`; keep blob_L in caller-owned
   std::vector<uint8_t>.
2. Free ctx/model (established reload pattern; blob_L survives).
3. At position 1, reload the identical [L,L+1) configuration, then
   `llama_state_seq_set_data(ctx, blob_L, size, 0)` BEFORE decoding.
4. Decode position 1 (batch pos=1, seq 0, embd=pos-1 handoff, token=846), then
   re-capture blob_L (position-1 state; retained for future extension, not needed
   for the 2-token target).

PLE predecessor metadata: layer 1's PLE binder reads predecessors from mem_attn
cells (`mctx->get_prev_tokens`), but a recurrent layer's own context never writes
attn cells, so at position 1 its predecessor lookup would see an empty cache
(stock sees token 248045 in the shared full-graph cache). Diagnostic seam change:
`CORTEX_FLASHNEXT_PLE_PREV` env supplies the predecessor token ids (oldest-first,
-1 for missing) to `llm_graph_input_qwen4exp_ple::set_input` in LAYER_RANGE mode
instead of the KV lookup. Metadata sourcing only; no change to PLE mathematics.
For position 1: CORTEX_FLASHNEXT_PLE_PREV="-1,248045" (n_gram-1 = 2 entries).

Layer-0 position-1 input is the token embedding of 846 replicated over the 4 HC
streams (stock `ggml_repeat_4d` of the embedding row), not the position-0
activation. Recurrent/KV state is NOT reset - it is restored from blob_0.

## 4. Validation plan checkpoints

A: golden single-token regressions (2-layer + 4-layer vs stock).
B: 2 tokens, layer 0 only (state update/export/restore/pos-1 execution).
C: 2 tokens, layers 0-1 (PLE predecessor metadata).
D: 2 tokens, layers 0-4 (incl. full-attention layer 3).
E: 2 tokens, all 48 layers; logits at positions 0 and 1 vs stock.

Bit-exact target: np.array_equal on every boundary and both logits vectors.
