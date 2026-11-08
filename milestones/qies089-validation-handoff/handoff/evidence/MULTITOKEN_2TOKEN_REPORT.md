# MULTITOKEN_2TOKEN_REPORT.md
QIES-049: multi-token state continuity. Branch `feature/flashnext-multitoken-state`
(from golden 4e8a5aa). Date: 2026-11-04.

## Headline result

Two consecutive tokens (248045 @ pos 0, 846 @ pos 1) through all 48 layers with
full persistent hybrid state — **bit-exact against stock llama.cpp at every
checkpoint**:

- All 48 layer outputs at position 0: EXACT (unchanged from golden milestone)
- All 48 layer outputs at position 1: EXACT vs the two-token stock oracle
- Logits position 0: EXACT (248,320 f32); predicted token 846, logit 21.0668354
- Logits position 1: EXACT (248,320 f32); predicted token 198, logit 26.4632492
  (stock: argmax=198 logit=26.4632492 — identical)
- REPACK=ON and REPACK=OFF builds: bit-identical to stock and to each other at
  both positions
- Stock determinism verified: pos-0 logits of the two-token stock run are
  bit-identical to the one-token stock run

## Checkpoints

| Checkpoint | Scope | Result |
|---|---|---|
| A | Golden single-token regressions (new binaries) | PASS (2-layer bit-exact, ON==OFF) |
| B | 2 tokens, layer 0 (state update/export/restore/pos-1) | PASS, layer-0 pos-1 EXACT vs stock |
| C | 2 tokens, layers 0-1 (PLE predecessor metadata) | PASS, all EXACT |
| D | 2 tokens, layers 0-4 (full-attention layer 3) | PASS, all EXACT |
| E | 2 tokens, all 48 layers + logits both positions | PASS (OFF and ON builds) |

Comparison method: numpy element-wise float32 (`np.array_equal`) of every
per-layer boundary (`cortex_layer{L}_output_p{0,1}` vs stock `hc_combine-L`)
and both logits vectors. No tolerances used; every comparison is bit-exact.

## State components preserved (see MULTITOKEN_STATE_INVENTORY.md)

Captured with `llama_state_seq_get_data(ctx, blob, size, 0)` after each
layer's position-0 transaction and restored with
`llama_state_seq_set_data(..., 0)` before that layer's position-1 decode, in
caller-owned `std::vector<uint8_t>` surviving the ctx/model teardown:

- Full-attention KV cells (12 layers) including per-cell positions and
  `llama_kv_cell_ext` token metadata (PLE predecessors) — verified included in
  `llama_kv_cache::state_write/read` (llama-kv-cache.cpp:2212-2450)
- QSA indexer cache (`llama_memory_hybrid_idx::state_write` incl. mem_idx)
- Gated-DeltaNet SSM state rows (all recurrent layers; the per-layer graph
  touches only its own layer's rows)
- Convolution histories and the PLE conv history row (recurrent cache rows)
- Sequence/cell mapping (pos, seq bitmap) and recurrent rs_idx mapping

Not state (correctly NOT persisted): the HC stream activation handoff between
layers (per-token, caller-owned, recomputed each position); recurrent memory
is never reset between positions — it is restored from the captured blob.

PLE predecessor metadata: a recurrent layer's own context holds no attention
cells, so the stock KV-cell predecessor lookup is unavailable after reload.
Diagnostic seam: `CORTEX_FLASHNEXT_PLE_PREV="-1,248045"` (oldest-first, -1 =
missing) supplies the predecessor ids to
`llm_graph_input_qwen4exp_ple::set_input` for non-zero positions only
(position 0 keeps fresh-sequence behavior). Metadata sourcing only; PLE
mathematics unchanged.

Position-1 layer-0 input: token embedding of 846 (fetched via
`llama_cortex_token_embd_rows`), replicated over the 4 HC streams by the
graph — matching stock; layer-0 recurrent state restored, not reset.

## Defect found and fixed en route (diagnostic harness, not math)

Position-1 executions initially failed with garbage captures: the
two-print-per-name budget for host-capture diagnostics was a process-global
static (`host_print_count`), exhausted by the position-0 contexts, so
position-1 contexts never refreshed `flashnext_diag_owned` — the transaction
bridge then read stale storage. Fix: per-context print budget
(`flashnext_diag_print_count`, llama-context.h/.cpp). The ggml graph itself
computed correctly throughout (verified by direct artifact readback).

## Memory and runtime (two-token, 48 layers)

- State blobs: 48 x 118,070,660 B (~5.5 GiB total) held in caller storage
  between positions; measured process RSS peaked at ~2.8 GiB mid-run
  (system used 5.9 / 9.8 GiB, no swap pressure, no OOM).
- Weights remain streamed (pread arena); one layer resident at a time.
- Runtime: position 0 ~110 s + position 1 ~110 s (228 s total, OFF build;
  218 s ON build) on the 2-core host; ~2.3 s/layer/position including the
  ~0.7 s per-layer model reload.
- RSS wrapper artifact PEAK_RSS_KB=2296 measures the timeout wrapper, not the
  binary; binary RSS measured by direct /proc polling.

## Exact files changed (branch feature/flashnext-multitoken-state)

- `runtime-flashnext-layer-exp/examples/flashnext-owned-layer.cpp` — two-token
  chain mode (state capture/restore per layer, emb1 fetch, PLE_PREV handling,
  pos-1 exports + final head, diagnostics)
- `runtime-flashnext-layer-exp/examples/flashnext-stock-two-layer.cpp` —
  two-token stock oracle (decodes stock argmax at pos 1, exports both logits)
- `runtime-flashnext-layer-exp/src/models/qwen4exp.cpp` —
  `CORTEX_FLASHNEXT_PLE_PREV` predecessor supply (non-zero positions only)
- `runtime-flashnext-layer-exp/src/llama-graph.cpp` — `CORTEX_FLASHNEXT_DEBUG_INPUT`
  upload log (diagnostic)
- `runtime-flashnext-layer-exp/src/llama-context.{h,cpp}` — per-context diag
  print budget fix
- Snapshot diff: MULTITOKEN_STATE_DIFF.diff (sha256 86b4b901…)

Test tooling (kept outside git): run-chain48.sh (CHAIN_TWO_TOKEN=1),
compare_multitoken.py, regression_check.py.

## Remaining limitations

- Validated for exactly two tokens. Three or more tokens need the PLE
  predecessor list extended per position (mechanism in place: the env list is
  per-position data) and re-verification; state blobs for later positions
  accumulate (~5.5 GiB per position snapshot unless trimmed to per-layer
  slices).
- State blobs carry ALL layers' memory (global memory layout) — ~118 MB per
  layer per snapshot. Trimming to per-layer slices would need a custom
  serializer; not required for the two-token target.
- Continuous generation, chat, and GPU acceleration are out of scope.
- The per-layer reload cost (~0.7 s) is paid twice per layer (once per
  position); a persistent-context design could halve it but was not needed.

## Reproduction

```
CHAIN_TWO_TOKEN=1 ./run-chain48.sh mt-E ./build-repack-off/bin/llama-flashnext-owned-layer 48 mt-E.log
python3 compare_multitoken.py mt-E 48
STOCK_TWO_TOKEN=1 STOCK_LOGITS_OUT=$PWD/stock2tok_logits_p0.bin ./run-bounded.sh stock stock2tok <stock-binary> stock2tok.log
```
