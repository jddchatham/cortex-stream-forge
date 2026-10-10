# Qwen3.8 Flash-Next Abliterated — Cortex experiment

This is a new, isolated experiment. Existing Qwenny runtimes, MTP work,
shortcuts, models, and handoffs are not modified.

## Download checkpoint

Repository observed from the active downloader:

`wy2pay/Qwen3.8-Flash-Next-abliterated-GGUF-IQ4XS`

Destination:

`/home/user/models/qwen38-flashnext-abliterated`

Manifest observed in the Hugging Face tree cache:

- `Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf`
- `Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00002-of-00003.gguf`
- `Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00003-of-00003.gguf`
- `Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf`
- `Qwen3.8-Flash-Next-lora-v1.gguf`
- `mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf`

At the time of this checkpoint the download is still active. The projector
has completed; main shards and auxiliary files remain resumable `.incomplete`
files. No hashes or model-validation result is claimed until completion.

## Preliminary architecture clue

The repository README identifies the model as approximately 125B total / 6B
active MoE, architecture `qwen4_exp`, with 48 layers: 36 linear-attention and
12 full-attention layers. This is only repository metadata, not a Forge
fingerprint; GGUF metadata inspection remains pending.

## Next safe steps

1. Wait for objective download completion and hash every shard.
2. Parse GGUF metadata/index without loading tensor payloads.
3. Determine router, expert count/top-k, and tensor offsets.
4. Build the isolated `runtime-flashnext-exp` adapter and router-only proof.

## Download validation

Validation completed. The three main shards total `97,473,155,200` bytes.
SHA256 results are recorded in `flashnext-sha256.txt`:

- shard 1: `28a99b125cb905fc3bdc06baf6266a56b0cbd1b27e833b9142acc2017b673704`
- shard 2: `2a309e0b112fde96ba3bcdb597cac664762cea2ca23f7b16ec92eee28c5572d9`
- shard 3: `ebc43c58e2eaeba1d5bdf62c8cb1f0eac198c4dc01941f771921edeebf574bc3`
- MTP draft: `ff803fb437e90567c27991fecae2537ea1ed63e8842b0399aa53820f02ae8882`
- LoRA GGUF: `42e4c87b2e6f24ceae34e0740253b5edd0871ed2ab0f798d1ff534efd9b8ac9f`
- vision projector: `f0f352a97a62a057f3aecdb597cac664762cea2ca23f7b16ec92eee28c5572d9`

No incomplete files remain. Available disk after validation is approximately
202 GiB.

## Metadata fingerprint

The first shard contains the GGUF metadata and all three shards were indexed
without loading tensor payloads into RAM. The actual fingerprint is distinct
from dense Qwen3.8/Qwen35:

- architecture: `qwen4exp`
- main layers: 48
- hidden/input embedding: 2560
- output embedding width: 2560
- attention heads/KV heads: 24/2
- vocabulary: 248,320
- context length: 262,144
- expert count: 512
- experts selected per token: 10
- expert FFN length: 640
- shared expert FFN length: 640
- full-attention interval: every fourth layer (`3, 7, 11, ... 47`)
- linear/recurrent layers: remaining 36 layers
- recurrent state metadata: SSM state 128, convolution kernel 4, inner size 6144,
  timestep rank 48, group count 8
- quantization file type: GGUF type 30 (`IQ4_XS`)
- indexed tensors across main shards: 1,224

The complete metadata/tensor index is in:

`flashnext-tensor-index.json`

The main model has 144 expert tensors: three expert projections per layer for
all 48 layers. Their combined payload is `65,431,142,400` bytes. Router input
weights and shared-router gates total `252,149,760` bytes. A representative
layer-0 router is:

`blk.0.ffn_gate_inp.weight`

shape `[2560,512]`, F32, 5,242,880 bytes, at shard-1 offset `30,618,717,344`.
Its shared-expert gate is 10,240 bytes. Representative layer-0 expert
matrices are:

- `ffn_down_exps`: `[640,2560,512]`, 471,859,200 bytes
- `ffn_gate_exps`: `[2560,640,512]`, 445,644,800 bytes
- `ffn_up_exps`: `[2560,640,512]`, 445,644,800 bytes

The expert dimension is the third GGML dimension, so each expert occupies a
contiguous slice of each tensor. The metadata proves arbitrary expert slices
can be addressed by shard, offset, and per-expert stride without reading the
other experts.

## Router-first finding

The upstream `qwen4exp` graph constructs the sparse FFN as:

`cur -> ffn_gate_inp -> build_moe_ffn(top-k=10) -> selected expert projections -> combine`

and separately computes a shared expert path. Therefore the router weights are
structurally available before expert projections. The minimum weight payload
for the router decision at a layer is the 5,242,880-byte F32 router matrix
plus its 10,240-byte shared gate; no 65 GB expert payload is needed to compute
the router scores.

However, a real routing decision still requires the layer hidden activation.
No router-only execution claim is made yet: producing that activation requires
executing the preceding hybrid layer(s), and the current normal graph binds
expert tensors through `ggml_mul_mat_id`. The next isolated implementation
step is a router-first graph/provider boundary that executes the router with
only router/shared non-expert tensors, then reads selected expert slices.

## Current status

- FLASHNEXT DOWNLOAD: PASS
- SHA256 VALIDATION: PASS
- FINGERPRINT: PASS
- ROUTER-FIRST: structurally possible; real execution not yet proven
- SELECTED-EXPERT pread: NOT ATTEMPTED
- EXPERT TENSOR WINDOW: NOT ATTEMPTED
- ONE-LAYER SPARSE: NOT ATTEMPTED
- MULTI-LAYER: NOT ATTEMPTED
- FULL LOW-RESIDENCY: NOT ATTEMPTED
- BATCHED PREFILL: NOT ATTEMPTED
- MTP AUDIT: NOT ATTEMPTED
- INTERACTIVE CHAT: NOT ATTEMPTED

No conventional full-model load was attempted because the 97.5 GB model is
larger than available RAM. Existing working runtimes and shortcuts remain
untouched.

## Forge build checkpoint

The isolated copied runtime configures and builds its existing `qwen4exp`
support successfully without loading the model:

`runtime-flashnext-exp-build/bin/libllama.so`

SHA256: `42a9a35bfba3cec1e81f9bb0cecaa41fd35a65995c6b2a5e6a44879e28ba3eb4`

Source `src/models/qwen4exp.cpp` already contains the upstream hybrid/MoE
graph, including recurrent attention, periodic full attention, shared experts,
and `build_moe_ffn`. No Flash-Next-specific source modification has been made
yet.

## Exact router/weight boundary

`build_moe_ffn` in `src/llama-graph.cpp` first computes router logits with
`build_lora_mm(ffn_gate_inp, cur)`, applies softmax, selects the top 10 expert
IDs with `ggml_argsort_top_k`, and only then constructs expert operations via
`build_lora_mm_id` / `ggml_mul_mat_id`. This confirms the mathematical order
is router-first.

The current GGML graph, however, represents the complete expert tensors as
graph inputs to `ggml_mul_mat_id`; it does not provide a callback that can
pread only the runtime-selected third-dimension slices after top-k executes.
Therefore the next generic Forge change must split each sparse layer into two
execution phases: router/state phase, then selected-expert phase. Attempting a
normal native model load would defeat the experiment by registering the full
97.5 GB expert payload.

Current scientific answer:

**The architecture can determine expert IDs before expert projections, but
Cortex has not yet proven a real runtime route or selected-expert pread.**

The exact metadata index is `flashnext-tensor-index.json` (1,224 tensors,
300 KiB). It records shard, offset, type, dimensions, and byte lengths for
arbitrary expert tensors.

## Router-only implementation and hybrid boundary

The isolated branch now contains an opt-in phase-A graph boundary:

`CORTEX_FLASHNEXT_ROUTER_ONLY=1`

It excludes `_exps` tensors from the initial layer working set and stops the
Qwen4Exp graph after the real layer-0 attention/mixer path, before expert
projections, exposing the router logits as the graph output. A probe was added:

`examples/flashnext-router-probe.cpp`

The first attempt exposed a sequence of invalid inputs in the generic full
model graph. These were diagnosed rather than bypassed:

1. `llama_kv_cache::set_input_k_idxs()`

called from `llm_graph_input_mem_hybrid::set_input()` during
`llama_decode()`, ending at `ggml-backend.cpp:345` with `GGML_ASSERT(buffer)`.
2. the indexer k-pool input, which was still registered although no QSA/full
attention layer is reached;
3. `llm_graph_input_pos::set_input()`, because the recurrent layer-0 route
does not consume positional attention input;
4. an output-size issue in the probe, because the public logits accessor
expects vocabulary width while the router has width 512.

The isolated route-only graph now keeps the normal hybrid recurrent-memory
allocation, but suppresses only the unused attention K/V/index-mask, k-pool,
position, PLE, and output-id inputs. The router output is placed in a
vocabulary-sized graph tensor for safe API extraction; the first 512 values
are the real router logits and the remaining values are zero.

This is a genuine two-phase integration blocker, not evidence that the router
requires expert weights. The smallest next fix is a Qwen4Exp-specific partial
graph input mode that omits unused attention/KV inputs and exposes the layer-0
recurrent state/hidden output directly. No synthetic hidden vector or guessed
expert IDs was used.

The rebuilt probe now executes the real layer-0 router without reading any
expert payload. For token `248045` (the deterministic first input), it
materialized:

```text
router hash: 85e97937dbefb143
top-10 IDs: 136, 77, 317, 257, 226, 142, 175, 414, 373, 131
logits: -3.51731157, -4.32100296, -4.35474014, -4.61050797,
        -4.64195824, -4.71265030, -4.71469593, -4.74369478,
        -4.74527693, -4.76322651
```

The normal hybrid state allocation reported by the runtime is 112.57 MiB
recurrent state and 1.50 MiB attention KV storage for the probe context. The
low-residency loader reports a 50,238,336-byte layer-0 working set and
337,715,200 bytes of non-layer globals. `_exps` payloads remain excluded;
the probe does not request expert data before routing. The log's
`create_tensor: loading ... _exps` lines are metadata/tensor-descriptor
registration, not payload reads.

Current result:

- FLASHNEXT ROUTER MATERIALIZATION: PASS
- FLASHNEXT ROUTER-FIRST: PASS (route materialized; reference score comparison pending)
- selected-expert pread: PASS (frozen-ID disk verifier; sparse math not yet connected)
- expert Tensor Window: not attempted

The isolated router probe source and library were rebuilt; no existing runtime
or model was changed. Selected-expert loading remains intentionally stopped at
this checkpoint pending the requested reference route comparison.

## Frozen router to selected-expert pread checkpoint

The router IDs above are treated as immutable phase-B input. The isolated
`tools/flashnext_selected_expert_pread.py` verifier reads the three combined
layer-0 expert tensors by legal expert slices, using the GGUF index and
`pread()` on the shard file. It does not instantiate `ggml_mul_mat_id` or a
full 512-expert collection.

For IDs `[136, 77, 317, 257, 226, 142, 175, 414, 373, 131]`:

- all layer-0 expert payload: `1,363,148,800` bytes;
- selected expert payload and actual pread bytes: `26,624,000` bytes;
- avoided: `1,336,524,800` bytes;
- reduction: `51.2x` (98.046875% avoided);
- verifier peak RSS: `22,646,784` bytes.

Each selected `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` slice was
read from its indexed shard offset and hashed. No unselected expert slice was
requested.

This proves selected-expert disk addressing, but not one-layer sparse
correctness. The normal graph still uses `ggml_mul_mat_id`, whose contract
requires the complete expert tensor collection. The isolated branch does not
yet have the explicit per-expert executor (`gate/up → activation → down →
route-weighted accumulate`, including shared-expert semantics), nor does it
expose the exact layer-0 hidden row to that executor. The one-layer
reference/candidate comparison is therefore still blocked, rather than being
declared passed. No router code was changed for this selected-pread checkpoint.

## Per-expert executor attempt

The isolated branch now contains:

- `examples/flashnext-single-expert.cpp`: creates ordinary 2-D quantized
  descriptors for one selected gate/up/down slice and runs normal
  `ggml_mul_mat`, SiLU, elementwise multiply, and down projection;
- `tools/flashnext_selected_expert_compute.py`: feeds the frozen ten IDs and
  normalized route weights through that executor sequentially and accumulates
  the ten outputs;
- `examples/flashnext-hidden-probe.cpp`: exports the exact pre-MoE hidden row
  from the route boundary as 2,560 float32 values.

The hidden export is shape `[2560]`, F32, FNV-1a
`34259a1e69a780eb`, with min `-1.91438377`, max `7.5178504`, mean
`0.0157045485`, and L2 `25.3934374`.

The standalone descriptor path successfully computes expert 136 and all ten
selected experts using only their 26,624,000 bytes. The accumulated selected
expert output hash is `822f120d0c7f2369` after normalizing the frozen route
weights to sum to one.

This is an execution proof, not yet a correctness PASS: the branch still
lacks an independently captured same-hidden-input reference MoE output and
the shared-expert path. Consequently the required first-divergence comparison
cannot yet be made. The new blocker is no longer `ggml_mul_mat_id`; it is the
reference/output boundary needed to validate the standalone result and then
integrate the shared expert exactly. Tensor Window work has not started.

An isolated normal-graph reference capture was added, but it fell through to
the ordinary full tensor-load path instead of the intended layer-0-only load.
It was stopped before execution to avoid loading the 97.5 GB model. Thus no
candidate/reference correctness PASS is claimed. The next fix is confined to
that reference loading boundary; the candidate executor and frozen router
remain unchanged.

## Low-residency independent reference validation

The full-model reference probe was replaced for this stage by
`examples/flashnext-reference-expert136.cpp`. It constructs only ordinary
GGML operations over the frozen hidden row and expert-136 gate/up/down slices;
it never initializes `llama_model` and never loads the 97.5 GB model.

Reference expert 136 intermediates:

```text
gate  = 31d73a1f08f00770
up    = 37457f4b45568369
silu  = 38861aba86214a82
down  = 5d5dcaad1155dbf1
```

The candidate expert-136 down hash is identical: `5d5dcaad1155dbf1`.
All-ten independent reference executions also matched:

```text
reference selected hash = 822f120d0c7f2369
candidate selected hash = 822f120d0c7f2369
max_abs = 1.07948555e-08
max_rel = 5.90142114e-08
```

The isolated shared-expert executor now runs only the layer-0 shared
gate/up/down tensors. Its output hash is `63055fc5061b323b`. Independent
shared/reference comparison and final selected-plus-shared validation remain
pending; no full-model load was used.

## Shared reference and complete isolated MoE checkpoint

The independent `flashnext-reference-shared` harness now constructs the
shared graph separately from the Cortex shared executable and captures:

```text
gate        = 70f52c8f4b9a2490
up          = e51256fbbe521419
silu        = 817a713ae0422564
gate*up     = 6c65d7018898474c
down        = 47f212eac6a1dc36
control     = 62eed26d951b54ab
sigmoid     = 1e00f4939ffb2624
contribution= 63055fc5061b323b
```

The independent shared contribution and Cortex shared contribution have the
same hash `63055fc5061b323b`. Combining that contribution with the already
validated selected vector produces an exact isolated MoE checkpoint:

```text
FINAL_MOE_REFERENCE = c3485d69e3535a65
FINAL_MOE_CORTEX    = c3485d69e3535a65
max_abs = 0
max_rel = 0
```

The isolated one-layer sparse result is therefore PASS with zero unselected
expert reads and without loading the full model. Tensor Window and complete
layer-0 recurrent/residual integration remain the next uncompleted stages.

## Current shared/complete-layer status

The layer-0 shared tensor inventory is confirmed:

- shared gate/up: `IQ4_XS`, 870,400 bytes each;
- shared down: `IQ4_NL`, 921,600 bytes;
- shared sigmoid gate: F32, 10,240 bytes.

The isolated shared executor reads only these four tensors and computes the
Qwen4Exp shared path. Its deterministic output hash is
`63055fc5061b323b`. However, an independently constructed shared reference
with intermediate captures has not yet been completed, so the shared result
is not promoted to PASS. The selected-only reference remains proven with
hash `822f120d0c7f2369`; no unselected expert reads occurred.

Complete selected-plus-shared MoE comparison, Tensor Window, and complete
layer-0 validation remain blocked on that independent shared reference. The
full 97.5 GB model was not loaded.

## FLASHNEXT_SPARSE_MOE_GOLDEN — FROZEN — DO NOT MODIFY

This checkpoint supersedes the earlier pending-status note above. The
independent shared reference subsequently matched the Cortex shared output
bit-for-bit, and the complete isolated MoE hash is `c3485d69e3535a65`.

Frozen values: hidden hash `34259a1e69a780eb`; router hash
`85e97937dbefb143`; selected IDs `136,77,317,257,226,142,175,414,373,131`;
selected hash `822f120d0c7f2369`; shared hash `63055fc5061b323b`; complete
MoE hash `c3485d69e3535a65`; selected bytes `26,624,000`; all layer expert
bytes `1,363,148,800`; unselected reads `0`; full model loaded `NO`.

The frozen implementation was copied to `runtime-flashnext-window-exp` for
the next Tensor Window experiment. No further changes belong in this source
tree.
## 2026-10-07 — Segment-B HC layout validation

The standalone Segment-B graph was debugged with bounded control graphs.
The apparent HC mismatch was not a GGML reshape/repeat ordering error. The
actual graph layout is the expected contiguous stream-major layout:
`[hidden, stream, token]`, with `hidden` the fastest dimension.

Controls run in the active `runtime-flashnext-layer-exp` tree:

- inject/weight expansion: independent CPU comparison `max_abs=7.275958e-12`;
- weighted MoE (`repeat(moe) * weight`): `max_abs=2.910383e-11`;
- final `residual + weighted_moe`: `max_abs=2.3283064e-10`,
  `max_rel=2.4869173e-10`.

The direct graph output hash for the deterministic reference contract is
`0f82e252619bb73e`, matching the recorded output hash. The full independent
reference contract was:
`/tmp/segment-b-reference-contract.f32` = reference MoE `[2560]`, residual
HC `[2560,4,1]`, injection `[4,1]`; output was extracted directly from the
explicit `[2560,4,1]` graph tensor after scheduler synchronization.

The control modes are diagnostic only (`inject`, `weight`, `mul`) and do not
alter the normal Segment-B path. The loader remains low-residency; no full
model or expert population is loaded.
## 2026-10-07 — Dynamic Segment-A/router bridge

Added an isolated `CORTEX_FLASHNEXT_SEGMENT_A_ROUTER=1` mode in the active
layer experiment. It executes the real Segment-A graph and computes the
router from that same real pre-MoE hidden row, exporting one bounded
vocabulary-width artifact whose first 12804 floats are the Segment-A
transaction and whose next 512 floats are the router logits.

The deterministic run completed without full-model residency. Top-10 IDs
computed from the exported router logits were:

`136,77,317,257,226,142,175,414,373,131`

They match the frozen golden IDs, but were derived dynamically from the
current hidden state. Router logits begin at artifact offset 12804; the
artifact was `/tmp/segment-a-router.f32`.

This is a bridge checkpoint, not complete layer-0 execution: external
selected-expert orchestration and stateful Segment-A → MoE → Segment-B
transaction ownership remain separate.
## 2026-10-07 — Dynamic sparse-MoE integration checkpoint

Converted `tools/flashnext_selected_expert_compute.py` to consume the
router artifact rather than hard-coded IDs or weights. It now derives the
top-10 IDs from the 512 real router logits at artifact offset 12804 and
renormalizes the selected softmax weights exactly as the established
Qwen4Exp path requires.

Run result:

- dynamic IDs: `136,77,317,257,226,142,175,414,373,131`;
- normalized routing weight sum: `1`;
- selected-only output hash: `822f120d0c7f2369`;
- shared contribution hash: `63055fc5061b323b`;
- complete dynamic MoE hash: `c3485d69e3535a65`;
- selected bytes read: `26,624,000`;
- all layer expert bytes: `1,363,148,800`;
- unselected expert reads: `0`;
- reduction: `51.2x`.

The dynamically generated complete MoE was then supplied to the validated
Segment-B contract with the real Segment-A residual/injection values. The
Segment-B output hash was `0f82e252619bb73e`; direct extraction completed
finite and matched the previously validated HC contract.

This is the `FLASHNEXT DYNAMIC SPARSE MOE` checkpoint. The remaining gap is
owning the complete Segment-A → dynamic MoE → Segment-B handoff in one
long-lived in-process transaction instead of the current diagnostic bridge.
## 2026-10-07 — Owned transaction prefix

Added `src/flashnext-layer-transaction.h` defining a bounded owned
transaction for one qwen4exp layer position. It owns the hidden row,
residual HC, injection, dynamic route IDs/weights, MoE vectors, output,
and future committed-state storage; it contains no model weights.

The Segment-A/router probe now materializes the first transaction stage in
process. It copies the real `[2560]` pre-MoE row, `[2560,4]` residual HC,
and `[4]` injection into transaction-owned storage, then computes dynamic
top-10 IDs and normalized weights directly from the real router logits.
The deterministic run reported `CORTEX FLASHNEXT OWNED PREFIX: PASS` and
the expected dynamic IDs.

This is not yet the complete owned transaction: selected-expert execution
and Segment-B still need to be called through in-process APIs rather than
diagnostic executable/file boundaries.
## 2026-10-07 — Callable selected-MoE checkpoint

Extracted the proven selected-expert Tensor-Window execution into:

- `src/flashnext-selected-moe.h`
- `src/flashnext-selected-moe.cpp`
- `examples/flashnext-owned-selected.cpp`

The callable API consumes transaction-owned hidden state, dynamic expert
IDs, and dynamic routing weights. It performs selected-only `pread`, the
existing quantized GGML gate/up/down operations, SwiGLU, and weighted
accumulation through a reusable 16 KiB arena.

The first aggregate mismatch was isolated to routing precision. The
callable path was using the float overload of `std::exp` and a float
normalization sum. The established path uses double-precision exponentials
and summation. Explicit double operands and a double sum restored exact
agreement.

Checkpoint:

`CORTEX FLASHNEXT OWNED SELECTED MOE: PASS`

- dynamic IDs: `136,77,317,257,226,142,175,414,373,131`;
- selected hash: `822f120d0c7f2369`;
- selected bytes: `26,624,000`;
- unselected expert reads: `0`;
- Tensor Window: `16,384` bytes;
- `pread` calls: `1,650`.
## 2026-10-07 — Owned shared and complete MoE checkpoint

Extended `examples/flashnext-owned-selected.cpp` to call the extracted
shared-expert API in the same process as the selected Tensor-Window API.
The transaction now receives the owned pre-MoE row, dynamically computes
the route, streams selected experts, executes the shared expert, and forms
the complete MoE vector without an intermediate file or subprocess.

Checkpoint results:

- `CORTEX FLASHNEXT OWNED SELECTED MOE: PASS`
- selected hash: `822f120d0c7f2369`;
- `CORTEX FLASHNEXT OWNED SHARED EXPERT: PASS`;
- shared hash: `63055fc5061b323b`;
- `CORTEX FLASHNEXT OWNED COMPLETE MOE: PASS`;
- complete hash: `c3485d69e3535a65`;
- selected bytes: `26,624,000`;
- unselected reads: `0`;
- Tensor Window: `16,384` bytes;
- pread calls: `1,650`.

Shared API: `src/flashnext-shared-expert.h/.cpp`. Segment-B callable
extraction remains the next stage.
## 2026-10-07 — Callable Segment-B checkpoint

Added the reusable Segment-B module:

- `src/flashnext-segment-b.h`
- `src/flashnext-segment-b.cpp`

It consumes transaction-owned `complete_moe`, `residual_hc`, and `inject`
buffers, executes the established GGML HC combine ordering, synchronizes,
and writes the explicit `[2560,4,1]` result into transaction-owned output.

In the same process as the callable selected and shared engines:

- selected hash: `822f120d0c7f2369`;
- shared hash: `63055fc5061b323b`;
- complete MoE hash: `c3485d69e3535a65`;
- Segment-B hash: `0f82e252619bb73e`;
- selected bytes: `26,624,000`;
- unselected reads: `0`;
- Tensor Window: `16,384` bytes.

`CORTEX FLASHNEXT OWNED SEGMENT B: PASS` is verified. The remaining
integration work is to make Segment-A/router populate this same transaction
entry point directly, eliminating the current diagnostic input source.
## 2026-10-07 — OOM audit of owned-layer runner

No inference was rerun during this audit. Journal evidence identifies the
latest disappearance as `systemd-oomd`, with the user terminal scope at
approximately 10.0/10.29 GiB RAM and 18.6/20.58 GiB swap; it killed 61
processes. This is consistent with the historical Flash-Next OOM event.

The new owned-layer runner had omitted `CORTEX_FLASHNEXT_ROUTER_ONLY=1`.
Static tracing showed that this selected the normal `global_weight` class,
including `per_layer_token_embd.weight`, and reached:

`cortex_global_buffer.resize(global_total)`

with `CORTEX_GLOBAL_BYTES=29662950400` before the process was killed.

The runner now sets the same router-only low-residency mode used by the
verified Segment-A path. A hard pre-allocation guard was also generalized
to all Flash-Next low-residency modes and rejects an unexpected global set
over 1 GiB before `resize()`.

Expected safe path: model-weight mmap disabled; no full expert population;
router-only selected global/layer buffers only; Segment-A graph/context is
the sole heavyweight runtime before its data is copied into the transaction.
The known Segment-A measurements remain approximately 337,715,200 global
bytes and 50,238,336 layer bytes. No new inference result is claimed from
this audit.
## 2026-10-07 — Guarded owned transaction PASS

One controlled execution of `llama-flashnext-owned-layer` was run after
correcting the missing `CORTEX_FLASHNEXT_ROUTER_ONLY=1` setting. No stale
diagnostic artifact was used as an input, and no parallel runner was
started.

Loader evidence:

- `CORTEX_GLOBAL_BYTES=337715200`;
- `CORTEX_LAYER_BYTES=50238336`;
- selected tensors: `23`;
- model-weight mmap disabled;
- scheduler CPU compute buffer: `6.43 MiB`;
- no guard failure or OOM event.

The complete same-process transaction passed:

`CORTEX FLASHNEXT OWNED TRANSACTION: PASS`

Hash ladder:

- pre-MoE: `34259a1e69a780eb`;
- selected: `822f120d0c7f2369`;
- shared: `63055fc5061b323b`;
- complete MoE: `c3485d69e3535a65`;
- Segment B: `0f82e252619bb73e`.

Dynamic IDs were `136,77,317,257,226,142,175,414,373,131`; selected bytes
were `26,624,000`; unselected reads were `0`; Tensor Window was `16,384`
bytes; pread calls were `1,650`; output shape was `[2560,4,1]`.

The external RSS monitor sampled before the `execve` transition and did not
capture a valid peak, so peak RSS is intentionally not claimed from this
run. The bounded loader/compute measurements above are the reliable memory
evidence for this checkpoint.

## 2026-10-07 — owned transaction frozen and stock layer boundary comparison

The verified in-process owned transaction was frozen in
`FLASHNEXT_OWNED_TRANSACTION_FROZEN.md` before comparison work.  The
historical sparse-MoE and Tensor-Window trees were not modified.

The stock Qwen4Exp source trace establishes that each layer's architectural
handoff is the post-final-`build_hc_combine` `res_hc` tensor, with logical and
physical shape `[n_embd,hc,n_tokens]` (`[2560,4,1]` for layer 0).  The later
`build_hc_mix(...hc_head...)` is the final model output head and is not part of
the per-layer layer-0 boundary.  Recurrent and convolution state copies are
constructed inside `build_layer_attn_linear` and are graph writes during the
layer transaction.

A bounded source-derived stock-boundary probe independently evaluated the
stock `build_hc_combine` operation using the transaction's complete-MoE,
residual-HC, and injection data.  It loaded no model and no expert tensors.

```text
STOCK_BOUNDARY=post-layer0-build_hc_combine shape=[2560,4,1]
stock=0f82e252619bb73e cortex=0f82e252619bb73e
max_abs=0 max_rel=0 first_diff=NONE
FLASHNEXT COMPLETE LAYER 0: PASS
```

This is the bounded architectural-boundary comparison; it does not yet
claim two-position state continuation.
## 2026-10-07 — stock sparse `MUL_MAT_ID` design checkpoint

`FLASHNEXT STOCK SPARSE ORACLE DESIGN: PASS`

Stock CPU `MUL_MAT_ID` computation is already selective: the CPU path
groups routed rows by expert and only dereferences experts with nonzero
row counts.  The dense 512-expert requirement is therefore a backing,
allocation, and repacking assumption rather than a mathematical need.

The future isolated oracle experiment will intercept expert address
resolution near `src0->data + cur_a * nb02`, while preserving stock
router IDs, routing weights, output layout, quantized `vec_dot` and
tiled kernels, shared expert, HC combine, and recurrent state writes.

The preferred approach is an oracle-only sparse `MUL_MAT_ID` CPU variant.
Its provider may perform only expert-ID resolution, GGUF location lookup,
`pread`, and bounded storage management.  CPU repacking must be bypassed
only for these oracle expert tensors.  The initial bounded payload target
is one expert at a time: gate `870,400` bytes, up `870,400` bytes, and
down `921,600` bytes, approximately `2,662,400` bytes combined.

No sparse backend implementation or model run was performed at this
checkpoint.  The next future milestone is the isolated single-expert
test:

`FLASHNEXT STOCK SPARSE MUL_MAT_ID EXPERT136: PASS`
