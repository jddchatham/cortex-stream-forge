# Qwenny internal MTP experiment

This is a new experimental tree copied from the frozen Qwenny checkpoint:

`runtime-qwenny-mtp-exp/`

No frozen runtime, checkpoint, shortcut, model, or existing handoff was
modified.

## GGUF inventory

Model blob:

`/home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6`

Metadata:

- architecture: `qwen35`
- `qwen35.nextn_predict_layers`: `1`
- main runtime layers: `64`
- GGUF block count including NextN: `65`
- quantization: Q4_K_M overall

The GGUF does contain an MTP/NextN block. The NextN-specific tensors are:

| Tensor | Dimensions | GGML type | Bytes | Absolute offset |
|---|---:|---:|---:|---:|
| `blk.64.nextn.eh_proj.weight` | `[10240,5120]` | 12 | 29,491,200 | 16,781,141,280 |
| `blk.64.nextn.enorm.weight` | `[5120]` | 0 | 20,480 | 16,810,632,480 |
| `blk.64.nextn.hnorm.weight` | `[5120]` | 0 | 20,480 | 16,810,652,960 |
| `blk.64.nextn.shared_head_norm.weight` | `[5120]` | 0 | 20,480 | 16,810,673,440 |

NextN-specific payload is 29,552,640 bytes. The MTP block also reuses the
normal layer-64 attention/FFN tensors and the model embedding/output tensors.
All 15 tensors under `blk.64.*` total 263,313,408 bytes and 424,699,392
logical elements.

The MTP block does not contain its own embedding table or output projection in
this GGUF. Upstream Qwen35 MTP explicitly falls back to the target model's
`tok_embd` and `output` tensors when `nextn.embed_tokens` and
`nextn.shared_head_head` are absent. This means the MTP helper is not an
independent little model: it is a target-coupled NextN head.

## Upstream implementation map

The copied runtime already contains the upstream-style implementation:

- `src/models/qwen35.cpp`: `graph_mtp`
- `src/llama-context.cpp`: MTP context type, NextN embedding extraction, and
  `llama_set_embeddings_nextn`
- `src/llama-batch.cpp`: MTP input-batch sizing and hidden-state handoff
- `common/speculative.cpp`: `draft-mtp` controller and draft verification
- `examples/speculative-simple/`: upstream integration example

`graph_mtp` consumes target hidden state `h` plus the next token embedding,
normalizes both, concatenates them, applies `nextn.eh_proj`, runs the appended
layer-64 attention/FFN path, applies `nextn.shared_head_norm`, and uses the
shared target LM head for logits. The input interface is therefore:

```text
target hidden state [n_embd]
+ next token ID or embedding
→ MTP graph
→ target-vocabulary logits
```

This structure is compatible in principle with an internal helper, but it
requires target layer-64 weights, target embedding/output weights, and a
separate MTP context/KV state. It cannot be treated as a standalone small
model.

## Status

## Experimental provider extension

The isolated branch now permits `llama_model_base::cortex_rebind_layer(64)`
when low-residency mode is enabled. In this branch only, the rebind list also
includes the Qwen35 `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`, and optional
shared-head tensors. This treats GGUF block 64 as an auxiliary streamed
segment while retaining the target-layer tensors used by `graph_mtp`. The
frozen Qwenny tree was not changed.

The isolated `llama` library rebuild passes. A complete MTP execution probe is
not yet claimed: the upstream MTP controller creates a second MTP context
against the target model, and the current low-residency entry point does not
yet expose a safe hidden-state capture/import command for that context. The
next safe implementation step is a dedicated probe that captures a target
hidden row, loads block 64 through the new rebind path, executes `graph_mtp`,
and compares logits before speculative verification.

Forensic inventory: PASS.

Isolated MTP-capable build: PASS (`runtime-qwenny-mtp-exp-build`, target
`llama-speculative-simple`; the modified `llama` library was rebuilt after the
auxiliary-layer rebind extension).

MTP graph execution, MTP Tensor-Window execution, self-draft, and batched
verification remain unverified. No shortcut or user-facing runtime was
created.

## Stage-A result

The target hidden interface is a float32 `h` row of width `n_embd` (5120)
passed as the MTP graph input, paired with the token ID whose embedding is
combined with that row. `graph_mtp` also consumes the MTP context's position
input and attention state. It does not consume the ordinary final LM logits
as its hidden input.

The existing Qwenny segmented executable captures the ordinary layer-boundary
hidden rows, but does not expose the exact target `h_nextn` artifact produced
by a target context configured with `embeddings_nextn`. Consequently no hidden
row was imported and no MTP logits were claimed. This is an interface blocker,
not evidence that the MTP mathematics or GGUF tensors are invalid.

The isolated source change builds successfully. The next required probe must
use a target context with `llama_set_embeddings_nextn(ctx, true, false)`,
capture `llama_get_embeddings_nextn_ith(...)`, then feed that row and the
corresponding token/position into the MTP context. It must also rebind block
64 before graph execution. No speculative decoding or shortcut was added.

## Dedicated h_nextn probe result

Added `examples/qwen38-mtp-batched-bridge.cpp` and built
`runtime-qwenny-mtp-exp-build/bin/llama-qwen38-mtp-batched-bridge`.
The probe uses the exact nine-token formatted `Hi` sequence, processes all
positions per layer, enables `embeddings_nextn` on the final target segment,
and exports `qwen38-mtp-hnextn.f32`.

Artifact: shape `[5120]`, float32, 20,480 bytes; final prompt position 8;
FNV-1a hash `5ca187f40e5b4748`; min/max `-13.4974708557` / `28.8281040192`;
mean `0.02113269504`; L2 norm `119.180788154`; NaN/Inf none.

Source tracing confirms this is the final target `output_norm` result exposed as
`res->t_h_nextn` before LM-head projection. It is a transformed target hidden
row, not an ordinary pre-norm layer boundary and not an MTP-partially-processed
value.

The independent MTP context currently fails during graph reservation because
`graph_mtp` sees `model.layers[64].nextn.eh_proj == nullptr`, despite the GGUF
tensor and metadata index being present. The failure occurs before MTP
arithmetic and before logits. The experimental rebind is ordered before MTP
context construction, so the remaining issue is tensor-descriptor construction
in the low-residency model path: the required NextN descriptors are not being
instantiated for the MTP context.

Status: genuine `h_nextn` capture PASS; artifact export PASS; independent MTP
execution blocked at missing `nextn.eh_proj`; streamed MTP logits not produced;
speculative decoding not started.

Probe hashes: source `c36aebcc689c4e53abb78487f2d2aa56d412efc4765d6757dad66f83fc282b39`;
executable `9d96fc290617f8cab559ce9053ed9c085a5efc3410671d7237129c6a614b5c31`.
