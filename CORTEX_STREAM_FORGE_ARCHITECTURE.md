# Cortex Stream Forge Architecture

## Core contract

Forge separates architecture-neutral streaming mechanics from model mathematics. `ForgeEngine` owns discovery, adapter selection, a `StreamPlan`, model identity, verification dispatch, diagnostic records, and the lifecycle rule that a reusable weight buffer is not overwritten until scheduled reads finish. An adapter owns the graph builder and state semantics for its architecture.

The lifecycle is:

1. Read GGUF metadata and identify `general.architecture`.
2. Select a registered adapter or fail explicitly.
3. Build an unmodified monolithic oracle.
4. Describe a segment: layer range, resident weights, activation boundary, recurrent state, and KV requirements.
5. Load actual GGUF tensor bytes for the selected layer into one reusable external buffer and rebind tensor data pointers.
6. Construct and execute the adapter graph window.
7. Wait for scheduler completion before reading the hidden boundary or reusing the buffer.
8. Preserve activations, recurrent state, KV cache, and model state independently of external weights.
9. Rebind the same buffer for the next segment.
10. Compare the streamed candidate with the oracle and report the first divergent operation.

## Adapter interface

The current Python contract is represented by `Adapter` in `forge/adapters.py`; a native backend can implement the same conceptual interface:

- `matches(architecture)`: strict GGUF architecture selection.
- `layer_count()` and `layer_spec(index)`: layer type plus exact tensor names.
- `weights_for_layer(index)`: actual tensors, never magic offsets.
- `state_requirements(index)`: recurrent planes, KV cache, and persistent state.
- `build_segment(start, end, inputs)`: architecture-specific graph construction.
- `set_segment_input(hidden, positions, state)`: boundary binding.
- `get_segment_output()`: boundary output after scheduler completion.
- `commit_state()`: recurrent/KV/model-state handoff.
- `cleanup_segment()`: synchronization and lifetime barriers before rebinding.
- `verify(model)`: independent oracle/candidate regression.

Qwen3.5's adapter must expose recurrent DeltaNet and full-attention layer kinds separately. The core only sees `recurrent_state` and `kv_cache` requirements; it does not implement DeltaNet, short convolution, RoPE, or attention mathematics. Llama/DeepSeek uses the ordinary KV-cache path. Qwen2.5 and BitNet retain their frozen reference runtime semantics.

## Memory model

The external buffer is one reusable, aligned byte region sized from the selected tensors' runtime metadata. Weight residency is temporary. Activations are scheduler/backend-owned. Recurrent state, KV cache, and persistent model state have independent lifetimes and must never be stored in the weight buffer. A segment is safe to replace only after `decode`/scheduler execution returns and all backend work has completed.

This supports a model larger than a comfortable resident weight set: load one layer, execute it, synchronize, reuse the buffer, and load the next layer. Future native implementations may replace the current vector with a bounded allocator without changing the adapter contract.

### Low-residency GGUF provider

`forge/gguf.py` retains a `TensorDescriptor` index containing each tensor's
name, dimensions, GGML type, relative GGUF offset, byte size, and absolute file
offset. It reads only the GGUF header, metadata, and descriptor table; payloads
are not mapped. `forge/weight_provider.py` opens the file with `os.open`, uses
chunked `pread` for selected descriptor ranges, and copies them into one
reusable `bytearray`. `load_layer()` selects tensors by the structural
`blk.<layer>.` prefix and returns memoryview slices suitable for rebinding;
`release()` drops logical residency while retaining capacity for reuse.

The Qwen3.8 proof test loads layers 0 and 1 sequentially, confirms the same
buffer is reused, and measures RSS. The 16.81 GB GGUF's layer 0 is
252,263,296 bytes; measured peak RSS growth was 297,451,520 bytes (about 284
MiB). This proves the safe provider/index boundary, not yet native GGML graph
execution. Native integration must occur before
`llama_model_base::load_tensors()` invokes ordinary `load_all_data()`, because
the existing reference external-buffer hook runs after that loader has already
registered the full model.

The Forge-local runtime also contains an opt-in audit hook,
`CORTEX_LOW_RESIDENCY_LAYER=N`, in `src/llama-model-loader.cpp`. It restricts
`load_all_data()` payload reads to `blk.N.*` and leaves the default path
unchanged. Testing showed that this alone is insufficient for native execution:
the preceding `llama_model_base::load_tensors()` phase still creates or assigns
backend buffers for the complete tensor set. The native probe was therefore
terminated under a hard timeout before inference, with memory returning to its
pre-test level. The next native change must make backend buffer creation and
tensor allocation segment-aware too, while retaining metadata, tokenizer data,
activations, recurrent state, and KV state independently.

The Forge-local native source now goes one step further for the explicit probe:
`llama_model_base::load_tensors()` sets `ml.no_alloc`, enumerates only
`blk.<N>.*` descriptors, reads their ranges through
`llama_model_loader::load_data_range()`, and binds those bytes to a reusable
external vector. This is the native equivalent of the Python provider's
descriptor/read/rebind sequence. It is intentionally not wired into the
ordinary chat CLI: that CLI still constructs a complete model graph and needs
embeddings, output tensors, and all layer objects. A dedicated segmented graph
entry must consume the selected layer plus separately owned hidden/state
buffers before full inference can be safely attempted on this host.

## Verification and diagnostics

Every adapter must preserve an ordinary oracle. Forge compares final output and boundary state, then walks execution checkpoints in order: boundary input, norm, projection, attention/recurrent output, residual, FFN, and boundary output. `FirstDivergence` carries operation, tensor name, metadata, source/destination addresses, max absolute/relative error, mismatch count, and can be extended with hashes.

The known failure signatures are:

- stale boundary: read happened before deferred scheduler nodes completed;
- projection mismatch: wrong tensor binding, type, stride, or layer selection;
- attention mismatch: KV-cache ownership/position mismatch;
- recurrent mismatch: state planes were not preserved or output/state storage overlapped;
- 9B Qwen3.5 reshape failure: fixed KV-head assumptions instead of per-layer metadata.

The oracle is never edited to accommodate a candidate.

## Existing proven references

- Qwen3.5 generic external hook and segmented graph: `../cortex-stream-qwen35-src/src/llama-model.cpp`, `src/models/qwen35.cpp`, `src/models/llama.cpp`, `src/llama-graph.cpp`.
- Qwen3.5 recurrent/KV regressions: `../cortex-stream-qwen35-src/examples/qwen35-recurrent-window-test.cpp` and `qwen35-window-test.cpp`.
- DeepSeek/Llama regression: `../cortex-stream-qwen35-src/examples/deepseek-window-test.cpp`.
- Qwen2.5/BitNet frozen path: `../cortex-stream-qwen-src` and its existing Phase24 result.

Forge does not modify these paths.

## Adding an architecture

1. Inspect GGUF metadata and tensor names; record SHA256 and dimensions.
2. Confirm the ordinary runtime can load and run a monolithic oracle.
3. Implement a strict adapter and layer descriptors.
4. Add actual tensor collection/rebinding through the generic buffer owner.
5. Add the graph input/output boundary without changing model equations.
6. Preserve all state objects in the runtime memory subsystem.
7. Write an oracle/candidate test that compares a minimal layer window first.
8. Add a full-output regression and first-divergence diagnostics.
9. Register the adapter and document the exact runtime/source reference.

If an architecture needs new GGML operations, implement those in its adapter/runtime layer first. Do not contaminate the core with architecture-specific conditionals.

## Current implementation status

Forge's Python core and CLI are deliberately a standalone control plane over the proven runtime adapters. It is not yet a replacement for llama.cpp's native graph library. The initial milestone proves the reusable project boundary and dispatches the existing verified tests; the next milestone is moving the same adapter contract behind a native C++ graph backend while retaining the old projects unchanged. The Qwen3.8 fingerprint is registered as `qwen38` with a Forge-local runtime extension, but remains unverified on this 9.6 GiB host because its GGUF is 16.81 GB.

The Forge-local runtime now reaches actual Qwen3.8 isolated graph execution:

### Qwen3.8 sequential low-residency execution

The Forge-local Qwen3.8 path is implemented in the copied runtime under
`runtime-qwen35-src`. `llama_model_load_from_file` enters low-residency mode
when `CORTEX_LOW_RESIDENCY_LAYER` is set: GGUF metadata and tensor descriptors
are retained, while global tensors and the selected `blk.N.*` ranges are read
with range loads into separate reusable buffers. `llama_cortex_rebind_layer`
then rebinds the selected layer's tensor data before graph execution.

`examples/qwen38-lowres-sequential-chat.cpp` is the complete segmented path.
For each token and layer it sets the segment range, rebinds that layer, creates
a bounded context, restores that layer's serialized state, executes one graph,
extracts the hidden boundary or logits, serializes the state, and frees the
context. The state vector is retained per layer, so DeltaNet/short-convolution
state and full-attention KV state are not reset when the weight buffer is
reused. The full 64-layer trunk passed for two sequential steps and produced
tokens `13` and `220`. The one-token interactive path is
`examples/qwen38-lowres-chat.cpp`; `bin/qwen38-chat` is its diagnostics-silent
launcher wrapper.

The measured Qwen3.8 low-residency allocation was a 252,263,296-byte maximum
layer buffer plus a 1,758,126,080-byte global/final-weight buffer, rather than
the 16.81 GB GGUF payload. The provider's layer probe measured approximately
284 MiB RSS growth. These are bounded working-set measurements; the GGUF stays
file-backed/on disk and is never loaded as one resident tensor set.
low-residency mode disables whole-file mmap prefetch, avoids full backend weight
allocation, and binds global tensors plus one selected layer. The dedicated
`llama-qwen38-lowres-segment-test` passes recurrent layer 0→1 and
full-attention layer 3→4. Sequential 64-layer rebinding, recurrent/KV state
transfer, final logits, and interactive chat remain unverified.
