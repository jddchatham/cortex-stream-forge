# Qwen3.8 explicit GGUF `pread()` streaming experiment

This is a Forge-local experiment. It is isolated in:

- source: `runtime-qwen38-pread-exp/`
- build: `runtime-qwen38-pread-exp-build/`

The golden Qwen3.8 executable, direct-mmap runtime, Omarchy entries, other
model implementations, and existing handoff documents were not modified.

## Frozen reference

The Qwen3.8 model blob is:

`/home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6`

Size is 16,810,714,400 bytes and its SHA256 is
`6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6`.

The frozen known-working behavioral reference supplied for this experiment is
the output fragment `, I want to build a website.`. Exact prompt/template
settings for that historical run were not altered or inferred here.

## Explicit provider design

When `CORTEX_PREAD_ARENA=1` is present in this experiment only:

1. GGUF metadata and tensor descriptors are loaded normally.
2. `llama_model_loader::use_mmap` is forced off.
3. `init_mappings()` skips construction of `llama_mmap` objects.
4. `load_data_range()` uses the already-open `llama_file::file_id()` and
   positional `pread()` calls into the caller-provided destination.
5. The selected layer uses one `posix_memalign()`-allocated arena, created
   once at the required layer capacity and reused for the layer payload.
6. The graph executes before any arena overwrite. The existing graph/scheduler
   boundary is the synchronization boundary.

The provider is architecture-independent: it consumes the GGUF tensor index
(`file index`, absolute tensor offset, byte count). Qwen3.8 mathematics remains
in the existing model graph code.

## Current verification

Build:

```bash
cmake -S runtime-qwen38-pread-exp \
  -B runtime-qwen38-pread-exp-build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build runtime-qwen38-pread-exp-build \
  --target llama-qwen38-lowres-segment-test -j2
```

Recurrent layer test:

```bash
env LD_LIBRARY_PATH="$PWD/runtime-qwen38-pread-exp-build/bin" \
  CORTEX_PREAD_ARENA=1 \
  runtime-qwen38-pread-exp-build/bin/llama-qwen38-lowres-segment-test \
  /home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6 \
  0 1
```

Result: PASS, hidden checksum `12.5023612`; layer payload 252,263,296 bytes;
global payload 1,758,126,080 bytes; model buffer reports 0 MiB; mappings stage
was approximately 12 ms and did not create a model mmap.

Full-attention layer test:

```bash
env LD_LIBRARY_PATH="$PWD/runtime-qwen38-pread-exp-build/bin" \
  CORTEX_PREAD_ARENA=1 \
  runtime-qwen38-pread-exp-build/bin/llama-qwen38-lowres-segment-test \
  /home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6 \
  3 4
```

Result: PASS, hidden checksum `70.9446554`; layer payload 233,760,768 bytes;
same zero model-buffer behavior.

Both runs completed in roughly 4.7–5.8 seconds on the two-core host. These are
layer proofs, not yet full generation.

## What is not proven yet

- a standalone one-`MUL_MAT` harness with an independent output hash;
- two successive layers using one arena in one process;
- complete 64-layer explicit-pread generation;
- exact RSS/PSS and live `/proc` mapping capture during graph execution;
- bytes read per generated token and storage-vs-compute timing;
- global embedding/output tiled projection through the explicit provider.

The next safe milestone is a multi-layer segmented executable that records the
arena generation/address and explicit read byte totals, followed by recurrent
state and full-attention/KV comparison against the frozen runtime.

## TENSOR-WINDOW PROOF: PASS

The standalone `examples/qwen38-pread-tensor-window.cpp` harness proves that a
whole quantized GGUF tensor larger than the streaming arena can execute by
complete quantized-row tiles:

```bash
env LD_LIBRARY_PATH="$PWD/runtime-qwen38-pread-exp-build/bin" \
  runtime-qwen38-pread-exp-build/bin/llama-qwen38-pread-tensor-window \
  /home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6 \
  $((8*1024*1024))
```

The tensor is `blk.0.ffn_down.weight`, Q6_K, dimensions `[17408,5120]`,
absolute GGUF offset `1829844256`, and payload size `73,113,600` bytes. Its
quantized row size is `14,280` bytes. An 8 MiB aligned arena streamed it as 9
tiles, reusing the same arena address and synchronizing after every CPU graph
computation. The completed output matched the full-tensor reference exactly:

`TENSOR-WINDOW PROOF: PASS max_abs=0 max_rel=0`

Progressive arena results:

| Arena | Result | Tile rows |
|---:|---|---:|
| 8 MiB | PASS | 587 |
| 4 MiB | PASS | 293 |
| 1 MiB | PASS | 73 |
| 64 KiB | PASS | 4 |
| 16 KiB | PASS | 1 |
| 8 KiB | structurally impossible | less than one 14,280-byte row |

The smallest verified allocation was 16 KiB, with 14,280 bytes of actual tile
payload capacity; the largest successfully streamed tensor was 73,113,600
bytes. Splitting is by complete quantized rows, preserving Q6_K block
boundaries; arbitrary byte slicing was not used. This is an operation-level
tensor-window proof, not yet a replacement for the model graph's large
projection execution.
