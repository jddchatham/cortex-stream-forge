# Cortex Stream Forge

This repository is the Cortex native bounded-memory Qwen4Exp/Flash-Next
runtime handoff. It contains GGUF tensor indexing, bounded range/pread
loading, PLE row streaming, selected-expert loading, recurrent state support,
and the native layer transaction harness.

## Main executable

The existing two-layer chain is implemented by:

`runtime-flashnext-layer-exp/examples/flashnext-owned-layer.cpp`

CMake target:

`llama-flashnext-owned-layer`

## Build

```bash
cmake -S runtime-flashnext-layer-exp -B build \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2 --target llama-flashnext-owned-layer
```

## Existing 1→2 chain

Run only inside a verified 4 GiB RAM / 4 GiB swap scope:

```bash
CORTEX_FLASHNEXT_CHAIN_1_2=1 \
./build/bin/llama-flashnext-owned-layer \
  /path/to/main-model-shard.gguf \
  /path/to/flashnext-tensor-index.json
```

The runner executes physical layer 1, copies its output into owned storage,
releases the layer-1 context/model, loads physical layer 2, and feeds the
owned activation into the layer-2 transaction. It reports transaction and
state-commit status; it is not an independent mathematical oracle.

## Verified baseline

- Layer-1 HC input hash: `8a01a4bbde70529d`
- Layer-1 output hash: `5d40cc4ca311dd41`
- Layer-1 output: `[2560,4,1]`, 40,960 F32 bytes

## Finish next

1. Complete the current native build.
2. Run `llama-flashnext-owned-layer` with `CORTEX_FLASHNEXT_CHAIN_1_2=1`.
3. Verify layer-1/layer-2 handoff bytes, state ownership, finite outputs,
   RSS/PSS/swap, and reproducibility.
4. Add an independent reference comparison before claiming mathematical
   equivalence.

The full handoff report is maintained separately as
`/tmp/cortex-phoenix-complete-handoff.txt` from the original packaging task.
