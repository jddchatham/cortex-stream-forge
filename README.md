# Cortex Stream Forge

Cortex Stream Forge is a standalone orchestration layer for the proven Cortex external-weight streaming technique. It identifies a GGUF architecture, selects an adapter, plans segmented execution, keeps a reusable external transformer-weight buffer, preserves activations and model state, and dispatches oracle/candidate verification.

Forge is intentionally separate from the existing Cortex/BitNet source trees. The current adapters bind to those verified runtimes and regression binaries; they do not rewrite their model mathematics or copy architecture-specific behavior into the core.

## QUORTEX FlashNext CPU text generation (QIES-072)

The `feature/flashnext-first-text-generation` milestone adds a low-residency CPU CLI for the 48-layer Qwen3.8 FlashNext model. On the tested model, the CLI applies the GGUF template's native `enable_thinking=false` user-only branch and generated `4` for “What is 2 + 2? Answer with only the number.” Stock llama.cpp and QUORTEX matched all 28 captured full-logit comparisons byte-for-byte, including the native end-of-generation decision. The detailed evidence is in [`docs/QIES072_NONTHINKING_FIRST_ANSWER.md`](docs/QIES072_NONTHINKING_FIRST_ANSWER.md).

Build the CPU-only target with REPACK enabled (the model shards and tensor index must be supplied separately):

```sh
cmake -S runtime-flashnext-layer-exp -B build-quortex \
  -DGGML_CUDA=OFF -DGGML_CPU_REPACK=ON \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-quortex --target llama-flashnext-chat -j2
```

Run a single question with a fresh disk-backed state directory:

```sh
export QUORTEX_MODEL=/path/to/model-00001-of-00003.gguf
export QUORTEX_TENSOR_INDEX="$PWD/flashnext-tensor-index.json"
run_dir=$(mktemp -d /tmp/quortex-question.XXXXXX)
CORTEX_FLASHNEXT_LOADER_CACHE=1 \
CORTEX_FLASHNEXT_EXPERT_ARENA_BYTES=1048576 \
CORTEX_FLASHNEXT_LAYER_RANGE=1 \
OMP_NUM_THREADS=1 \
  build-quortex/bin/quortex \
  --model "$QUORTEX_MODEL" \
  --tensor-index "$QUORTEX_TENSOR_INDEX" \
  --prompt "What is 2 + 2? Answer with only the number." \
  --max-tokens 16 \
  --state-dir "$run_dir/state" \
  --logits-dir "$run_dir/logits"
```

The measured reference run used about 5.86 GiB peak process RSS and took roughly 37 minutes, with most time in prompt prefill. This remains a correctness-focused, CPU-only diagnostic CLI; broader chat-template support and faster interactive generation are not claimed. Keep the expert arena at 16 KiB unless explicitly selecting the validated 1 MiB option as above.

## Current adapter status

| Architecture | Adapter | Evidence |
|---|---|---|
| BitNet / BitNet b1.58 | `bitnet` | frozen production reference registration; model-presence regression |
| Qwen2 / Qwen2.5 | `qwen2` | Phase24 bit-exact regression dispatch |
| Qwen3.5 | `qwen35` | recurrent and full-attention handoff regressions already verified |
| Qwen3.8 27B | `qwen38` | low-residency 64-layer sequential execution and one-token chat verified |
| Llama / DeepSeek-R1 Llama | `llama` | DeepSeek hidden-state handoff and external-layer byte comparison verified |

The adapters are evidence-backed dispatch points, not claims that every model sharing an architecture is automatically compatible. Each model still requires an oracle and a streamed regression.

Qwen3.8 cannot use the ordinary whole-model resident loader on this host. Its Forge-local low-residency path is verified below.

## Qwen3.8 low-residency status

The Forge-local Qwen3.8 runtime now uses the metadata/index plus direct range
loads, a reusable layer buffer, and one short-lived graph context per layer.
All 64 runtime layers execute sequentially; each layer is rebound, executed,
serialized for recurrent/KV state, and released before the next layer. A two
step full-trunk run passed and selected tokens `13` then `220`. A bounded
interactive run on `Hi` produced a response token (`,`). The 16.81 GB model
payload is not made resident as one allocation: the observed layer buffer is
252,263,296 bytes and the global/final-weight buffer is 1,758,126,080 bytes.

The interactive wrapper is `bin/qwen38-chat`; it suppresses runtime diagnostics
on stderr and uses the local Ollama blob. The Omarchy entry is
`Qwen3.8 27B Abliterated Chat`. The existing model launchers are unchanged.

### Safe low-residency milestone

```bash
python3 tests/test_weight_provider.py
```

The provider indexes GGUF metadata without mapping the model, loads Qwen3.8
layer 0, reuses the same resident buffer for layer 1, and reports RSS. This is
the foundation for native graph integration; complete Qwen3.8 inference is not
claimed until the native loader can use this provider before full-model
`load_all_data()`.

The regression also compares a selected tensor's bytes against a direct
`pread` at its indexed absolute GGUF offset, then loads a recurrent layer and
the full-attention layer 3 through the same buffer. The measured Qwen3.8
working set remains approximately 284 MiB RSS growth.

The Forge-local native segment test reaches actual Qwen3.8 graph execution:

```bash
cmake -S runtime-qwen35-src -B runtime-qwen35-build -DCMAKE_BUILD_TYPE=Release
cmake --build runtime-qwen35-build --target llama-qwen38-lowres-segment-test -j2
runtime-qwen35-build/bin/llama-qwen38-lowres-segment-test <GGUF> 0 1
runtime-qwen35-build/bin/llama-qwen38-lowres-segment-test <GGUF> 3 4
```

The recurrent and full-attention isolated segments pass without whole-file
prefetch. Complete sequential stateful chat remains unverified.

## Usage

~~~bash
cd /home/user/bitnet-cortex/cortex-stream-forge
./cortex-forge --model /home/user/Downloads/DeepSeek-R1-Llama-8B-Abliterated-Q4_K_M.gguf --plan
./cortex-forge --model /home/user/Downloads/DeepSeek-R1-Llama-8B-Abliterated-Q4_K_M.gguf --verify
./cortex-forge --model /home/user/Downloads/DeepSeek-R1-Llama-8B-Abliterated-Q4_K_M.gguf --chat
~~~

The current Forge chat path uses the verified newer `llama-simple-chat` runtime and the GGUF's tokenizer/template metadata. Normal chat does not set Phase24 or diagnostic variables.

## Boundaries

The existing handoffs and source trees are references and remain untouched by Forge. See `CORTEX_STREAM_FORGE_ARCHITECTURE.md` for the lifecycle, ownership model, adapter contract, and extension process.

## FlashNext (Qwen3.8) 48-layer single-token milestone

Branch `milestone/flashnext-48layer-bitexact` validates full 48-layer,
single-token FlashNext inference through the low-residency streaming engine,
bit-exact against the stock graph (see `milestone-flashnext-48layer/`).

- **Scope**: one token (248045) at position 0 through all 48 layers
  (36 recurrent + 12 full-attention at layers 3,7,...,47; PLE at layer 1),
  final HC mixer, and output projection. All 48 layer outputs and the final
  248,320-float logits are element-wise bit-identical to the stock reference;
  predicted token and logit match stock exactly.
- **Hardware constraints**: validated on a 9.8 GB RAM host against a 44.7 GB
  IQ4_XS shard. Weights stream through pread tiles (one resident layer at a
  time); measured peak process RSS ~2.2 MB with the OS page cache carrying the
  streaming load. ~1.9 s/layer (91 s for 48 layers). REPACK=ON and OFF builds
  produce bit-identical results.
- **Numerical validation**: `milestone-flashnext-48layer/regression_check.py`
  (golden 2-layer), `compare_chain48.py <tag> <N>` (per-layer vs stock
  captures), checkpoints N=2/4/8/16/32/48 all bit-exact. The host-side sparse
  MoE and final head mirror the stock ggml op-for-op (see
  HC_DIVERGENCE_REPORT.md for the corrected HC transaction capture).
- **Remaining limitations**: multi-token generation is not yet validated - it
  needs recurrent/KV state carryover across per-layer reloads and PLE
  predecessor-token metadata. Per-layer reload costs ~0.7 s of the ~1.9 s/layer.
- **Artifacts**: `FULL_48_LAYER_REPORT.md` (full results), `CHAIN_48LAYER_DIFF.diff`
  and `HC_TX_FIX_DIFF.diff` (source provenance), `run-chain48.sh` (bounded
  checkpoint runner), `regression_check.py`, `compare_chain48.py`.
