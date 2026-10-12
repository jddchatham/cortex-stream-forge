# Cortex Stream Forge

Cortex Stream Forge is a standalone orchestration layer for the proven Cortex external-weight streaming technique. It identifies a GGUF architecture, selects an adapter, plans segmented execution, keeps a reusable external transformer-weight buffer, preserves activations and model state, and dispatches oracle/candidate verification.

Forge is intentionally separate from the existing Cortex/BitNet source trees. The current adapters bind to those verified runtimes and regression binaries; they do not rewrite their model mathematics or copy architecture-specific behavior into the core.

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

The older Forge-local Qwen3.8 adapter uses metadata/index range loads and a
reusable layer buffer for isolated experiments. Those adapter results are not
evidence that the current Qwen4Exp canonical-manifest path provides complete
end-to-end chat or token generation.

The interactive wrapper and Omarchy entry are historical adapter artifacts;
they are not presented as a working frontend for the current canonical
residency-manifest development path.

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

## Cortex Stream Forge development status

Cortex Stream Forge's current goal is bounded-memory streaming inference for
large GGUF models on low-memory CPUs. The Qwen4Exp work now includes a
metadata-only descriptor phase that uses the runtime model structure rather
than a separate tensor-family table.

The verified Qwen4Exp descriptor inventory contains 1,224 runtime tensor
descriptors across 48 physical layers: 1,074 arena-resident descriptors, 144
externally streamed expert descriptors, and six global or separately managed
descriptors. The fixed per-layer arena is 67,851,136 bytes with 64-byte
alignment. A canonical runtime-derived residency manifest is generated before
mapping or payload materialization and is consumed by the planner, activation,
and rebind validation paths.

Current non-inference regressions pass for descriptor extraction, residency
transitions, selection-aware rebinding, HC handoff, range validation, and
format checks. Genuine end-to-end prompt-to-token generation has **not** been
demonstrated by this development branch.

Known blockers are documented in `DEVELOPMENT_STATUS.md`: full MoE graph
integration with external expert streaming, the final HC head, bounded
vocabulary projection, strict residency commit ordering, and supervised
launcher timeout/PID cleanup.
