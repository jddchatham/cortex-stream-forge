# Cortex Stream Forge Model Fingerprints

## Definition

A model fingerprint is a structural description derived from GGUF metadata and, when needed, tensor-name inspection. It answers what a model is: architecture identifier, family, layer count and sequence, dimensions, attention/KV layout, recurrent or convolution state, tensor naming convention, graph operations, and persistent state. It does not contain execution code.

An adapter answers how to execute that fingerprint. The Forge core matches a fingerprint to an adapter, owns the reusable external weight buffer and lifetime barriers, and asks the adapter to construct and execute a segment. This prevents Qwen3.5 mathematics, DeepSeek mathematics, and BitNet kernels from entering the core.

## Derivation and workflow

`forge/gguf.py` reads only the GGUF header and metadata key/value table. It never validates tensor payloads and never loads full model weights. `forge/fingerprints.py` turns those values into a `Fingerprint`. `ForgeEngine` hashes the file, selects an adapter by the architecture field, and exposes the fingerprint in `--plan` JSON.

For a new file:

1. Run `cortex-forge --model FILE --plan`.
2. Inspect architecture, dimensions, layer types, tensor conventions, and state requirements.
3. Match only if the complete structural contract is compatible.
4. Establish an ordinary monolithic oracle.
5. Implement the segmented adapter and actual tensor rebinding.
6. Verify activation/state handoff and reusable-buffer reuse.
7. Add the fingerprint and a regression before declaring support.

## Verified fingerprint: Qwen2.5

- GGUF architecture: `qwen2`.
- Verified family: Qwen2/Qwen2.5 standard transformer.
- Local reference: `/home/user/models/qwen/Qwen2.5-Coder-3B-Instruct-abliterated-Q4_K_M.gguf`.
- Structural class: full-attention decoder layers with RMSNorm, RoPE, attention KV cache, and gated FFN tensor convention `blk.N.*`.
- State: activation boundary and KV cache; no recurrent or short-convolution state.
- Adapter: `qwen2`; verification dispatches the frozen Phase24 implementation.
- Experimental proof: `mono_ok=1`, `seg0_ok=1`, `seg1_ok=1`, zero layer error, reused buffer metadata, unchanged KV metadata.

## Verified fingerprints: Qwen3.5

### 0.8B

- Architecture: `qwen35`, 24 layers, hidden 1024, FFN 3584, attention head count 8.
- Layer sequence: recurrent DeltaNet layers except full attention at 3, 7, 11, 15, 19, 23; per-layer KV-head metadata is zero for recurrent layers and two for full attention.
- State: DeltaNet recurrent matrix state, short-convolution state, and full-attention KV cache.
- Required operations: RMSNorm, SSM_CONV, SSM_SCAN/GatedDeltaNet, RoPE/M-RoPE, attention, residual, FFN.
- Experimental proof: recurrent-state handoff and full-attention/KV handoff both pass with max error <= 1e-5; external rebinding matches at representative recurrent/full-attention layers.

### 9B

- Architecture: `qwen35`, 32 layers, hidden 4096, FFN 12288, head count 16.
- Full-attention layers use four KV heads; recurrent layers use the hybrid state path.
- External buffers observed: 135,971,584 bytes for recurrent layer 0 and 97,943,552 bytes for full-attention layer 3.
- Experimental proof: monolithic inference succeeds; external recurrent layer 0 and full-attention layer 3 outputs are byte-identical to the oracle.
- Generic compatibility requirement: use `hparams.n_head_kv(layer)` rather than a fixed KV-head count.

## Verified fingerprint: DeepSeek-R1 Llama 8B Abliterated

- File: `/home/user/Downloads/DeepSeek-R1-Llama-8B-Abliterated-Q4_K_M.gguf`.
- SHA256: `3d82513481d8edadf815cc5db5073c20d2910c01bad99f4df04b184fb9b05690`.
- Architecture: `llama`, 32 layers, hidden 4096, FFN 11008, 32 attention heads, 8 KV heads, head dimension 128.
- Structural class: standard full-attention Llama decoder with RMSNorm, RoPE, KV cache, and `blk.N.*` Q/K/V/O plus FFN tensors.
- External layer residency: nine tensors, 138,936,320 bytes at layer 0.
- Experimental proof: monolithic oracle works; layer-0 external candidate is byte-identical; segmented hidden-state handoff passes max error <= 1e-5.
- Adapter: `llama`; DeepSeek is not disguised as Qwen.

## Verified fingerprint: BitNet4 / BitNet b1.58

- Architecture identifier: `bitnet-b1.58` in the local BitNet4 GGUF.
- Local reference: `/home/user/models/bitnet/bitnet-abliterated-i2s-run17.gguf`.
- Structural class: BitNet quantized transformer with architecture-specific BitNet matmuls and frozen production state behavior.
- Adapter: `bitnet`; it is a registration boundary around the existing production implementation, not a rewrite.
- Preservation rule: Forge does not copy or modify BitNet kernels. Existing BitNet regression evidence remains the source of truth.

## Adapter mapping

| Fingerprint family | Forge adapter | Core state contract |
|---|---|---|
| BitNet | `bitnet` | activations, KV cache, frozen BitNet state |
| Qwen2/Qwen2.5 | `qwen2` | activations, KV cache |
| Qwen3.5 hybrid | `qwen35` | activations, DeltaNet state, short-convolution state, KV cache |
| Qwen3.8 hybrid extension | `qwen38` | same state domains, four-section RoPE, `ssm_dt.bias` mapping |
| Llama/DeepSeek | `llama` | activations, KV cache |

The registry intentionally does not claim generic support for every model sharing a string. A fingerprint is accepted only when its layer sequence, tensor convention, dimensions, and state operations match the adapter contract and a regression passes.

## Independent implementation guidance

Another AI should read the fingerprint JSON first, inspect the corresponding frozen source reference, implement the adapter's graph/state methods, and then use a one-layer oracle/candidate test to locate the first divergence. It must not edit the oracle, use magic offsets, put persistent state into the reusable weight buffer, or add architecture-specific branches to the core. Once the test passes, add the exact fingerprint, model SHA256, command, and tolerance to this registry and the adapter regression suite.
