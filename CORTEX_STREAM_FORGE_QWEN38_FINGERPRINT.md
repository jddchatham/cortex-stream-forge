# Qwen3.8 27B Abliterated Fingerprint

## Identity

- Ollama model: `huihui_ai/Qwen3.8-abliterated`
- Manifest: `/home/user/.ollama/models/manifests/registry.ollama.ai/huihui_ai/Qwen3.8-abliterated/latest`
- GGUF blob: `/home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6`
- GGUF SHA256: `6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6`
- Size: 16,810,714,400 bytes
- GGUF: v3, 32-byte alignment, 866 tensors

## Metadata fingerprint

- Architecture identifier: `qwen35` (the model is called Qwen3.8, but its GGUF architecture key is qwen35).
- Parameter count from the runtime metadata: 27.32B.
- GGUF block count: 65; runtime trunk count: 64 plus one MTP/next-token block.
- Hidden size: 5120.
- FFN size: 17408.
- Attention heads: 24.
- KV heads: 4.
- Key/value head dimensions: 256.
- SSM inner size: 6144.
- SSM state size: 128.
- SSM timestep rank: 48.
- SSM group count: 16.
- Short convolution kernel: 4.
- Full-attention interval: 4.
- RoPE dimension sections: four entries `[11, 11, 10, 0]`; the fourth plane is zero.
- Layer schedule: recurrent Gated DeltaNet layers at indices 0,1,2,4,5,6,...; full attention at 3,7,11,...,63; block 64 is the additional MTP block.
- State: recurrent DeltaNet matrix state and short-convolution state for recurrent layers; attention KV cache for full-attention layers.
- Tensor convention: `blk.N.*`, but the recurrent timestep tensor is `blk.N.ssm_dt.bias`, not the older `blk.N.ssm_dt` name.
- Layer-0 recurrent tensor set observed: attention gate/norm/QKV, FFN down/gate/up, post-attention norm, `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm`, and `ssm_out`.
- GGUF-derived layer-0 recurrent tensor bytes: 252,263,296 bytes across 14 tensors. A representative full-attention layer 3 totals 233,760,768 bytes across 11 tensors.

## Adapter decision

The existing Qwen3.5 adapter is computationally close but cannot consume this fingerprint unchanged. Qwen3.8 was registered as a distinct Forge adapter key, `qwen38`, while reusing the Qwen3.5 graph implementation in a Forge-local runtime copy.

The smallest demonstrated compatibility extensions are:

1. Accept four RoPE section metadata entries. The existing graph already stores four entries and consumes the first three sections.
2. Resolve `blk.N.ssm_dt` first and fall back to `blk.N.ssm_dt.bias` when the former is absent. The graph uses the resolved tensor as the timestep bias.
3. Keep per-layer schedule handling from `full_attention_interval`; a scalar KV-head metadata value must not be interpreted as every layer being full attention.

These changes are in the Forge-local copy:

`runtime-qwen35-src/src/models/qwen35.cpp`

The original `/home/user/bitnet-cortex/cortex-stream-qwen35-src` was not modified.

## Oracle and streaming status

The unmodified existing Qwen3.5 runtime rejected the model before weight execution with:

`key qwen35.rope.dimension_sections has wrong array length; expected 3, got 4`

The Forge-local compatibility runtime was rebuilt. It then passed metadata/tensor registration and reached CPU repacking/graph reservation, reporting the correct 27.32B dimensions. It exposed the next model-specific tensor-layout difference (`ssm_dt.bias`), which was fixed in the Forge-local copy.

Full monolithic inference and streamed oracle/candidate comparison are not claimed on this host. The host has approximately 9.6 GiB RAM, while the GGUF is 16.81 GB. The first unconstrained attempt was stopped during repacking/graph setup to avoid exhausting the machine. No Qwen3.8 external-layer result is reported as verified.

The Forge-local runtime now bypasses that limitation for isolated segments. In
low-residency mode it calls `init_mappings(false)` (avoiding whole-file mmap
prefetch), uses no full backend weight allocation, and reads global tensors
plus one selected layer into a reusable external buffer.

The dedicated native segment executable is:

~~~bash
cmake -S runtime-qwen35-src -B runtime-qwen35-build -DCMAKE_BUILD_TYPE=Release
cmake --build runtime-qwen35-build --target llama-qwen38-lowres-segment-test -j2
runtime-qwen35-build/bin/llama-qwen38-lowres-segment-test <GGUF> 0 1
runtime-qwen35-build/bin/llama-qwen38-lowres-segment-test <GGUF> 3 4
~~~

Observed results:

~~~text
qwen38 lowres segment: PASS layer=0 end=1 hidden_checksum=12.5023612
qwen38 lowres segment: PASS layer=3 end=4 hidden_checksum=70.9446554
~~~

Construction timings were approximately metadata 0.2 s, vocabulary 0.7 s,
tensor registration 0.01 s, mappings 0.01 s, and selected-layer binding 1.9–2.1
s. Layer 0 used 2,010,389,376 bytes and layer 3 used 1,991,886,848 bytes.

The complete sequential 64-layer rebind, recurrent-state transfer, KV transfer,
final logits, and interactive chat remain unverified. The native isolated
segment entry is now proven; no Qwen3.8 Omarchy shortcut has been created.

This is a genuine resource blocker, not an oracle modification or numerical mismatch. A machine with sufficient RAM must run the following before registration is promoted from fingerprinted to verified:

~~~bash
model=/home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6
exe=/home/user/bitnet-cortex/cortex-stream-forge/runtime-qwen35-build/bin/llama-simple
$exe -m "$model" -p hello -n 1 -ngl 0 > /tmp/q38-oracle.out 2>/tmp/q38-oracle.log
CORTEX_STREAM_EXTERNAL_LAYER=0 $exe -m "$model" -p hello -n 1 -ngl 0 > /tmp/q38-ext0.out 2>/tmp/q38-ext0.log
cmp /tmp/q38-oracle.out /tmp/q38-ext0.out
~~~

Then add a Qwen3.8 recurrent/full-attention segmented regression using the same state handoff method as the Qwen3.5 tests. Until those commands pass, `qwen38` verification intentionally returns an unverified status.

## Reproduction of fingerprinting

~~~bash
python3 /home/user/bitnet-cortex/cortex-stream-forge/cortex-forge \
  --model /home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6 \
  --plan
~~~

This performs GGUF header/key-value/tensor-name inspection and SHA256 hashing without loading model weights into a runtime context.

## Historical provisional status

The earlier resource-blocker and no-launcher notes in this document describe the
pre-low-residency state. They are retained as history; the updated status below
records the completed Forge-local path.

## Updated implementation status

The earlier provisional status above is superseded by the Forge-local
low-residency implementation. The GGUF remains the same blob and fingerprint;
the `qwen38` adapter now routes to a dedicated segmented executor based on the
Qwen3.5 graph implementation, without pretending the model is Qwen3.5.

The loader uses metadata-only tensor descriptors and direct GGUF range reads.
`llama_cortex_rebind_layer()` loads only the selected `blk.N.*` tensors into a
reusable layer buffer; global/final tensors have a separate buffer. The
sequential executor runs layers 0 through 63 one at a time, carrying a
serialized state vector per layer. This preserves Gated DeltaNet and
short-convolution state for recurrent layers and KV state for full-attention
layers. The full trunk passed for two steps, with selected tokens `13` and
`220`; layer 0, layer 1, and full-attention layer 3 provider/graph checks also
pass. The observed maximum layer buffer is 252,263,296 bytes and the global
buffer is 1,758,126,080 bytes.

Reproduction from the Forge root:

```bash
cmake --build runtime-qwen35-build -j2
runtime-qwen35-build/bin/llama-qwen38-lowres-sequential-chat \
  /home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6
printf 'Hi\n/exit\n' | CORTEX_QWEN38_MAX_TOKENS=1 bin/qwen38-chat
```

The second command produced a response token on this host. Normal chat
diagnostics are suppressed by `bin/qwen38-chat`. The Omarchy shortcut is the
new `qwen38-27b-abliterated-chat` entry in
`~/.config/omarchy/extensions/omarchy-menu.jsonc`; the pre-existing Qwen3.5,
Qwen3.5 9B, DeepSeek, and Forge entries were not changed.
