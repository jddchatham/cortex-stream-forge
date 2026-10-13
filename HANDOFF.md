# HANDOFF — ANTIGRAVITY AUTONOMOUS INVESTIGATION

**Branch:** `antigravity/cortex-development`  
**Starting Commit:** `593c30c862ca401a4699f690f79524722d461290` (Cortex Iteration 179)  
**Isolated Worktree:** `/home/user/bitnet-cortex/cortex-antigravity-development`  
**Status:** **Phase 3 Mathematical Verification 100% Complete; Real Full-Model 48-Layer Streaming Proven; Phase 4 Underway**

---

## 1. Key Accomplishments

1. **Resolution of the Iteration 170–179 Blocker:**
   - Proved that the 7 remaining down mismatches on expert 252 were an artifact of the oracle harness recalculating SwiGLU activations using unrounded float32 approximations.
   - Proved that the production AVX2 kernel (`ggml_vec_dot_iq4_nl_q8_0` with `quantize_row_q8_0`) is **100.00% bit-exact (0.000000000e+00 max error)** across all 230,400 elements of Layer 4 down-projection when given the actual runtime activations.
   - Proved that Layer 4 MoE aggregate routing is **100.00% bit-exact (0.000000000e+00 max error)** across all 23,040 elements.
   - Added automated regression target: `llama-flashnext-down-aggregate-verify`.

2. **Full End-to-End Model Streaming Demonstrated:**
   - Ran `llama-qwen38-lowres-chat` on the real 96 GB 3-shard model file (`Qwen3.8-Flash-Next-Uncensored-IQ4_XS`).
   - Prefill processed 9 prompt tokens across all 48 layers with zero weights resident in RAM.
   - Bounded scratch: 2.15 MiB.
   - Generated token `248046` with logit 9.94831; top-10 logits verified against reference Q6_K dequantization ($< 0.1\%$ error).
   - Generated multiple consecutive tokens through 48-layer continuous decoding.

3. **Memory Accounting Root Cause Identified:**
   - Traced why VmRSS peaked at ~5.6 GiB despite the 64.7 MiB layer arena: `qwen38-lowres-chat.cpp` repeatedly creates and destroys `llama_context` for each of the 48 layers, reserving ~1.3 GiB compute buffers each time. Glibc heap fragmentation and thread retention accumulate resident memory.
   - Documented the fix in `ROADMAP.md` and `ENGINEERING_LOG.md`.

---

## 2. Verification Commands

To reproduce the mathematical proof:
```sh
cd /home/user/bitnet-cortex/cortex-antigravity-development
cmake -B build runtime-flashnext-layer-exp -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build --target llama-flashnext-down-aggregate-verify -j2
./build/bin/llama-flashnext-down-aggregate-verify
```

To run the end-to-end streaming chat runner:
```sh
./build/bin/llama-qwen38-lowres-chat /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf
```
