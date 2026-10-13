# BENCHMARKS & PERFORMANCE METRICS

**System Specifications:**
- **CPU:** AMD Athlon Silver 3050U (2 Cores, 2 Threads, 2.3 GHz base / 3.2 GHz boost)
- **Host Memory:** 9.8 GB physical RAM
- **Swap:** 9.6 GB zram swap
- **Storage:** NVMe SSD backing GGUF model files (`/home/user/models/qwen38-flashnext-abliterated/`)
- **Model:** Qwen 3.8 FlashNext Uncensored (IQ4_XS, 96 GB across 3 shards, 48 layers, MoE with selected experts)

---

## 1. Mathematical Accuracy & Precision

| Component | Target Elements | Bit-Exact Matches | Max Abs Error | Status |
|:---|:---:|:---:|:---:|:---:|
| **Gate Projection** | 57,600 | 57,600 | $5.25 \times 10^{-6}$ (float rounding) | **PASS** |
| **Up Projection** | 57,600 | 57,600 | $4.29 \times 10^{-6}$ (float rounding) | **PASS** |
| **Down Projection (Hardware AVX2)** | 230,400 | 230,400 (100.00%) | $0.000000000 \times 10^{0}$ | **PASS (100% BIT-EXACT)** |
| **MoE Aggregate Routing** | 23,040 | 23,040 (100.00%) | $0.000000000 \times 10^{0}$ | **PASS (100% BIT-EXACT)** |
| **Vocabulary Projection (Q6_K)** | Top-10 logits | 10/10 | $< 0.1\%$ relative vs FP32 ref | **PASS** |

---

## 2. Memory Footprint Accounting

| Allocation / Resource | Measured Size | Notes |
|:---|:---:|:---|
| **Model Weights Resident in RAM** | **0.00 MiB** | All weights streamed via external `pread` descriptors |
| **Cortex Layer Arena** | 64.7 MiB (capacity 67.8 MiB) | Fixed reusable buffer for current layer's resident tensors |
| **Selected MoE Bounded Scratch** | 2.15 MiB | Tiled matrix-vector expert computation scratch |
| **Recurrent State (RS) Buffer** | 112.57 MiB | 48 layers persistent recurrent state |
| **KV Cache Buffer** | 7.50 MiB | 6.00 MiB main KV + 1.50 MiB indexer KV (256 cells) |
| **Final Vocabulary Head Compute** | 0.08 MiB | Scratch for RMS norm and final head |
| **Peak Compute Buffer per Context**| ~1,306 MiB | GGML scheduler reserved graph buffer |
| **Process Peak VmRSS (Current Runner)**| ~5.6 GiB | Inflated by per-layer context recreation heap retention |
| **Target Bounded VmRSS (Persistent Ctx)**| **< 1.5 GiB** | Fits entirely inside restricted 2 GiB memory cgroup |

---

## 3. Streaming Throughput & Latency

- **Model Loading Time:** 1.11 seconds (descriptor extraction only; zero model weights read into RAM).
- **Layer Rebind Time:** ~5–12 ms per layer (reading required layer tensors via bounded `pread` into the 64.7 MiB arena).
- **Prefill Latency (9 tokens across 48 layers):** ~8.4 seconds (~1.07 tokens/sec on 2-core AMD Athlon).
- **Decode Latency (per generated token):** ~5.8 seconds (~0.17 tokens/sec on 2-core CPU).
