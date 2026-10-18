# CORTEX OPERATION FASTER STREAMING — MILESTONE 4 REPORT
## Evaluation of Asynchronous Double-Buffered Weight Prefetching Under Low-Memory Constraints

- **Date:** October 18, 2026
- **Repository:** `jddchatham/cortex-stream-forge`
- **Branch:** `small-model/faster-streaming`
- **Frozen Baseline Tags:**
  - `cortex-true-streaming-v1` (`f125c5b`) — Milestone 2 (True streaming, 24 contexts)
  - `cortex-lean-context-v1` (`d158bb5`) — Milestone 3 (Lean context, 1 context)
- **Target Model:** Qwen 3.5 0.8B GGUF (`sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5`, 988 MiB)
- **Host Hardware:** AMD Athlon Silver 3050U (2 cores, 2 threads, 8 GB RAM, NVMe/eMMC storage)

---

## 1. Executive Summary

In Milestone 4 (**Operation Faster Streaming**), we evaluated whether **asynchronous double-buffered `pread()` weight prefetching** can accelerate streaming autoregressive inference toward 1 token/sec or beyond on dual-core hardware without compromising bounded-memory operation or numerical correctness.

### Key Conclusions:
1. **Correctness Confirmed:** Asynchronous double-buffered layer prefetching was implemented and validated with **0 / 248,320 prefill logit differences (exact $0.000\text{e}+00$ maximum error)** and **100% token-for-token matches across 6-, 32-, and 64-token golden reference tests**.
2. **Memory Ceiling Preserved:** Under Lean Context single-context execution, dual-arena double buffering raised peak RSS by only +9.53 MiB (from 447.63 MiB to 457.16 MiB), remaining comfortably within the 477 MiB hard ceiling.
3. **Severe Hardware Bottleneck on Dual-Core CPU:**
   - On the 2-core AMD Athlon Silver 3050U, running an asynchronous I/O thread in parallel with 2-thread GGML compute induced **severe core scheduling contention, cache thrashing, and OS thread serialization**.
   - As a result, synchronous execution takes **410.01 ms/tok (2.44 tok/s)**, whereas asynchronous prefetch degraded latency to **638.35 ms/tok (1.57 tok/s)** — a **35.8% throughput penalty**.
4. **Engineering Verdict:** Asynchronous prefetching **must remain disabled by default** on low-core machines (1–2 cores). The runtime retains optional double-buffered prefetching via explicit API calls, but synchronous streaming is preserved as the high-performance production default.
5. **Next Breakthrough Direction:** Compute optimization (specifically GDN state vectorization and kernel fusion) and eliminating redundant layer weight transfers are the true high-yield avenues for speedup.

---

## 2. Phase 0: Milestone 3 Frozen Baseline & Reproducibility

Prior to developing Milestone 4, Milestone 3 was formally audited, verified, and tagged:
- **Git Tag:** `cortex-lean-context-v1` pointing to commit `d158bb5`.
- **Preceding Tag:** `cortex-true-streaming-v1` at `f125c5b` remains strictly untouched.
- **Regression Parity:**
  - `q35_test_arithmetic_6`: Prefill Diff 0 / 248,320, 6 / 6 tokens match (`PASS`)
  - `q35_test_reasoning_32`: Prefill Diff 0 / 248,320, 32 / 32 tokens match (`PASS`)
  - `q35_test_code_64`: Prefill Diff 0 / 248,320, 64 / 64 tokens match (`PASS`)
- **Token-20 Reasoning Audit:** The previous divergence reported at token 20 was confirmed to be a test artifact of multi-context test runner isolation and position metadata indexing in attention layers, completely eliminated by proper position handling in the unified context.

---

## 3. Phase 1: Performance Baseline & Profile Breakdown

We profiled the exact breakdown of time spent during synchronous decode on the AMD Athlon Silver 3050U:

| Component | Time (ms/step) | Percentage of Decode Step |
| :--- | :--- | :--- |
| **GGML Compute Execution** | 249.11 – 272.53 ms | **65.8%** |
| **Synchronous `pread()` Weights** | 127.73 – 136.52 ms | **33.8%** |
| **Context Routing & Overhead** | 1.55 – 1.90 ms | **0.4%** |
| **Total Step Latency** | **378.38 – 410.01 ms** | **100.0%** (~2.44 – 2.64 tok/s) |

### Memory and I/O Profile:
- **Transferred Bytes per Token:** 24 layers $\times$ ~22.98 MiB = **551.54 MiB/token**.
- **Page Cache Behavior:** When page cache holds the 988 MiB model, positional `pread()` takes ~5.3 ms per layer (~128 ms total per token). In uncached conditions, flash read throughput bounds latency, but CPU compute remains twice as long as the read time.

---

## 4. Phase 2: Double-Buffered Asynchronous Prefetch Implementation

An optional double-buffered prefetching mechanism was integrated into `llama-model.cpp`:

1. **Dual Aligned Arenas:**
   - Arena 0 (`cortex_pread_arena`): 22.98 MiB fixed buffer.
   - Arena 1 (`cortex_prefetch_arena`): 22.98 MiB fixed buffer.
2. **Asynchronous Dispatch (`llama_cortex_prefetch_layer`):**
   - While the GGML compute graph processes layer $N$ using active arena $A$, a background thread (`std::async`) initiates positional `pread()` calls for layer $N+1$ into alternate arena $1-A$.
3. **Safe Synchronization (`llama_cortex_wait_prefetch` & `cortex_rebind_layer`):**
   - When layer $N+1$ begins, `cortex_rebind_layer` checks if layer $N+1$ was prefetched. It waits for the future to complete, validates read bytes, swaps active arena index `active_arena_idx = 1 - active_arena_idx`, and binds layer tensor pointers.
   - If no prefetch was scheduled or a short read occurred, synchronous fallback is automatically invoked.
   - Destructors safely drain in-flight futures before deallocating memory.

---

## 5. Phase 3: Correctness and Numerical Parity Verification

The asynchronous double-buffered implementation was tested against stock `llama.cpp` using the full regression suite (`cortex_regression_prefetch.cpp`):

| Test Case | Test Description | Prefill Logit Differences | Max Absolute Error | Token ID Match | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`q35_test_arithmetic_6`** | Arithmetic (2+2) | **0 / 248,320** | **0.000000e+00** | **6 / 6 tokens** | **PASS** |
| **`q35_test_reasoning_32`** | Step-by-Step Reasoning | **0 / 248,320** | **0.000000e+00** | **32 / 32 tokens** | **PASS** |
| **`q35_test_code_64`** | Factorial Code Generation | **0 / 248,320** | **0.000000e+00** | **64 / 64 tokens** | **PASS** |

Generated text output under prefetch is bitwise identical:
```
<think>

</think>

4
```

---

## 6. Phase 4: Comparative Benchmarks & Performance Analysis

### Multi-Run Benchmark Results (Median of 3 Runs)

| Benchmark Metric | Synchronous Baseline | Asynchronous Double-Buffer | Delta / Impact |
| :--- | :--- | :--- | :--- |
| **Decode Step Latency** | **410.01 ms** | **638.35 ms** | **+228.34 ms (+55.7% slower)** |
| **Pread / Rebind Wait Latency**| **136.52 ms** | **325.16 ms** | **+188.64 ms (+138.2%)** |
| **Compute Execution Latency** | **272.53 ms** | **302.79 ms** | **+30.26 ms (+11.1%)** |
| **Decode Throughput** | **2.44 tok/s** | **1.57 tok/s** | **-0.87 tok/s (-35.8%)** |
| **Peak VmRSS** | **447.63 MiB** | **457.16 MiB** | **+9.53 MiB (+2.1%)** |
| **Peak `RssAnon`** | **435.97 MiB** | **447.68 MiB** | **+11.71 MiB** |
| **Peak `RssFile`** | **9.40 MiB** | **9.48 MiB** | **+0.08 MiB** |
| **Swap Activity** | **0 kB** | **0 kB** | **Zero swap** |

### Why Did Asynchronous Prefetch Make Execution 35.8% Slower?
1. **Core Starvation on Dual-Core CPU:** The Athlon 3050U possesses only 2 physical CPU cores and 2 threads. GGML utilizes both cores for matrix multiplication. Spawning an asynchronous thread for `pread()` forces the Linux CFS scheduler to time-slice between intensive matrix multiplication and kernel I/O, resulting in frequent context switching.
2. **Memory Bus and L3 Cache Eviction:** While the compute thread is reading hidden states and recurrent state arrays from L2/L3 cache, the I/O thread is concurrently streaming 23 MiB of raw weights across the memory bus, evicting cache lines needed by the active layer.
3. **Thread Synchronization Latency:** Waiting on `std::future::get()` introduces synchronization and thread wake-up latency (~10–15 ms per layer transition) that outweighs the theoretical overlap.

---

## 7. Phase 5: Architectural Decision & Next Steps

### Verdict:
- **Do not enable asynchronous prefetch by default on dual-core hardware.**
- Synchronous `pread()` streaming provides the cleanest, fastest (2.44–2.64 tok/s), and most memory-efficient (~430–447 MiB RSS) execution path on 2-core machines.
- The double-buffering prefetch code is preserved in `libllama` as an opt-in capability for multi-core systems (4+ cores) where dedicated I/O threads do not preempt compute threads.

### Next Recommended Optimization (Compute Breakthrough):
1. **SSM/GDN SIMD Vectorization:** Since compute consumes ~66% of step latency (250–270 ms per token), optimizing recurrent state math via AVX2/FMA vectorization will yield direct speedup without memory overhead.
2. **Selective / Static Residency for Hot Weights:** Layers or tensors with heavy reuse (such as norm scales or shared attention projections) can be selectively pinned if memory allows.
