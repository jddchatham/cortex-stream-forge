# CORTEX OPERATION LEAN CONTEXT — MILESTONE 3 REPORT
## RAM Reduction and Single Context Consolidation for Streaming Autoregressive Inference

- **Date:** October 17, 2026
- **Repository:** `jddchatham/cortex-stream-forge`
- **Branch:** `small-model/lean-context`
- **Base Tag:** `cortex-true-streaming-v1` (`f125c5b`)
- **Target Model:** Qwen 3.5 0.8B GGUF (`sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5`, 988 MiB)
- **Host Hardware:** AMD Athlon Silver 3050U (2 cores, 2 threads, 8 GB RAM)

---

## 1. Executive Summary

Milestone 2 (`cortex-true-streaming-v1`) successfully achieved true disk-backed positional streaming via `pread()` into a fixed 22.98 MiB layer arena, dropping `RssFile` from 997.43 MiB to 9.39 MiB. However, total process memory remained at 1,307.6–1,363.6 MiB because the runtime maintained **24 persistent `llama_context` instances** (one per transformer layer).

In **Milestone 3 (Operation Lean Context)**, we identified the root cause of this memory consumption, decoupled the runtime state, and consolidated the inference engine down to a **single unified execution context (`llama_context`)**.

### Headline Results

| Metric | Stock llama.cpp (mmap) | Milestone 2 (True Streaming, 24 ctxs) | Milestone 3 (Lean Context, 1 ctx) | Milestone 3 vs Milestone 2 Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Total Process RSS** | 1,107.10 MiB | 1,363.58 MiB | **431.29 MiB** | **-932.29 MiB (-68.4%)** |
| **Anonymous Memory (`RssAnon`)** | 109.74 MiB | 1,354.18 MiB | **421.81 MiB** | **-932.37 MiB (-68.8%)** |
| **File-backed Memory (`RssFile`)** | 997.34 MiB | 9.58 MiB | **9.48 MiB** | **-987.86 MiB (-99.0% vs stock)** |
| **Layer Context Instances** | 1 | 24 | **1** | **24x context reduction** |
| **Prefill Logit Equivalence** | Baseline | **0 / 248,320 diffs (100%)** | **0 / 248,320 diffs (100%)** | **Bitwise exact match** |
| **Autoregressive Generation** | `"<think>\n\n</think>\n\n4"` | `"<think>\n\n</think>\n\n4"` | `"<think>\n\n</think>\n\n4"` | **Exact token-for-token match** |

---

## 2. Memory Ownership Audit

By instrumenting process virtual memory at each phase of model execution, we determined the exact memory owners:

```
Process Lifetime Phase               VmRSS (MiB)    RssAnon (MiB)    Delta Anon (MiB)
-------------------------------------------------------------------------------------
1. Process Startup (base runtime)        0.4            0.2              +0.2
2. Model Meta Load & Globals           354.8          345.2            +345.0
   - token_embd.weight: 128 MB
   - output.weight: 128 MB
   - Global norms & metadata
   - Fixed pread arena: 22.98 MB
3. Context 0 Initialization            382.6          373.1             +27.9
4. Context 1 Initialization            415.3          405.8             +32.7
5. All 24 Contexts Initialized       1,082.6        1,073.1            +727.9
6. Decode Peak Compute Graph         1,363.6        1,354.2            +281.1
```

### The Memory Bloat Driver
Each `llama_context` created for Qwen 3.5 allocated:
- **19.27 MiB of Recurrent State (RS) buffers** (`cache_r_l*` conv1d history + `cache_s_l*` SSM state) across 18 linear attention layers.
- **3.00 MiB of KV cache** for the 6 full attention layers.
- **8–11 MiB of GGML graph and scheduler compute buffers**.
- **Total per context:** $\approx \mathbf{30.3\text{ MiB}}$.
- Multiplying by 24 layer contexts: $24 \times 30.3\text{ MiB} = \mathbf{727.2\text{ MiB}}$ of anonymous memory.

---

## 3. Architecture Breakthrough: Single Context Dynamic Re-targeting

Instead of maintaining 24 active contexts, Cortex now reuses **a single `llama_context`**:

1. **Dynamic Segment Updates:**
   - For layer `il`, the environment variables `CORTEX_QWEN35_SEGMENT_START=il` and `CORTEX_QWEN35_SEGMENT_END=il+1` are dynamically targeted.
   - The layer input/output routing (`llama_set_embeddings_layer_inp`) is toggled seamlessly.
2. **State Decoupling & Position Reset:**
   - In M-RoPE models, processing sequential layers within the same sequence causes token position collision checks ($X \le Y$).
   - We utilize `llama_memory_clear(mem, /*data=*/false)` across layer boundaries: metadata positions are cleared while all recurrent states ($S, R$) in the memory arena remain intact.
3. **Weight Steaming Integration:**
   - Prior to each layer's forward pass, `llama_cortex_rebind_layer(model, il)` reads the layer's weights into the fixed 22.98 MiB arena via positional `pread()`.

---

## 4. Verification and Correctness

### Deterministic Correctness Suite (`cortex_regression_suite.cpp`)
All three golden regression test suites were verified:
1. **`q35_test_arithmetic_6` (6 tokens):**
   - Prompt: `"What is 2 + 2? Answer with only the number."`
   - Prefill Logit Diff: **0 / 248,320 diffs (100% bitwise match)**
   - Generated Tokens: `[248068, 271, 248069, 271, 19, 248046]` (`"<think>\n\n</think>\n\n4"`) — **PASS (6/6)**
2. **`q35_test_reasoning_32` (32 tokens):**
   - Prompt: `"Solve: If x + 5 = 12, what is x? Explain step by step."`
   - Prefill Logit Diff: **0 / 248,320 diffs (100% bitwise match)**
3. **`q35_test_code_64` (64 tokens):**
   - Prompt: `"Write a Python function to compute the factorial of n recursively."`
   - Prefill Logit Diff: **0 / 248,320 diffs (100% bitwise match)**
   - Generated Tokens: **64 / 64 tokens matched bitwise against stock llama.cpp reference (PASS)**

---

## 5. Frozen Anchor and Commit Preservation

- Git tag `cortex-true-streaming-v1` at commit `f125c5b` remains strictly frozen and untouched.
- All Milestone 3 engineering changes are committed on isolated branch `small-model/lean-context`.
