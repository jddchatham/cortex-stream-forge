# Qwen3.8 Experimental Optimization Log

This document covers only the separate runtime under
`runtime-qwen35-exp/`. The golden Qwen3.8 runtime, `bin/qwen38-chat`, and the
existing Omarchy entry are not modified.

## Direct mmap benchmark

The experimental `CORTEX_DIRECT_MMAP=1` path binds immutable layer tensors
directly to their live GGUF mappings. It removes the mapped-range to external
layer-buffer copy while retaining the segmented graph and per-layer state
handoff.

Command:

```bash
CORTEX_DIRECT_MMAP=1 \
  runtime-qwen35-exp-build2/bin/llama-qwen38-lowres-sequential-chat \
  /home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6
```

Measured full 64-layer, two-step result:

| step | total | rebind | context creation | restore | decode | save | token |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54.2 s | 0.06 s | 3.2 s | 0 | 42.7 s | 2.2 s | 13 |
| 1 | 64.8 s | 0.14 s | 4.4 s | 10.0 s | 40.7 s | 2.1 s | 220 |

The run passed against the golden token sequence (`13`, `220`). RSS remained
bounded and returned to baseline after exit. The next bottleneck is CPU graph
execution; the former ~35.5-second weight-copy cost is gone.

## Partial-state experiment — rejected

The separate runtime was changed temporarily to use
`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` for per-layer save/restore. Step 0 still
produced `13`, but step 1 produced `198` instead of the golden `220`. This
shows that Qwen3.8’s per-layer state handoff includes state not represented by
the partial-state API, such as full-attention/KV or related context state.

The experimental source was restored to the full `llama_state_get_data` /
`llama_state_set_data` API. The partial-state path is not production code.

## Context pooling — rejected

Retaining all 64 contexts exceeded the host memory budget. Reusing a small
pool across different layers caused a layer-specific graph/output assertion.
It is not used.

## Current decision

No fast launcher or Omarchy entry has been created. The only usable launcher
remains the frozen golden baseline. Further optimization must reduce CPU graph
execution or introduce a safe layer-independent graph representation without
sharing stale recurrent/KV state.

## CPU operator profile

The experimental CPU backend was instrumented with the opt-in environment
variable `CORTEX_CPU_PROFILE=1`. It records worker-0 elapsed time per GGML
operator, a useful wall-time proxy because graph workers synchronize around
each node.

Full 64-layer, two-token direct-mmap profile:

| operator | executions | worker-0 time |
|---|---:|---:|
| `MUL_MAT` | 1,119 | 72.9 s |
| `SSM_CONV` | 96 | 1.80 s |
| `FLASH_ATTN_EXT` | 32 | 0.117 s |
| `GATED_DELTA_NET` | 96 | 0.119 s |
| `GET_ROWS` | 321 | 0.090 s |
| `CPY` | 192 | 0.029 s |

The graph contains the expected per-layer matrix multiplications; no large
unexpected copy, attention, or scheduler component was found. Thread testing
on the two-CPU host also passed the golden token sequence:

| configuration | step 0 decode | step 1 decode |
|---|---:|---:|
| 1 thread | 42.7 s | 45.1 s |
| 2 threads | 41.7 s | 41.0 s |

Two threads remain the experimental default. The full model is CPU-saturated;
the dominant `MUL_MAT` work is the Qwen3.8 computation itself rather than
duplicated graph setup. No fast launcher was created.

## CPU kernel and quantization research

The host is an AMD Athlon Silver 3050U with two physical CPUs and runtime
features AVX2, FMA, BMI1, BMI2, and F16C. The experimental build uses `-O3`,
`-march=native`, and OpenMP. The individual `GGML_AVX2`, `GGML_FMA`, and
`GGML_BMI2` cache variables being `OFF` is not a fallback: with `GGML_NATIVE=ON`
the GGML CMake logic emits `-march=native`, and the generated compile commands
confirm that flag.

For Q4_K tensors, the CPU type table selects
`ggml_vec_dot_q4_K_q8_K`. The source also contains the AVX2-supported tiled
K-quant path, enabled by default through `GGML_CPU_TILED_MM`. A controlled
one-layer comparison passed in both modes, but the full 64-layer comparison
was the meaningful result:

| path | step 0 | step 1 | tokens |
|---|---:|---:|---|
| default tiled K-quant | 54.2 s | 64.8 s | `13`, `220` |
| `GGML_CPU_TILED_MM=0` fallback | 56.8 s | 68.4 s | `13`, `220` |

The fallback is slower and is not retained as the preferred path. An explicit
non-native `-mavx2 -mfma -mbmi2` build was started as a separate comparison,
but concurrent build invocations corrupted that experimental build directory
at link time; it produced no benchmark and no conclusion. The native build was
not changed.

The local Ollama store contains only the 16.81 GB Q4 model for Qwen3.8. No
lower-bit 27B model was downloaded or added to the test set. Official/reputable
GGUF listings do exist online, including Q3_K_M-sized variants, but they would
be separate model fingerprints and require an explicit download before testing.
The upstream GGML CPU source documents Q4_K/Q3_K/Q2_K quantized dispatch and
the x86 quantized kernels; the experimental source already includes these
paths. No third-party kernel was copied into Forge.

Current fastest verified configuration remains:

```bash
CORTEX_DIRECT_MMAP=1 CORTEX_THREADS=2 \
  runtime-qwen35-exp-build2/bin/llama-qwen38-lowres-sequential-chat \
  /home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6
```

It preserves the golden tokens and bounded mmap-based memory behavior. No
`Qwen3.8 CPU Fast` shortcut is justified or created.
