# QUORTEX CPU streaming validation

This source snapshot contains the QUORTEX CPU-only layer-streaming executor and its validated support changes: 48-layer single-token execution, hybrid-state continuity for generation, context-owned diagnostic capture cleanup, opt-in immutable loader metadata/vocabulary reuse, and bounded selected-expert staging with a 16 KiB default and optional 1 MiB arena (`CORTEX_FLASHNEXT_EXPERT_ARENA_BYTES=1048576`).

## Validation status

The external QIES-062 run report records a fresh REPACK=ON, metadata-cache-enabled, 1 MiB-arena, 32-token run. All 32 complete 248,320-float logits vectors and the greedy token history matched the preserved stock llama.cpp oracle bit-for-bit. The report also records completed layer captures and state-store integrity checks. These results apply to that recorded CPU configuration; this document does not claim that every layer boundary was independently compared with a stock layer-boundary oracle for all 32 positions.

The QIES-063 investigation measured total context creation from preserved timing data (about 263 ms per context) but did not add internal constructor-phase timers. It made no source changes and established no constructor optimization. Cross-model context reuse remains unproven because scheduler buffers, hybrid memory, output buffers, model tensor references, and diagnostic captures have context/model lifetimes.

The unexplained QIES-061 SIGKILL remains unresolved. Its cause must not be inferred from the successful QIES-062 rerun.

Detailed logs, resource samples, hashes, and comparisons are maintained outside this source repository in the QUORTEX diagnostic workspace. This summary does not replace those run artifacts.
