# Cortex Iteration 159 — GGUF-backed layer-4 numerical oracle

## Scope

Work continued on an isolated `cortex-iteration-159` branch preserving the
dirty Iteration 158 state. No whole-model conventional loading, arithmetic
change, routing change, or streamed-layer selection change was made.

## Real input transaction

The real runner used the existing three-layer streamed configuration
(`CORTEX_MOE_STREAM_LAYERS=2,3,4`) and captured physical layer 4. The graph
boundary reported shape `[2560,1,9,1]` for the activation, `[10,9,1,1]` for
IDs, and `[640,10,9,1]` for gate output. All fixtures came from the same run:

| fixture | bytes | SHA-256 |
|---|---:|---|
| activation | 92160 | `5d31b8081e24c92a09fcf96cb6063cdfd7208db79f328488dff635b0d48bb4d` |
| selected IDs | 360 | `af142844978613f976515c4f0636194dbd023fb1413ec8948437a837d1d5fabb` |
| streamed up | 230400 | `6f1fbe3c98aa9ff7da6d7b8d950e880c46fda96f58eb0bcbb502c6d4f9d6a5b4` |
| streamed down | 921600 | `a6e18e3f98abc131545a7aa60d3a4b2a5b1d39c355ab5b142fce2dd013ab6a74` |
| streamed aggregate | 92160 | `cfde8a3685b1ec67816268508a0226eddd7a5e8ba7a511d374c2492bb773104c` |

The runner completed with exit 0, token `248046`, selected logit `11.2143`.

## Oracle implementation

The existing standalone scalar evaluator was extended with an explicit layer
argument. It reads only selected expert rows directly from the descriptor
index and GGUF files using bounded `pread()` buffers. It independently decodes
IQ4_XS and IQ4_NL rows, independently quantizes the captured activation using
Q8_K/Q8_0 scalar logic, computes gate/up, applies SiLU(gate)*up, computes down,
and performs routed aggregation. It does not call the streamed provider's
projection implementation.

The layer-4 GGUF descriptors were the verified entries in
`/tmp/cortex-iter153-layer2-4-index.json`; the conventional zero-filled
scheduler buffers were not used.

## Numerical result

Command:

```text
runtime-flashnext-layer-exp-build/bin/llama-flashnext-l2-moe-reference \
  /tmp/cortex-iter153-layer2-4-index.json \
  /tmp/cortex-iter159-l4-x.bin /tmp/cortex-iter159-l4-ids2.bin \
  /tmp/cortex-iter159-l4-refup.bin /tmp/cortex-iter159-l4-refdown.bin \
  /tmp/cortex-iter159-l4-refout.bin 4
```

| stage | elements | max abs | RMS | first mismatch >1e-4 | status |
|---|---:|---:|---:|---:|---|
| gate | 57600 | 2.99211 | 0.444845 | 6400 | FAIL |
| up | 57600 | 2.08128 | 0.367934 | 6400 | FAIL |
| down | 230400 | 3.27526 | 0.0743469 | 0 | FAIL |
| aggregate | 23040 | 1.05916 | 0.0612104 | 0 | FAIL |

The first gate/up mismatch at element 6400 corresponds to the next expert
slice in the flattened `[output, expert, token]` layout. This narrows the next
investigation to selected-expert ordering, row/tile indexing, or output-layout
mapping rather than the first expert's scalar arithmetic. Down divergence at
element 0 may also reflect the preceding gate/up mismatch and should not yet
be treated as an independent down-decoder defect.

No tolerance was relaxed. The oracle and streamed path are not numerically
equivalent at this stage.

## Validation status

| check | status |
|---|---|
| real layer-4 activation/ID capture | PASS |
| canonical GGUF row access | PASS |
| bounded oracle execution | PASS |
| gate projection equivalence | FAIL |
| up projection equivalence | FAIL |
| down projection equivalence | FAIL |
| routed aggregate equivalence | FAIL |
| full layer-4 residual equivalence | NOT TESTED |
| full-model streaming inference | NOT TESTED |
| provider regression | PASS in prior/current experimental build |
| bounded memory | PASS for row-bounded oracle design; full RSS not measured here |

## Next action

Compare the exact flattened output ordering and selected-ID-to-row mapping for
the second expert slice, then validate one complete gate row using the same
captured activation and GGUF bytes. Do not change production arithmetic or
claim layer-4 correctness until that mapping discrepancy is resolved.
