# Cortex Iteration 156 — Real-Activation Layer-4 Numerical Forensics

Date: 2026-10-13
Branch: `cortex-iteration-156`
Base: preserved dirty Iteration 155 state; no commit or push.

## A/B setup

Both runs used the same GGUF, prompt, backend, runtime settings, and combined
layer-2/3/4 descriptor registry. The only toggle was:

```text
A: CORTEX_MOE_STREAM_LAYERS=2,3
B: CORTEX_MOE_STREAM_LAYERS=2,3,4
```

The new test-only `CORTEX_MOE_CAPTURE_LAYER=4` selector emitted layer-qualified
fixtures. Layer-4 input and selected-ID fixtures are bit-identical:

```text
activation SHA-256: 5d31b8081e24c92a09fcf96cb6063cdfd7208db79f328488dff635b0d48bb4d9
IDs SHA-256:        af142844978613f976515c4f0636194dbd023fb1413ec8948437a837d1d5fabb
```

Both have input shape `[2560,1,9,1]`, IDs shape `[10,9,1,1]`, and the same
layer-4 transaction metadata.

## Execution results

Control A completed successfully:

```text
exit 0, token 248046, logit 10.7801
```

Streamed B completed successfully:

```text
exit 0, token 248046, logit 11.2143
```

Layer 4 gate/up/down streaming callbacks completed in B. No additional layer
was enabled.

## First proven forensic boundary

The conventional A capture is invalid. Its synchronized captured gate output,
down output, and residual output are all zero-filled, while A completed real
inference and B produced nonzero corresponding tensors. Nonzero counts:

| Fixture | A nonzero | B nonzero |
|---|---:|---:|
| Gate output | 0 / 57,600 | 6,400 / 57,600 |
| Down output | 0 / 230,400 | 25,600 / 230,400 |
| Layer output | 0 / 23,040 | 2,560 / 23,040 |

Therefore direct A/B numerical errors are not valid evidence of a production
projection mismatch. The first demonstrated divergence is the capture boundary:
the conventional graph result is being read from a stale, unmaterialized, or
recycled buffer. This must be repaired before interpreting layer-4 arithmetic.

For completeness, comparing the invalid A buffers against B gives misleading
values such as gate max abs `5.80044`, RMS `0.160893`, and down max abs
`3.27607`, RMS `0.0743789`; these are explicitly rejected as model-error
measurements because A is all zero.

Fixture hashes:

```text
A gate: 2a589ae1f2fa2a6328223ff195a29c9244bec633dca49139f6f231e1d79c0eb2
B gate: 6e8f8a3aacbfa2c9bcfdb4f8d5a2fd45085cf202008b89817957f27b22394ad7
A down: 0b150fd32588b1daca5569992ebe559c0102c837306b1af4c44d35128ec58366
B down: a6e18e3f98abc131545a7aa60d3a4b2a5b1d39c355ab5b142fce2dd013ab6a74
```

## Provider validation

The existing real-weight layer-4 provider regression remains passing:

```text
one token:   max_abs=3.19351e-6, RMS=7.9655e-7
three token: max_abs=3.86965e-6, RMS=9.53667e-7
```

That test uses deterministic nonzero provider inputs, not the invalid A graph
capture, so it does not resolve the end-to-end A/B discrepancy.

## Status

| Check | Status | Evidence |
|---|---|---|
| Preserve Iteration 155 | PASS | isolated branch, dirty state retained |
| Identical layer-4 input | PASS | activation and ID hashes match |
| A completed | PASS | exit 0, token/logit recorded |
| B completed | PASS | exit 0, token/logit recorded |
| B nonzero layer-4 captures | PASS | nonzero gate/down/output fixtures |
| A trustworthy layer-4 captures | FAIL | all captured outputs are zero |
| First model arithmetic divergence | NOT TESTED | A boundary is invalid |
| Full-logit equivalence | NOT TESTED | invalid intermediate reference |
| Provider regression | PASS | established tolerance bounds |
| Memory optimization | NOT TESTED | out of scope |
| `git diff --check` | PASS | clean |
| Full-model streaming | NOT TESTED | not demonstrated |

## Next action

Replace the conventional A capture with a backend-safe retained graph output or
post-compute staging copy at the actual conventional layer-4 node. Verify A has
nonzero gate/down/residual values before any A/B numerical comparison. Do not
change provider arithmetic or enable additional streaming layers until that
capture is trustworthy.
