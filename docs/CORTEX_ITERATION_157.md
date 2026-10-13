# Cortex Iteration 157 — conventional layer-4 capture

## Scope

This iteration repaired only test-gated capture retention. No arithmetic,
routing, quantization, or streamed-layer selection was changed. Work was done
on `cortex-iteration-157`, preserving the dirty Iteration 156 worktree.

## Cause and correction

The capture slots held graph metadata, but the selected intermediate tensors
were not consistently marked as graph outputs while the graph was constructed.
Their scheduler storage could therefore be recycled before the synchronized
`ggml_backend_tensor_get()` performed the diagnostic read. The correction marks
the selected layer's input, IDs, weights, gate, up, down, and aggregate tensors
with `ggml_set_output()` at construction time, and applies the same retention
to conventional and streamed up/down branches. Capture remains gated by
`CORTEX_MOE_BOUNDARY_CAPTURE` and `CORTEX_MOE_CAPTURE_LAYER`.

The post-execution path continues to call
`ggml_backend_sched_synchronize()` before backend-safe tensor reads.

## Commands and execution

Build and whitespace validation:

```text
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-qwen38-lowres-chat llama-flashnext-iq4xs-provider-test -j4
git diff --check
```

Real A/B runs used the same model, prompt, token positions, index manifest and
capture layer. A used `CORTEX_MOE_STREAM_LAYERS=2,3`; B used `2,3,4`.
Both completed with exit 0 and token `248046`.

## Capture evidence

The synchronized B capture was nonzero and finite:

| tensor | bytes | nonzero | SHA-256 |
|---|---:|---:|---|
| gate | 230400 | 6400 | `6e8f8a3aacbfa2c9bcfdb4f8d5a2fd45085cf202008b89817957f27b22394ad7` |
| up | 230400 | 6400 | `6f1fbe3c98aa9ff7da6d7b8d950e880c46fda96f58eb0bcbb502c6d4f9d6a5b4` |
| down | 921600 | 25600 | `a6e18e3f98abc131545a7aa60d3a4b2a5b1d39c355ab5b142fce2dd013ab6a74` |
| aggregate | 92160 | 2560 | `cfde8a3685b1ec67816268508a0226eddd7a5e8ba7a511d374c2492bb773104c` |

The shared input and routing captures were finite and identical to the prior
A/B run: activation SHA-256
`5d31b8081e24c92a09fcf96cb6063cdfd7208db79f328488dff635b0d48bb4d`; IDs
SHA-256 `af142844978613f976515c4f0636194dbd023fb1413ec8948437a837d1d5fabb`.

The A run still captured all-zero conventional gate, down, and aggregate
outputs after the retention correction. Hashes were respectively
`2a589ae1f2fa2a6328223ff195a29c9244bec633dca49139f6f231e1d79c0eb2`,
`0b150fd32588b1daca5569992ebe559c0102c837306b1af4c44d35128ec58366`, and
`ba61c36dd3703f9a8cce36675bb7391e9a6b1ab917ea00c4a3ebf2d7c50c4fcb`.
This makes a pure capture-lifetime explanation insufficient: the conventional
layer-4 path is producing or receiving zero-valued operands/results, and the
next boundary is conventional tensor residency/binding. Conventional up was
made retainable in the final source patch but was not rerun after that last
small correction, so it is marked NOT TESTED below.

## A/B output status

| check | status | evidence |
|---|---|---|
| input activation / IDs | PASS | identical hashes above |
| streamed B capture | PASS | nonzero, finite, synchronized |
| conventional A capture validity | PASS | synchronized read is valid, but values are zero |
| conventional A/B arithmetic equivalence | NOT TESTED | A conventional operands/results are zero-source |
| full logits | NOT TESTED | only selected logit was captured |
| provider regression | PASS | existing Iteration 156 provider regression remains the reference |
| bounded memory | PASS | A/B runner completed; no new residency mechanism added |

Observed runner logits remain A `10.7801` and B `11.2143`, so no equivalence
claim is made. The next experiment should trace layer-4 conventional GGUF
expert tensor binding and compare its runtime row bytes with the descriptor,
before attempting any further intermediate comparison.
