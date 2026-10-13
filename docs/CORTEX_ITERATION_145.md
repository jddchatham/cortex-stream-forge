# Cortex Iteration 145 — complete bounded aggregate validation

Branch: `cortex-iteration-145`
Base: local Iteration 144 state; no commit or push

## Reference correction

The provider test reference in `examples/flashnext-iq4xs-provider-test.cpp`
was corrected to independently reproduce GGML-compatible float32 Q8_K and
Q8_0 activation encoding. It does not call production quantization or vec-dot
functions. The reference now:

- uses float32 operation boundaries for Q8_K scale and reciprocal;
- uses the Q8_0 FP16 scale conversion;
- computes gate/up from independently decoded GGUF rows and quantized
  activations;
- computes SiLU(gate)*up;
- quantizes that activation through the independent Q8_0 path;
- computes down and routed expert aggregation independently.

The production provider and reference use the same deterministic nonzero input,
expert IDs 0/3, routing weights 0.1, and real GGUF rows. No production
arithmetic or model data was changed.

## Commands and result

```text
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-flashnext-iq4xs-provider-test -j4
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  flashnext-tensor-index.json
```

Build exit: 0. Provider test exit: 0.

One-token comparison:

```text
max_abs=3.09898e-06
rms=7.49719e-07
max_rel=2.12121
```

Three-token comparison:

```text
max_abs=3.67127e-06
rms=8.08196e-07
max_rel=3.84638
```

The absolute/RMS acceptance bound is `max_abs <= 1e-5` and `RMS <= 2e-6`.
This bound is applied to the complete routed aggregate because relative error
is ill-conditioned for outputs near zero. Relative error remains reported and
was not used to manufacture a pass.

Largest three-token absolute errors:

```text
index 5698: provider=0.00514050061 reference=0.00513682934 abs=3.67127359e-06
index 4519: provider=-0.00617424306 reference=-0.00617067656 abs=3.56649980e-06
index 3972: provider=-0.00626880908 reference=-0.00626535388 abs=3.45520675e-06
```

The input is nonzero and the output is nontrivial; zero-only comparisons were
not used.

## Coverage boundaries

The complete provider path executes gate, up, SiLU/gate multiplication, down,
expert weighting, and aggregation. The independent reference covers the same
complete routed aggregate and agrees within the stated absolute/RMS bounds.

Separate complete output-vector captures for gate, up, and down are not yet
implemented. Therefore those stage-specific claims remain NOT TESTED even
though their operations execute as part of the aggregate path. The prior
`llama-flashnext-l2-moe-reference` fixture run is historical evidence only; it
uses earlier reference fixtures and reported large mismatches, so it was not
used as a current pass criterion.

## Status

- Deterministic nonzero input: **PASS**
- Corrected independent Q8_K reference: **PASS** from Iteration 144 byte test
- Corrected independent Q8_0 reference: **PASS** at the aggregate comparison
- Complete routed aggregate: **PASS** for 1-token and 3-token cases
- Gate stage vector comparison: **NOT TESTED**
- Up stage vector comparison: **NOT TESTED**
- Down stage vector comparison: **NOT TESTED**
- Gate/up activation intermediate comparison: **NOT TESTED** as a separate
  captured vector
- Selected-expert weighting and final aggregation: **PASS** in aggregate
- Historical failing reference: **PRESERVED**, not relabeled
- Bounded provider scratch: **PASS** at the configured 16 KiB arena
- Full RSS/memory-forensics rerun: **NOT TESTED** in this iteration
- Full-model streaming inference: **NOT TESTED**
- `git diff --check`: **PASS**

## Next action

Add bounded stage-output capture for all 640 gate/up rows and 2560 down rows
for one deterministic token/expert set. Compare those vectors independently
before treating stage-level gate/up/down equivalence as complete. Preserve the
current aggregate regression and do not alter the production quantization path.
