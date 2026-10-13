# Cortex Iteration 166 — Corrected Multi-Token MoE Numerics

## Preservation

- Branch: `cortex-iteration-166`
- Created from the dirty Iteration 165 worktree without reset, clean, stash,
  merge, commit, or push.
- Existing unrelated changes remain in place.

## Addressing audit

The provider input layout is token-major with `KSEL=10` selected-expert
slots. Gate/up callback input is populated in slot 0 of each token block;
down input is populated for every token/expert slot after the gate/up
combination.

The audited access paths are:

| Projection | Input offset | Output offset |
|---|---:|---:|
| Gate/up | `t * KSEL * input_width` | `(t*KSEL+s) * n_rows` |
| Down | `(t*KSEL+s) * input_width` | `(t*KSEL+s) * n_rows` |

The prior gate/up expression `t * input_width` was wrong for `t > 0`: it
addressed token 0's unused expert-slot tail. The corrected expression is used
for both gate and up. Bounds are consistent with the callback buffer layout:
gate/up input capacity is `n_tokens*KSEL*2560`, and down input capacity is
`n_tokens*KSEL*640`.

## Real multi-token execution

Command:

```sh
timeout --kill-after=5s 120s bash -lc "printf 'hello\\n/exit\\n' | env \
  CORTEX_FLASHNEXT_TOKEN_Q8_CAPTURE=1 \
  CORTEX_MOE_STREAM_LAYERS=2,3,4 \
  CORTEX_MOE_STREAM_GATE=1 \
  CORTEX_MOE_STREAM_UP_DOWN=1 \
  CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter153-layer2-4-index.json \
  runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
  /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

- Runner exit: PASS (0)
- Layer-4 gate/up: 9 tokens × 10 IDs, output 230400 bytes each, callback PASS
- Layer-4 down: 9 tokens × 10 IDs, output 921600 bytes, callback PASS
- Token 0 gate/up: 2560/2560 nonzero
- Token 1 gate/up: 2560/2560 nonzero; first/last `-0.021405/0.29972`
- Token 0 down: 640/640 nonzero
- Token 1 down: 640/640 nonzero; first/last `-0.11674/-0.0132519`
- Final token: `248046`
- Selected logit: `13.0355`
- Provider scratch: `2,150,400` bytes

This proves the Iteration 165 token-1 zero-input condition is eliminated in
the real nine-token execution. The run was repeated from the same executable
configuration only once in this iteration; repeatability beyond the prior
baseline was not independently quantified.

## Provider regression

```sh
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  /tmp/cortex-iter153-layer2-4-index.json 4
```

Result: PASS, exit 0.

- Three-token max absolute error: `3.4757e-06`
- Three-token RMS: `9.24548e-07`
- Existing acceptance bounds: max `1e-5`, RMS `2e-6`

## Independent oracle status

The available independent GGUF-backed oracle fixtures from Iteration 159 were
single/multi-token capture infrastructure, but no fresh synchronized token-0
and token-1 layer-4 gate/up/down oracle output comparison was executed here.
Therefore complete numerical equivalence is not claimed. The old
conventional layer-4 path remains invalid because its expert buffers were
zero-filled.

| Stage | Status | Evidence |
|---|---|---|
| Token addressing audit | PASS | Corrected and audited offsets above |
| One-token provider regression | PASS | Existing regression executable |
| Multi-token gate projection execution | PASS | 9-token callback and nonzero token-1 input |
| Multi-token up projection execution | PASS | 9-token callback and nonzero token-1 input |
| Multi-token down projection execution | PASS | 9-token callback and nonzero token-1 input |
| Independent token-0 gate/up/down oracle | NOT TESTED | No fresh synchronized oracle run |
| Independent token-1 gate/up/down oracle | NOT TESTED | No fresh synchronized oracle run |
| Aggregate numerical equivalence | NOT TESTED | Provider regression is not the real layer-4 oracle |
| Full final-logit equivalence | NOT TESTED | No valid conventional oracle |
| Bounded memory | PASS for scratch only | Scratch measured; whole RSS not measured here |
| Full-model streaming inference | NOT TESTED | Only configured runner path exercised |

## Validation

`git diff --check`: PASS (exit 0).

## Conclusion and next milestone

Iteration 165 resolved the demonstrated token-1 provider addressing defect,
and Iteration 166 confirms corrected nonzero multi-token gate/up/down execution
through layer 4. The next milestone is a fresh synchronized independent
GGUF-backed oracle comparison for all token-0/token-1 gate, up, SiLU×up, down,
and weighted aggregate tensors. No additional streamed layers or memory
optimization should be enabled before that comparison passes.
