# Cortex Iteration 168 — Down-Projection Activation and Q8_0 Forensics

## Preservation and execution

- Branch: `cortex-iteration-168`
- Iteration 167 work and unrelated dirty changes preserved.
- No reset, clean, stash, commit, push, merge, or production change.
- Build: `cmake --build runtime-flashnext-layer-exp-build --target llama-qwen38-lowres-chat -j4` — PASS.
- Real runner exit: `0`.
- Token: `248046`; selected logit: `13.0355`.

## Down-input capture

The debug-only `CORTEX_MOE_DOWN_FORENSICS=1` capture records the exact F32
buffer passed to the layer-4 down provider before any backend transfer:

`/tmp/cortex-iter168-l4-down-input.bin`

- Shape: `[token=9, selected-slot=10, width=640]`
- Bytes: `230400`
- Nonzero: `57600/57600`
- SHA-256: `bf99dae2881b9acc94e6d898f51522186c2300abe66b9db0b3d6a69bc27dc012`

The provider uses down offset `(t*KSEL+s)*input_width`; the capture confirms
all token/expert slots are populated and bounded.

Using the fresh independent gate/up oracle outputs, independently computing
`SiLU(gate)*up` in F32, and comparing against the captured provider down input:

| Token | Slots | Max abs | RMS | Status |
|---:|---:|---:|---:|---|
| 0 | 10/10 | `2.41252e-05` | `4.47522e-07` | FAIL under `1e-5` |
| 1 | 10/10 | `1.52042e-06` | `8.53661e-08` | PASS |
| all 9 | 90/90 | `2.41252e-05` | `1.77287e-07` | FAIL under `1e-5` |

Token 1—the original problematic token—therefore has a correctly aligned
down input within the established tolerance. The larger token-0 error is
consistent with propagation of small gate/up provider-vs-oracle differences,
but no arithmetic change is justified here.

## Q8_0 capture

The fresh CPU-boundary run enabled `CORTEX_FLASHNEXT_TOKEN_Q8_CAPTURE=1` and
`CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE=1`. It emitted 20 write-once down Q8_0
records for layer 4 token 1/expert 257, each with:

- Q8_0 row bytes: `680`
- F32 source count: `640`
- Capture marker: `status=PASS`

Representative fixture names use the existing key format
`/tmp/cortex-iter162-p2-l4-t1-e257-{q,f32}.bin`. A complete byte-for-byte
comparison against a separately implemented Q8_0 encoder was not executed in
this iteration; therefore Q8_0 byte identity is NOT TESTED, not PASS.

## Stage result

Iteration 167’s fresh oracle comparison remains the numerical reference:

- Gate token 0: PASS, max abs `5.24521e-06`.
- Gate token 1: PASS, max abs `3.57628e-06`.
- Up token 0: PASS, max abs `4.29153e-06`.
- Up token 1: PASS, max abs `3.45707e-06`.
- Down token 0: FAIL, max abs `3.275258`.
- Down token 1: FAIL, max abs `0.512695`.
- Aggregate: FAIL downstream of down.

The down-input comparison proves the first major boundary is not a token
stride or expert-slot population loss. Remaining candidates are Q8_0
quantization/dot semantics, IQ4_NL row interpretation, or a smaller
provider/oracle activation rounding difference. No production arithmetic was
changed.

## Status

| Check | Status |
|---|---|
| Down F32 token/expert layout | PASS |
| Token-1 down-input identity | PASS within `1e-5` |
| Token-0 down-input identity | FAIL under established max bound |
| Q8_0 capture emitted | PASS |
| Q8_0 byte equivalence | NOT TESTED |
| IQ4_NL row identity | NOT TESTED in this iteration |
| Down projection equivalence | FAIL |
| Aggregate equivalence | FAIL |
| Provider regression | PASS (prior max `3.4757e-06`, RMS `9.24548e-07`) |
| Bounded scratch | PASS (`2,150,400` bytes) |
| Full-model streaming | NOT TESTED |

`git diff --check`: PASS.

## Next action

Use one exact token-1/expert-257 Q8_0 capture to independently encode the
captured 640-element F32 row, compare all 680 bytes, then compare its IQ4_NL
weight row and scalar dot result against the CPU result. This is the smallest
discriminating test remaining before changing any down implementation.
