# Cortex Iteration 144 — activation quantization byte validation

Branch: `cortex-iteration-144`
Base: local Iteration 143 state; no commit or push

## Fresh deterministic capture

The existing deterministic nonzero provider test was run with:

```text
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-flashnext-iq4xs-provider-test -j4
rm -f /tmp/cortex-iter140-projection-*.bin
CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE=1 \
  runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  flashnext-tensor-index.json
```

The provider test still exits 7 because its historical full-MoE reference
threshold is exceeded. The fresh row captures are valid, nonzero, and complete.

Gate record:

```text
record bytes: 14608
projection/type: 0 / IQ4_XS
width: 2560
weight bytes: 1360
activation bytes: 2920
CPU dot: -0.023141495883464813
record SHA-256: c0780799ebe5d638d1575bb8f94d5a74357bd0862634d51d54a1553815103800
F32 input SHA-256: 08ceec947f45ed5b4a0e93482cd82ccb7e4d41726e2596e56b3f249e1ebd5c33
production Q8_K SHA-256: 1c95a4bf8eea8dba15645daacd82f43dea73c55f002a5de1212e32e93dae1925
```

The captured original F32 input is the same deterministic provider input used
for the failing full projection test. The production activation record is
2920 bytes: ten Q8_K blocks of 292 bytes each.

## Byte comparison

The original independent encoder from the Iteration 143 path used Python/C++
double-precision intermediates for the Q8_K scale calculation. Its output was:

```text
independent old SHA-256: d0cba3f27bc58665bfa5abe66eb476a1f7c1dda0193e34fbce19e924354c3953
byte mismatches: 10 / 2920
first mismatch: byte 0
```

All ten mismatches are the first four-byte `d` field of each 256-element
Q8_K block. The quantized values and block sums otherwise match. Repeating the
independent calculation with explicit float32 rounding at the GGML operation
boundaries (`iscale = float(-127 / max)`, `d = float(1 / iscale)`, and float32
input multiplication before `nearest_int`) produced:

```text
corrected independent SHA-256: 1c95a4bf8eea8dba15645daacd82f43dea73c55f002a5de1212e32e93dae1925
byte mismatches: 0 / 2920
```

This proves the old activation reference was numerically implemented with the
wrong precision boundary. It does not prove the complete 640-row gate matrix,
because the diagnostic intentionally accepts one row per projection.

## Row-level arithmetic

The gate record’s exact production quantized activation and weight bytes are
the same operands used by the independent scalar row calculation. The CPU dot
and scalar result agree within the Iteration 141 row-level tolerance. The
production Q8_K activation byte mismatch was therefore isolated to the old
encoder’s scale serialization, not weight layout, row indexing, or CPU dot
arithmetic.

Up and down fresh records were also emitted:

```text
up record SHA-256:   67e2dba3888d0add00ad8ffcbd6423b81298355dca4742e12325e5442fa20ffb
up activation SHA:   1c95a4bf8eea8dba15645daacd82f43dea73c55f002a5de1212e32e93dae1925
down record SHA-256: 739942a37418a660939f3bdcf113e4539d25a3136093d336d577ec79e9f550ef
down activation SHA: 4360471cf6e2283a04b6416bdb70104310ff1630ad36d792cbd57968b28f28e7
```

Only the gate Q8_K bytes were independently compared in this iteration.

## Status

- Isolated Iteration 144 branch: **PASS**
- Production Q8_K capture: **PASS**
- F32 input identity/hash: **PASS**
- Old independent Q8_K bytes: **FAIL** (10 scale-byte-field mismatches)
- Corrected float32 Q8_K bytes: **PASS**, 0/2920 mismatches
- Gate complete projection equivalence: **NOT TESTED** (only one row captured)
- Up activation byte equivalence: **NOT TESTED**
- Down Q8_0 activation equivalence: **NOT TESTED**
- Full gate/up/down aggregation: **NOT TESTED**
- Zero-source diagnostics: **PRESERVED**, not reclassified
- Scheduler/RSS optimization: **NOT TESTED**
- `git diff --check`: **PASS**

## Smallest justified next change

Replace the independent Q8_K/Q8_0 reference encoder’s double-precision scale
path with explicit GGML-compatible float32 intermediates, then capture a bounded
set of complete gate rows and compare full output vectors. Do not alter
production arithmetic or scheduler ownership based on this row-level result.
