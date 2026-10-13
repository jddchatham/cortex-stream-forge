# Cortex Iteration 143 — projection error and memory forensics

Branch: `cortex-iteration-143`
Base: local Iteration 142 state on top of checkpoint `c227bd0`

## Preservation

The branch was created with `git switch -c cortex-iteration-143`; no reset,
stash, clean, checkout, commit, or push was used. Existing dirty Iterations
124–142 files and generated artifacts remain preserved and unstaged.

## Gate/reference reproduction

Build and test:

```text
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-flashnext-iq4xs-provider-test -j4
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  flashnext-tensor-index.json
```

Build exit: 0. Test exit: 7.

Observed output:

```text
IQ4XS_PROVIDER tokens=1 max_abs=7.4076e-05 max_rel=6.20632 rms=1.66929e-05 bytes=26624000
```

The deterministic input is nonzero (`0.01 * (1 + i % 31)`), uses real GGUF
layer-0 expert IDs 0/3 and routing weights 0.1. The provider output is finite.

Top absolute errors from the same run:

```text
rank index provider       reference       abs error       relative error
1    475   0.0004081833   0.0003341073    7.407597e-05    0.221713
2    1784 -0.0000257941   0.0000401701    6.596423e-05    1.642123
3    1492  0.0020434926   0.0019859981    5.749450e-05    0.028950
4    1897 -0.0006732736  -0.0007286112    5.533762e-05    0.075949
5    1002 -0.0026607302  -0.0026067062    5.402393e-05    0.020725
```

### Verified

- Real GGUF rows are used by both paths.
- Input dimensions are 2560 and output dimensions are 2560.
- The bounded provider executes and reads real rows.
- Iteration 141 independently validated exact captured production full-row
  quantized operands and CPU dot products for gate, up, and down.
- The failure is reproducible and is not accepted by relaxing tolerance.

### First remaining boundary

The existing full reference decodes weights and then locally re-encodes F32
activations through `q8k_roundtrip`/`q8_0_roundtrip` in
`flashnext-iq4xs-provider-test.cpp`. Production quantizes inside the GGML CPU
matmul. Since the exact production quantized bytes passed in Iteration 141,
the first unproven boundary in this full projection test is the independent
activation re-encoding. Weight-row decoding is not the leading explanation for
this failure. Exact production quantized bytes for this deterministic full
projection fixture have not yet been captured, so this remains the leading
hypothesis rather than a final root-cause claim.

### Not tested

- Ten largest errors for up/down separately.
- Sign-changing, sparse, and dense alternative vectors.
- Complete gate/up/down independent projection agreement.
- Gate/up activation and routed final aggregation agreement.

## Memory forensics

The real bounded runner was executed with a Python `/proc` monitor that sampled
`/proc/<pid>/status` and `smaps_rollup` while feeding `hello` and `/exit`.
The process exited normally:

```text
exit=0
elapsed=26.903 s
VmHWM=5,461,568 kB
Swap observed from smaps_rollup=1,323,436 kB
```

The run log retained the established token/logit result and approximately
1308.27 MiB scheduler CPU compute reservation. The provider scratch capacity
remained 2,150,400 bytes.

### Verified distinction

- Scheduler reservation: approximately 1308.27 MiB, a planned CPU compute
  buffer measurement.
- Process high-water RSS: approximately 5.21 GiB, measured independently.
- Process swap: approximately 1.26 GiB from `smaps_rollup`.
- These are not interchangeable measurements.
- No allocation reduction or ownership rewrite was attempted.

The monitor sampled process-local `VmHWM` and `smaps_rollup`; a complete
system-wide `/proc/meminfo` before/after swap accounting was not collected in
this run. Therefore the process-specific swap observation is valid, while its
contribution to system-wide swap pressure is not separately quantified.

## Zero-source producer trace

The source passed into the provider is explicit in the current implementation:

- Gate/up: `tx.pre_moe_hidden`.
- Down: local `act = silu(gate) * up`.

The Iteration 142 run counted:

```text
IQ4_XS: 1080 calls with 2560 nonzero source values; 8640 all-zero calls
IQ4_NL:  570 calls with  640 nonzero source values; 4560 all-zero calls
```

This identifies the producer-consumer boundary but does not prove whether the
zero values are deliberate placeholder/context work or a genuine upstream
initialization defect. Full transaction/token provenance for every call is not
yet present.

## Status

- Reproduce deterministic nonzero failure: **PASS**
- Gate full projection equivalence: **FAIL**
- Up full projection equivalence: **NOT TESTED**
- Down full projection equivalence: **NOT TESTED**
- Activation quantizer divergence: **HYPOTHESIS**
- GGUF weight-row divergence: **NOT SUPPORTED by Iteration 141 evidence**
- Memory high-water reproduction: **PASS**
- Scheduler reservation measurement: **PASS**
- Process swap measurement: **PASS**
- Bounded-memory target: **FAIL** for measured RSS/swap, despite bounded
  provider scratch
- Complete zero-source root cause: **NOT TESTED**
- Full model streaming: **NOT TESTED**
- `git diff --check`: **PASS**

## Smallest safe next action

Capture the exact production Q8_K/Q8_0 activation bytes for this deterministic
nonzero vector at the CPU boundary, then compare those bytes against the local
encoder before changing arithmetic. In parallel, add bounded token/context
identity to the zero-source counters. Do not optimize scheduler allocation or
alter production projection arithmetic until those two boundaries are proven.
