# Cortex Iteration 142 — nonzero projection validation

Branch: `cortex-iteration-142`
Base checkpoint: `c227bd0` (pushed as `cortex-iteration-141`)

## Checkpoint

Iteration 141 was committed and pushed before this work. The Iteration 142
branch was created from that commit. Pre-existing Iterations 124–140 worktree
changes and generated files remain untouched and unstaged.

## Zero-source trace

The diagnostic `source_nonzero` value is collected at the CPU `MUL_MAT`
boundary from the actual F32 `src1` passed to `tiled_matvec`.

Provider source mapping is explicit:

- Gate and up receive `tx.pre_moe_hidden`.
- Down receives the provider-local `act`, computed as
  `silu(gate) * up`.

The fresh real run produced 14,850 bounded CPU projection calls:

```text
IQ4_XS gate/up: 1080 calls with 2560 nonzero source values
               8640 calls with zero source values
IQ4_NL down:    570 calls with 640 nonzero source values
               4560 calls with zero source values
```

This proves where the zero observations occur, but not why upstream calls are
zero. They are not the accepted nonzero capture records and do not establish a
model defect. The next required discriminator is to tag every provider
transaction with the originating token/context and compare the zero source to
the corresponding `pre_moe_hidden` or `silu(gate)*up` fixture. A synthetic
nonzero input can validate the provider independently, but cannot prove that
the model's zero inputs are correct.

## Deterministic nonzero provider test

Command:

```text
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-flashnext-iq4xs-provider-test llama-flashnext-l2-moe-reference -j4
python3 .../proc VmHWM wrapper \
  runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  flashnext-tensor-index.json
```

The test uses deterministic nonzero input values `0.01 * (1 + i % 31)`, real
GGUF layer-0 IQ4_XS/IQ4_NL rows, selected expert IDs 0/3, and routing weight
0.1. It exercised the real bounded provider and its separate scalar reference.

Observed result:

```text
IQ4XS_PROVIDER tokens=1 max_abs=7.4076e-05 max_rel=6.20632 rms=1.66929e-05 bytes=26624000
exit=7
elapsed=0.859 s
VmHWM=11404 kB
```

The provider returned finite output, but the existing independent full-MoE
reference exceeded its `max_rel <= 5e-4` threshold. This is a numerical FAIL,
not a successful equivalence result. The next correction must localize the
first complete projection/output element and account for the exact production
Q8_K/Q8_0 quantization and accumulation semantics; row-dot agreement alone is
insufficient.

## Real bounded runner and memory

Command used:

```text
python3 /proc VmHWM+smaps_rollup wrapper \
  runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
  /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1
```

The wrapper fed `hello` and `/exit`, enabled the existing streamed gate/up/down
diagnostics and used `/tmp/cortex-iter120-layer2-index.json`.

```text
exit=0
elapsed=26.903 s
VmHWM=5,461,568 kB
Swap observed from smaps_rollup=1,323,436 kB
```

The runner produced the established token/logit result in its log and the
bounded provider scratch capacity remained 2,150,400 bytes. Scheduler
reservation and RSS are separate measurements: the log reports approximately
1308.27 MiB scheduler CPU compute reservation, while the observed process
high-water RSS was approximately 5.21 GiB with swap activity. No memory cap was
changed and no complete expert tensor was intentionally loaded by the bounded
provider path.

## Status

- Zero-source producer classification: **PASS** (producer paths identified;
  root cause unresolved).
- Deterministic nonzero input fixture: **PASS**.
- Complete gate projection equivalence: **FAIL** (existing reference threshold
  exceeded; earliest complete-output divergence not yet isolated).
- Complete up projection equivalence: **NOT TESTED**.
- Complete down projection equivalence: **NOT TESTED**.
- Gate/up activation combination: **NOT TESTED** independently.
- Selected-expert weighting/final aggregation: **NOT TESTED** independently.
- Scheduler reservation accounting: **PASS** as a measurement; approximately
  1308.27 MiB reserved.
- Actual RSS/swap measurement: **PASS**.
- Bounded-memory target: **FAIL** for the measured process high-water RSS;
  scratch itself remained bounded, but RSS and swap were substantial.
- Full-model streaming inference: **NOT TESTED**.

## Checks

```text
git diff --check: PASS
```

No Iteration 142 commit or push was performed.
