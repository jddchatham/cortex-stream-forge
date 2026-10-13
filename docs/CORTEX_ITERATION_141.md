# Cortex Iteration 141 — independent review reconciliation

Date: 2026-10-13
Branch: `cortex-iteration-141`
Baseline: `2d0b8ffcdfc6df77b1baa95027b8db4f06cc8f4d`

## Scope

This checkpoint reconciles the Iteration 140 gate discrepancy, fixes the
gate/up capture alias, and records a fresh bounded layer-2 capture. No commit,
push, production deployment, or arithmetic change was made.

## Source correction

`src/flashnext-selected-moe.cpp` previously mapped both gate and up to capture
projection 0 (`type == DOWN_TYPE ? 2 : 0`). It now maps gate/up/down to
projection IDs 0/1/2. The atomic record header is version 2 and includes an
invocation ID plus layer, token, expert, row, and tile fields. Accepted records
remain one per projection and are not overwritten.

The current provider call has no token-position argument, so the emitted token
field is `0`; this is explicit metadata, not an inferred position. The fresh
records identify layer 2, expert 322, row 0, tile 0, and distinct invocation
IDs 0/4860/9720.

## Exact run

```text
cmake --build runtime-flashnext-layer-exp-build --target llama-qwen38-lowres-chat -j4
rm -f /tmp/cortex-iter140-projection-*.bin
timeout --kill-after=5s 100s bash -c \
  "printf 'hello\\n/exit\\n' | env CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE=1 \
   CORTEX_MOE_STREAM_GATE=1 CORTEX_MOE_STREAM_UP_DOWN=1 \
   CORTEX_MOE_BOUNDARY_CAPTURE=1 \
   CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
   runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
   /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

Build exit: 0. Runner exit: 0. Elapsed wall time: 24 seconds. The first
`/usr/bin/time` attempt was not a model run because this environment does not
provide `/usr/bin/time`; it exited 127 before launching the command.

Fresh capture log: `/tmp/cortex-iter141-run2.log`.

## Capture records

The C++ record header is 88 bytes (`<IIIIQQQQf4xQiiiii`), followed by the
complete quantized weight row, complete quantized activation row, and F32
activation row.

| projection | type / activation | width | record bytes | dot | SHA-256 |
|---|---|---:|---:|---:|---|
| gate (0) | IQ4_XS / Q8_K | 2560 | 14608 | -0.2227051705121994 | `d85bf32ae40c94d4d483f8305ab78437bfc9673226d1517b36c28cff1d749e19` |
| up (1) | IQ4_XS / Q8_K | 2560 | 14608 | 0.2050902098417282 | `aff9328de5caebc3d9f1dfa4305c0f8c4a0c45d16a356f0049ce005427283f53` |
| down (2) | IQ4_NL / Q8_0 | 640 | 3688 | -0.08832758665084839 | `189b4f59b13a66163ddcf770b640e81ccb9e175dd94f67ab3646dbbf853c8cf0` |

All records have magic `0x43585231`, version 2, layer 2, expert 322, row 0,
tile 0, and token field 0. Record payload sizes and hashes were checked after
the run; no stale files were used.

## Independent scalar reconciliation

The reference used the exact record payloads and independent scalar decoding
of the official IQ4_XS/Q8_K and IQ4_NL/Q8_0 layouts from `ggml-cpu/quants.c`.
It did not call production vec-dot or quantization functions.

| projection | CPU dot | scalar dot | absolute error | relative error |
|---|---:|---:|---:|---:|
| gate | -0.2227051705121994 | -0.2227051918040707 | 2.1292e-08 | 9.5606e-08 |
| up | 0.2050902098417282 | 0.2050902271576564 | 1.7316e-08 | 8.4431e-08 |
| down | -0.0883275866508484 | -0.0883275741074385 | 1.2543e-08 | 1.4201e-07 |

These results prove the corrected Kimi explanation for the gate value. The
original mismatch came from the prior scalar decoder/fixture interpretation,
not from a NaN in these fresh records. The current payloads are finite,
complete, and nonzero where accepted. Antigravity's contamination hypothesis
is not supported for the accepted records, although historical fixtures are
not retroactively reclassified.

## Zero-activation observation

The same run emitted 14,850 CPU operand diagnostics:

```text
type 23 (IQ4_XS): 1080 calls with 2560 nonzero source values,
                  8640 calls with zero nonzero-count
type 20 (IQ4_NL):  570 calls with 640 nonzero source values,
                  4560 calls with zero nonzero-count
```

Thus 13,200 calls had an all-zero F32 source. This is not sufficient to call
an inference defect: the run also produced the three accepted nonzero records,
and the zero calls occur in repeated provider/context work. The current source
does not yet attach token position and full transaction identity to every
provider call, so whether those zero inputs are deliberate zero-state/test
contexts or an actual upstream initialization defect remains unresolved.
The next discriminating test is a provider-local nonzero deterministic input
fixture with the same layer/expert/operation metadata, followed by comparison
of its gate/up/down/final aggregation outputs to independent scalar results.

## Memory

The run retained the existing bounded provider scratch configuration. Scheduler
diagnostics in the log report approximately 1308.27 MiB CPU compute-buffer
reservation; this is a reservation measurement, not a separately measured RSS
value. No full expert tensor was loaded by this checkpoint. Peak RSS was not
measured because `/usr/bin/time` is unavailable in this environment.

## Validation status

- Gate scalar discrepancy reconciliation: **PASS**.
- Capture identity correction: **PASS** for projection identity and fresh
  atomic records; token-position propagation remains incomplete.
- Nonzero gate/up/down operand records: **PASS**.
- Independent full-row dot products: **PASS** for all three accepted rows.
- Full gate/up/down projection equivalence: **NOT TESTED**.
- Final MoE aggregation equivalence: **NOT TESTED**.
- Zero-activation root cause: **UNRESOLVED**.
- Bounded scratch architecture: **PASS** for this run; scheduler reservation
  remains documented and unchanged.
- Full model streaming inference: **NOT TESTED**.

## Hashes and checks

```text
llama-qwen38-lowres-chat  00da662c12f26b3b362186876c82bab7835b42a2f73c688feb0eaf0aa914fbf9
libllama.so               523a496662b57ec638112e0ee8071ecd7362151047a38032a2046693262fbddf
git diff --check           PASS
```

The worktree contains pre-existing Iteration 124–140 changes and generated
files. They were preserved and were not staged, committed, or pushed.
