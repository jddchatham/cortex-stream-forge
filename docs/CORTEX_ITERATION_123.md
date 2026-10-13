# Cortex Stream Forge — Iteration 123

## Scope

This iteration traced the real layer-2 gate `GGML_OP_MUL_MAT_ID` execution
without changing production arithmetic, weights, routing, or streaming.

Repository/checkout:

- Branch: `integration/canonical-residency-manifest-20261011`
- HEAD before diagnostics: `385144efbca026ac48417aed1d94c473e2a7f11b`
- `git diff --check`: pass

## Executed trace

Command:

```sh
timeout --kill-after=5s 180s bash -c \
  "printf 'hello\\n/exit\\n' | env CORTEX_MOE_BOUNDARY_CAPTURE=1 \
   CORTEX_MOE_KERNEL_CAPTURE=1 CORTEX_MOE_KERNEL_TRACE=1 \
   runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
   /home/user/models/qwen38-flashnext-abliterated/\
   Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

Relevant execution evidence:

```text
CORTEX_MOE_DISPATCH_ENTRY ... src0_data=... weight_hash136=f15c5115137197e3 activation0=0.0782519 ne=[640,10,9,1]
CORTEX_MOE_DISPATCH_EXTRA handled=0 ... first=0.0149312
CORTEX_MOE_DISPATCH_AFTER_STOCK first=0 id0=322 ...
CORTEX_MOE_KERNEL_CAPTURE ... bytes=230400 nonzero=0 finite=57600 min=0 max=0
CORTEX_CHAT TOKEN position=8 id=248046 logit=10.9936
```

The CPU extra/repack path did not handle this node (`handled=0`); the stock
CPU `GGML_OP_MUL_MAT_ID` implementation executed. The destination was
nonzero before stock computation, then zero immediately after the stock
operation and remained zero after the worker barrier.

## First concrete failure

The layer-2 gate weight storage presented to the CPU operation is not the
real GGUF payload. The runtime first-row capture
`/tmp/cortex-iter123-runtime-gate-row0.bin` is 136 zero bytes. The matching
GGUF row at descriptor offset `33017459616` contains 134 nonzero bytes.

```text
runtime SHA-256: b707241545a346265aab1ffb32ff64b55bf8f8dc1b56a46ef33ce3d15db11d33
GGUF    SHA-256: af687585072d2201455360eee8be1199364fb8da505c42538cf4ef7a44de474f
byte differences: 134/136, first byte: 0
```

Therefore the earliest proven loss is before the dot product: the canonical
layer-2 gate tensor is bound to zero-filled storage instead of its real
IQ4_XS payload. The stock kernel receives a valid nonzero activation and a
valid expert ID (`322`), but has no valid weight bytes to multiply.

## Replay

The completed kernel destination fixture remains the Iteration 122 file:
`/tmp/cortex-iter122-l2-gate-kernel-output.bin`.

It is 230400 bytes, all 57600 values finite and zero. Bounded replay against
that unchanged fixture reports:

```text
max_abs=2.94772 rms=0.322075 mismatches=57586/57600 exit=7
```

The original inference path still exits 0 and returns token `248046`, logit
`10.9936`; this does not establish gate correctness.

## Remaining uncertainty and next action

The exact binding call that leaves the layer-2 canonical expert tensor
zero-filled is not changed here. The next discriminating test should trace
the tensor through model loading, `cortex_rebind_layer(2)`, scheduler
reservation, and PREAD binding, then bind the verified layer-2 gate payload at
the earliest valid ownership boundary and re-run the same kernel capture.

Diagnostics are test-gated by `CORTEX_MOE_KERNEL_TRACE` and
`CORTEX_MOE_KERNEL_CAPTURE`.

## Build/evidence hashes

- Runner exit code: `0`
- Build logs: `/tmp/cortex-iter123-build5.log`
- Run log: `/tmp/cortex-iter123-run5.log`
- Runner SHA-256: `00da662c12f26b3b362186876c82bab7835b42a2f73c688feb0eaf0aa914fbf9`
- `libllama.so` SHA-256: `73ac0b14b76231ea1a3cd01202c2bda9e4081fdf3a895c420927547bec15b537`
- `libggml-cpu.so` SHA-256: `4298f72a2bdc0ab04c41487c126ab9c423e5341daa1ff366d97e0c8f1eb7cef0`

Full regressions and deployment validation remain pending. No commit or push
was performed for Iteration 123.
