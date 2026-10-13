# Cortex Stream Forge — Iteration 124

## Result

The first invalid transition is established in `llama_model_base::cortex_rebind_layer()`:
the Qwen4Exp rebind path replaces the active selection with canonical
`arena_resident` records, while the layer-2 expert matrices are explicitly
classified `EXTERNALLY_STREAMED`. They are therefore not selected, not read by
the rebind `pread()` loop, and remain unbound when the graph is constructed.

The graph still uses those canonical expert leaves through
`GGML_OP_MUL_MAT_ID`, so scheduler allocation supplies zero-filled storage for
the full tensors. No safe bounded repair was applied in this iteration.

## Verification

Checkout:

- Branch: `integration/canonical-residency-manifest-20261011`
- HEAD: `85ed203e0a5f4b9ed749dd77a70f97bc9ed18d73`
- `git diff --check`: PASS

Command:

```sh
timeout --kill-after=5s 180s bash -c \
  "printf 'hello\\n/exit\\n' | env CORTEX_TRACE_EXPERT_TIMELINE=1 \
   CORTEX_MOE_BOUNDARY_CAPTURE=1 CORTEX_MOE_KERNEL_CAPTURE=1 \
   runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
   /home/user/models/qwen38-flashnext-abliterated/\
   Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

Relevant trace:

```text
CORTEX_REBIND checkpoint=ENTRY layer=2 selected_count=22
CORTEX_EXPERT_TIMELINE ... name=blk.2.ffn_gate_exps.weight present=NO storage=EXTERNALLY_STREAMED data=(nil) buffer=(nil)
CORTEX_EXPERT_TIMELINE ... name=blk.2.ffn_up_exps.weight present=NO storage=EXTERNALLY_STREAMED data=(nil) buffer=(nil)
CORTEX_EXPERT_TIMELINE ... name=blk.2.ffn_down_exps.weight present=NO storage=EXTERNALLY_STREAMED data=(nil) buffer=(nil)
CORTEX_REBIND checkpoint=EXIT layer=2 result=PASS selected_count=22 bytes=50238336
```

The layer-2 expert GGUF descriptors are present in the metadata index, but
their PREAD path is not entered because they are absent from `selected`.
Their known descriptors are:

| Tensor | Type | Payload | GGUF absolute offset |
|---|---:|---:|---:|
| `blk.2.ffn_gate_exps.weight` | IQ4_XS | 445644800 | 33017459616 |
| `blk.2.ffn_up_exps.weight` | IQ4_XS | 445644800 | 33469227936 |
| `blk.2.ffn_down_exps.weight` | IQ4_NL | 471859200 | 32544678816 |

At CPU dispatch, the graph still consumes the canonical gate tensor. The
runtime first row is zero while the GGUF row is nonzero:

```text
runtime SHA-256: b707241545a346265aab1ffb32ff64b55bf8f8dc1b56a46ef33ce3d15db11d33
GGUF SHA-256:    af687585072d2201455360eee8be1199364fb8da505c42538cf4ef7a44de474f
runtime nonzero bytes: 0/136
GGUF nonzero bytes:    134/136
```

The CPU operation receives a valid activation and expert ID, but the
destination becomes all zero after stock `GGML_OP_MUL_MAT_ID` execution. The
unchanged Iteration 122 destination fixture remains 230400 bytes and all
57600 values are zero; bounded replay mismatches 57586/57600.

## Why no repair was applied

Binding the canonical leaves directly would require making the full selected
expert matrices resident (approximately 1.36 GiB for layer 2), violating the
bounded-memory design. The existing selected-expert PREAD provider is not a
backend binding for the canonical full tensor and is not currently connected
to `ggml_mul_mat_id()`.

The smallest safe next action is to replace the graph dependency with a
bounded selected-expert operation/provider, preserving the exact GGML input,
expert IDs, quantization, and output layout. Directly attaching undersized
selected-expert storage to the canonical descriptors would be invalid.

## Evidence

- Run log: `/tmp/cortex-iter124-run.log`
- Build/run status: runner exit code `0`
- Historical destination fixture: `/tmp/cortex-iter122-l2-gate-kernel-output.bin`
- Full regressions and deployment validation: pending
- No commit or push performed for Iteration 124
