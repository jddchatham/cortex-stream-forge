# Cortex Stream Forge — Iteration 125

## Result

The layer-2 gate dependency was replaced behind the test-only
`CORTEX_MOE_STREAM_GATE=1` gate.  The canonical expert tensor remains a
metadata-only GGML leaf; it is not assigned a short buffer.  Instead, the
graph now contains a CPU `GGML_OP_MAP_CUSTOM3` node whose dependencies are the
real gate activation and selected-expert ID tensors.  Its callback invokes the
existing bounded `flashnext_execute_selected_gate_rows_layer()` provider and
writes the correctly shaped F32 result `[640,10,n_tokens,1]` for the normal
downstream MoE operations.

The callback uses the real layer-2 activation and IDs computed by the graph,
loads only the selected IQ4_XS rows from the GGUF index named by
`CORTEX_FLASHNEXT_TENSOR_INDEX`, and runs with one CPU task so the provider's
bounded scratch ownership lasts through the complete operation.  The stock
`MUL_MAT_ID` gate leaf is not expanded into the executed graph in this mode.

## Commands and results

Build:

```sh
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-qwen38-lowres-chat -j2
```

Exit code: `0`.

Baseline:

```sh
timeout --kill-after=5s 60s bash -c \
  "printf 'hello\\n/exit\\n' | env \
   runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
   /home/user/models/qwen38-flashnext-abliterated/\
   Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

Streamed gate:

```sh
timeout --kill-after=5s 60s bash -c \
  "printf 'hello\\n/exit\\n' | env CORTEX_MOE_STREAM_GATE=1 \
   CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
   runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
   /home/user/models/qwen38-flashnext-abliterated/\
   Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1"
```

Both commands exited `0` and returned:

```text
CORTEX_CHAT TOKEN position=8 id=248046 logit=10.9936
```

The streamed run emitted:

```text
CORTEX_MOE_STREAM result=PASS layer=2 tokens=9 ids=10 output_bytes=230400
```

The captured streamed gate output was 230400 bytes / 57600 F32 values,
57600 nonzero and finite values, with range `[-1.7813099623, 2.9477152824]`.
The existing replay executable compared that captured fixture against the
same bounded provider over all 57600 elements: `max_abs=0`, `rms=0`,
`mismatches=0`, exit code `0`.  This confirms provider self-consistency and
real nonzero selected-row execution; an independent scalar IQ4_XS gate oracle
was not added in this iteration.

The streamed run reported `bytes=521472000` for the later vocabulary scan and
`scratch=2150400` bytes.  Scheduler reservations remain 1308.27 MiB for the
prompt graphs and 0.08 MiB for the generation graph; this patch does not yet
remove the unrelated prompt-graph reservations.

Historical Iteration 122 evidence is unchanged: the stock gate destination
was all zero and its bounded replay mismatch was 57586/57600.  The new
streamed fixture is nonzero and must not overwrite that historical fixture.

## Files and validation

Changed source:

- `runtime-flashnext-layer-exp/src/llama-graph.cpp`

New documentation:

- `docs/CORTEX_ITERATION_125.md`

Relevant evidence:

- `/tmp/cortex-iter125-build-final.log`
- `/tmp/cortex-iter125-baseline.log`
- `/tmp/cortex-iter125-streamed.log`
- `/tmp/cortex-iter125-capture.log`
- `/tmp/cortex-iter125-hashes.txt`
- `/tmp/cortex-iter120-l2-gate-activation.bin`
- `/tmp/cortex-iter120-l2-gate-ids.bin`
- `/tmp/cortex-iter120-l2-gate-output.bin`

Executable SHA-256:

```text
00da662c12f26b3b362186876c82bab7835b42a2f73c688feb0eaf0aa914fbf9  llama-qwen38-lowres-chat
15d5a8bd47f555f0f35b7829e516c45e2cbaa1b9bdb851b90f80f848cef98831  libllama.so
```

`git diff --check` passes.  No commit or push was performed.  Existing dirty
worktree changes were preserved.

## Remaining limitation and next action

Only the gate projection is integrated.  The up and down expert projections
still use their canonical graph leaves and therefore remain unresolved for
bounded production execution.  The next safe step is to apply the same
real-input, shape-preserving boundary to the up projection, then the down
projection, comparing each against an independently computed real-weight
reference before replacing the next canonical dependency.  This iteration
does not claim end-to-end bounded MoE or full inference success.
