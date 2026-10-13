# CORTEX Stream Forge — Iteration 137

## Verified call graph

```text
layer-2 MAP_CUSTOM3 callback
  -> flashnext_execute_selected_projection_rows_layer()
  -> tiled_matvec() in src/flashnext-selected-moe.cpp
  -> ggml_new_tensor_2d(weight/activation)
  -> ggml_mul_mat()
  -> ggml_backend_graph_compute()
  -> CPU backend MUL_MAT operation
```

The bounded provider does not call `vec_dot` directly. It constructs a
per-tile `GGML_OP_MUL_MAT` graph and submits it to its own CPU backend. The
outer operation is `MAP_CUSTOM3`; the provider graph operation is `MUL_MAT`.

## Runtime evidence

With `CORTEX_FLASHNEXT_CPU_DISPATCH_TRACE=1`, the real runner emitted:

```text
CORTEX_DISPATCH provider_compute type=23 rows=640 width=2560 op=ggml_mul_mat
CORTEX_DISPATCH graph_construct op=29 ne=[12,1,1,1]
CORTEX_DISPATCH graph_complete type=23 status=PASS
CORTEX_DISPATCH provider_compute type=20 rows=2560 width=640 op=ggml_mul_mat
CORTEX_DISPATCH graph_construct op=29 ne=[40,1,1,1]
CORTEX_DISPATCH graph_complete type=20 status=PASS
```

The runner exited `0`, completed gate/up/down streaming, and returned token
`248046`, logit `10.8127`. This proves provider callback, provider compute,
graph construction, backend submission and completion. No `MUL_MAT_ID` is used
inside the bounded provider graph.

## Capture conclusion

The previous `capture_once=0` diagnostics were attached to the provider graph,
but the CPU-internal capture was placed below an unverified dispatch path.
Runtime evidence now identifies the correct provider operation boundary. Stock
CPU vec-dot use remains unproven because the CPU backend dispatch entry and
vec-dot call were not instrumented by this trace.

## Status

PROVIDER CALLBACK: PASS
ACTUAL COMPUTE FUNCTION IDENTIFIED: PASS
GGML OPERATION IDENTIFIED: PASS
CPU DISPATCH PATH VERIFIED: PASS
STOCK VEC-DOT USED: UNPROVEN
CAPTURE FAILURE ROOT CAUSE: PROVEN (capture was attached below an unverified CPU path)
BOUNDED MEMORY: PASS
FULL MODEL STREAMING: NOT TESTED

Next capture location: CPU backend `GGML_OP_MUL_MAT` dispatch immediately before
the selected type-trait implementation, using the provider-local diagnostic
context. No production arithmetic change is justified.

`git diff --check`: PASS.
