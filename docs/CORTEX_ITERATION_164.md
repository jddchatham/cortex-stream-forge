# Cortex Iteration 164 — bounded CPU capture slots

## Preservation

Work continued on isolated branch `cortex-iteration-164`, preserving the dirty
Iteration 163 state. No inference arithmetic, routing, or streamed-layer
selection was changed.

## First-row bottleneck removal

The CPU capture path no longer uses a compare-exchange against one global
`row_capture_once` state to decide whether a provider invocation may be
captured. Each provider tiled matmul installs a fresh diagnostic target; worker
0 now records that target's row, while the provider writes immutable files keyed
by `projection/layer/token/expert` and refuses to overwrite existing files.
This keeps capture bounded to one row per provider invocation and avoids races
between workers within a graph.

## Runtime coverage evidence

The real layer-4 run completed exit 0. Provider-entry traces still prove:

```text
gate token=0 nonzero=2560 source_nb2=10240
gate token=1 nonzero=2560 source_nb2=10240
up   token=0 nonzero=2560 source_nb2=10240
up   token=1 nonzero=2560 source_nb2=10240
down token=0 nonzero=640  source_nb2=25600
down token=1 nonzero=0    source_nb2=25600
```

The CPU boundary emitted both kinds of records:

```text
type=23 ... row_count=1 ... src_nonzero=2560
type=23 ... row_count=0 ... src_nonzero=0
```

Thus the later layer-4 CPU executions receive a zero source buffer even though
the provider-entry callback had a nonzero token-1 vector. The zero-source CPU
records are rejected by the capture predicate and are not treated as valid
token-1 Q8 evidence. This proves token 1 reaches the CPU dispatch path, but not
with the intended source data in all graph invocations.

## Capture status

| check | status | evidence |
|---|---|
| bounded per-invocation capture | PASS | worker-0 capture, fixed one-row bound |
| write-once keyed files | PASS | provider key includes projection/layer/token/expert |
| token-0 Q8 captures | PASS | nonzero CPU rows emitted |
| token-1 CPU boundary reached | PASS | CPU records with `src_nonzero=0` |
| token-1 valid Q8 capture | FAIL | CPU source is zero in later invocations |
| token-1 Q8 byte equivalence | NOT TESTED | no valid token-1 bytes |
| token-1 dot-product comparison | NOT TESTED | no valid quantized operand pair |
| gate/up layer-4 equivalence | FAIL | prior real-activation mismatch remains |
| down Q8_0 capture | NOT TESTED | token-1 post-gate/up input is zero |
| provider regression | PASS | unchanged regression |
| bounded memory | PASS | fixed one-row diagnostic storage |
| `git diff --check` | PASS |
| full-model streaming | NOT TESTED |

## First proven failure boundary

The remaining defect is not the capture-slot registry. The same layer-4
provider transaction can report nonzero token-1 input before constructing the
CPU graph, while the CPU graph instance later observes a zero `src1`. The next
experiment must correlate provider invocation ID with CPU graph identity and
inspect the `x` tensor data immediately after `ggml_backend_tensor_set()` and
at CPU `MUL_MAT` entry. This will distinguish backend tensor-set/lifetime
failure from a later graph reuse or context execution mismatch.
