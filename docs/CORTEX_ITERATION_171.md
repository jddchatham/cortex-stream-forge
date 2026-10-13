# Cortex Iteration 171 — Down Output Placement and Routed Aggregation

## Preservation

- Branch: `cortex-iteration-171`
- Iteration 170 and unrelated dirty changes preserved.
- No reset, clean, stash, merge, commit, or push.
- Fresh runner and oracle logs: `/tmp/cortex-iter171-run.log`,
  `/tmp/cortex-iter171-oracle.log`.

## Fresh nine-token oracle

Fresh layer-4 boundary fixtures were regenerated from the current streamed
runner and passed to the existing bounded GGUF oracle. The oracle exited 0,
but numerical status is based on comparisons, not exit status:

```text
CORTEX_L2_REF gate n=57600 max_abs=5.24521e-06 rms=4.10268e-07 mismatches=0
CORTEX_L2_REF up   n=57600 max_abs=4.29153e-06 rms=3.47447e-07 mismatches=0
CORTEX_L2_REF down n=230400 max_abs=3.27526 rms=0.0792954 mismatches=229698 first=0
CORTEX_L2_REF aggregate n=23040 max_abs=1.05916 rms=0.0629292 mismatches=22918 first=0
```

The streamed graph output is stored as GGML `[channel, expert-slot, token]`
while the oracle vectors are flattened `[token, expert-slot, channel]`.
After applying that proven transpose, gate/up still pass, while down remains
the only failed stage. The prior raw-layout comparison was not used as proof.

## Exact CPU target re-check

The fresh exact record remains:

`/tmp/cortex-iter170-l4-p2-t1-e257-r0.bin`

Identity: layer 4, down, token 1, expert 257, row 0. The captured CPU dot is
`0.006443969905376434`. The independently decoded canonical IQ4_NL row and
captured Q8_0 activation produce `0.006443967811833318`, absolute error
`2.09354e-09`.

The Iteration 170 diagnostic correction copies `wdata`, the pointer consumed by
the CPU vec-dot, rather than padded `src0->data`. A fresh corrected-row hash
rerun was not performed in this iteration, but the exact dot evidence remains
PASS and production arithmetic is unchanged.

## First remaining divergence

The exact individual CPU dot is correct, but the existing full down oracle
does not reproduce the corresponding down vector. This means the remaining
failure is after or around the reference's down-row/output mapping, not a
proven IQ4_NL/Q8_0 CPU arithmetic defect. The current diagnostics do not yet
capture enough keyed dot-to-destination records to prove whether the provider
output placement or the oracle's down reconstruction is responsible.

Routing aggregation is therefore downstream and NOT independently accepted.

## Runtime and regression

- Real runner: PASS, exit 0.
- Token: `248046`.
- Logit: `13.0355`.
- Provider regression: PASS, prior max abs `3.4757e-06`, RMS `9.24548e-07`.
- Scratch: `2,150,400` bytes.
- `git diff --check`: PASS.

| Stage | Status |
|---|---|
| Fresh gate oracle | PASS |
| Fresh up oracle | PASS |
| Exact token-1 CPU IQ4_NL/Q8_0 dot | PASS |
| Full down oracle | FAIL |
| Down dot-to-output placement | NOT TESTED |
| Routed aggregate | NOT TESTED |
| Full layer-4 equivalence | FAIL |
| Full-model streaming | NOT TESTED |

## Next action

Use the corrected `wdata` capture to persist one keyed CPU record per selected
down row and compare its destination element against a reference built from
the same row and Q8_0 bytes. In parallel, audit the oracle's down output
flattening and expert-slot index using the exact token-1 expert-257 mapping.
Do not change the down kernel or routing until those two representations are
identity-matched.
