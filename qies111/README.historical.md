# QIES111 — synthetic multi-layer CPU orchestration

This is isolated experimental code. Numerical workers see synthetic fixtures,
never the GGUF. The unchanged QIES104 real execution guard remains locked even
on a host that has 6 GiB MemAvailable.

Preserved prerequisite reproduction (from this worktree root):

```sh
env -u LD_LIBRARY_PATH python3 -I -B qies109/run.py
env -u LD_LIBRARY_PATH python3 -I -B qies110/run.py
env -u LD_LIBRARY_PATH QIES110_STORAGE_RESULTS=storage-results-q111 python3 -I -B qies110/storage_tests.py
env -u LD_LIBRARY_PATH python3 -I -B qies110/sanitizer_tests.py
```

Those evidence directories are retained. The storage/sanitizer scripts refuse
to overwrite existing results; repeat in a fresh isolated worktree or use a new
storage output name. The inherited sanitizer directory was preserved separately
as `qies110/sanitizer-results-inherited` before the fresh replay.

Pinned SDK state defaults to sibling `qies105-publication/.qies105-build/state.json`;
`QIES106_SDK_STATE` can supply a verified relocated state file. The build rechecks
all stock source hashes and copies only the three required GGML CPU libraries.
No llama runtime, GPU dependency or model parser is linked into numerical actors.

```sh
env -u LD_LIBRARY_PATH ./pandacode-run.sh --layers 2 4 8 --boundary
# Stretch campaign; use only after the smaller numerical/fault cases pass:
env -u LD_LIBRARY_PATH ./pandacode-run.sh --layers 80
```

The exact model schedule is `layer-schedule.json`: four sliding-window layers,
then a global layer, repeated sixteen times. Q5_K FFN applies to layers 0–14 and
65–79; the middle fifty layers use Q4_K. Two deterministic synthetic weight
containers are shared read-only, with a layer-bound plan per layer. Weights are
intentionally repeated within each profile; this is not eighty independently
sampled models. All dimensions and graph operators remain real-width.

The controller starts a fresh worker per layer, passes the previous layer's
activation and the same original embedding, and mounts that layer's previous
KV checkpoint read-only. The existing 1 MiB scratch arena is reused within each
layer and released with its process. Scratch is bounded, but is not retained at
the same address across layer subprocesses; process isolation has startup cost.

All per-layer checkpoints for a token stage under `.UNCOMMITTED`. Only completion
of every layer, file/directory synchronization and atomic directory publication
produce `.COMMITTED`. The old complete token stays immutable. `COMMIT.json` binds
layer IDs, positions, capacities, state hashes and capture manifests. Unsupported
or corrupt checkpoints fail closed. A failure during/after publication sync can
have an uncertain durability outcome: restart verifies published state; it does
not claim an atomic rollback of storage faults or exactly-once external output.
There is no text/token sampling or terminal delivery protocol here.

Progress is logged and results are resumable without repeating verified layers:

```sh
env -u LD_LIBRARY_PATH ./pandacode-run.sh --no-build --layers 80 --resume qies111/runs/EXACT-RUN-DIRECTORY
```

Use the original layer/boundary arguments. Source/build/controller identities
must match. Interrupted outputs are preserved under distinct names. A changed
implementation requires a new campaign. `INPUT.json`/`OUTPUT.json` and checkpoint
hashes are verified before cached work is reused. There is no automatic desktop
shutdown or system configuration change.

`contract_tests.py` tests controller failure/ownership with simulated state only.
`fault_tests.py RUN-DIRECTORY` requires a completed 2-layer numerical campaign and
injects failures into real-width synthetic workers. It preserves authoritative
prior checkpoints and checks uncommitted output cannot become the next token.

The 256/1,280-slot capacities are deliberate bounded tests. Sliding-window
expiration clears stale entries, but this is not a wrapping KV ring and execution
past capacity is rejected. Boundary history at position 1023 is seeded synthetic
history, not 1,023 previously evaluated tokens.

Accounting note: `METRICS.json` controller sample keys ending `_kib` were
initially named incorrectly; the inherited monitor returns **bytes**. Raw files
are preserved. `verify_results.py` corrects the units in the final audit. This
label correction changes neither runtime measurements nor resource guards.
