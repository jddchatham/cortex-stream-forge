# QIES107 complete synthetic Ricursive FFN

No real model inputs or model arithmetic. QIES104 real admission and its 6 GiB
MemAvailable gate remain unchanged. Fixed procedural dense synthetic matrices only.

From this isolated worktree:

```sh
./pandacode-run.sh --smoke  # one complete q5 branch; seven exact taps
./pandacode-run.sh          # q5/q4 x three vectors, repeat3, 100 faults, 6 refusals
```

The QIES106 replay must already have seven passing byte-exact comparisons whose
output hashes match the preserved original. It is available independently:

```sh
python3 -I -B qies106/run.py
```

SDK provenance, CPU/headers/library hashes, compiler commands and numerical source
hashes are in each campaign's build manifest. The source-built baseline GGML
0.25.3 and independent stock GGML0.29.0 come from the exact QIES105 revisions.
An alternate source-built SDK can be named with `QIES106_SDK_STATE`; use the pinned
source-only build command in `qies106/README.md`, not a system/global install.
Requirements are C++17, OpenSSL, Linux bubblewrap/unprivileged namespaces, Python3
and those CPU libraries. No NumPy, GPU dependencies or model download is needed.

## Contract

Input is the already constructed `ffn_inp` residual from the stock Ricursive graph.
Both hidden RMS scales are nontrivial deterministic vectors. The exact order is:

1. RMSNorm(input, epsilon1e-6) times FFN gamma.
2. Gate and up quantized matvecs, 5120 to40960.
3. `ggml_swiglu_clamp(gate, up, 20)`. It contains the SiLU and elementwise multiply;
   there is no second multiplication after it. Gate is upper-clamped, up is
   symmetrically clamped, exactly as in upstream.
4. Down matvec,40960 to5120.
5. RMSNorm(down, epsilon1e-6) times post-FFN gamma.
6. Add the original `ffn_inp` residual.

Profiles q5 and q4 match actual layer0 and layer15 tensor descriptors respectively.
Every projection retains its full inner dimension, uses upstream CPU kernels and
streams128 output rows. Capture taps are input-norm, gate, up, active, down,
post-norm and final. All sizes/dtypes/weight identities/every output byte must match
the independent full-graph stock observer. Zero tolerances; mismatches stop.

Two nonzero input vectors exercise all clamp branches; a zero vector tests the
RMS epsilon path. An additional three identical branches in one worker test
bounded lifetimes and deterministic repeated use (not attention/KV continuation).

## Fixture construction and independent reference

257 small256-element templates per projection, independently quantized by each
worker's upstream library, define dense full-width matrices with a deterministic
block-selection hash. Gate/up use distinct seeds and larger amplitude to trigger
clamping; down uses another seed. Every quantized byte is filled into each actual
matrix/tile, hashed and consumed by GGML. This fixture factory only reduces the
cost of generating synthetic data. It does not compress real model weights, skip
rows, reduce dimensions, replace matrix multiplication, requantize real weights,
or represent an inference performance result.

The reference source `reference_ffn.cpp` compiles no candidate graph/scheduler.
It builds a single upstream whole-FFN graph and allocates all three full matrices.
Shared code defines inputs, ownership, telemetry and captures only. There is no
`libllama` dependency or whole-model load. Source order is derived from the exact
pinned `src/models/ricursive.cpp`, not guessed from the five requested tap names.

## Memory, failure and output boundary

The candidate reuses one1 MiB CPU scratch buffer across stages/repeats and only one
weight tile. GGML tensors/buffers are scoped to each projection/elementwise graph;
activations are freed after their final consumer. Three tiny synthetic block caches
are bounded input factories, not resident full matrices. Residual survives until
final add. Peak weight/CPU-buffer counters and RSS/PSS/anon/file-backed measurements
are logged at actual live points and after cleanup.

All workers require a synthetic-only namespace with no home or real-model mounts.
Private numerical worker files have no host executable permission and namespace-
only RPATH. Real roles/files/absence of the namespace marker fail closed.

QIES107's additional model-free monitor does not alter QIES106's monitor or any real
inference policy: RAM/swap/disk/OOM/timeout checks apply to its own tree. The full
reference has a calculated412.5 MiB of q5 weights, so its newly defined synthetic
RSS ceiling is640 MiB; the streaming ceiling is256 MiB. Synthetic preflight is
1.75 GiB MemAvailable for that reference,1.5 GiB for the small candidate, with a
1 GiB runtime stop. These are not permission to run real inference below6 GiB.
No cgroup, rlimit, swap, allocator setting or OS configuration is changed.

A fault before/after each major allocation/compute/capture boundary returns failure,
leaves zero owned resources, and removes only that invocation's provisional capture
files. Captures are exclusively staged; final publication uses renameat2(NOREPLACE).
A final directory plus successful job result is required for acceptance. This is a
model-free diagnostic transaction, not production checkpoint/crash recovery.

Campaigns write unique `runs/` directories. Original QIES106 source/results are
preserved in `QIES106-preserved.tar.gz` with its manifest and original directories
untouched. No commit/merge/push or frozen-archive modification is performed.
