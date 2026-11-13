# QIES106 procedural-synthetic streaming experiment

This is an isolated experiment at real Ripple tensor dimensions. There is **no
real-model loader**, model filename argument, or real-execution unlock. QIES104
admission and its original 6 GiB real-inference gate remain byte-identical.

From this detached worktree:

```sh
./pandacode-run.sh
# Optional shorter campaign (still includes cleanup and interface rejection tests):
python3 -I -B qies106/run.py --case ffn_gate
# Separate metadata-only graph-buffer/workspace calculations:
python3 -I -B qies106/plan_only.py
```

The default campaign builds the two private workers, runs independent whole-matrix
stock GGML and streamed candidate projections for five fixed cases, checks two
additional Q-projection tile sizes, and tests four cleanup faults and six rejected
entry requests. Every output byte and the quantized synthetic weight hashes must
match. A mismatch or resource violation stops the campaign; no tolerance fallback.
Every campaign has a new results directory. Interrupted `.partial` files and missing
result receipts are not accepted as success or silently resumed.

Requires Linux, bubblewrap with working unprivileged namespaces, C++17, OpenSSL and
the source-built CPU SDKs recorded by QIES105. No NumPy or model download is needed.
The default SDK identity file is the existing sibling
`qies105-publication/.qies105-build/state.json`. For another checkout use the
`QIES106_SDK_STATE` environment variable to name the `state.json` emitted by
`python3 -I -B qies105/replay.py build --out /persistent/ssd/sdk --jobs 1`.
That source-only command builds pinned baseline GGML and independent upstream
llama.cpp CPU libraries; it does not load a model. No global install is required.
Only three GGML CPU libraries are copied into this experiment, hash-verified; no
`libllama`, GPU backend or canonical retired executable is restored.

Synthetic data are deterministic dyadic F32 values from `fixture.h`, quantized by
the upstream quantizers **only to construct test inputs**. Every matrix element is
represented, not sampled or replaced by a small matrix. A future real source must
read verified quantized byte ranges verbatim and never requantize actual weights.
Candidate and reference generate their own quantized inputs; matching SHA256 proves
both numerical paths receive exactly the same quantized weight bytes. The reference
uses one full-matrix stock graph; it does not call the candidate scheduler/kernel.

The numerical workers are non-executable host files with namespace-only RPATHs.
They run through the ELF interpreter inside a synthetic-only filesystem exposing
no home/model directories. Only built-in case names, tile sizes and test faults
are accepted. Existing real dispatch is always denied. Additional monitoring of
these synthetic jobs stops its own process tree on unsafe available RAM/swap/disk,
OOM events, timeout or its small process RSS budget. It does not override any
existing real-model resource policy or change cgroups, swap, rlimits or the kernel.

Evidence: each run's `summary.json`, per-job `result.json`, `/proc` monitor JSONL,
worker phase telemetry, hashes and output bytes. `memory-accounting.json` is
**calculated**, not measured residency. `plan_only.py` calls
`ggml_gallocr_reserve_n_size(no_alloc=true)` and `ggml_graph_plan`; it never calls
buffer reserve/allocation or compute. Its large-context results are virtual
allocation plans, not multi-gigabyte resident experiments.

A 4 GiB machine cannot satisfy the unchanged 6 GiB MemAvailable real-execution gate.
These tests establish synthetic operation-level bounds, not complete-layer,
real-weight, full-model or 4 GiB hardware qualification. QIES107 should assemble
and independently compare the complete synthetic real-width FFN/residual branch.
