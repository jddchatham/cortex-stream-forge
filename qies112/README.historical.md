# QIES112 synthetic autoregressive reproduction

New isolated worktree; no real model arithmetic is enabled. The exact Ripple
configuration is H=5120, V=100352, 80 blocks, scaled IQ4_XS token embeddings,
F32 final RMS normalization and an **untied** Q6_K head. Token IDs are synthetic
inputs/outputs, not a demonstration of meaningful natural-language generation.

The baseline launcher verifies preserved QIES111 source/evidence and uses its
unchanged numerical APIs/guards to replay 2 layers, boundary layers and80 layers
with two tokens. It replaces only the historical artifact-path lookup; it does
not replace or lower resource admission. Original 5,148 captures/1,584 weight
hashes remain preserved. See preservation.json and baseline-active.json.

Pinned CPU SDK state is the same QIES105 source-built SDK used by QIES111.
`QIES106_SDK_STATE` can point to a verified relocated SDK JSON. Neither model
metadata nor a full llama model is initialized for each layer. The two synthetic
layer-weight containers are prepared once and reused read-only. Existing
per-layer process/backend/context setup remains for crash isolation and bounded
residency; it is not claimed to be optimal or a persistent-context redesign.

```sh
env -u LD_LIBRARY_PATH python3 -I -B qies112/baseline_replay.py
env -u LD_LIBRARY_PATH python3 -I -B qies112/build.py
env -u LD_LIBRARY_PATH python3 -I -B qies112/endpoint_tests.py
env -u LD_LIBRARY_PATH ./pandacode-run.sh --tokens 2
# Extend the same accepted sequence; verified previous decisions are not reevaluated:
env -u LD_LIBRARY_PATH ./pandacode-run.sh --tokens 4 --resume qies112/runs/EXACT-RUN
```

Extend to8/16 only after the previous milestones pass. Scripts intentionally
refuse to overwrite prior test result directories; reproduce in a fresh worktree
or preserve outputs under new paths. Numerical output remains in sealed bwrap
namespaces with only synthetic fixtures, activations and read-only prior KV.

Candidate and reference run their80-layer branches in separate child controllers
at the same time. Each numerical worker keeps the inherited unchanged synthetic
memory/disk/OOM safeguards and CPU thread count1. Additional generation-tree
observation stops if aggregate sampled RSS exceeds1GiB, system reserves are
crossed, an ancestor OOM event rises, either branch fails or a stack step exceeds
1200seconds. No real-weight6GiB policy or cgroup/rlimit/system settings change.

The parent samples its own full process tree for simultaneous RSS/PSS, including
both branches and controllers. It also records controller-only RSS/PSS and host
MemAvailable/swap/disk minima. These are direct concurrent samples, not sums of
individual unrelated peaks. RSS can double-count shared pages; PSS apportions
them. Shared ancestor cgroup history is not isolated worker accounting.

Each step independently gathers/scales embeddings, runs all80 layers, applies
final RMS normalization, projects all100352 logits, and picks the lowest-ID
finite argmax. Branches must match every captured layer/KV tensor, embedding,
normalization, all logits and selected ID before a durable decision is published.
The selected ID is fed back at the next step; the last prediction is explicitly
pending evaluation, so KV next_position counts evaluated input tokens.

A layer-stack `.COMMITTED` directory by itself is **not** a generation decision.
Only `decision-N.json`, after successful endpoints/comparisons and synchronized
checkpoint references, authorizes feedback. The previous global decision remains
valid after a head/stack failure. Orphan completed stacks can be verified/reused
on restart without silently advancing the global sequence. Decision publication
uses exclusive atomic hard-linking of a synchronized temporary file and directory
synchronization. No exactly-once terminal-delivery or universal storage-rollback
claim is made. `CANCEL` in the run root stops before the next uncommitted step;
an active step finishes at its transaction boundary. SIGTERM/error may leave
uncommitted artifacts, retained for diagnosis. Child controllers get a process
parent-death signal; numerical bwrap children also use die-with-parent.

Explicit stops: selected EOS100265; requested prediction count; boundary
cancellation; invalid IDs/dimensions/state; comparison failure; resource/OOM/
timeout/branch error. The numerical test uses initial BOS100257 and supports
2/4/8/16 predictions in the bounded256-slot cache. This is not unbounded KV-ring
generation; long sliding-window continuation remains outside QIES112.

Tests: endpoint_tests.py (numerics/faults/refusals), sanitizer_tests.py (new caller
ASan/UBSan), integration_fault_tests.py (real step control with simulated stacks),
decision_tests.py (explicitly simulated stopping cases), termination_test.py
(intentional synthetic SIGTERM). ASan instruments the new caller, not prebuilt
GGML shared libraries. The historical crash remains unresolved.
