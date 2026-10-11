# Supervised hard-memory gate

Forge verifies a tiny non-model child inside a delegated user systemd scope.
The child reads its own unified cgroup files. PASS requires finite exact
`memory.max` and `memory.swap.max` readback plus zero OOM events. RSS sampling
and RLIMIT_AS are not enforcement evidence. The workload wrapper reads the
same limits in the actual child immediately before `execv`; a prior probe
alone is not sufficient.

One observed Lenovo run (the evidence in this PR) with a 64 MiB probe:
`memory.max=67108864`, `memory.swap.max=0`, `memory.current=5148672`,
`memory.peak=5173248`, and `low=0 high=0 max=0 oom=0 oom_kill=0
oom_group_kill=0`. Exit status was 0. Earlier observations are not combined
with this run.
This proves only the tiny probe; no model was loaded and it is not a model
budget.

After review, the smallest genuine diagnostic is the existing one-layer
`llama-qwen38-lowres-segment-test` (or equivalent one-layer owned-layer range).
It requires a separate working-set estimate and reviewed larger scope; this
change intentionally does not launch it.
