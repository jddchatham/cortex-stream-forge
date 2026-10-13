# Cortex Iteration 150 — Remaining Prompt-Graph Expert Allocations

Date: 2026-10-13
Branch: `cortex-iteration-150`
Working base: Iteration 149 dirty state; no commit or push performed.

## Preservation

The Iteration 149 worktree was dirty before this iteration. I created
`cortex-iteration-150` with `git switch -c`, preserving all existing changes and
untracked evidence. No reset, clean, stash, checkout, merge, or force operation
was used.

## Context inventory

The allocator profile from the unchanged runner records several kinds of graph
reservation:

| Context family | Observed reservation | Evidence |
|---|---:|---|
| Layer-2 streamed MoE context | 9.59 MiB | `/tmp/cortex-iter149-alloc.log`, callbacks PASS |
| Other prompt/layer contexts | 1308.03–1308.27 MiB | same allocator log |
| Final generation context | 0.08 MiB | same allocator log |

The large contexts repeatedly allocate full expert leaves. Representative
records include `blk.46.ffn_down_exps.weight` (471,859,200 bytes),
`blk.46.ffn_gate_exps.weight` (445,644,800 bytes), and
`blk.46.ffn_up_exps.weight` (445,644,800 bytes), followed by equivalent
allocations for layer 47. The allocation offsets and frees show that these are
scheduler-owned payload intervals, not merely virtual tensor metadata.

The Iteration 149 mechanism is explicitly gated by `il == 2` in
`build_moe_ffn()`. That is intentional: the current provider index contains
only the three layer-2 descriptors from `/tmp/cortex-iter120-layer2-index.json`,
and the callback passes layer `2` to the PREAD provider. Applying the shape-only
representation to layers 0–47 without corresponding real descriptors, row
offsets, and provider lifetime support would create a graph that advertises
valid shapes but cannot supply valid weights. That would violate the external
ownership contract.

## Source-level ownership result

The successful layer-2 path replaces its three `ggml_mul_mat_id()` shape probes
with explicit F32 shape tensors. Its callback consumes `selected_experts`, the
real activation, and bounded PREAD rows. The other contexts still take the
ordinary `build_lora_mm_id()` path, whose `ggml_mul_mat_id()` nodes reference
the canonical full expert leaves. Because those leaves have `data=NULL` and
`buffer=NULL` during reservation, GGML allocates their declared payloads.

This explains why the local layer-2 reservation falls to 9.59 MiB while the
overall run continues to reserve approximately 1308 MiB: the runner constructs
additional physical-layer contexts, including contexts whose MoE leaves are not
served by the layer-2-only provider.

## Validation commands

Build:

```text
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-qwen38-lowres-chat llama-flashnext-iq4xs-provider-test -j4
```

Exit: 0.

Deterministic provider regression:

```text
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  flashnext-tensor-index.json
```

Exit: 0.

```text
one token:   max_abs=3.09898e-6, RMS=7.49719e-7
three token: max_abs=3.67127e-6, RMS=8.08196e-7
```

Real runner:

```text
printf 'hello\n/exit\n' | env CORTEX_MOE_STREAM_GATE=1 \
  CORTEX_MOE_STREAM_UP_DOWN=1 \
  CORTEX_FLASHNEXT_TENSOR_INDEX=/tmp/cortex-iter120-layer2-index.json \
  runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
  /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1
```

Exit: 0. Final observed output: token `248046`, logit `10.8127`; provider
scratch `2,150,400` bytes. Layer-2 gate, up, and down callbacks all reported
PASS.

Allocator profile evidence: `/tmp/cortex-iter149-alloc.log`. It shows a 9.59
MiB layer-2 context but later 1308.27 MiB reservations and full expert-leaf
allocations in other contexts.

Monitored run evidence: `/tmp/cortex-iter149-rss2.log` and
`/tmp/cortex-iter149-rss2.tsv`.

Observed peak values:

```text
VmRSS/VmHWM: 5,700,444 / 5,706,596 kB
VmSwap:      1,495,324 kB
```

These are not a measured improvement over the historical Iteration 149
baseline (~5.70 GiB was observed there as well, with environmental variation),
and no whole-process RSS reduction is claimed.

## Status

| Requirement | Status | Finding |
|---|---|---|
| Preserve Iteration 149 work | PASS | isolated branch, dirty files preserved |
| Inventory reservation contexts | PASS | allocator log identifies small and large contexts |
| Identify remaining expert allocation sites | PASS | ordinary `build_lora_mm_id()` paths in other contexts/layers |
| Apply shape tensors to eligible layer-2 path | PASS | already validated in Iteration 149 |
| Generalize to all remaining contexts safely | NOT TESTED | no complete all-layer descriptor/provider contract |
| One-token regression | PASS | within tolerance |
| Three-token regression | PASS | within tolerance |
| Layer-2 reservation reduction | PASS | 1308 MiB-class local graph to 9.59 MiB |
| Whole-run reservation reduction | FAIL | later contexts still reserve 1308.03–1308.27 MiB |
| Whole-process RSS reduction | NOT DEMONSTRATED | measured peak remains ~5.70 GiB |
| Full-model streaming inference | NOT TESTED | only bounded layer-2 provider path validated |
| `git diff --check` | PASS | clean |

## Next smallest justified action

Build a complete per-layer expert descriptor index and prove provider lifetime
and numerical equivalence for one additional physical MoE layer. Then apply the
same shape-tensor substitution only to that layer's graph contexts and measure
the reservation. A global condition or allocator suppression is unsafe until
all conventional consumers and fallback paths are removed or supplied with
real external storage.
