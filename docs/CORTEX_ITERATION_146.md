# Cortex Iteration 146 — memory residency forensics

Branch: `cortex-iteration-146`
Checkpoint base: pushed Iteration 145 commit `7174649`

## Checkpoint

Iteration 145 was committed and pushed to:

```text
https://github.com/jddchatham/cortex-stream-forge/tree/cortex-iteration-145
```

The checkpoint included only the provider reference test and Iteration 142–145
documents. Iteration 146 is local and uncommitted. Pre-existing dirty files
and generated artifacts remain untouched.

Checkpoint validation:

```text
cmake --build runtime-flashnext-layer-exp-build \
  --target llama-flashnext-iq4xs-provider-test -j4
runtime-flashnext-layer-exp-build/bin/llama-flashnext-iq4xs-provider-test \
  flashnext-tensor-index.json
git diff --cached --check
```

Build exit: 0. Deterministic provider regression exit: 0. Cached diff check:
PASS. The regression retained the Iteration 145 bounds (`max_abs <= 1e-5`,
`RMS <= 2e-6`).

## Monitored baseline command

The real runner was executed once with the same model and input as the prior
baseline. A Python monitor sampled `/proc/<pid>/status`,
`/proc/<pid>/smaps_rollup`, and `/proc/meminfo` every 200 ms, while stdout and
stderr were written to `/tmp/cortex-iter146-run.log`.

```text
python3 <bounded /proc monitor> \
  runtime-flashnext-layer-exp-build/bin/llama-qwen38-lowres-chat \
  /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1
```

Input sent: `hello`, then `/exit`. Exit: 0. Elapsed: 25.882 s.

## Memory measurements

```text
VmRSS peak:          5,622,252 kB
VmHWM final:         5,647,288 kB
smaps Rss peak:      5,624,804 kB
smaps Pss peak:      5,620,299 kB
Private_Dirty peak:  5,614,616 kB
VmSwap peak:           364,436 kB
smaps Swap peak:       364,436 kB
final VmRSS:         1,759,436 kB
final smaps Rss:     1,694,056 kB
final VmSwap:           16,236 kB
```

At the time of this measurement `/proc/meminfo` reported approximately
5,883,576 kB `MemAvailable`, 20,096,492 kB `SwapTotal`, and 14,967,876 kB
`SwapFree`. The sampled process-specific swap reached 364,436 kB; this does
not establish that all system swap activity came from this process.

The scheduler log reported approximately 1308.27 MiB CPU compute-buffer
reservation. The provider scratch arena remained 2,150,400 bytes. These are
distinct from committed resident pages and process swap.

## Phase observations

The monitor classified samples using emitted runtime markers. It observed 103
samples: 4 startup, 21 scheduler/graph-reservation, and 78 selected-MoE or
later execution samples. The peak occurred during the late execution interval,
not as a final retained allocation. The process returned to approximately
1.76 GiB RSS by teardown, showing that the peak is mostly temporary or
reclaimable rather than a permanently retained provider scratch buffer.

`llama_context::sched_reserve()` explicitly constructs and reserves a worst-case
prompt graph, a token-generation graph, and a final prompt graph. This explains
the planned compute reservation, but does not by itself explain the additional
multi-gigabyte resident peak. The current run did not include allocator-level
interval attribution for every GGML leaf and graph temporary, so ownership of
the remaining peak is not proven.

## Allocation attribution

### Verified

- Scheduler reservation is approximately 1308.27 MiB.
- Provider scratch is 2,150,400 bytes.
- Peak RSS and private dirty pages are approximately 5.6 GiB.
- Peak process swap is approximately 356 MiB.
- RSS falls substantially by teardown.
- No memory-cap increase or production arithmetic change was made.

### Not yet proven

- Whether the additional resident peak is duplicate model tensor storage,
  prompt graph temporaries, a backend allocator arena, or a combination.
- Exact tensor-level lifetimes accounting for the 5.6 GiB peak.
- Whether any model-mapped file pages are being copied into anonymous memory.

Because the largest owner/lifetime is not yet proven, no memory optimization was
attempted. Reducing scheduler reservation or changing tensor ownership now
could invalidate the validated execution path.

## Numerical regression

The Iteration 145 deterministic provider regression passed before the memory
run. The real runner exited 0 and preserved its established streamed inference
output. No numerical source was changed in Iteration 146.

## Status

- Iteration 145 checkpoint: **PASS**, pushed as `7174649`
- Memory monitor execution: **PASS**
- Scheduler reservation measurement: **PASS**
- Process RSS/HWM/swap measurement: **PASS**
- Allocation-owner attribution: **NOT TESTED**
- Safe memory reduction: **NOT TESTED**
- Numerical regression after memory change: **NOT APPLICABLE** (no memory
  change made)
- Bounded provider scratch: **PASS**
- Bounded total residency target: **FAIL** for the measured 5.6 GiB peak
- Full-model streaming inference: **NOT TESTED**
- `git diff --check`: **PASS**

## Smallest next experiment

Add a test-gated allocator interval trace around `ggml_backend_sched_reserve`
and graph execution, recording tensor identity, backend buffer, offset, size,
first/last use, and release/reuse events. Reconcile those intervals with the
5.6 GiB RSS peak before changing scheduler or model-tensor ownership.
