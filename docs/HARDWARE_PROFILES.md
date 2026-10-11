# Hardware profiles: design proposal

Profiles describe **resource policies**, not separate model implementations. No profile configuration parser or runtime cache policy is claimed to exist yet. Avoid restructuring the current engine until a minimal configuration abstraction is reviewed.

## Reference profile: potato-2gb

- Reference machine: AMD Athlon Silver 3050U, CPU-only x86.
- Hard ceiling: 2,147,483,648 bytes RAM, zero swap.
- Verify the **actual child process** is in the enforced cgroup before execution.
- Record cgroup peak and OOM events, RSS/PSS, elapsed time, bytes read, exit status, and numerical checkpoints.
- Historical supervised diagnostic timeout: 195 seconds. Choose a justified timeout for each new test.
- This is a stress/regression target, **not** a universal project requirement.

## Proposed profiles (not implemented)

| Profile | Illustrative resource envelope | Possible strategy |
| --- | --- | --- |
| `potato-2gb` | 2 GiB, zero swap | Minimal resident layers, streamed PLE and experts |
| `desktop-16gb` | 16 GiB RAM | Cache selected tensors subject to measured headroom |
| `workstation` | Hardware-defined | Larger caches, optional future compute backends |

Do not interpret profile names as benchmarked capabilities. Actual allocations and throughput remain to be measured.

## Minimum future configuration fields

- `memory_limit_bytes`
- `swap_limit_bytes`
- `backend` (initially CPU)
- `weight_cache_budget_bytes`
- `io_strategy`
- `timeout_seconds`

Any configuration must fail safely when the selected model, tensor manifest, or memory budget cannot be validated. Do not introduce a configuration parser solely for documentation.

## Contributor hardware reports

Include CPU, RAM, swap, storage type, OS/kernel, model shard identities, exact build/source revision, profile/budget, command, measured peak RSS/cgroup memory, bytes read, numerical result, and tokens/s **only if decoding actually occurred**.
