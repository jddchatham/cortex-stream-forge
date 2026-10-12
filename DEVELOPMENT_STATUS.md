# Cortex Stream Forge development status

This branch records the current bounded-streaming runtime work. It is not a
claim of complete chat inference.

## Implemented

- Runtime-derived Qwen4Exp descriptor-only extraction.
- Deterministic descriptor manifest with GGUF shard and offset metadata.
- Classification into `ARENA_RESIDENT`, `EXTERNALLY_STREAMED`, and
  `GLOBAL_OR_SEPARATELY_MANAGED`.
- Fixed 67,851,136-byte layer arena with 64-byte alignment.
- Explicit residency validity and layer-transition checks.
- Selection-aware rebinding and HC activation handoff validation.
- Metadata-only Python planner consuming the runtime manifest.
- Bounded selected-expert tile streaming through the existing provider.
- Actual-child cgroup preflight and persistent diagnostic scripts.

## Verified without inference

- 1,224 descriptors discovered from Qwen4Exp runtime tensor members.
- 1,074 arena-resident descriptors.
- 144 externally streamed expert descriptors.
- Six separately managed global descriptors.
- All 48 physical layers represented.
- Layer requirements match the expected 22/28/23 tensor classes and
  50,238,336 / 67,851,136 / 45,153,792-byte totals.
- Descriptor-only execution reports zero weight-payload reads, zero weight
  arena allocations, and zero backend weight-buffer allocations.
- Residency, rebinding, HC handoff, range, and `git diff --check` regressions
  pass.

## Still experimental

- The external selected-MoE provider is verified independently, but its
  connection to the normal Qwen4Exp chat graph is not complete.
- Final HC normalization/head execution is not proven end to end.
- Vocabulary projection remains without a verified bounded output-matrix
  streaming path in the interactive chat executable.
- Runtime manifest integration has not been validated by a real inference
  transaction in this branch.

## Known limitations and blockers

1. Complete MoE graph integration is required so graph execution does not
   reference nonresident expert tensors.
2. The final HC head must stream or otherwise bind its required global tensors.
3. `output.weight` needs a bounded vocabulary projection implementation.
4. Residency should publish `VALID(layer)` only after all rebinding succeeds.
5. The supervisor must record the actual workload PID and explicitly terminate
   and reap timed-out children.
6. Current launcher tests require execution inside the configured 2 GiB,
   zero-swap systemd scope.

## Next milestones

1. Complete non-inference MoE boundary integration tests.
2. Add and test bounded final HC-head execution.
3. Add tiled vocabulary projection with explicit memory accounting.
4. Tighten residency commit ordering and failure-state tests.
5. Repair supervisor PID/timeout cleanup and rerun only harmless launcher tests.
6. Request one separately approved supervised end-to-end token attempt.

## How to help

Contributors can help by reviewing the runtime descriptor ownership model,
adding synthetic graph tests for the MoE and final-head boundaries, validating
bounded vocabulary-projection designs, and improving cgroup-v2 supervision.
No contributor should add model weights, inference logs, binaries, credentials,
or claims of generated text without reproducible evidence.
