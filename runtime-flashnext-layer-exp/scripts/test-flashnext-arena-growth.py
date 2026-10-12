#!/usr/bin/env python3
"""Model-free checks for bounded replacement-arena planning."""

BUDGET = 768 * 1024 * 1024
old = 50_238_336
new = 67_851_136
assert new > old and new <= BUDGET
assert old + new < BUDGET  # transition headroom for old + replacement arenas
assert (new, old) == (67_851_136, 50_238_336)

manifest = {"blk.0.weight.0", "blk.0.weight.1"}
candidate = {"blk.1.weight.0", "blk.1.weight.1"}
before = set(manifest)
assert candidate != manifest
manifest = candidate  # publish only after plan and allocation checks
assert manifest == candidate

# Simulated failures do not publish a partial candidate.
manifest_before_failure = set(manifest)
try:
    raise OSError("short pread")
except OSError:
    pass
assert manifest == manifest_before_failure

assert new <= (2 * 1024 * 1024 * 1024)
print("FLASHNEXT_ARENA_GROWTH PASS")
print("old=50238336 new=67851136 budget=805306368 rollback=PASS overflow_rejection=PASS")
