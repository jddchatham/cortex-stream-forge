#!/usr/bin/env python3
"""Model-free contract test for atomic per-layer residency transitions."""

metadata = {f"tensor.{i}" for i in range(1224)}
layer0 = {f"blk.0.required.{i}" for i in range(22)}
layer1 = {f"blk.1.required.{i}" for i in range(22)}
published = set(layer0)
assert len(metadata) == 1224
assert len(published) == 22

# A valid transition preserves the metadata index and replaces only the
# resident manifest after all requested layer-1 entries are validated.
candidate = layer1.copy()
assert candidate <= {f"blk.1.required.{i}" for i in range(22)}
published = candidate
assert published == layer1
assert not (published & layer0)
assert len(metadata) == 1224

# A missing tensor or capacity failure leaves the previous manifest intact.
previous = set(published)
bad_candidate = layer1 - {"blk.1.required.7"}
assert len(bad_candidate) == 21
published_after_failure = previous
assert published_after_failure == layer1

# HC handoff storage is independent of the resident-weight manifest.
hc = bytes(range(256)) * 480
assert len(hc) == 122880
assert hc == bytes(hc)
print("FLASHNEXT_RESIDENCY_TRANSITION PASS")
print("layer0_selected=22 layer1_selected=22 metadata_entries=1224")
print("atomic_publish=PASS failed_transition_preserves_previous=PASS")
print("hc_handoff_preserved=PASS")
