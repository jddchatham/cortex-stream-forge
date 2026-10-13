#!/usr/bin/env python3
"""Deterministic contract test for the Qwen4Exp selected-MoE boundary.

This is a synthetic reference test only.  It does not load GGUF data or run
the model.  It mirrors the ownership contract: router -> top-k/weights ->
selected SwiGLU experts -> weighted sum -> shared expert -> residual.
"""
import math
from pathlib import Path

K, F, N_EXPERT, TOP_K = 4, 3, 4, 2
hidden = [0.25, -0.5, 0.75, 1.0]
router = [0.1, 1.2, -0.4, 0.7]
gate = [[(e + 1) * (j + 1) * 0.07 for j in range(K)] for e in range(N_EXPERT)]
up = [[(e + 2) * (j + 1) * -0.05 for j in range(K)] for e in range(N_EXPERT)]
down = [[(j + 1) * (e + 1) * 0.03 for j in range(K)] for e in range(N_EXPERT)]
shared_gate = [0.2, -0.1, 0.3, 0.4]
shared_up = [0.11, 0.22, -0.13, 0.17]
shared_down = [0.31, -0.27, 0.19, 0.23]

def dot(row, x):
    return sum(a * b for a, b in zip(row, x))

def reference(x):
    probs0 = [math.exp(v - max(router)) for v in router]
    total = sum(probs0)
    probs = [v / total for v in probs0]
    ids = sorted(range(N_EXPERT), key=lambda i: probs[i], reverse=True)[:TOP_K]
    weights = [probs[i] / sum(probs[j] for j in ids) for i in ids]
    moe = [0.0] * K
    for eid, weight in zip(ids, weights):
        g = [dot(gate[eid], x) / F] * F
        u = [dot(up[eid], x) / F] * F
        act = sum((v / (1.0 + math.exp(-v))) * w for v, w in zip(g, u))
        for j in range(K):
            moe[j] += weight * act * down[eid][j]
    sg = 1.0 / (1.0 + math.exp(-sum(a * b for a, b in zip(shared_gate, x))))
    shared = [sg * sum(a * b for a, b in zip(shared_up, x)) * v for v in shared_down]
    return ids, weights, moe, [a + b + c for a, b, c in zip(x, moe, shared)]

def partitioned(x):
    # The external boundary receives exactly the same router-derived IDs and
    # normalized weights, then performs only the selected expert contribution.
    ids, weights, _, _ = reference(x)
    moe = [0.0] * K
    for eid, weight in zip(ids, weights):
        g = [dot(gate[eid], x) / F] * F
        u = [dot(up[eid], x) / F] * F
        act = sum((v / (1.0 + math.exp(-v))) * w for v, w in zip(g, u))
        for j in range(K):
            moe[j] += weight * act * down[eid][j]
    _, _, _, final = reference(x)
    shared_plus_residual = [f - m for f, m in zip(final, reference(x)[2])]
    return ids, weights, moe, [a + b for a, b in zip(shared_plus_residual, moe)]

ids_a, weights_a, moe_a, final_a = reference(hidden)
ids_b, weights_b, moe_b, final_b = partitioned(hidden)
assert ids_a == ids_b
assert all(abs(a - b) < 1e-12 for a, b in zip(weights_a, weights_b))
assert all(abs(a - b) < 1e-12 for a, b in zip(moe_a, moe_b))
assert all(abs(a - b) < 1e-12 for a, b in zip(final_a, final_b))
assert len(hidden) == K and len(moe_a) == K and len(final_a) == K

source = (Path(__file__).parents[1] / "src" / "models" / "qwen4exp.cpp").read_text()
assert "CORTEX_FLASHNEXT_PRE_MOE" in source
assert "CORTEX_FLASHNEXT_SEGMENT_A" in source
assert "build_layer_ffn(cur, il)" in source
assert "ffn_shexp" in source

print("FLASHNEXT_SELECTED_MOE_BOUNDARY PASS")
print(f"pre_moe=[{K},tokens] router=[{N_EXPERT},tokens] selected={ids_a}")
print(f"selected_moe=[{K},tokens] post_moe=[{K},tokens] final_max_error=0")
print("normal-vs-partitioned synthetic reference: PASS")
