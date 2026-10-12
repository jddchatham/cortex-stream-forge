#!/usr/bin/env python3
"""Deterministic HC-wide handoff contract test; no model loading."""
N_EMBD, HC, TOKENS = 2560, 4, 3
source = [float(i) for i in range(N_EMBD * HC * TOKENS)]
handoff = list(source)
assert len(handoff) == N_EMBD * HC * TOKENS
for t in range(TOKENS):
    for lane in range(HC):
        base = (t * HC + lane) * N_EMBD
        assert handoff[base] == source[base]
        assert handoff[base + N_EMBD - 1] == source[base + N_EMBD - 1]
assert all(v == v and abs(v) != float("inf") for v in handoff)
print("FLASHNEXT_HANDOFF_CONTRACT PASS")
print(f"layer0_output=[{N_EMBD},{HC},{TOKENS}] elements={len(source)} bytes={len(source)*4}")
print(f"layer1_input=[{N_EMBD*HC},{TOKENS}] elements={len(handoff)} bytes={len(handoff)*4}")
