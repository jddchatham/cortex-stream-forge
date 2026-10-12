#!/usr/bin/env python3
"""Static ownership regression for the Qwen4Exp bounded layer rebind list."""
from pathlib import Path

text = (Path(__file__).resolve().parents[1] / "src/llama-model.cpp").read_text(encoding="utf-8")
required = [
    "l.hc_attn_norm", "l.hc_attn_down", "l.hc_attn_up", "l.hc_attn_inject",
    "l.hc_ffn_norm", "l.hc_ffn_down", "l.hc_ffn_up", "l.hc_ffn_inject",
    "l.ple_key", "l.ple_value", "l.ple_norm_key", "l.ple_norm_query",
    "l.ple_norm_conv", "l.ple_conv1d", "l.ffn_gate_inp",
    "l.ffn_gate_exps", "l.ffn_up_exps", "l.ffn_down_exps",
]
missing = [name for name in required if name not in text]
if missing:
    raise SystemExit("rebind contract failed; missing: " + ", ".join(missing))
if "CORTEX_FLASHNEXT_ALLOCATOR_DIAGNOSTICS" not in (
        (Path(__file__).resolve().parents[1] / "ggml/src/ggml-alloc.c").read_text(encoding="utf-8")):
    raise SystemExit("allocator diagnostics are not compiled at the assertion boundary")
print("FLASHNEXT_REBIND_CONTRACT PASS")
print("HC, PLE, indexer, recurrent and MoE layer weights included")
print("invalid allocator assignments emit tensor metadata before assertion")
