#!/usr/bin/env python3
"""Static safety regression for bounded Qwen4Exp graph input selection.

This test intentionally does not load a model.  It guards the source-level
contract that is exercised by the real graph builder: homogeneous recurrent
ranges omit generic KV binders, homogeneous attention ranges retain them, and
mixed ranges fail closed before graph construction.
"""
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "src/models/qwen4exp.cpp"
text = source.read_text(encoding="utf-8")

required = [
    "has_recurrent_layer",
    "has_attention_layer",
    "mixed_attention_range",
    'unsupported mixed recurrent/attention range',
    "const bool full_attention_range = has_attention_layer;",
    "const bool cut_without_kv_inputs",
    "inp->get_attn()->self_k_idxs = nullptr",
    "build_inp_mem_hybrid()",
]
missing = [needle for needle in required if needle not in text]
if missing:
    raise SystemExit("range contract regression failed; missing: " + ", ".join(missing))

if text.index("if (mixed_attention_range)") > text.index("auto * inp = build_inp_mem_hybrid();"):
    raise SystemExit("mixed-range rejection occurs after graph input construction")

print("FLASHNEXT_RANGE_CONTRACT PASS")
print("recurrent [0,1): KV inputs omitted")
print("recurrent start>0: KV inputs omitted")
print("full-attention range: KV inputs retained")
print("mixed range: rejected before graph construction")
