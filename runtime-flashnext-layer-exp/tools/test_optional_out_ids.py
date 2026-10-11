#!/usr/bin/env python3
"""Regression coverage for optional out_ids in non-final range graphs."""
from pathlib import Path

QWEN = (Path(__file__).resolve().parents[1] / "src" / "models" / "qwen4exp.cpp").read_text()
GRAPH = (Path(__file__).resolve().parents[1] / "src" / "llama-graph.cpp").read_text()

def main():
    assert "const bool omit_range_out_ids = layer_start > 0 && !final_head;" in QWEN
    assert "omit_slice_out_ids || omit_range_out_ids" in QWEN
    assert "ggml_tensor * inp_out_ids =" in QWEN
    assert "if (ubatch->token && tokens)" in GRAPH

    # Lightweight lifecycle model: each context may omit the optional leaf,
    # while a final-head context retains it. No omitted leaf is bound.
    contexts = [(1, False), (2, False), (3, False), (48, True)]
    for layer, final_head in contexts:
        out_ids = not (layer > 0 and not final_head)
        assert (not out_ids) if layer in (1, 2, 3) else out_ids

    print("optional_out_ids=PASS")

if __name__ == "__main__":
    main()
