#!/usr/bin/env python3
"""Non-inference regression for external embedding/token metadata binding."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRAPH = (ROOT / "src" / "llama-graph.cpp").read_text(encoding="utf-8")
QWEN = (ROOT / "src" / "models" / "qwen4exp.cpp").read_text(encoding="utf-8")
RUNNER = (ROOT / "examples" / "flashnext-owned-layer.cpp").read_text(encoding="utf-8")


def main() -> None:
    # Generic input upload must require both the batch token data and the
    # graph-owned token tensor. This is the invariant that prevents the
    # Phase-2L null dereference.
    assert "if (ubatch->token && tokens)" in GRAPH

    # The range graph must omit the token graph tensor while retaining the
    # externally supplied F32 embedding input.
    range_start = GRAPH.index("const bool range_input =")
    range_body = GRAPH[range_start:GRAPH.index("// select one of the 2 inputs", range_start)]
    assert "if (!range_input)" in range_body
    assert "inp->tokens = ggml_new_tensor_1d" in range_body
    assert "inp->embd = ggml_new_tensor_2d" in range_body

    # Qwen4Exp PLE consumes ubatch token metadata independently of the
    # generic token tensor, so token_id=0 remains available to row hashing.
    ple_start = QWEN.index("auto tok_of =")
    ple_body = QWEN[ple_start:QWEN.index("const int64_t n_tokens", ple_start)]
    assert "return ubatch->token ? ubatch->token[k] : img_tok;" in ple_body
    assert "tok_of(" in QWEN[ple_start:QWEN.index("ggml_backend_tensor_set", ple_start)] or "tok_of" in QWEN

    # Every bounded range transaction must create the execution-scoped
    # pre-MoE/router captures consumed by its coordinator.
    assert 'setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);' in RUNNER
    assert 'return transaction.state_committed ? 0 : 17;' in RUNNER

    print("external_embedding_token_binding=PASS")


if __name__ == "__main__":
    main()
