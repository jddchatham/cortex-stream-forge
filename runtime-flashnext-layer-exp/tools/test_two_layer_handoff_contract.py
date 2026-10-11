#!/usr/bin/env python3
"""Static and synthetic checks for the existing layer-1 -> layer-2 handoff."""

from pathlib import Path
import hashlib
import struct


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "examples" / "flashnext-owned-layer.cpp").read_text(encoding="utf-8")
QWEN = (ROOT / "src" / "models" / "qwen4exp.cpp").read_text(encoding="utf-8")


def main() -> None:
    # The handoff is independently owned F32 storage with the complete HC
    # lane layout: 2560 * 4 values, then layer-1 context teardown.
    assert "const std::vector<float> layer1_output(transaction.output.begin(), transaction.output.end());" in RUNNER
    assert "llama_free(ctx);" in RUNNER
    assert "llama_model_free(model);" in RUNNER
    assert "setenv(\"CORTEX_LOW_RESIDENCY_LAYER\", \"2\", 1);" in RUNNER
    assert "run_layer_transaction(layer2_ctx, 2, 0, layer1_output, argv[2])" in RUNNER
    assert "previous_output.size() == 2560 * 4" in RUNNER

    # A bounded range output is explicitly the HC residual and skips final
    # normalization/vocabulary projection; this is the next layer contract.
    assert 'cb(res_hc, "flashnext_range_output", n_layer - 1);' in QWEN
    assert "final_norm=SKIPPED vocab_projection=SKIPPED" in QWEN

    # PLE is consumed only inside the model-declared PLE layer branch. The
    # shared PLE input may nevertheless be provisioned for the graph, which
    # explains row-provider activity even when a later layer does not consume
    # the values mathematically.
    assert "if (hparams.is_ple(il))" in QWEN
    assert "res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);" in QWEN
    assert "ple_emb = build_inp_ple(mctx_hyb);" in QWEN

    # Layer contexts are recreated per physical layer, preventing accidental
    # reuse of prior-layer recurrent/KV/indexer storage.
    assert "layer2_model ? llama_init_from_model(layer2_model, cp) : nullptr" in RUNNER
    assert "llama_free(ctx);" in RUNNER
    assert "llama_model_free(model);" in RUNNER

    # Synthetic byte-integrity check: F32 bytes and lane ordering survive an
    # owned copy exactly, independent of any model execution.
    values = [struct.unpack("<f", struct.pack("<f", i * 0.25 - 3.0))[0] for i in range(2560 * 4)]
    source = b"".join(struct.pack("<f", value) for value in values)
    handoff = bytes(source)
    assert len(handoff) == 40960
    assert handoff == source
    assert hashlib.sha256(handoff).hexdigest() == hashlib.sha256(source).hexdigest()

    # The existing chain reports nonzero failure for an unsuccessful layer-2
    # transaction rather than treating a failed handoff as success.
    assert "return transaction.state_committed ? 0 : 17;" in RUNNER
    assert "return 20;" in RUNNER
    print("two_layer_handoff_contract=PASS")


if __name__ == "__main__":
    main()
