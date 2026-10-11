#!/usr/bin/env python3
"""Non-inference checks for the explicit consecutive-layer range contract."""
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "examples" / "flashnext-owned-layer.cpp"
TEXT = SOURCE.read_text(encoding="utf-8")

def require(fragment: str) -> None:
    assert fragment in TEXT, f"missing range contract fragment: {fragment}"

def main() -> None:
    require('CORTEX_FLASHNEXT_RANGE_START')
    require('CORTEX_FLASHNEXT_RANGE_END')
    require('RANGE_START must be >= 1 in external-HC mode')
    require('RANGE_END must be greater than RANGE_START')
    require('RANGE_END exceeds model layer count')
    require('legacy chain flags cannot be combined')
    require('std::numeric_limits<uint32_t>::max()')
    # The range is no longer restricted to hard-coded end points.
    require('for (uint32_t layer = start + 1; layer < end; ++layer)')
    require('target_layer = requested_range.start')
    require('LAYER_START must equal RANGE_START')
    require('FLASHNEXT_LEGACY_CHAIN_REJECTED')
    # The first transaction is external input; later transactions use owned
    # prior output and preserve the existing teardown boundary.
    require('const std::vector<float> layer1_output')
    require('llama_free(ctx);')
    require('run_layer_transaction(layer2_ctx, 2, 0, layer1_output')
    require('return transaction.state_committed ? 0 : 17;')
    print('layer_range_contract=PASS')

if __name__ == '__main__':
    main()
