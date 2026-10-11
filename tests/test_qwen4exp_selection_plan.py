import pathlib
import unittest

ROOT = pathlib.Path(__file__).parents[1]
WRAPPER = (ROOT / "runtime-flashnext-layer-exp/examples/qwen38-lowres-segment-test.cpp").read_text()
LOADER = (ROOT / "runtime-flashnext-layer-exp/src/llama-model.cpp").read_text()

PLAN_MANIFEST = {
    "blk.1.hc_attn_norm.weight": 40960,
    "blk.1.hc_attn_down.weight": 1740800,
    "blk.1.hc_attn_inject.weight": 21760,
    "blk.1.hc_ffn_norm.weight": 40960,
    "blk.1.hc_ffn_down.weight": 1740800,
    "blk.1.hc_ffn_inject.weight": 21760,
    "blk.1.attn_qkv.weight": 18022400,
    "blk.1.attn_gate.weight": 8355840,
    "blk.1.ssm_conv1d.weight": 163840,
    "blk.1.ssm_dt.bias": 192,
    "blk.1.ssm_a": 192,
    "blk.1.ssm_beta.weight": 65280,
    "blk.1.ssm_alpha.weight": 65280,
    "blk.1.ssm_norm.weight": 512,
    "blk.1.ssm_out.weight": 8355840,
    "blk.1.ple_key.weight": 13926400,
    "blk.1.ple_value.weight": 3481600,
    "blk.1.ple_norm_key.weight": 40960,
    "blk.1.ple_norm_query.weight": 40960,
    "blk.1.ple_norm_conv.weight": 40960,
    "blk.1.ple_conv1d.weight": 81920,
    "blk.1.ffn_gate_inp.weight": 5242880,
    "blk.1.ffn_gate_inp_shexp.weight": 10240,
    "blk.1.ffn_gate_shexp.weight": 870400,
    "blk.1.ffn_up_shexp.weight": 870400,
    "blk.1.hc_attn_up.weight": 1843200,
    "blk.1.hc_ffn_up.weight": 1843200,
    "blk.1.ffn_down_shexp.weight": 921600,
}

class Qwen4ExpSelectionPlanTests(unittest.TestCase):
    def test_segment_enables_bounded_selector(self):
        self.assertIn('CORTEX_FLASHNEXT_LAYER_RANGE', WRAPPER)
        self.assertIn('CORTEX_FLASHNEXT_ROUTER_ONLY', WRAPPER)
        self.assertIn('CORTEX_FLASHNEXT_PRE_MOE', WRAPPER)

    def test_plan_is_before_allocation(self):
        marker = 'CORTEX_FLASHNEXT_PLAN_ONLY'
        self.assertIn(marker, WRAPPER)
        self.assertIn('CORTEX_SELECTION_PLAN', LOADER)
        self.assertLess(LOADER.index('CORTEX_SELECTION_PLAN'), LOADER.index('cortex_global_buffer.resize(global_total)'))

    def test_plan_reports_tensor_identity(self):
        self.assertIn('CORTEX_SELECTION_TENSOR', LOADER)

    def test_invalid_range_is_rejected(self):
        self.assertIn('layer < 0 || end != layer + 1 || end > 48', WRAPPER)

    def test_strict_plan_rejects_unexpected_totals(self):
        self.assertIn('CORTEX_FLASHNEXT_PLAN_STRICT', WRAPPER)
        self.assertIn('CORTEX_SELECTION_PLAN_FAIL', LOADER)
        self.assertIn('expected_global = layer0_plan ? 337715200 : 0', LOADER)
        self.assertIn('expected_layer = layer0_plan ? 50238336 : 67851136', LOADER)
        self.assertIn('expected_tensors = layer0_plan ? 23 : 28', LOADER)

    def test_layer_zero_preserves_token_lookup_but_range_does_not(self):
        self.assertIn('forge_layer == 0 &&', LOADER)
        self.assertIn('std::strncmp(name, "token_embd", 10)', LOADER)
        self.assertIn('external', LOADER)

    def test_two_stage_plan_totals_are_explicit(self):
        self.assertIn('expected_global = layer0_plan ? 337715200 : 0', LOADER)
        self.assertIn('expected_layer = layer0_plan ? 50238336 : 67851136', LOADER)
        self.assertIn('expected_tensors = layer0_plan ? 23 : 28', LOADER)

    def test_strict_plan_rejects_unvalidated_ranges(self):
        self.assertIn('forge_layer != 0 && forge_layer != 1', LOADER)
        self.assertIn('unsupported_layer_range', LOADER)

    def test_manifest_has_exact_compiled_selection(self):
        self.assertEqual(len(PLAN_MANIFEST), 28)
        self.assertEqual(sum(v for k, v in PLAN_MANIFEST.items() if k.startswith('blk.')), 67851136)

    def test_historical_layer_zero_manifest_totals_are_preserved(self):
        self.assertIn('expected_global = layer0_plan ? 337715200 : 0', LOADER)
        self.assertIn('expected_layer = layer0_plan ? 50238336 : 67851136', LOADER)
        self.assertIn('expected_tensors = layer0_plan ? 23 : 28', LOADER)

    def test_plan_reports_measurements_without_execution(self):
        self.assertIn('CORTEX_PLAN_METRICS', WRAPPER)
        self.assertIn('model_payload_allocation=SKIPPED', WRAPPER)
        self.assertIn('graph_execution=SKIPPED', WRAPPER)

if __name__ == '__main__':
    unittest.main()
