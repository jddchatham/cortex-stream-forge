import pathlib
import unittest

ROOT = pathlib.Path(__file__).parents[1]
WRAPPER = (ROOT / "runtime-flashnext-layer-exp/examples/qwen38-lowres-segment-test.cpp").read_text()
LOADER = (ROOT / "runtime-flashnext-layer-exp/src/llama-model.cpp").read_text()

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

if __name__ == '__main__':
    unittest.main()
