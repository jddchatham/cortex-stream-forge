import pathlib
import unittest

ROOT = pathlib.Path(__file__).parents[1]
WRAPPER = (ROOT / "runtime-flashnext-layer-exp/examples/qwen38-lowres-segment-test.cpp").read_text()
LOADER = (ROOT / "runtime-flashnext-layer-exp/src/llama-model.cpp").read_text()


class Qwen4ExpSelectionPlanTests(unittest.TestCase):
    def test_nonzero_segment_enables_flashnext_selector(self):
        self.assertIn('setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1)', WRAPPER)
        self.assertIn('setenv("CORTEX_FLASHNEXT_ROUTER_ONLY", "1", 1)', WRAPPER)
        self.assertIn('setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1)', WRAPPER)

    def test_plan_flag_is_optional_and_preallocation(self):
        self.assertIn('"--plan"', WRAPPER)
        marker = 'if (std::getenv("CORTEX_FLASHNEXT_PLAN_ONLY") != nullptr)'
        self.assertIn(marker, LOADER)
        self.assertLess(LOADER.index(marker), LOADER.index("pimpl->cortex_global_buffer.resize(global_total)"))

    def test_plan_reports_tensor_identity_and_sizes(self):
        self.assertIn("CORTEX_SELECTION_PLAN", LOADER)
        self.assertIn("CORTEX_SELECTION_TENSOR", LOADER)


if __name__ == "__main__":
    unittest.main()
