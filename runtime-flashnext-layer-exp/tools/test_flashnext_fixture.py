import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from flashnext_fixture import FixtureError, compare_f32, load_manifest


class FixtureTests(unittest.TestCase):
    def make_fixture(self):
        root = Path(tempfile.mkdtemp())
        hc = struct.pack("<10240f", *([1.0] * 10240))
        (root / "hc_input.bin").write_bytes(hc)
        (root / "state.bin").write_bytes(b"state")
        def rec(name):
            data = (root / name).read_bytes()
            return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        manifest = {
            "fixture_version": 1,
            "graph_boundary": "physical_layer_1_pre_moe",
            "model_shards": [{"path": "model.gguf", "sha256": "0" * 64}],
            "sequence": {"id": 0, "head": 0, "n_rs": 1, "rs_size": 1, "rs_zero": 0, "rs_s_copy": [0]},
            "position": {"index": 0, "token_ids": [], "consumed": False},
            "ple_rows": {"status": "metadata_only", "indices": [0], "rows": []},
            "hc_input": {"file": "hc_input", "dtype": "f32_le", "shape": [2560, 4, 1]},
            "files": {"hc_input": rec("hc_input.bin"), "sequence_state": rec("state.bin")},
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root

    def test_compare_exact_and_mismatch(self):
        a = struct.pack("<2f", 1.0, 2.0)
        self.assertTrue(compare_f32(a, a)["byte_exact"])
        r = compare_f32(a, struct.pack("<2f", 1.0, 3.0))
        self.assertEqual(r["first_differing_element"], 1)

    def test_valid_fixture(self):
        root = self.make_fixture()
        self.assertEqual(load_manifest(root)["fixture_version"], 1)

    def test_tamper_rejected(self):
        root = self.make_fixture()
        (root / "state.bin").write_bytes(b"tampered")
        with self.assertRaises(FixtureError):
            load_manifest(root)


if __name__ == "__main__":
    unittest.main()
