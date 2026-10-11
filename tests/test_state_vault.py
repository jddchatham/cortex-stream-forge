import hashlib
import os
import tempfile
import time
from pathlib import Path

import unittest

from tools.state_vault import VaultError, recover, recover_components, read_payload, validate, write_checkpoint, write_components


META = {"model_fingerprint": "model-test", "config_fingerprint": "cfg-test", "sequence_id": 0, "position": 17}


class VaultTests(unittest.TestCase):
  def test_vault_roundtrip_and_recovery(self):
    with tempfile.TemporaryDirectory(prefix="cortex-vault-") as td:
        root = Path(td); payload = bytes((i * 17) & 255 for i in range(118070660))
        t0 = time.monotonic(); p1 = write_checkpoint(root, 1, META, payload); wt = time.monotonic() - t0
        cp = validate(p1, META); self.assertEqual(cp.payload_size, len(payload))
        out = root / "payload.bin"; t0 = time.monotonic(); read_payload(cp, out); rt = time.monotonic() - t0
        self.assertEqual(out.read_bytes(), payload)
        write_checkpoint(root, 2, META, b"new generation")
        # Corrupt newest payload; recovery must return generation 1.
        newest = root / "checkpoint-00000000000000000002.cvs"
        with newest.open("r+b") as f: f.seek(-1, os.SEEK_END); f.write(b"X")
        t0 = time.monotonic(); recovered = recover(root, META); recovery = time.monotonic() - t0
        self.assertEqual(recovered.generation, 1)
        self.assertTrue(wt >= 0 and rt >= 0 and recovery >= 0)


  def test_rejects_truncation_header_and_metadata(self):
    with tempfile.TemporaryDirectory(prefix="cortex-vault-") as td:
        root = Path(td); p = write_checkpoint(root, 1, META, b"payload")
        raw = p.read_bytes()
        p.write_bytes(raw[:-1])
        with self.assertRaises(VaultError): validate(p, META)
        p.write_bytes(raw); p.write_bytes(b"bad")
        with self.assertRaises(VaultError): validate(p, META)
        p.write_bytes(raw)
        with self.assertRaises(VaultError): validate(p, {**META, "model_fingerprint": "wrong"})


  def test_interrupted_temp_is_ignored(self):
    with tempfile.TemporaryDirectory(prefix="cortex-vault-") as td:
        root = Path(td); write_checkpoint(root, 1, META, b"good")
        (root / ".checkpoint-00000000000000000002.interrupted.tmp").write_bytes(b"partial")
        self.assertEqual(recover(root, META).generation, 1)

  def test_multi_component_commit_and_crash_recovery(self):
    with tempfile.TemporaryDirectory(prefix="cortex-vault-components-") as td:
        root = Path(td); meta = {"model_fingerprint": "m", "config_fingerprint": "c", "sequence_id": 0, "position": 2}
        first = {"recurrent": b"r1" * 1000, "attention_kv": b"a1" * 1000, "position_meta": b"p1"}
        write_components(root, 1, meta, first)
        second = {"recurrent": b"r2" * 1000, "attention_kv": b"a2" * 1000, "indexer": b"i2", "position_meta": b"p2"}
        write_components(root, 2, meta, second)
        self.assertEqual(recover_components(root, meta)["generation"], 2)
        bad = root / "generation-00000000000000000003"
        bad.mkdir(); (bad / "recurrent.bin").write_bytes(b"r3")
        (root / "generation-00000000000000000003.manifest").write_text(
            '{"model_fingerprint":"m","config_fingerprint":"c","sequence_id":0,"position":2,"generation":3,'
            '"components":{"recurrent":{"bytes":2,"sha256":"00"},"attention_kv":{"bytes":2,"sha256":"00"}}}')
        self.assertEqual(recover_components(root, meta)["generation"], 2)
        (root / "generation-00000000000000000002" / "recurrent.bin").write_bytes(b"corrupt")
        self.assertEqual(recover_components(root, meta)["generation"], 1)


if __name__ == "__main__":
    unittest.main()
