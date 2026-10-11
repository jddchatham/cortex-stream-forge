#!/usr/bin/env python3
"""Create the non-model synthetic input fixture for the first supervised run."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


def splitmix32(x: int) -> int:
    x = (x + 0x9E3779B9) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) * 0x85EBCA6B & 0xFFFFFFFF
    x = (x ^ (x >> 13)) * 0xC2B2AE35 & 0xFFFFFFFF
    return (x ^ (x >> 16)) & 0xFFFFFFFF


def main(out: str) -> None:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    seed = 0xC0DEC0DE
    values = []
    for i in range(10240):
        u = splitmix32(seed + i)
        values.append(((u / 4294967295.0) * 2.0 - 1.0) * 0.01)
    payload = struct.pack("<10240f", *values)
    (root / "hc_input.bin").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    manifest = {
        "fixture_version": 1,
        "origin": "synthetic_splitmix32",
        "seed": seed,
        "graph_boundary": "physical_layer_1_pre_moe",
        "physical_layer": 1,
        "model_vocabulary_size": 248320,
        "model_shards": [
            {"path": "/home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf", "sha256": "28a99b125cb905fc3bdc06baf6266a56b0cbd1b27e833b9142acc2017b673704"},
            {"path": "/home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00002-of-00003.gguf", "sha256": "2a309e0b112fde96ba3bcba5a6b58cc05e5df7bb7fad5a990eaa51df335b0e43"},
            {"path": "/home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00003-of-00003.gguf", "sha256": "ebc43c58e2eaeba1d5bdf62c8cb1f0eac198c4dc01941f771921edeebf574bc3"},
        ],
        "tensor_index": {"path": "/home/user/bitnet-cortex/cortex-stream-forge/flashnext-tensor-index.json", "sha256": "bc78052954dc3ab6d37ad189a2643c68cc8ad6ecfab5a6bb76361dc0b94f22c7"},
        "sequence": {"id": 0, "head": 0, "n_rs": 1, "rs_size": 1, "rs_zero": 0, "rs_s_copy": [0], "initialization": "fresh_zero"},
        "position": {"index": 0, "token_ids": [0], "consumed": True},
        "attention_state": {"initialization": "empty_new_sequence", "consumption": "UNVERIFIED"},
        "indexer_state": {"initialization": "empty_new_sequence", "consumption": "UNVERIFIED"},
        "ple_rows": {"status": "derived_by_runtime_from_token_id", "token_id": 0, "position": 0, "indices": []},
        "hc_input": {"file": "hc_input", "dtype": "f32_le", "shape": [2560, 4, 1], "elements": 10240},
        "files": {"hc_input": {"path": "hc_input.bin", "bytes": len(payload), "sha256": sha}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(root), "bytes": len(payload), "sha256": sha, "token_id": 0}, sort_keys=True))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
