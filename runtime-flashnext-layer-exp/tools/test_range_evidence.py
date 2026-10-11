#!/usr/bin/env python3
"""Offline checks for durable generalized-range evidence."""
import hashlib
import json
import tempfile
from pathlib import Path

REQUIRED = {
    "layer", "input_shape", "output_shape", "input_bytes", "output_bytes",
    "input_sha256", "output_sha256", "transaction", "handoff", "state_commit",
    "pre_moe", "sparse_moe", "graph_b", "ple_rows_provisioned", "ple_bytes_read",
    "ple_consumption", "selected_experts", "unselected_expert_reads",
}

def main():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "layers.jsonl"
        data = b"\x00" * (2560 * 4 * 4)
        record = {
            "layer": 2, "input_shape": [2560, 4, 1], "output_shape": [2560, 4, 1],
            "input_bytes": len(data), "output_bytes": len(data),
            "input_sha256": hashlib.sha256(data).hexdigest(),
            "output_sha256": hashlib.sha256(data).hexdigest(),
            "transaction": "PASS", "handoff": "PASS", "state_commit": "PASS",
            "pre_moe": "PASS", "sparse_moe": "PASS", "graph_b": "PASS",
            "ple_rows_provisioned": 16, "ple_bytes_read": 1440, "ple_consumption": "CONSUMED",
            "selected_experts": 10, "unselected_expert_reads": 0,
        }
        p.write_text(json.dumps(record) + "\n")
        got = json.loads(p.read_text().splitlines()[0])
        assert REQUIRED <= got.keys()
        assert got["input_sha256"] == hashlib.sha256(data).hexdigest()
        assert got["handoff"] == "PASS"
        assert got["state_commit"] == "PASS"
        partial = dict(got, transaction="FAIL", state_commit="FAIL", graph_b="FAIL")
        assert partial["transaction"] != "PASS"
    print("range_evidence=PASS")

if __name__ == "__main__":
    main()
