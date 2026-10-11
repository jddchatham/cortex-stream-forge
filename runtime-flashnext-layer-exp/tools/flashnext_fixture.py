#!/usr/bin/env python3
"""Versioned, fail-closed fixture and comparison helpers for layer-1 work."""
from __future__ import annotations

import hashlib
import json
import math
import struct
import argparse
from pathlib import Path
from typing import Any

FIXTURE_VERSION = 1
HC_BYTES = 10240 * 4


class FixtureError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def load_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), "missing manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(f"invalid manifest: {exc}") from exc
    _require(manifest.get("fixture_version") == FIXTURE_VERSION, "unsupported fixture version")
    _require(manifest.get("graph_boundary") == "physical_layer_1_pre_moe", "wrong graph boundary")
    _require(isinstance(manifest.get("model_shards"), list) and manifest["model_shards"], "missing model shards")
    _require(isinstance(manifest.get("files"), dict), "missing file records")
    _require(manifest.get("sequence", {}).get("id") == 0, "only sequence 0 is supported")
    _require(manifest.get("position", {}).get("index") is not None, "missing position")

    for logical, record in manifest["files"].items():
        _require(isinstance(record, dict), f"invalid file record: {logical}")
        rel = record.get("path")
        _require(isinstance(rel, str) and rel and not Path(rel).is_absolute(), f"unsafe path: {logical}")
        path = (root / rel).resolve()
        _require(root.resolve() in path.parents, f"file escapes fixture: {logical}")
        _require(path.is_file(), f"missing payload: {logical}")
        size = path.stat().st_size
        _require(size == record.get("bytes"), f"length mismatch: {logical}")
        _require(sha256_file(path) == record.get("sha256"), f"hash mismatch: {logical}")

    hc = manifest.get("hc_input", {})
    _require(hc.get("dtype") == "f32_le", "HC input must be little-endian F32")
    _require(hc.get("shape") == [2560, 4, 1], "unexpected HC shape")
    _require(hc.get("file") == "hc_input", "HC file record missing")
    _require(manifest["files"]["hc_input"]["bytes"] == HC_BYTES, "HC input must contain 10240 F32 values")
    return manifest


def compare_f32(actual: bytes, expected: bytes, *, atol: float = 0.0, rtol: float = 0.0) -> dict[str, Any]:
    _require(len(actual) % 4 == 0 and len(expected) % 4 == 0, "F32 payload is not aligned")
    _require(len(actual) == len(expected), "F32 payload length mismatch")
    a = struct.unpack("<%df" % (len(actual) // 4), actual)
    e = struct.unpack("<%df" % (len(expected) // 4), expected)
    first = None
    max_abs = 0.0
    max_rel = 0.0
    sum_sq = 0.0
    for i, (x, y) in enumerate(zip(a, e)):
        _require(math.isfinite(x) and math.isfinite(y), f"non-finite value at {i}")
        diff = abs(x - y)
        max_abs = max(max_abs, diff)
        max_rel = max(max_rel, diff / max(abs(y), 1e-30))
        sum_sq += diff * diff
        if first is None and diff > atol + rtol * abs(y):
            first = i
    return {
        "bytes": len(actual),
        "elements": len(a),
        "actual_sha256": hashlib.sha256(actual).hexdigest(),
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
        "byte_exact": actual == expected,
        "first_differing_element": first,
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "rms_error": math.sqrt(sum_sq / len(a)) if a else 0.0,
    }


def validate_fixture(root: str | Path) -> dict[str, Any]:
    manifest = load_manifest(root)
    return {"valid": True, "fixture_version": manifest["fixture_version"],
            "graph_boundary": manifest["graph_boundary"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="validate a Flash-Next layer fixture")
    parser.add_argument("fixture")
    args = parser.parse_args()
    print(json.dumps(validate_fixture(args.fixture), sort_keys=True))

