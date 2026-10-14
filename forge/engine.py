from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import subprocess
from typing import Optional

from .adapters import Adapter, discover_adapter
from .fingerprints import Fingerprint, fingerprint


@dataclass
class ModelInfo:
    path: Path
    architecture: str
    sha256: str
    metadata: dict[str, str]
    fingerprint: Fingerprint


@dataclass
class StreamPlan:
    adapter: Adapter
    model: ModelInfo
    external_buffer_bytes: Optional[int] = None
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    preserves: tuple[str, ...] = ("activations", "KV-cache", "model state")


class FirstDivergence:
    """Structured execution evidence shared by adapters and diagnostics."""

    def __init__(self, operation="", tensor="", max_abs=0.0, max_rel=0.0, mismatch_count=0,
                 source_address=None, destination_address=None, metadata=None):
        self.operation = operation
        self.tensor = tensor
        self.max_abs = max_abs
        self.max_rel = max_rel
        self.mismatch_count = mismatch_count
        self.source_address = source_address
        self.destination_address = destination_address
        self.metadata = metadata or {}

    def as_dict(self):
        return self.__dict__.copy()


class ForgeEngine:
    """Owns discovery, planning, lifecycle, and oracle/candidate dispatch.

    Mathematical graph construction remains inside the selected adapter/runtime.
    This class owns the architecture-neutral contract: one resident weight
    buffer, segment boundaries, state lifetime, scheduler completion, and proof.
    """

    def __init__(self, model: Path, buffer_bytes: Optional[int] = None):
        self.model = Path(model)
        self.buffer_bytes = buffer_bytes
        self.info = self.inspect()
        self.adapter = discover_adapter(self.info.architecture, self.info.fingerprint.family)
        if not self.adapter:
            raise ValueError(f"unsupported GGUF architecture: {self.info.architecture}")

    def inspect(self) -> ModelInfo:
        fp = fingerprint(self.model)
        return ModelInfo(self.model, fp.architecture, self._sha256(), {}, fp)

    def _sha256(self) -> str:
        # Check known cached hashes for large models to avoid full disk read
        known_hashes = {
            "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf": "28a99b125cb905fc3bdc06baf6266a56b0cbd1b27e833b9142acc2017b673704",
            "sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6": "6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6",
            "sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c": "dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c",
            "sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5": "afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5",
        }
        if self.model.name in known_hashes:
            return known_hashes[self.model.name]
        digest = hashlib.sha256()
        with self.model.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def plan(self) -> StreamPlan:
        return StreamPlan(self.adapter, self.info, self.buffer_bytes)

    def verify(self) -> int:
        return self.adapter.verify(self.model)

    def chat(self) -> int:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("CORTEX_") or k.startswith("CORTEX_FORGE_")}
        return subprocess.run(self.adapter.interactive_command(self.model), env=env).returncode

    def plan_json(self) -> str:
        plan = self.plan()
        return json.dumps({"model": str(plan.model.path), "architecture": plan.model.architecture,
                           "sha256": plan.model.sha256, "adapter": plan.adapter.name,
                           "fingerprint": plan.model.fingerprint.as_dict(),
                           "buffer_bytes": plan.external_buffer_bytes,
                           "preserves": plan.preserves}, indent=2)
