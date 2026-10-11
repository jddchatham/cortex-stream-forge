#!/usr/bin/env python3
"""Opaque, disk-backed checkpoint vault; no model/runtime dependencies."""
from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"CORTEXSV"
VERSION = 1
_PREFIX = struct.Struct("<8sIQ")  # magic, version, header length
_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class Checkpoint:
    model_fingerprint: str
    config_fingerprint: str
    sequence_id: int
    position: int
    generation: int
    payload_size: int
    payload_sha256: str
    path: Path


class VaultError(ValueError):
    pass


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except OSError:
        pass


def _header(meta: dict, size: int, digest: str) -> bytes:
    required = {"model_fingerprint", "config_fingerprint", "sequence_id", "position", "generation"}
    if set(meta) != required:
        raise VaultError("invalid metadata fields")
    h = dict(meta, payload_size=size, payload_sha256=digest)
    raw = json.dumps(h, sort_keys=True, separators=(",", ":")).encode()
    return _PREFIX.pack(MAGIC, VERSION, len(raw)) + raw


def _digest_file(path: Path) -> tuple[int, str]:
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            size += len(block)
            sha.update(block)
    return size, sha.hexdigest()


def write_checkpoint(vault: Path, generation: int, meta: dict, payload: bytes | os.PathLike[str]) -> Path:
    """Write one generation atomically. Payload may be bytes or a source file."""
    vault.mkdir(parents=True, exist_ok=True)
    final = vault / f"checkpoint-{generation:020d}.cvs"
    payload_tmp = vault / f".payload-{generation:020d}.{os.getpid()}.tmp"
    tmp = vault / f".checkpoint-{generation:020d}.{os.getpid()}.tmp"
    sha = hashlib.sha256()
    size = 0
    try:
        with payload_tmp.open("wb") as out:
            if isinstance(payload, (str, os.PathLike)):
                source = Path(payload).open("rb")
                close_source = True
            else:
                source = None
                close_source = False
            try:
                while True:
                    block = source.read(_CHUNK) if source else payload[size:size + _CHUNK]
                    if not block:
                        break
                    out.write(block); sha.update(block); size += len(block)
            finally:
                if close_source:
                    source.close()
            out.flush(); os.fsync(out.fileno())
        header = _header(dict(meta, generation=generation), size, sha.hexdigest())
        with tmp.open("wb") as out, payload_tmp.open("rb") as src:
            out.write(header)
            while True:
                block = src.read(_CHUNK)
                if not block: break
                out.write(block)
            out.flush(); os.fsync(out.fileno())
        os.replace(tmp, final)
        _fsync_dir(vault)
        return final
    finally:
        if tmp.exists(): tmp.unlink()
        if payload_tmp.exists(): payload_tmp.unlink()


def validate(path: Path, expected: dict | None = None) -> Checkpoint:
    with path.open("rb") as f:
        prefix = f.read(_PREFIX.size)
        if len(prefix) != _PREFIX.size: raise VaultError("truncated prefix")
        magic, version, hlen = _PREFIX.unpack(prefix)
        if magic != MAGIC or version != VERSION: raise VaultError("invalid magic/version")
        try: h = json.loads(f.read(hlen))
        except Exception as e: raise VaultError("invalid header") from e
        required = {"model_fingerprint", "config_fingerprint", "sequence_id", "position", "generation", "payload_size", "payload_sha256"}
        if set(h) != required: raise VaultError("invalid header fields")
        if expected and any(h[k] != v for k, v in expected.items()): raise VaultError("metadata mismatch")
        if not isinstance(h["generation"], int) or h["generation"] < 0 or h["payload_size"] < 0: raise VaultError("invalid generation metadata")
        start = _PREFIX.size + hlen
        actual_size, digest = _digest_file_range(path, start)
        if actual_size != h["payload_size"]: raise VaultError("payload length mismatch")
        if digest != h["payload_sha256"]: raise VaultError("payload checksum mismatch")
    return Checkpoint(**h, path=path)


def _digest_file_range(path: Path, start: int) -> tuple[int, str]:
    sha = hashlib.sha256(); size = 0
    with path.open("rb") as f:
        f.seek(start)
        while True:
            b = f.read(_CHUNK)
            if not b: break
            size += len(b); sha.update(b)
    return size, sha.hexdigest()


def recover(vault: Path, expected: dict | None = None) -> Checkpoint:
    candidates = sorted(vault.glob("checkpoint-*.cvs"), reverse=True)
    for path in candidates:
        try: return validate(path, expected)
        except (OSError, VaultError): continue
    raise VaultError("no valid checkpoint")


def read_payload(cp: Checkpoint, destination: Path) -> None:
    with cp.path.open("rb") as src:
        prefix = src.read(_PREFIX.size); _, _, hlen = _PREFIX.unpack(prefix)
        src.seek(hlen, os.SEEK_CUR)
        with destination.open("wb") as dst:
            while True:
                b = src.read(_CHUNK)
                if not b: break
                dst.write(b)


def write_components(vault: Path, generation: int, meta: dict, components: dict[str, bytes | os.PathLike[str]]) -> Path:
    """Atomically commit a complete multi-component token checkpoint."""
    required = {"model_fingerprint", "config_fingerprint", "sequence_id", "position"}
    if set(meta) != required or generation < 0 or not components:
        raise VaultError("invalid manifest metadata or components")
    vault.mkdir(parents=True, exist_ok=True)
    temp = vault / f".generation-{generation:020d}.{os.getpid()}.tmp"
    final = vault / f"generation-{generation:020d}"
    try:
        temp.mkdir()
        records = {}
        for name, source in components.items():
            if not name or "/" in name or "\\" in name:
                raise VaultError("invalid component identifier")
            path = temp / f"{name}.bin"
            sha = hashlib.sha256(); size = 0
            with path.open("wb") as out:
                src = Path(source).open("rb") if isinstance(source, (str, os.PathLike)) else None
                try:
                    while True:
                        block = src.read(_CHUNK) if src else source[size:size + _CHUNK]
                        if not block: break
                        out.write(block); sha.update(block); size += len(block)
                finally:
                    if src: src.close()
                out.flush(); os.fsync(out.fileno())
            records[name] = {"bytes": size, "sha256": sha.hexdigest()}
        manifest = dict(meta, generation=generation, components=records)
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        (temp / "manifest.json.tmp").write_bytes(raw)
        with (temp / "manifest.json.tmp").open("rb") as f: os.fsync(f.fileno())
        (temp / "manifest.json.tmp").rename(temp / "manifest.json")
        _fsync_dir(temp)
        os.replace(temp, final)
        _fsync_dir(vault)
        # The manifest is the visible commit marker; rename it last within the generation.
        (final / "manifest.json").rename(vault / f"generation-{generation:020d}.manifest")
        _fsync_dir(vault)
        return vault / f"generation-{generation:020d}.manifest"
    finally:
        if temp.exists():
            import shutil; shutil.rmtree(temp)


def _validate_components_manifest(path: Path, expected: dict | None = None) -> dict:
    try:
        manifest = json.loads(path.read_text())
        required = {"model_fingerprint", "config_fingerprint", "sequence_id", "position", "generation", "components"}
        if set(manifest) != required or not isinstance(manifest["components"], dict):
            raise VaultError("invalid component manifest")
        if expected and any(manifest[k] != v for k, v in expected.items()): raise VaultError("metadata mismatch")
        generation = manifest["generation"]
        if not isinstance(generation, int) or generation < 0: raise VaultError("invalid generation")
        directory = path.parent / f"generation-{generation:020d}"
        for name, rec in manifest["components"].items():
            if not name or "/" in name or set(rec) != {"bytes", "sha256"}: raise VaultError("invalid component record")
            component = directory / f"{name}.bin"
            if not component.is_file(): raise VaultError("missing component")
            size, digest = _digest_file(component)
            if size != rec["bytes"] or digest != rec["sha256"]: raise VaultError("component mismatch")
        return manifest
    except (OSError, json.JSONDecodeError) as e:
        raise VaultError("unreadable manifest") from e


def recover_components(vault: Path, expected: dict | None = None) -> dict:
    """Return newest complete generation; never combines generations."""
    paths = sorted(vault.glob("generation-*.manifest"), reverse=True)
    for path in paths:
        try: return _validate_components_manifest(path, expected)
        except VaultError: continue
    raise VaultError("no valid component generation")
