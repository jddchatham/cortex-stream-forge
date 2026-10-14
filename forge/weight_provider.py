"""Bounded, file-backed GGUF tensor loading for Forge.

This deliberately does not mmap or materialize the model.  GGUF descriptors
remain small metadata objects; only the selected tensors are copied into one
reusable resident byte buffer.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from .gguf import Reader, TensorDescriptor


def rss_bytes():
    with open("/proc/self/status", encoding="ascii") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


@dataclass
class ResidentLayer:
    descriptors: tuple
    buffer: bytearray
    slices: dict

    def view(self, name):
        start, end = self.slices[name]
        return memoryview(self.buffer)[start:end]


class WeightProvider:
    """Index GGUF metadata and stream selected tensor ranges from disk."""

    def __init__(self, path, chunk_bytes=8 * 1024 * 1024):
        self.path = Path(path)
        self.reader = Reader(self.path).read()
        self.index = self.reader.tensor_index
        self.chunk_bytes = chunk_bytes
        self._fd = os.open(self.path, os.O_RDONLY)
        self._buffer = bytearray()
        self._capacity = 0

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def buffer_capacity(self):
        return self._capacity

    def descriptor(self, name):
        return self.index[name]

    def _read(self, descriptor, destination):
        remaining = descriptor.nbytes
        source = descriptor.absolute_offset
        at = 0
        while remaining:
            count = min(remaining, self.chunk_bytes)
            data = os.pread(self._fd, count, source)
            if len(data) != count:
                raise IOError(f"short read for {descriptor.name}: {len(data)} != {count}")
            destination[at:at + count] = data
            source += count
            at += count
            remaining -= count

    def load(self, names):
        descriptors = tuple(self.index[name] for name in names)
        total = sum(t.nbytes for t in descriptors)
        if total > self._capacity:
            self._buffer = bytearray(total)
            self._capacity = total
        slices = {}
        at = 0
        for descriptor in descriptors:
            end = at + descriptor.nbytes
            self._read(descriptor, memoryview(self._buffer)[at:end])
            slices[descriptor.name] = (at, end)
            at = end
        return ResidentLayer(descriptors, self._buffer, slices)

    def load_layer(self, layer, include_names=None):
        prefix = f"blk.{layer}."
        names = [t.name for t in self.reader.tensor_descriptors if t.name.startswith(prefix)]
        if include_names:
            names.extend(include_names)
        return self.load(dict.fromkeys(names))

    def release(self):
        """Release logical residency while retaining capacity for reuse."""
        return self._capacity
