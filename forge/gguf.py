"""Small metadata-only GGUF reader used by Forge discovery.

It reads the GGUF header and key/value table only; tensor payloads are never
validated or loaded. This keeps fingerprinting safe for large models.
"""
import struct
from dataclasses import dataclass
from math import prod
from pathlib import Path


class GGUFError(ValueError):
    pass


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    dims: tuple
    ggml_type: int
    offset: int
    nbytes: int
    data_offset: int

    @property
    def absolute_offset(self):
        return self.data_offset + self.offset


_TYPE_LAYOUT = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50),
    20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 54), 36: (32, 18),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2),
}


def tensor_nbytes(dims, ggml_type):
    """Return the GGML payload size without touching the payload."""
    try:
        block, type_size = _TYPE_LAYOUT[ggml_type]
    except KeyError as exc:
        raise GGUFError(f"unsupported GGML tensor type {ggml_type}") from exc
    if not dims:
        return type_size
    rows = prod(dims[1:]) if len(dims) > 1 else 1
    columns = dims[0]
    if columns % block:
        raise GGUFError(f"tensor dimensions {dims} are not aligned to block {block}")
    return rows * (columns // block) * type_size


class Reader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.metadata = {}
        self.version = 0
        self.tensor_count = 0
        self.alignment = 32
        self.tensor_descriptors = []
        self.tensor_names = []

    def _str(self, f):
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def _value(self, f, typ):
        formats = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                   6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if typ == 8:
            return self._str(f)
        if typ == 9:
            subtype = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [self._value(f, subtype) for _ in range(n)]
        if typ not in formats:
            raise GGUFError(f"unsupported GGUF metadata type {typ}")
        return struct.unpack(formats[typ], f.read(struct.calcsize(formats[typ])))[0]

    def read(self):
        with self.path.open("rb") as f:
            if f.read(4) != b"GGUF":
                raise GGUFError(f"not a GGUF file: {self.path}")
            self.version, self.tensor_count, n_kv = struct.unpack("<IQQ", f.read(20))
            for _ in range(n_kv):
                key = self._str(f)
                typ = struct.unpack("<I", f.read(4))[0]
                self.metadata[key] = self._value(f, typ)
            raw_tensors = []
            for _ in range(self.tensor_count):
                name = self._str(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
                ggml_type = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                raw_tensors.append((name, dims, ggml_type, offset))
                self.tensor_names.append(name)
            descriptor_end = f.tell()
        self.alignment = int(self.metadata.get("general.alignment", 32))
        self.data_offset = (descriptor_end + self.alignment - 1) // self.alignment * self.alignment
        self.tensor_descriptors = [
            TensorDescriptor(name, dims, typ, offset, tensor_nbytes(dims, typ), self.data_offset)
            for name, dims, typ, offset in raw_tensors
        ]
        self.tensor_index = {t.name: t for t in self.tensor_descriptors}
        return self
