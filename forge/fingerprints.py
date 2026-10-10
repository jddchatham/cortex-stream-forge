from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .gguf import Reader


@dataclass
class Fingerprint:
    architecture: str
    family: str
    layer_count: int
    layer_types: list[str]
    hidden: int | None
    attention_heads: int | None
    kv_heads: Any
    ffn: int | None
    recurrent_state: bool
    convolution_state: bool
    kv_cache: bool
    tensor_convention: str
    required_ops: list[str]
    persistent_state: list[str]
    metadata: dict[str, Any]
    tensor_names: list[str]

    def as_dict(self):
        return asdict(self)


def _first(m, *keys):
    for key in keys:
        if key in m:
            return m[key]
    return None


def fingerprint(path: Path) -> Fingerprint:
    reader = Reader(path).read()
    m = reader.metadata
    arch = str(m.get("general.architecture", "unknown")).lower()
    layers = int(_first(m, f"{arch}.block_count", "llama.block_count", "qwen35.block_count") or 0)
    hidden = _first(m, f"{arch}.embedding_length", "llama.embedding_length", "qwen35.embedding_length")
    heads = _first(m, f"{arch}.attention.head_count", "llama.attention.head_count", "qwen35.attention.head_count")
    kv = _first(m, f"{arch}.attention.head_count_kv", "llama.attention.head_count_kv", "qwen35.attention.head_count_kv")
    ffn = _first(m, f"{arch}.feed_forward_length", "llama.feed_forward_length", "qwen35.feed_forward_length")
    if arch == "qwen35":
        interval = int(m.get("qwen35.full_attention_interval", 0) or 0)
        if isinstance(kv, list):
            kv_list = kv
        elif interval > 0:
            # Newer Qwen3.8 metadata uses a scalar KV-head value and the
            # interval to describe the hybrid schedule.
            kv_list = [kv if (i + 1) % interval == 0 else 0 for i in range(layers)]
        else:
            kv_list = [kv] * layers
        types = ["full_attention" if int(x or 0) else "recurrent_deltanet" for x in kv_list]
        family = "Qwen3.8" if "blk.0.ssm_dt.bias" in reader.tensor_names or len(kv_list) == 65 else "Qwen3.5"
        return Fingerprint(arch, family, layers, types, hidden, heads, kv, ffn, True,
                           True, True, "blk.N.* Qwen3.5 tensor names",
                           ["RMSNorm", "SSM_CONV", "SSM_SCAN", "GatedDeltaNet", "RoPE", "attention"],
                           ["recurrent state", "short-convolution state", "attention KV cache"],
                           {k: v for k, v in m.items() if k.startswith("qwen35.")}, reader.tensor_names[:80])
    if arch in {"llama", "qwen2", "qwen2.5"}:
        family = "Qwen2/Qwen2.5" if arch.startswith("qwen2") else "Llama/DeepSeek"
        return Fingerprint(arch, family, layers, ["full_attention"] * layers, hidden, heads, kv, ffn,
                           False, False, True, f"blk.N.* {arch} tensor names",
                           ["RMSNorm", "RoPE", "attention", "FFN"], ["attention KV cache"],
                           {k: v for k, v in m.items() if k.startswith(arch + ".")}, reader.tensor_names[:80])
    if arch in {"bitnet", "bitnet-b1.58"}:
        return Fingerprint(arch, "BitNet", layers, ["bitnet_transformer"] * layers, hidden, heads, kv, ffn,
                           False, False, True, "blk.N.* BitNet tensor names",
                           ["BitNet quantized matmul", "RMSNorm", "RoPE", "attention", "FFN"],
                           ["attention KV cache"], {k: v for k, v in m.items() if k.startswith(arch + ".")}, reader.tensor_names[:80])
    return Fingerprint(arch, "unknown", layers, [], hidden, heads, kv, ffn, False, False, False,
                       "unknown", [], [], {}, [])
