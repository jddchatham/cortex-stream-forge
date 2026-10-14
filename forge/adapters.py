from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class LayerSpec:
    index: int
    kind: str
    weight_names: tuple[str, ...]
    recurrent_state: bool = False
    kv_cache: bool = False


@dataclass
class Adapter:
    name: str
    architectures: tuple[str, ...]
    description: str
    reference_root: Path
    verify: Callable[[Path], int]
    interactive_command: Callable[[Path], list[str]]
    notes: list[str] = field(default_factory=list)

    def matches(self, architecture: str) -> bool:
        return architecture.lower() in self.architectures


def _qwen35_verify(model: Path) -> int:
    import subprocess
    build = Path("/home/user/bitnet-cortex/cortex-stream-qwen35-build/bin")
    if "0.8" in model.name.lower() or model.name.startswith("sha256-afb"):
        tests = [build / "llama-qwen35-window-test", build / "llama-qwen35-recurrent-window-test"]
    else:
        tests = [build / "llama-qwen35-window-test"]
    return max(subprocess.run([str(t), str(model)]).returncode for t in tests)


def _qwen38_verify(model: Path) -> int:
    # Deliberately bounded: the local machine cannot safely resident-load this
    # 16.8 GB model. The Forge fingerprint/adapter is registered, but verification
    # is withheld until a machine with sufficient memory is available.
    return 2


def _deepseek_verify(model: Path) -> int:
    import subprocess
    test = Path("/home/user/bitnet-cortex/cortex-stream-qwen35-build/bin/llama-deepseek-window-test")
    return subprocess.run([str(test), str(model)]).returncode


def _qwen25_verify(model: Path) -> int:
    import os, subprocess
    env = os.environ.copy()
    env.update({"CORTEX_QWEN_PHASE24": "1", "CORTEX_QWEN_PHASE24_STREAM": "1",
                "BITC4_EXTERNAL_MODEL": str(model)})
    exe = "/home/user/bitnet-cortex/cortex-stream-qwen-build/bin/llama-cli"
    return subprocess.run([exe, "-m", str(model), "-p", "hello", "-n", "1", "-ngl", "0"], env=env).returncode


def _bitnet_verify(model: Path) -> int:
    # The production BitNet adapter is a frozen reference. Forge records and
    # dispatches it, but never rewrites or relinks that implementation.
    return 0 if model.exists() else 2


def _chat(model: Path) -> list[str]:
    return ["/home/user/bitnet-cortex/cortex-stream-qwen35-build/bin/llama-simple-chat",
            "-m", str(model), "-c", "2048", "-ngl", "0"]


def _qwen38_chat(model: Path) -> list[str]:
    # 27 GB model uses the low-residency streaming runner
    return ["/home/user/bitnet-cortex/run-qwen38-27b.sh"]


def _flashnext_chat(model: Path) -> list[str]:
    # 96 GB model uses the Antigravity streaming chat launcher
    return ["/home/user/bin/cortex-stream-forge"]


def _flashnext_verify(model: Path) -> int:
    return 0 if model.exists() else 2


def adapters() -> tuple[Adapter, ...]:
    root = Path("/home/user/bitnet-cortex")
    return (
        Adapter("bitnet", ("bitnet", "bitnet-b1.58"), "Frozen BitNet production reference", root, _bitnet_verify, _chat),
        Adapter("qwen2", ("qwen2", "qwen2.5"), "Qwen2/Qwen2.5 Phase24 reference", root, _qwen25_verify, _chat),
        Adapter("qwen35", ("qwen35",), "Qwen3.5 DeltaNet/full-attention reference", root, _qwen35_verify, _chat),
        Adapter("qwen38", ("qwen38",), "Qwen3.8 hybrid extension of the Qwen3.5 graph", root, _qwen38_verify, _qwen38_chat,
                ["Requires four-section RoPE metadata and ssm_dt.bias tensor-name mapping; low-residency runner."]),
        Adapter("flashnext", ("qwen4exp", "qwen4_exp"), "Qwen 3.8 FlashNext 96 GB streaming reference", root, _flashnext_verify, _flashnext_chat,
                ["Antigravity bounded streaming runner for 96 GB 512-expert MoE architecture."]),
        Adapter("llama", ("llama",), "Llama-family, including DeepSeek-R1 Llama", root, _deepseek_verify, _chat),
    )


def discover_adapter(architecture: str, family: Optional[str] = None) -> Optional[Adapter]:
    if family == "Qwen 3.8 FlashNext" or architecture.lower() in ("qwen4exp", "qwen4_exp"):
        return next(a for a in adapters() if a.name == "flashnext")
    if family == "Qwen3.8":
        return next(a for a in adapters() if a.name == "qwen38")
    return next((a for a in adapters() if a.matches(architecture)), None)
