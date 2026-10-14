import argparse
import sys
from pathlib import Path
from .engine import ForgeEngine
from .weight_provider import WeightProvider, rss_bytes


KNOWN_MODELS = [
    ("Qwen2.5", "/home/user/models/qwen/Qwen2.5-Coder-3B-Instruct-abliterated-Q4_K_M.gguf"),
    ("Qwen3.5 0.8B", "/home/user/.ollama/models/blobs/sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5"),
    ("Qwen3.5 9B", "/home/user/.ollama/models/blobs/sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c"),
    ("DeepSeek-R1 Llama 8B", "/home/user/Downloads/DeepSeek-R1-Llama-8B-Abliterated-Q4_K_M.gguf"),
    ("BitNet4", "/home/user/models/bitnet/bitnet-abliterated-i2s-run17.gguf"),
    ("Qwen 3.8 FlashNext — 96 GB", "/home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf"),
    ("Qwen 3.8 FlashNext — 27 GB", "/home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6"),
]


def main():
    p = argparse.ArgumentParser(prog="cortex-forge")
    p.add_argument("--model", type=Path)
    p.add_argument("--select", action="store_true", help="select from locally known supported models")
    p.add_argument("--oracle", action="store_true", help="run the adapter's monolithic/oracle verification")
    p.add_argument("--verify", action="store_true", help="run oracle-vs-streamed adapter verification")
    p.add_argument("--chat", action="store_true", help="launch interactive chat")
    p.add_argument("--plan", action="store_true", help="print the adapter/stream plan")
    p.add_argument("--buffer-bytes", type=int)
    p.add_argument("--stream-layer", type=int, help="safely load one GGUF layer without full-model loading")
    args = p.parse_args()
    if args.select:
        available = [(name, Path(path)) for name, path in KNOWN_MODELS if Path(path).exists()]
        if not available:
            p.error("no known supported local models are present")
        print("Cortex Stream Forge supported local models:", file=sys.stderr)
        for i, (name, path) in enumerate(available, 1):
            print(f"  {i}. {name} — {path}", file=sys.stderr)
        print("Select model: ", end="", file=sys.stderr, flush=True)
        choice = int(input())
        args.model = available[choice - 1][1]
    if args.model is None:
        p.error("--model is required unless --select is used")
    if args.stream_layer is not None:
        before = rss_bytes()
        with WeightProvider(args.model) as provider:
            resident = provider.load_layer(args.stream_layer)
            peak = rss_bytes()
            capacity = provider.buffer_capacity
            print(f"low_residency layer={args.stream_layer} tensors={len(resident.descriptors)} "
                  f"bytes={sum(t.nbytes for t in resident.descriptors)} buffer_capacity={capacity} "
                  f"rss_before={before} rss_peak={peak} rss_growth={peak - before}")
        return 0
    engine = ForgeEngine(args.model, args.buffer_bytes)
    if args.plan or not (args.oracle or args.verify or args.chat):
        print(engine.plan_json())
    if args.oracle or args.verify:
        return engine.verify()
    if args.chat:
        return engine.chat()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
