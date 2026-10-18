import subprocess
import resource
import time

def measure(cmd, name):
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    t1 = time.perf_counter()
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f"=== {name} ===")
    print(f"Return code: {p.returncode}")
    print(f"Total Wall Time: {t1 - t0:.3f} s")
    print(f"Peak RSS: {ru.ru_maxrss / 1024.0:.2f} MiB")

measure(["/tmp/cortex_stock_qwen35_reference", "/home/user/.ollama/models/blobs/sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5"], "Stock llama.cpp Reference")
measure(["/home/user/bitnet-cortex/cortex-small-model/cortex-small-streaming", "/home/user/.ollama/models/blobs/sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5"], "Cortex Small Streaming")
