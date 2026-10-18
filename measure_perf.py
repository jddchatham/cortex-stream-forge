import subprocess
import time

MODEL_PATH = "/home/user/.ollama/models/blobs/sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5"

def measure(cmd, name):
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    max_rss = 0
    anon_rss = 0
    file_rss = 0
    pid = p.pid
    while p.poll() is None:
        try:
            with open(f"/proc/{pid}/status") as f:
                for l in f:
                    if "VmHWM" in l or "VmRSS" in l:
                        val = int(l.split()[1])
                        if val > max_rss: max_rss = val
                    if "RssAnon" in l:
                        val = int(l.split()[1])
                        if val > anon_rss: anon_rss = val
                    if "RssFile" in l:
                        val = int(l.split()[1])
                        if val > file_rss: file_rss = val
        except Exception:
            pass
        time.sleep(0.005)
    p.wait()
    t1 = time.perf_counter()
    print(f"=== {name} ===")
    print(f"Return code: {p.returncode}")
    print(f"Total Wall Time: {t1 - t0:.3f} s")
    print(f"Peak RSS:        {max_rss / 1024.0:.2f} MiB")
    print(f"Peak RssAnon:    {anon_rss / 1024.0:.2f} MiB")
    print(f"Peak RssFile:    {file_rss / 1024.0:.2f} MiB\n")

measure(["/tmp/cortex_stock_qwen35_reference", MODEL_PATH], "Stock llama.cpp Reference (mmap)")
measure(["/home/user/bitnet-cortex/cortex-small-model/cortex-small-streaming", MODEL_PATH], "Cortex Operation True Streaming (Bounded pread arena)")
