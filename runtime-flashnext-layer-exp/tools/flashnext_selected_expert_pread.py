#!/usr/bin/env python3
"""Read only the frozen layer-0 selected experts from the GGUF shards.

This is intentionally an I/O boundary probe.  It does not construct the
normal ggml_mul_mat_id graph and therefore cannot accidentally materialize the
other 502 experts.
"""
import argparse
import hashlib
import json
import os
import resource
from collections import defaultdict

SELECTED = [136, 77, 317, 257, 226, 142, 175, 414, 373, 131]
ROLES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")

def expert_role(name):
    for marker in ROLES:
        if f".{marker}.weight" in name:
            return marker
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1 << 20)
    args = ap.parse_args()
    with open(args.index) as f:
        doc = json.load(f)
    root = doc["model_dir"]
    selected = set(SELECTED)
    rows = []
    all_expert_bytes = 0
    selected_bytes = 0
    for t in doc["tensors"]:
        name = t["name"]
        if not name.startswith(f"blk.{args.layer}."):
            continue
        role = expert_role(name)
        if not role:
            continue
        if t["bytes"] % 512:
            raise RuntimeError(f"expert tensor is not divisible by 512: {name}")
        per_expert = t["bytes"] // 512
        all_expert_bytes += t["bytes"]
        selected_bytes += per_expert * len(selected)
        for eid in selected:
            rows.append((eid, role, dict(t, offset=t["offset"] + eid * per_expert, bytes=per_expert,
                                         name=f"{name}[expert={eid}]")))
    rows.sort(key=lambda x: (x[0], x[1]))
    fds = {}
    total_read = 0
    digests = defaultdict(hashlib.sha256)
    try:
        for eid, role, t in rows:
            path = os.path.join(root, t["shard"])
            if path not in fds:
                fds[path] = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            fd = fds[path]
            remaining = t["bytes"]
            off = t["offset"]
            while remaining:
                n = min(remaining, args.chunk)
                data = os.pread(fd, n, off)
                if len(data) != n:
                    raise RuntimeError(f"short pread {t['name']}: {len(data)} != {n}")
                digests[(eid, role)].update(data)
                total_read += len(data)
                off += len(data)
                remaining -= len(data)
            print(f"PREAD expert={eid} role={role} name={t['name']} shard={t['shard']} offset={t['offset']} bytes={t['bytes']} sha256={digests[(eid, role)].hexdigest()}")
    finally:
        for fd in fds.values():
            os.close(fd)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(f"SELECTED_IDS={','.join(map(str, SELECTED))}")
    print(f"ALL_EXPERT_BYTES_LAYER={all_expert_bytes}")
    print(f"SELECTED_EXPERT_BYTES={selected_bytes}")
    print(f"ACTUAL_PREAD_BYTES={total_read}")
    print(f"EXPERT_BYTES_AVOIDED={all_expert_bytes - selected_bytes}")
    print(f"REDUCTION_FACTOR={all_expert_bytes / selected_bytes:.6f}")
    print(f"PEAK_RSS_BYTES={rss}")
    print("FLASHNEXT SELECTED-EXPERT PREAD: PASS")

if __name__ == "__main__":
    main()
