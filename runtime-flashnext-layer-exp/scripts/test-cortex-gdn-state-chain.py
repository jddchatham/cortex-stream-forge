#!/usr/bin/env python3
"""Validate the exported layer-0 GDN state chain from a stateful run."""
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def equal(a, b):
    if a.stat().st_size != b.stat().st_size:
        return False, -1
    with a.open("rb") as fa, b.open("rb") as fb:
        offset = 0
        while True:
            xa, xb = fa.read(1024 * 1024), fb.read(1024 * 1024)
            if xa != xb:
                for i, (va, vb) in enumerate(zip(xa, xb)):
                    if va != vb:
                        return False, offset + i
                return False, offset + min(len(xa), len(xb))
            if not xa:
                return True, -1
            offset += len(xa)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cortex-iter102")
    log = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/cortex-iter102.log")
    replay = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    if not log.exists():
        print("GDN_CHAIN FAIL missing_log", log); return 2

    text = log.read_text(errors="replace")
    ordinals = [int(x) for x in re.findall(r"CORTEX_GDN_EXPORT ordinal=(\d+) role=persistent_s", text)]
    positions = [int(x) for x in re.findall(r"CORTEX_CHAT TOKEN position=(\d+)", text)]
    expected_ord = [0, 49, 98, 147]
    expected_pos = [8, 9, 10, 11]
    if ordinals[:4] != expected_ord or positions[:4] != expected_pos:
        print("GDN_CHAIN FAIL mapping", "ordinals", ordinals[:4], "positions", positions[:4]); return 3
    mapping = dict(zip(expected_pos, expected_ord))
    print("GDN_CHAIN mapping", " ".join(f"pos{p}=ord{o}" for p, o in mapping.items()))

    states = {}
    pres = {}
    for pos, ordinal in mapping.items():
        suffix = "" if ordinal == 0 else f"-{ordinal}"
        pre = root.with_name(root.name + f"-s_before{suffix}.bin")
        post = root.with_name(root.name + f"-state_after{suffix}.bin")
        cache = root.with_name(root.name + f"-persistent_s{suffix}.bin")
        for role, path in (("pre", pre), ("post", post), ("cache", cache)):
            if not path.exists():
                print("GDN_CHAIN FAIL missing", pos, role, path); return 4
            print(f"GDN_CHAIN pos={pos} role={role} bytes={path.stat().st_size} sha256={digest(path)}")
        ok, first = equal(post, cache)
        print(f"GDN_CHAIN transition=post_to_cache pos={pos} cmp={'PASS' if ok else 'FAIL'} first_diff={first}")
        if not ok or digest(post) != digest(cache):
            print("GDN_CHAIN FAIL inconsistent post/cache evidence", pos); return 5
        states[pos], pres[pos] = post, pre

    for left, right in ((8, 9), (9, 10), (10, 11)):
        ok, first = equal(states[left], pres[right])
        print(f"GDN_CHAIN transition=pos{left}_to_pos{right} cmp={'PASS' if ok else 'FAIL'} first_diff={first}")
        if not ok or digest(states[left]) != digest(pres[right]):
            print("GDN_CHAIN FAIL state handoff", left, right); return 6

    if replay:
        for pos, ordinal in list(mapping.items())[1:]:
            suffix = "" if ordinal == 0 else f"-{ordinal}"
            with tempfile.TemporaryDirectory(prefix=f"cortex-gdn-replay-{pos}-") as td:
                base = Path(td) / "fixture"
                for role in ("q", "k", "v", "g", "b", "s_before", "result", "state_after"):
                    src = root.with_name(root.name + f"-{role}{suffix}.bin")
                    shutil.copyfile(src, Path(str(base) + f"-{role}.bin"))
                proc = subprocess.run([str(replay), str(base)], capture_output=True, text=True, timeout=30)
                print(f"GDN_CHAIN replay pos={pos} ordinal={ordinal} exit={proc.returncode}")
                for line in proc.stdout.splitlines():
                    if "compare=" in line or "PASS" in line:
                        print(line)
                if proc.returncode != 0:
                    print(proc.stderr); return 7
    print("GDN_CHAIN PASS transitions=3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
