#!/usr/bin/env python3
"""Run one bounded child and preserve its real status and cgroup evidence."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REQUIRED = {
    "CORTEX_FLASHNEXT_EXTERNAL_EMBEDDING": "1",
    "CORTEX_FLASHNEXT_LAYER_START": "0",
    "CORTEX_FLASHNEXT_LAYER_END": "1",
    "CORTEX_FLASHNEXT_LAYER_RANGE": "1",
    "CORTEX_FLASHNEXT_SSM_CONV_OPERAND_DIAGNOSTICS": "1",
    "CORTEX_FLASHNEXT_ALLOCATOR_DIAGNOSTICS": "1",
    "CORTEX_FLASHNEXT_REBIND_DIAGNOSTICS": "1",
}

def cgroup_path():
    rel = None
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        f = line.split(":", 2)
        if len(f) == 3 and f[0] == "0":
            rel = f[2]
            break
    mount = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        if " - cgroup2 " in line:
            mount = Path(line.split(" - ", 1)[0].split()[4])
            break
    if not rel or not mount:
        raise RuntimeError("cgroup_v2_unavailable")
    path = mount / rel.lstrip("/")
    if not path.is_dir():
        raise RuntimeError("cgroup_path_missing:" + str(path))
    return path

def telemetry(path):
    def read(name):
        value = (path / name).read_text().strip()
        if not value:
            raise RuntimeError("empty_cgroup_file:" + name)
        return value
    events = {}
    for line in read("memory.events").splitlines():
        key, value = line.split()
        events[key] = int(value)
    return {"path": str(path), "memory.max": read("memory.max"),
            "memory.swap.max": read("memory.swap.max"),
            "memory.current": read("memory.current"),
            "memory.peak": read("memory.peak"), "memory.events": events}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if not args.command or args.command[0] == "--":
        args.command = args.command[1:]
    missing = [k for k, v in REQUIRED.items() if os.environ.get(k) != v]
    evidence = Path(args.evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    if missing:
        (evidence / "launcher.json").write_text(json.dumps({"status": "PREFLIGHT_FAIL", "missing": missing}))
        return 78
    cg = cgroup_path()
    before = telemetry(cg)
    if before["memory.max"] != "2147483648" or before["memory.swap.max"] != "0":
        (evidence / "launcher.json").write_text(json.dumps({"status": "PREFLIGHT_FAIL", "telemetry": before}))
        return 78
    (evidence / "environment.json").write_text(json.dumps({k: os.environ.get(k) for k in REQUIRED}, sort_keys=True))
    with (evidence / "stdout.log").open("wb") as out, (evidence / "stderr.log").open("wb") as err:
        try:
            proc = subprocess.run(args.command, stdin=sys.stdin.buffer, stdout=out, stderr=err,
                                  timeout=195, check=False)
            status = proc.returncode
        except subprocess.TimeoutExpired:
            status = 124
    after = telemetry(cg)
    (evidence / "telemetry.json").write_text(json.dumps({"before": before, "after": after,
        "pid": os.getpid(), "child_status": status}, sort_keys=True))
    (evidence / "launcher.json").write_text(json.dumps({"status": "PASS" if status == 0 else "FAIL",
        "child_status": status, "pid": os.getpid(), "cgroup": after["path"]}, sort_keys=True))
    return status

if __name__ == "__main__":
    raise SystemExit(main())
