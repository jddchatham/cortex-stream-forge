"""Fail-closed tiny user-scope cgroup memory verification."""
from __future__ import annotations
import json, shutil, subprocess, time, uuid
from dataclasses import dataclass
from typing import Any

PROBE_CODE = r'''import json, os
def r(p):
    with open(p, encoding="ascii") as f: return f.read().strip()
cg = next((x[3:] for x in r("/proc/self/cgroup").splitlines() if x.startswith("0::")), None)
if cg is None: raise RuntimeError("unified cgroup unavailable")
root = "/sys/fs/cgroup" + cg
o = {"cgroup": "0::" + cg}
for n in ("memory.max", "memory.swap.max", "memory.current", "memory.peak", "memory.events"): o[n] = r(os.path.join(root, n))
print(json.dumps(o, sort_keys=True), flush=True)'''

EXEC_CODE = r'''import json, os, sys
def r(p):
    with open(p, encoding="ascii") as f: return f.read().strip()
cg = next((x[3:] for x in r("/proc/self/cgroup").splitlines() if x.startswith("0::")), None)
if cg is None: raise SystemExit("unified cgroup unavailable")
root = "/sys/fs/cgroup" + cg
o = {"cgroup": "0::" + cg}
for n in ("memory.max", "memory.swap.max", "memory.current", "memory.peak", "memory.events"): o[n] = r(os.path.join(root, n))
if o.get("memory.max") != "EXPECTED_MEMORY" or o.get("memory.swap.max") != "EXPECTED_SWAP": raise SystemExit("cgroup limits do not match requested workload limits")
events = dict((p[0], p[1]) for p in (x.split() for x in o["memory.events"].splitlines()) if len(p) == 2)
for k in ("oom", "oom_kill", "oom_group_kill"):
    if k not in events or events[k] != "0": raise SystemExit("required zero OOM evidence missing")
print("CORTEX_FORGE_SCOPE=" + json.dumps(o, sort_keys=True), flush=True)
if not sys.argv[1:]: raise SystemExit("missing workload argv")
os.execv(sys.argv[1], sys.argv[1:])'''

@dataclass(frozen=True)
class GateResult:
    status: str
    reason: str | None
    requested_memory_max: int
    requested_swap_max: int
    child: dict[str, Any] | None
    stdout: str
    stderr: str
    returncode: int | None
    duration_seconds: float
    @property
    def verified(self): return self.status == "PASS"

def _validate(c, mem, swap):
    required = ("cgroup", "memory.max", "memory.swap.max", "memory.current", "memory.peak", "memory.events")
    missing = [x for x in required if x not in c]
    if missing: return "missing child fields: " + ",".join(missing)
    if not c["cgroup"].startswith("0::/"): return "not a unified child cgroup"
    if c["memory.max"] != str(mem): return f"memory.max={c['memory.max']!r}, expected {mem}"
    if c["memory.swap.max"] != str(swap): return f"memory.swap.max={c['memory.swap.max']!r}, expected {swap}"
    for key in ("memory.current", "memory.peak"):
        try: int(c[key])
        except (TypeError, ValueError): return f"non-numeric {key}"
    events = {}
    for line in str(c["memory.events"]).splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[0] in events: continue
        try: events[fields[0]] = int(fields[1])
        except ValueError: continue
    for key in ("oom", "oom_kill", "oom_group_kill"):
        if key not in events: return f"missing memory.events field {key}"
        if events[key] != 0: return f"probe reported {key}={events[key]}"
    return None

def verify_hard_memory_gate(memory_max=64*1024*1024, memory_swap_max=0, timeout_seconds=20.0, runner=subprocess.run):
    start = time.monotonic()
    if memory_max <= 0 or memory_swap_max < 0:
        return GateResult("FAIL", "invalid requested limits", memory_max, memory_swap_max, None, "", "", None, 0)
    systemd, python = shutil.which("systemd-run"), shutil.which("python3")
    if not systemd or not python:
        return GateResult("UNAVAILABLE", "missing " + ("systemd-run" if not systemd else "python3"), memory_max, memory_swap_max, None, "", "", None, 0)
    argv = [systemd, "--user", "--scope", "--quiet", "--unit=cortex-forge-probe-" + uuid.uuid4().hex[:12], f"--property=MemoryMax={memory_max}", f"--property=MemorySwapMax={memory_swap_max}", python, "-c", PROBE_CODE]
    try: p = runner(argv, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as e:
        return GateResult("FAIL", "scope probe timed out", memory_max, memory_swap_max, None, str(e.stdout or ""), str(e.stderr or ""), None, time.monotonic()-start)
    except OSError as e:
        return GateResult("UNAVAILABLE", f"cannot launch scope: {e}", memory_max, memory_swap_max, None, "", str(e), None, time.monotonic()-start)
    out, err = p.stdout or "", p.stderr or ""
    if p.returncode != 0: return GateResult("UNAVAILABLE", f"systemd scope exited {p.returncode}", memory_max, memory_swap_max, None, out, err, p.returncode, time.monotonic()-start)
    try:
        lines = [x for x in out.splitlines() if x.strip()]
        child = json.loads(lines[-1]) if lines else None
        reason = _validate(child, memory_max, memory_swap_max) if isinstance(child, dict) else "invalid child report"
    except (ValueError, json.JSONDecodeError) as e:
        child, reason = None, f"invalid child report: {e}"
    return GateResult("PASS" if reason is None else "FAIL", reason, memory_max, memory_swap_max, child, out, err, p.returncode, time.monotonic()-start)

def require_hard_memory_gate(**kwargs):
    result = verify_hard_memory_gate(**kwargs)
    if not result.verified: raise RuntimeError("hard memory gate unavailable: " + (result.reason or result.status))
    return result

def verified_scope_argv(command, memory_max=64*1024*1024, memory_swap_max=0):
    """Build argv that verifies the eventual workload child before execv."""
    if not command or not all(isinstance(x, str) and x for x in command):
        raise ValueError("workload argv must be a non-empty string sequence")
    systemd, python = shutil.which("systemd-run"), shutil.which("python3")
    if not systemd or not python: raise RuntimeError("systemd-run and python3 are required")
    code = EXEC_CODE.replace("EXPECTED_MEMORY", str(memory_max)).replace("EXPECTED_SWAP", str(memory_swap_max))
    return [systemd, "--user", "--scope", "--quiet", "--unit=cortex-forge-workload-" + uuid.uuid4().hex[:12], f"--property=MemoryMax={memory_max}", f"--property=MemorySwapMax={memory_swap_max}", python, "-c", code, *command]

def result_json(result):
    return json.dumps({"status": result.status, "reason": result.reason, "requested_memory_max": result.requested_memory_max, "requested_swap_max": result.requested_swap_max, "child": result.child, "returncode": result.returncode, "duration_seconds": result.duration_seconds}, indent=2, sort_keys=True)
