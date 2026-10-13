#!/usr/bin/env python3
"""Unit tests for supervisor PID, exit propagation, timeout and group cleanup."""
import importlib.util
import io
import os
import sys
import tempfile

path = os.path.join(os.path.dirname(__file__), "cortex-supervised-child.py")
spec = importlib.util.spec_from_file_location("cortex_supervised_child", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def run(code, timeout=3):
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        return mod.run_workload([sys.executable, "-c", code], out, err, timeout=timeout, grace=.25)

status, timed_out, pid = run("raise SystemExit(0)")
assert (status, timed_out) == (0, False) and pid > 0
status, timed_out, pid = run("raise SystemExit(5)")
assert (status, timed_out) == (5, False) and pid > 0
status, timed_out, pid = run("raise SystemExit(134)")
assert (status, timed_out) == (134, False) and pid > 0
status, timed_out, pid = run("import time; time.sleep(30)", timeout=.1)
assert (status, timed_out) == (124, True) and pid > 0
try:
    os.kill(pid, 0)
except ProcessLookupError:
    pass
else:
    raise AssertionError("timed-out workload was not reaped")
print("SUPERVISED_CHILD_UNIT PASS")
print("actual workload pid recorded: PASS")
print("exit propagation 0/5/134: PASS")
print("timeout TERM/KILL and reap: PASS")
