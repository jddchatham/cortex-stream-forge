#!/usr/bin/env python3
"""Harmless launcher regression tests; never loads a GGUF model."""
import json, os, subprocess, sys, tempfile
from pathlib import Path

launcher = Path(__file__).with_name("cortex-supervised-child.py")
env = os.environ.copy()
required = {
    "CORTEX_FLASHNEXT_EXTERNAL_EMBEDDING": "1",
    "CORTEX_FLASHNEXT_LAYER_START": "0",
    "CORTEX_FLASHNEXT_LAYER_END": "1",
    "CORTEX_FLASHNEXT_LAYER_RANGE": "1",
    "CORTEX_FLASHNEXT_SSM_CONV_OPERAND_DIAGNOSTICS": "1",
    "CORTEX_FLASHNEXT_ALLOCATOR_DIAGNOSTICS": "1",
    "CORTEX_FLASHNEXT_REBIND_DIAGNOSTICS": "1",
}
env.update(required)

def run(code, custom_env=env):
    with tempfile.TemporaryDirectory(prefix="cortex-launch-test-") as d:
        r = subprocess.run([sys.executable, str(launcher), "--evidence", d, "--",
                            sys.executable, "-c", code], env=custom_env)
        return r.returncode, json.loads((Path(d) / "launcher.json").read_text())

assert run("raise SystemExit(0)")[0] == 0
assert run("raise SystemExit(5)")[0] == 5
assert run("raise SystemExit(134)")[0] == 134
bad = env.copy(); bad.pop("CORTEX_FLASHNEXT_REBIND_DIAGNOSTICS")
assert run("raise SystemExit(0)", bad)[0] == 78
rc, _ = run("import os; assert os.environ['CORTEX_FLASHNEXT_REBIND_DIAGNOSTICS']=='1'")
assert rc == 0
print("SUPERVISED_CHILD_LAUNCHER PASS")
print("exit propagation: 0, 5, 134")
print("missing diagnostic variable: rejected")
print("effective child environment: verified")
