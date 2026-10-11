import json, subprocess, unittest
from unittest.mock import patch
from forge.memory_gate import _validate, verify_hard_memory_gate, verified_scope_argv
GOOD = {"cgroup":"0::/user.slice/probe.scope", "memory.max":"67108864", "memory.swap.max":"0", "memory.current":"5000000", "memory.peak":"5000000", "memory.events":"low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0"}
class MemoryGateTests(unittest.TestCase):
    def test_valid(self): self.assertIsNone(_validate(GOOD, 67108864, 0))
    def test_unbounded_rejected(self): self.assertIn("memory.max", _validate(dict(GOOD, **{"memory.max":"max"}), 67108864, 0))
    def test_oom_rejected(self): self.assertIn("oom", _validate(dict(GOOD, **{"memory.events":"oom 1\noom_kill 1"}), 67108864, 0))
    def test_missing_oom_field_rejected(self): self.assertIn("missing memory.events field", _validate(dict(GOOD, **{"memory.events":"oom 0\noom_kill 0"}), 67108864, 0))
    def test_non_numeric_oom_field_rejected(self): self.assertIn("missing memory.events field", _validate(dict(GOOD, **{"memory.events":"oom nope\noom_kill 0\noom_group_kill 0"}), 67108864, 0))
    def test_missing_rejected(self): self.assertIn("missing", _validate({}, 67108864, 0))
    def test_success_checked(self):
        p = subprocess.CompletedProcess([], 0, json.dumps(GOOD), "")
        with patch("forge.memory_gate.shutil.which", return_value="/bin/tool"): self.assertTrue(verify_hard_memory_gate(runner=lambda *a, **k:p).verified)
    def test_failure_not_pass(self):
        p = subprocess.CompletedProcess([], 1, "", "permission denied")
        with patch("forge.memory_gate.shutil.which", return_value="/bin/tool"): self.assertFalse(verify_hard_memory_gate(runner=lambda *a, **k:p).verified)
    def test_workload_scope_wraps_actual_argv(self):
        with patch("forge.memory_gate.shutil.which", return_value="/bin/tool"):
            argv = verified_scope_argv(["/bin/true"])
        self.assertIn("/bin/true", argv)
        self.assertIn("CORTEX_FORGE_SCOPE=", "".join(argv))
if __name__ == "__main__": unittest.main()
