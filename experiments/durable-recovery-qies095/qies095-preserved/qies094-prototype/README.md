# QIES094 model-free recovery prototype

Run `../pandacode-run.sh` from this directory, or `./pandacode-run.sh` from the worktree root. Python 3 standard library only. No model, native engine, GPU library, package installation or inference launcher is used.

The wrapper makes a fresh results directory for each run. It never overwrites prior runs. `results.json` records deterministic fault receipts and measured overhead; `tests.log` records success or failure. SIGKILL tests kill only their own forked synthetic-fixture child, wait for it, and retain the fixture directory. No global cache dropping, reboot, services, memory settings or resource-guard actions occur.

`recovery.py` implements framed decision journaling, immutable single-file synthetic checkpoints, publication receipts, strict recovery, a separately durable file/directory simulation, and a single-writer POSIX backend. `test_recovery.py` exercises the model and real backend.

This is not a QUORTEX state format or resume loader. A durable output decision is not proof of external delivery or proof its token was evaluated. Ordinary stdout is never replayed automatically. Torn tails are reported and block further writes; no repair/truncation/segment rotation is implemented. Corrupt published references fail closed. All committed checkpoints are retained, with a 64-token quota and directory/file/journal caps; there is no garbage collector.

`QIES093-design-input.md` is an exact design snapshot. The report at the worktree root documents deliberate reduced scope and remaining integration blockers.
