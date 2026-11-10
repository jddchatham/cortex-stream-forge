# QIES095 recovery replay experiment

Run `./pandacode-run.sh` from the worktree root. Standard-library Python only; no model, native inference engine, GPU dependency or package installation. The wrapper uses isolated Python mode, disables bytecode writes, runs the unmodified preserved QIES094 suite and the new replay suite, and creates a new results directory each time.

`replay.py` imports the byte-verified QIES094 storage implementation from `../qies095-preserved/qies094-prototype/`. It adds a synthetic engine contract, explicit select/prepare, optional output attempt, pending evaluation and historical replay. `fixture.json` contains a fixed synthetic protocol reference vector, not model golden data.

`Session.replay(start,end)` validates the store and reconstructs committed synthetic state without selecting, publishing or emitting. Its authoritative position is the latest verified checkpoint; the reconstructed position may be an explicitly requested earlier endpoint. Pending decisions are returned separately, never counted as evaluated. `resume_pending()` publishes once if needed; calling it again after a completed commit is a no-op.

Recovery never calls a sink. `attempt_output(sink)` is an explicit live operation. A repeated attempt by the same coordinator is refused, but a restarted process cannot infer delivery from the journal. No exactly-once terminal guarantee or durable delivery ledger exists. Callers must choose how to handle ambiguous output before explicit replay/resume policy; the prototype does not implement that policy UI.

All storage limits and torn-tail/corruption refusal remain inherited from QIES094. Historical replay does not implement sparse checkpointing or compaction. Engine integration described in the report is a proposed contract, not production wiring.
