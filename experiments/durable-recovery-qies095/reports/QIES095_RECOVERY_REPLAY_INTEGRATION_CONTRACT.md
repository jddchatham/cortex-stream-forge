# QIES095 - Recovery Replay and Integration Contract

Date: 2026-11-09. Model-free experimental implementation and proposed native integration contract.

**Recommendation: GO for further model-free adapter qualification; NO-GO for production inference integration or real-state recovery claims.** All 23 test groups pass: the original 309 QIES094 fault outcomes/injections remain passing, plus 248 new multi-token/replay fault outcomes. Historical replay reconstructs evaluated state without selecting, publishing or emitting. A pending selected token remains separate until explicitly evaluated and durably published.

## 1. Isolation and preservation

Worktree: `/home/user/PandaCode Bots/meatball-84ccea00/qies095-worktree`.
Branch: `development/continuous-generation`.
HEAD: `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`.

A second isolated checkout of the existing branch was registered using `git worktree add --force`, because that branch was already checked out elsewhere. This does not force-push or rewrite history. All linked checkouts share the branch reference; commits/branch movement remain prohibited. This worktree's tracked engine files are the unchanged development baseline; uncommitted QIES091 production files from qies090-worktree were not copied over them.

QIES094 was preserved before implementation:

- `qies095-preserved/revision.txt`: exact source HEAD. The prototype is uncommitted, so HEAD alone is not its complete source identity.
- `status.txt`: original complete working-tree/untracked inventory.
- `refs.txt`: original refs for archive/tag/branch verification.
- `source-sha256.json`: individual hashes of all **701** original QIES094 source/report/evidence files.
- `qies094-snapshot.tar.gz`: byte-verified complete snapshot of those files, including earlier runs.
- `qies094-prototype/` and `QIES094_DURABLE_RECOVERY_PROTOTYPE.md`: accessible exact copies for regression and review.

All original QIES094 SHA256SUMS entries were verified before preservation; all archived files were rehashed against the new manifest. The new implementation imports the preserved storage module directly without editing it. The original worktree and prior milestone artifacts were not overwritten.

## 2. Implementation summary

| New file | Role |
| --- | --- |
| `qies095-prototype/replay.py` | Synthetic engine adapter, checked coordinator, streaming historical decision iterator |
| `qies095-prototype/test_replay.py` | Multi-token replay, boundary faults, drift/corruption checks, output ambiguity, resource measurements |
| `qies095-prototype/fixture.json` | Fixed synthetic token/state reference vector; not model golden data |
| `qies095-prototype/README.md` | Reproduction and scope |
| `pandacode-run.sh` | Fresh foreground run of preserved QIES094 and new QIES095 suites |
| `qies095-checks/complete-wil19L/` | Final combined passing run and full per-case receipts |
| `qies095-checks/verification.json`, `SHA256SUMS` | Protected identities, preservation and final artifact integrity |

`SyntheticEngine.restore/evaluate/verify/select` operate on an immutable small State(position, history, blob). Selection varies deterministically with history. `Session.select_and_prepare` journals the selection without advancing evaluated state. `resume_pending` evaluates the existing pending ID, verifies position/history/state, then calls the inherited immutable publisher. Repeated resume after successful publication is a no-op and does not call evaluate again.

`Session.replay(start,end)` validates the store, restores a verified retained checkpoint and streams recorded token decisions to the requested committed endpoint. It compares every reconstructed intermediate state/history/position with its verified checkpoint. It never reselects, writes a checkpoint or calls an output sink. All replay remains synthetic; no production decoder or state reader is imported.

The fixed reference contains evaluated IDs `[19,94,189,142,49,132,120,155,32,77,134,156]`, position 12, pending ID 50, and an independently calculated synthetic serialized-state hash. These are arbitrary synthetic protocol values, not Qwen outputs or a replacement for independent inference goldens.

## 3. Recovery invariants and semantics

1. **Selected is not evaluated.** Journal prepare leaves committed position p unchanged. At most one decision may be pending beyond the latest checkpoint; it references that checkpoint and input position p.
2. **Published state is authoritative.** Evaluated position advances only through the inherited file-sync, immutable-checkpoint publication, receipt-sync and directory-sync protocol. Failed/uncertain operations require fresh verified recovery.
3. **Replay is reconstruction, not a second commit.** It can evaluate historical tokens into private synthetic state, but committed files/position remain byte-identical. A caller requesting an earlier endpoint receives both reconstructed position and latest authoritative position.
4. **Recorded IDs are replay inputs.** Replay never calls selection; a test engine that throws on any selection still replays and resumes pending decisions successfully.
5. **Positions are checked at every step.** Before evaluation, state position equals input position and history length. After evaluation, position is p+1, history is H+[t], and the serialized state must match the expected synthetic transition/checkpoint. Injected position drift is rejected before publication.
6. **Pending advancement occurs once per committed transition.** After publication, pending disappears. If publication survives but live acknowledgment is lost, recovered receipt—not a stale in-memory flag—determines whether to evaluate again.
7. **Physical evaluation can repeat after a crash.** Reconstructing from an earlier checkpoint or retrying an uncommitted candidate may perform the same computation again. The guarantee is no double application to committed state, not exactly one physical invocation.
8. **Output is separate.** Recovery reports UNKNOWN delivery and NO_AUTOMATIC_REPLAY. A partial sink failure invalidates the coordinator; replay/resume never silently re-emits bytes.
9. **Errors stop the coordinator.** Invalid sessions cannot be reused. A fresh coordinator must restore through the verified store. Published corruption is rejected; no silent rewind to an older valid checkpoint.
10. **Limits remain finite.** The inherited 64-decision quota, bounded file/directory/journal sizes and one pending token remain intact. No unlimited context or continuous service is implied.

The real-filesystem test closes the storage handle and lock, reopens them, restores from a retained checkpoint, replays a committed suffix and consumes a pending token. This is a fresh storage/coordinator restart within the test process. New QIES095 process-kill tests were not added; the preserved QIES094 SIGKILL suite still runs unchanged.

### Output behavior

`attempt_output(sink)` is an explicit live operation after durable prepare. It returns a local-write result with `durable_output_ack=false` and external delivery UNKNOWN. A repeated attempt by the same coordinator is refused before calling the sink again. After process/coordinator restart that in-memory attempt marker is gone; it cannot certify prior delivery.

The prototype does not implement a durable delivery ledger or require a caller's replay/skip choice before state-only resume. The integration contract below requires that policy explicitly. Ordinary terminal/stdout consumers do not gain exactly-once delivery from this prototype. Choosing to resume state without replaying text can omit output; explicit replay can duplicate it.

## 4. Test matrix and results

Final combined runner: `qies095-checks/complete-wil19L/`.

| Suite/coverage | Result |
| --- | --- |
| Preserved QIES094 suite | 13/13 groups, 5.959 s |
| Original transaction survival/loss cases | 84 passing |
| Original mixed file/directory persistence cases | 84 passing |
| Original genesis interruption cases | 76 passing |
| Original recovery scan interruptions | 8 passing |
| Original I/O error injections | 28 passing |
| Original real SIGKILL tests | 29 passing, all children reaped |
| New QIES095 suite | 10/10 groups, 17.296 s |
| Coordinator boundaries: selection, journal sync, output, evaluation, publication, replay/each replay step | 23 boundaries x 2 survival outcomes = **46** passing |
| Storage/scan boundaries after four committed tokens | 101 boundaries x 2 survival outcomes = **202** passing |
| All retained restart checkpoints 0..12, plus pending token | 13 exact reconstructions; pending consumed once |
| Partial emission and repeated output attempt | Failure/ambiguity exposed; no implicit retry/replay |
| Engine position drift and corrupt middle checkpoint/journal | Rejected; no silent publication |
| Replay with selection disabled | Pass; recorded IDs only |
| Real close/reopen and pending continuation | Pass |
| Quota and stream bounds | 64 tokens accepted; next selection refused without mutation |
| Repeated bounded replay | 100 passes, no retained hook event log |

Total fault outcomes/injections across preserved and new matrices: **557 = 309+46+202**, separate from normal, negative and measurement assertions. Detailed receipts identify numeric fault index, boundary label, survival outcome, committed position, pending/torn status and visible byte count where applicable.

Fault coverage includes before/after token selection; actual underlying append/write/fsync boundaries; before/after evaluation; checkpoint/receipt publication; before/after output; before replay, after restore, and before/after each historical token. New faults begin after multiple committed tokens rather than only from genesis. Interruption leaves a verified old or new committed position; selected tokens are never counted merely because text may have been emitted.

The unchanged storage tests simulate loss/survival and selected independently surviving namespace/data outcomes. Neither that model nor SIGKILL proves power-loss behavior of a physical drive or every possible sector tear.

## 5. Engine integration API: proposed contract, not implemented production code

The native adapter must preserve the existing per-layer context lifecycle, full sequence serialization, compute-logits scheduling, greedy first-maximum rule, model arithmetic and state ownership. No persistent llama_context, tensor-operation or QIES086 change is authorized.

### Interfaces and ownership

```text
Engine.describe() -> EngineIdentity
Engine.restore(ReadOnlyCheckpoint, ExactMeta) -> PrivateStateOwner
Engine.select(ValidatedLogitsView) -> TokenId
Engine.evaluate_checked(PrivateStateOwner, StepInput) -> CandidateStateSet
Recovery.prepare_decision(SourceCheckpointId, p, token, ExactPiece) -> DurableDecision
Recovery.publish(CandidateStateSet, ExactMeta, JournalPrefix) -> CommitReceipt
Recovery.replay(SourceCheckpointId, target_p) -> ReconstructedPrivateState
Recovery.resume_pending(VerifiedSnapshot, OutputResolution) -> VerifiedSnapshot
```

These are logical API requirements, not declarations added to the engine.

**EngineIdentity** pins executable/library/source/native-state ABI, model shard hashes, index identity, tokenizer/template, context/sequence/thread configuration and relevant environment. Incompatibility is rejected before native restore. The current prototype pins only its synthetic identity; native identity qualification remains missing.

**ExactMeta** includes evaluated position p, complete token history H and hash, prompt/prefill cursor, generated sequence, optional pending decision ID, UTF-8 suffix, stop reason and required logits identity. It must cover PLE previous-token dependencies as well as recurrent/KV/indexer continuity. Any persistent component not covered by the existing native serialized state or proven deterministic reconstruction blocks integration.

**ReadOnlyCheckpoint / PrivateStateOwner:** the recovery layer owns immutable files and receipts. The engine may borrow verified bytes only for restore; borrowed spans cannot outlive their owner. Evaluation restores into private working state and never overwrites a published checkpoint. Checkpoints occur at whole-forward boundaries, not mid-layer HC handoffs. Existing per-layer llama_context creation/free and graph references must remain valid.

**StepInput** contains token ID, expected input p, history identity, sequence ID, source checkpoint/working-state identity and the original compute-logits request. Evaluate receives the recorded ID during replay; it must not sample. The candidate must describe p+1 and H+[t]. EOG is stopped before evaluation; reaching a token cap does not force evaluation of the final pending token. Cancellation stays cooperative at the existing boundaries.

**CandidateStateSet:** all 48 exact native full-state blobs plus required continuation metadata/logits. Existing serializer buffers can be consumed sequentially by the durable writer before release; do not accumulate a second full-state vector or apply QIES086. A partial layer failure invalidates only the private candidate, never the parent checkpoint. Only the recovery layer can acknowledge durable publication.

**Selection:** live engine/coordinator supplies a validated token ID from current finite, exact-size logits using the existing greedy rule. Native token-to-piece conversion and UTF-8 rules remain unchanged. Persist the exact decision against its source state before authorized streaming output. Historical replay uses that stored ID and bytes; it does not select or re-tokenize.

### Resume and replay handshake

1. Hold the single-writer lock and validate identities, receipt chain, checkpoint files and required journal prefix.
2. Restore a verified source into a private owner. Replay consecutive recorded inputs at exact positions. Verify each step's metadata and available checkpoint hashes; native logits/serialization determinism must be independently qualified, not assumed from the synthetic test.
3. Historical replay publishes nothing and emits nothing. It may reconstruct a current private owner or be discarded after verification. Never feed the historical sequence into an already advanced live owner.
4. Re-read/verify the latest authority before a live resume. If t is already covered by its commit receipt, do not evaluate it again. If t is pending at p, evaluate once from verified S(p), then publish p+1. If no pending token exists, restore current logits and return to original selection scheduling.
5. If evaluate/publication fails or its acknowledgment is uncertain, invalidate the owner and reopen through recovery arbitration. Do not clear flags and retry using mutated state.
6. OutputResolution must state HOLD, SKIP_UNKNOWN, EXPLICIT_REPLAY or an independently validated acknowledged-sink policy. HOLD performs no further live continuation. SKIP_UNKNOWN may omit displayed text; EXPLICIT_REPLAY may duplicate it. Return run/decision sequence/byte ranges and delivery uncertainty to callers. Never describe local stdout success as a durable consumer acknowledgment.

The prototype's publisher independently formats its small deterministic synthetic blob. It does not serialize arbitrary adapter-supplied native bytes. A native adapter must publish the supplied serialized result **without performing a second decode**. That interface still needs implementation and tests in separate model-free adapter work.

## 6. Measured resource bounds

Final combined run, six separate real-filesystem stores, three timed replays per size. Replay includes complete inherited recovery validation and private reconstruction. Python tracing is enabled; these are synthetic costs on this machine, not native inference estimates.

| Evaluated tokens replayed | Median replay ms | Journal bytes | All checkpoint bytes | Receipt bytes | Total logical bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.172 | 295 | 2,210 | 402 | 2,974 |
| 4 | 9.788 | 1,182 | 5,646 | 1,098 | 7,993 |
| 8 | 17.333 | 2,365 | 10,282 | 2,026 | 14,740 |
| 16 | 32.605 | 4,743 | 19,734 | 3,889 | 28,433 |
| 32 | 72.128 | 9,504 | 39,341 | 7,617 | 56,529 |
| 64 | 164.827 | 19,020 | 81,209 | 15,073 | 115,369 |

At 64 tokens the three replay times were 153.196, 164.827 and 177.268 ms. Journal growth is **295..298 bytes/token**, averaging 297.1875 bytes at 64, for the five-byte ASCII fixture pieces and current identities. Real tokenizer pieces/metadata change this cost.

The table retains every checkpoint including genesis, so checkpoint history metadata grows with position. It is not a fixed-size-per-token storage prediction. At 64 tokens allocated file data was 557,056 bytes, excluding filesystem metadata and device write amplification. No full model/state files were copied.

Peak incremental traced allocation across replay measurements: **699,492 bytes (0.667 MiB)**, below the 4 MiB prototype target. The per-case RSS/anonymous/PSS samples showed no within-case increase at this measurement resolution after prior tests had warmed the allocator. Across the six cases VmHWM went from 24,850,432 to 24,870,912 bytes. These are warmed-process observations, not proof of cold-start memory or large-native-state behavior.

The separate 100-replay bounded test, with 64 decisions retained on disk and replaying one final token, recorded a traced peak of 110,550 bytes, current retained allocation 21,145 bytes, and zero retained hook events. Measurement instrumentation and Python overhead are included in traced peaks; total process memory exceeds these incremental values.

`decisions()` streams one bounded frame at a time on the real filesystem and does not retain the replayed journal. The unchanged QIES094 validation pass still reads its bounded journal and retains at most 64 decision descriptors; this has **not** become an unlimited-memory service. The 532,480-byte journal scan cap, 32,768-byte checkpoint-file cap and 512-entry directory cap remain. Simulated backends use a bounded in-memory file image. No unbounded in-memory transcript or per-recovery event history is retained.

### Checkpoint frequency tradeoffs

Implemented policy is one checkpoint per evaluated synthetic token, all retained; replay deliberately starts from older retained checkpoints to validate reconstruction. **Sparse checkpoints are not implemented.** Deleting intermediate receipts/files or appending multiple pending decisions would violate the current chain and be rejected.

For a future coarser interval K, fewer full-state publications can reduce storage I/O, but up to K-1 evaluated steps plus a pending token may need replay after a crash. It needs a new explicit evaluated-but-not-checkpointed journal/state protocol; intermediate state cannot be called durably checkpoint-committed. Longer replay repeats model work, and output ambiguity remains. This is a design tradeoff, not a feature validated by the current timing table.

Using the historical ~5.31 GiB full state estimate, native per-step immutable replacement can entail GiB-scale writeback per step, compared with KiB-scale synthetic checkpoints here. Current measurements do not predict its latency, SSD wear, file-cache pressure, cgroup memory or resource fit. No physical-memory savings or real-model throughput result is claimed.

## 7. Reproduction and integrity

```sh
cd '/home/user/PandaCode Bots/meatball-84ccea00/qies095-worktree'
./pandacode-run.sh
```

The foreground wrapper creates a new `qies095-checks/complete-XXXXXX`, runs the exact preserved 13-group suite, then the 10-group replay suite. Existing runs remain untouched; nonzero test status stops the wrapper. All killed processes belong to the preserved harmless-fixture tests and are waited for. Standard-library Python only; no packages, native engine, model load or inference guard invocation.

```sh
cd '/home/user/PandaCode Bots/meatball-84ccea00/qies095-worktree/qies095-checks'
sha256sum -c SHA256SUMS
```

The final manifest covers this report, prototype/reference sources, preserved snapshot/hash inventory and final combined evidence. `verification.json` records original-source preservation, exact base/branch, refs, protected engine/guard identities and the successful run. Raw earlier development runs also remain available.

## 8. Remaining production risks and decision

- One synthetic blob is not 48 native hybrid-state components. Native state/ABI/model identity, logits, recurrent/KV/indexer/PLE/HC continuity and real bit-exactness remain unqualified.
- Durable delivery receipts, UTF-8 checkpoint suffix integration, EOS/cancellation/token-cap stop receipts and a caller-facing replay/skip policy are not added here.
- Torn journals remain read-only; repair/segment rotation, checkpoint retention/GC and explicit rewind from a corrupt newest published checkpoint remain absent.
- New failure matrices are deterministic simulations; the inherited SIGKILL suite is not a drive power-loss test. Arbitrary sector-level failures and every filesystem/device persistence behavior are not modeled.
- Full-chain validation and retained checkpoints are bounded but increasingly costly. Raising quotas without redesigning bounded validation/storage is not authorized.
- The native adapter needs candidate-state isolation and exact supplied-byte publication. It cannot call the original mutable state store on published files and claim this protocol's guarantees.
- Independent runtime oracle/golden/guard-adapter prerequisites remain unresolved. No memory-gate admission was attempted; no inference was run. Any future native run requires fresh >=6 GiB MemAvailable admission and all unchanged swap/disk/cgroup/PSI/runtime guard checks.

**GO for further non-inference qualification of the proposed adapter contract, including 48 tiny independent components and complete stop/output/UTF-8 semantics. NO-GO for production hookup or native recovery acceptance.** The current result proves bounded deterministic synthetic replay, selected-versus-evaluated separation and no silent committed-position drift under the tested boundaries.

No production engine code, decode scheduling, serialized hybrid-state ownership, resource guard or system configuration changed. No QIES086 application, persistent-context redesign, inference, commit, push or merge occurred.
