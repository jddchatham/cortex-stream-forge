# QIES094 - Durable Generation Recovery Prototype

Date: 2026-11-09. Model-free, CPU-only experimental validation infrastructure.

**Decision: CONDITIONAL GO for QIES095 as further non-inference protocol qualification; NO-GO for production engine integration or real-model recovery claims.** The prototype passes 13 test groups covering 309 fault outcomes/injections plus normal continuation, corruption, capacity, memory and filesystem tests. Output decisions and evaluated positions remain distinct; no exactly-once terminal-delivery claim is made.

## 1. Isolation and design review

Experimental worktree:

`/home/user/PandaCode Bots/meatball-84ccea00/qies094-worktree`

Branch: `development/continuous-generation`.
HEAD: `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`.

The branch was already checked out in `qies090-worktree`. To satisfy the requested branch name and isolated worktree simultaneously, `git worktree add --force` registered a second checkout of that same existing branch. This was not a force-push, reset, branch overwrite or history change. Both checkouts share the branch reference, so no commits or branch movement are permitted in either during this experiment. The original checkout's uncommitted QIES091/QIES092 implementation was not copied into production files here; this worktree's tracked engine remains the exact original development baseline.

Read QIES093 and retained an exact copy as `qies094-prototype/QIES093-design-input.md`. Source/helper/test hashes in the original checkout, the baseline executable and the resource guard were verified unchanged. `git diff --exit-code HEAD` in this worktree is empty: all prototype files are new, separately named files. Archive branches and milestone tags were not modified; the verification receipt records their refs.

### Design ambiguities resolved before implementation

1. A complete record may survive even when its writer never received the fsync acknowledgment. Live code acknowledges durability only after successful synchronization. Recovery may adopt a complete, checksum-valid, ancestry-valid surviving record; this is an uncertain operation resolved by recovery, not evidence that an unsynced write was guaranteed durable.
2. A torn trailing record cannot be silently truncated and then appended through. The prototype preserves it, reports the valid prefix and refuses further mutation. The design's future linked-segment repair is not implemented.
3. A generation file is not a committed checkpoint merely because rename succeeded. An independently synchronized publication receipt is required. A crash around receipt directory sync can recover the old or fully valid new checkpoint; live code must stop rather than guess.
4. QIES093's 48-layer directory layout is reduced to one immutable synthetic-state file plus one publication receipt in a flat dedicated run directory. This exercises publication ordering, not native component completeness. No claim is made that a single fixture substitutes for 48-layer semantics.
5. Published corruption is rejected rather than silently falling back to an older position. Retained older checkpoints remain evidence; operator-directed rewind is not implemented.

No unresolved contradiction prevented this reduced-scope prototype. Full streaming delivery receipts, native state identity and multi-component checkpoint integration remain explicitly deferred.

## 2. Files and implementation architecture

| File | Purpose |
| --- | --- |
| `qies094-prototype/recovery.py` | Standalone protocol, POSIX backend and deterministic persistence model; Python standard library only |
| `qies094-prototype/test_recovery.py` | Model faults, real filesystem and SIGKILL tests, corruption/capacity checks and measurements |
| `qies094-prototype/README.md` | Reproduction and scope limits |
| `qies094-prototype/QIES093-design-input.md` | Unmodified design snapshot |
| `pandacode-run.sh` | POSIX foreground runner; creates a fresh test-output directory and executes no inference |
| `qies094-checks/verification.json` | Branch/base/source/guard/binary identities and design hash |
| `qies094-checks/run-4lKmjv/` | Final passing logs, detailed fault receipts, retained killed-child fixtures and benchmark state |
| `qies094-checks/SHA256SUMS` | Source/report/evidence integrity manifest |

Key functions/classes:

- `Store.initialize`: exclusive new-store admission, synchronized identity/journal creation and genesis checkpoint.
- `Store.prepare`: verify current recovery state, append one selected-token decision, fsync the journal, then acknowledge the decision. Never advances evaluated position.
- `Store._publish`: private checkpoint write, sync, atomic rename, directory sync, readback verification, receipt write/sync/rename/directory sync, then coordinator acknowledgment.
- `Store.evaluate_pending`: synthetic deterministic transition, publishing exactly one next-position checkpoint from the existing pending decision.
- `Store.recover`: strict identity and ordered-prefix validation, full retained receipt/checkpoint chain validation, authoritative evaluated position and at most one pending decision.
- `RealFS`: single-writer flock, exclusive file creation, no symlink traversal for opened files, retry of interrupted/short writes, required file/directory synchronization.
- `MemoryFS`: independent volatile/durable file bytes and directory entries, with deterministic crash outcomes.
- `Hooks`: enumerated operation boundaries and injected interruption/I/O errors. Routine backends do not retain event traces; only test traces do.

The fixture transition is a deterministic small string tagged attention/recurrent/indexer and derived from the complete synthetic token history. These labels are test data, not actual attention, recurrent or indexer state.

## 3. Journal and checkpoint format

### Journal

One append-only `journal` file, whose creation and directory entry are synchronized at initialization.

Each frame contains:

- Fixed little-endian header `<8sHI`: 8-byte magic `QIES094J`, uint16 version 1, uint32 payload length.
- Canonical JSON payload with schema, session identity, output-decision sequence, evaluated input position, selected token ID, source checkpoint SHA-256, previous-frame SHA-256 and base64 exact piece bytes.
- 32 raw SHA-256 checksum bytes over header plus payload.

SHA-256 is integrity checking, not authentication against a party replacing all artifacts. Canonical decoding rejects duplicate JSON keys and noncanonical encodings. Token/sequence/position bounds and exact session/checkpoint ancestry are checked separately.

Configured prototype limits: 64 decisions, token IDs 0..255, piece <=1,024 bytes, header frame bound 2 MiB, journal scan cap 532,480 bytes, ordinary file cap 32,768 bytes, directory cap 512 entries. The actual fixture decision format is much smaller than the frame bound. These are prototype quotas, not new inference-guard thresholds or model-vocabulary limits.

**Durability point:** `prepare` returns success only after the complete append and a successful fsync on the journal. A successful write or close alone is not acknowledged as durable. The journal name is already durable. Any exception poisons that writer instance; it cannot continue by assuming an operation did not happen. A fresh recovery must resolve surviving state.

A final incomplete header/frame is reported as `torn_tail`; valid checkpoint recovery remains possible only if the checkpoint's required prefix is complete. Both prepare and pending evaluation refuse to proceed with that tail. A complete record with a bad checksum, invalid header/version, inconsistent reference or corrupted committed prefix fails closed. No truncation, repair or replay to stdout occurs.

### Immutable checkpoint and publication receipt

Files use unique attempt names, for example:

```text
identity
writer.lock
journal
checkpoint-000-000
commit-000
checkpoint-001-000.tmp
checkpoint-001-001
commit-001
```

A checkpoint has schema/native synthetic identity, session, generation filename, evaluated position, token history, synthetic state, parent receipt hash and exact journal boundary/hash. A receipt binds its position, parent receipt hash, checkpoint filename and full checkpoint SHA-256. The genesis is position zero with empty synthetic history. Candidate attempts get a new suffix, allowing a recovered pending decision to be evaluated without overwriting an orphan candidate.

Publication order:

1. Exclusive create of checkpoint temporary file; checked complete write and close.
2. Fsync the checkpoint file.
3. Rename to an unused final checkpoint name; fsync the run directory.
4. Read back and verify exact checkpoint bytes.
5. Exclusive create of a uniquely named temporary receipt; checked write/close and fsync.
6. Rename receipt to final `commit-NNN`; fsync the run directory.
7. Only now acknowledge COMMITTED. No in-memory position advance is authorized by steps 1..6 failing.

The flat layout keeps renames within one directory/filesystem. Initialization also syncs the parent of the dedicated run directory. A generation is ignored unless a final valid receipt references it. Old checkpoint/receipt files are never overwritten or removed. There is no garbage collector: the 64-token quota bounds normal growth, and attempt/directory limits bound repeated failed candidates.

The POSIX rename collision check relies on the exclusive writer lock and a private directory with no external concurrent modification. It is not a hostile multi-user transactional store. No directory-wide snapshot, reflink, global cache reset or full real-state copy is used.

## 4. Recovery behavior and invariants

Recovery validates session/schema identity before state use, scans a bounded journal, then checks the complete consecutive commit chain from genesis. Receipt hashes bind parents; checkpoints bind exact history, state and journal prefix. Newest mtime is never authority. A gap, malformed published receipt, corrupt checkpoint or incompatible identity rejects recovery rather than selecting an older checkpoint silently.

The returned result includes:

- Authoritative evaluated position/history/state.
- Last verified checkpoint and receipt identities.
- Pending selected decision, if exactly one exists after the checkpoint.
- Journal valid-prefix end/hash and torn-tail flag.
- `output_delivery = UNKNOWN; no external acknowledgment; no automatic replay`.

Preparing output does not advance position. Evaluating a pending token advances exactly once, only by a new published receipt. A checkpoint already containing that token has no pending token and refuses another evaluation without a new decision. A checksum-valid orphan file cannot advance position. Fresh recovery after interruption can evaluate an intact pending token using a new candidate attempt name; it cannot mix state from different attempts.

A durable decision records intended bytes and token identity. It does not prove stdout delivery, and the prototype deliberately has no terminal-writing or replay function. It does not implement QIES093's DELIVERY_INTENT/LOCAL_WRITE_COMPLETE journal types yet. Therefore it cannot decide whether bytes were delivered, omitted or duplicated externally. QIES095 must retain this distinction.

## 5. Fault injection and results

Final execution: **13 test groups passed in 5.879 seconds**. Detailed receipts are in `run-4lKmjv/results.json`; raw output is `tests.log`.

| Coverage | Result |
| --- | --- |
| 42 enumerated prepare/evaluate/scan boundaries, both unsynced-loss and unsynced-survival outcomes | 84/84 valid old/new recovery outcomes; no position drift |
| Independent directory-name/data survival variants at those same boundaries | 84/84 recovered consistent state; no silent mixed checkpoint |
| Initialization interruptions with two persistence outcomes | 76 outcomes: valid genesis or explicit refusal |
| Interruption during recovery scanning | 8/8; no filesystem mutation; subsequent clean scan succeeds |
| ENOSPC/EIO before/during writes, file fsync and directory fsync | 28/28; writer poisoned, no silent success, recoverable prior/new committed state |
| Real SIGKILL at every mutating/acknowledgment boundary in the transaction trace | 29/29 children terminated as intended, reaped, and artifacts recovered consistently |
| Real short writes and InterruptedError injection | Complete bytes reconstructed correctly; successful recovery |
| Real zero-progress write | Failure; writer invalidated |
| Single-writer lock | Second writer refused |
| Journal/checkpoint/receipt/identity corruption and wrong session/version | Rejected |
| Token-position/history/state mismatch, unsafe checkpoint path, duplicate/noncanonical JSON, corrupt complete trailing record | Rejected |
| Torn journal tail | Reported, preserved byte-for-byte, further writes refused |
| Multiple consecutive tokens and fresh-object pending continuation | Exact history/position and no double evaluation |
| Capacity | 64 tokens accepted; token 65 refused without new objects |
| Repeated recovery | 1,000 scans; no retained observer-event accumulation |

Total enumerated fault outcomes/injections: 309 (84+84+76+8+28+29), separate from normal/corruption/capacity/measurement assertions. Faults are deterministic by numeric operation index and recorded label. The synthetic transition and token values are fixed; no random seed is required.

Boundaries include before append, mid-journal write, after write/close before fsync, after journal fsync, during checkpoint write, after checkpoint sync before rename, after checkpoint publication, before/after directory synchronization, receipt publication and coordinator acknowledgment. An interrupted sync is modeled by both loss and survival possibilities around the operation. SIGKILL tests exercise process death near synchronization, not power failure inside the storage device. The model tests selected persistence extremes and independently surviving names/data; they do not exhaust every possible sector-level reorder/torn-write pattern.

The first test run found two real-backend failures: directory enumeration reused an exhausted descriptor offset. Fixed by opening a fresh directory descriptor per scan; subsequent repeated scans and all real-backend tests pass. That initial failed run and the intermediate passing run remain intact; neither was overwritten. A later audit disabled default hook-history retention and added mixed-persistence/repeated-scan tests before the final passing run.

## 6. Measured overhead

Measured on this worktree's btrfs filesystem, `/dev/mapper/root[/@home]`, using 32 one-byte synthetic decisions/checkpoints in one process. Times include Python and tracing overhead; they are illustrative local measurements, not QUORTEX performance predictions. Each step revalidates all retained history, so scan cost grows with token count.

| Operation | Median | Observed range |
| --- | --- | --- |
| Journal write + close + fsync only | 2.927 ms | 2.106..6.462 ms |
| Checkpoint publication only, including file/receipt/directory syncs and readback | 11.700 ms | 9.849..13.050 ms |
| Prepare including complete recovery validation | 21.986 ms | 3.052..46.931 ms |
| Evaluate/publish including recovery validation | 32.110 ms | 12.467..52.985 ms |
| Recovery scan | 19.069 ms | 1.392..40.128 ms |

Memory:

- Incremental traced Python peak during the 32-token benchmark: **738,586 bytes (0.705 MiB)**, below the 4 MiB prototype target.
- Traced retained allocation at end: 70,061 bytes, including bounded measurement lists.
- Observed anonymous RSS increased from 13,279,232 to at most 13,307,904 bytes: **28,672 bytes**.
- Observed PSS increased from 16,642,048 to at most 16,687,104 bytes: **45,056 bytes**.
- Process VmHWM increased from 24,678,400 to 24,772,608 bytes. Total Python process memory is larger than the incremental prototype target; the target is not a 4 MiB whole-process limit.
- In the separate 1,000-recovery test, traced peak was 15,024 bytes and current retained allocation 2,699 bytes; hook event history remained empty.
- Host MemAvailable and shared cgroup current/peak/swap were sampled, but other desktop activity makes their changes non-attributable. Cgroup current fell from 4,183,375,872 to 4,120,727,552 bytes; lifetime peak stayed 4,268,040,192; swap current stayed 42,606,592. These are shared-cgroup observations, not a prototype memory-saving result.

Disk:

- Logical growth per generated synthetic token: **1,659..1,749 bytes**; growth includes retained checkpoint/receipt and journal expansion.
- 32-token store: 55,784 logical file bytes, 286,720 allocated file bytes, 69 files including the lock. Filesystem metadata and physical device write amplification are not included in those numbers.
- 64-token in-memory capacity fixture: 113,366 logical bytes and 132 files (the simulation has no actual lock file). All old checkpoints are retained.
- The complete retained three-run evidence directory was approximately 2.4 MiB at inspection. No weights or real hybrid state were copied.

These measurements do not qualify the 2 MiB maximum-frame case, opaque 5.31 GiB state replacement, 48 simultaneously referenced files, multi-gigabyte page-cache pressure or long-running storage reclamation. Full engine checkpoint writeback cost remains unmeasured.

## 7. Reproduction and artifact integrity

Run from the isolated worktree:

```sh
cd '/home/user/PandaCode Bots/meatball-84ccea00/qies094-worktree'
./pandacode-run.sh
```

The wrapper runs in the foreground, creates a fresh `qies094-checks/run-XXXXXX`, and preserves all prior runs. It calls Python with isolated mode and bytecode writes disabled. No dependencies are installed. It exits nonzero on any test failure; inspect the new run's tests.log if it stops before printing results. No graphical preview applies to this CLI-only harness.

Check recorded artifacts:

```sh
cd '/home/user/PandaCode Bots/meatball-84ccea00/qies094-worktree/qies094-checks'
sha256sum -c SHA256SUMS
```

Source files, report, input design and passing evidence are included in the manifest. `verification.json` pins the successful run and protected source/helper/test/guard/baseline identities. No model identities are fabricated: the protocol identity is explicitly `SYNTHETIC-STATE-v1`.

## 8. Unsupported recovery cases and integration risks

1. Native QUORTEX 48-layer state, model/tokenizer/index compatibility, next-logits persistence, full recurrent/KV/indexer/PLE/HC continuity and runtime bit-exactness are untested. No native loader is involved.
2. No EOS/cancellation/token-budget stop receipts, UTF-8 suffix persistence, delivery-intent/local-write receipts, terminal replay policy UI, linked journal-segment rotation or repair are implemented. Token quota refusal is a storage safety bound, not model generation stopping semantics.
3. Torn tails permit read-only recovery but block continued evaluation/appending. Corrupt published objects reject the entire current recovery attempt; selecting an older checkpoint needs a future explicit rewind protocol.
4. Incomplete initialization needs a fresh dedicated store. Corrupted/forked publication histories and hostile concurrent directory mutation are unsupported; all writes require one cooperative owner.
5. No garbage collection or archival compaction. Quota/capacity refusal is deliberate. Published old state remains retained until a separately validated retention protocol exists.
6. Hashes provide accidental corruption detection, not cryptographic authentication. Close failures and directory-fsync failure at the actual kernel/device layer were not independently reproduced; injected I/O failures and before/after crash outcomes test control flow only.
7. SIGKILL does not flush-test a drive's volatile cache or prove power-loss behavior. No reboot, cache reset, filesystem remount or storage-device fault was performed. The in-memory persistence model does not model arbitrary partial-sector corruption; checksum tests cover selected corruption cases.
8. The prototype reads a bounded complete journal and retains bounded decision/checkpoint metadata. Its fixed 64-step cap makes memory growth finite; it is not an unbounded continuous-generation service. Recovery currently rescans all retained files and has increasing time cost.
9. Per-layer ownership adaptation and avoiding duplicate serialization buffers remain future design work. Synthetic low overhead cannot be extrapolated to full state; file-backed cache and cgroup pressure must be measured there.

## 9. QIES095 recommendation

**CONDITIONAL GO for further model-free qualification**, specifically a multi-component synthetic checkpoint (48 tiny independently hashed state entries), complete stop/pending/UTF-8 and output-receipt state machine, linked-segment continuation after a torn tail, stronger corruption/size-bound stress tests and an explicit retention/recovery policy. Preserve the existing single-file prototype and golden synthetic fixtures as an independently reproducible reference.

**NO-GO for wiring this directly into production inference or claiming crash-consistent QUORTEX recovery.** Before native integration, qualify the checkpoint/state-store adapter against exact baseline serialization and decode order, recover or independently generate qualified reference fixtures, and prove real continuation/logit equivalence under the unchanged >=6 GiB MemAvailable gate and all existing guards. No inference launch is part of QIES094 or authorized by its small-fixture results.

No production engine changes, decode scheduling changes, QIES086 application, persistent-context redesign, resource-guard changes, system configuration changes, inference, commit, push or merge occurred.
