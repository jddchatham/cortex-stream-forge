# QIES093 - Transactional Generation and Recovery Design

Date: 2026-11-09. Design and source audit only; nothing here is an implemented recovery guarantee.

**Recommendation: GO for a small, model-free QIES094 protocol prototype; NO-GO for production integration or crash-recovery claims.** Use immutable, versioned state checkpoints and a durable output-decision journal. Preserve emission before evaluation and the final unevaluated pending token. Treat evaluated position, selected-token count, and output delivery as different facts. Stdout cannot provide exactly-once crash recovery.

## 1. Audited baseline and scope

Worktree: `/home/user/PandaCode Bots/meatball-84ccea00/qies090-worktree`.
Branch: `development/continuous-generation`.
HEAD: `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`.

The working tree intentionally includes the preserved, uncommitted QIES091 implementation and QIES092 test extensions. The design audits those contents rather than pretending HEAD alone contains them. Read QIES091/QIES092, the generation helper, CLI state store and forward path, and the sequence/hybrid serialization implementations. Existing archive branches, tags and all source/test/binary/guard identities remain unchanged. Only this report was added.

Source evidence, relative to `runtime-flashnext-layer-exp/`:

- `examples/flashnext-generation.h`: progress ownership, advance protocol, checked conversion, UTF-8 handling, generate loop and progress writer.
- `examples/flashnext-chat.cpp:127`: capture_seq_state uses llama_state_seq_get_size/get_data; `:260`: state-store mappings; `:308`: restore_layer_state; `:326`: save_layer_state; `:583`: forward_step; `:884`: main progress/state ownership; `:969`: helper integration.
- `src/llama-context.cpp:4255`: raw sequence serialization delegates to memory state_write/read. It does not itself supply the proposed outer transaction/model identity envelope.
- `src/llama-memory-hybrid.cpp:190`: full sequence serialization includes attention and recurrent memory; partial-only flags omit attention. Keep the existing full-state flags. Restoring one component is not an atomic all-layer restore.

The relevant protection hashes are unchanged: CLI source `62fd09e0bbcd132b9fe04c87a9cb8e73572e8c0c3944db123cff02a9abd4e2b7`; helper `a0998f749bb3d6f826be91d34ded7c0beb1cdd8eb1048f46d5a0f230ccd3a4c4`; guard `ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2`.

## 2. Current behavior and failure windows

Let S(p) be all authoritative layer blobs after exactly p input tokens have been evaluated, H the evaluated token history, and L(p) the logits produced at the appropriate completed forward. Position p equals len(H), not the number of displayed tokens.

Current generation order is:

1. Validate finite vocabulary-sized L(p), write requested diagnostic selection output, select first-maximum greedy token t.
2. If native EOG, stop without converting or evaluating EOG. An unfinished UTF-8 suffix makes this an error.
3. Convert t with native token-to-piece, validate and combine with the held UTF-8 suffix.
4. Persist metadata with output_valid=false; write/flush the complete prefix to stdout.
5. Append t to generated IDs, retain any incomplete suffix, set pending_token=t and output_valid=true; persist metadata.
6. If cancelled, stop here. If the generation budget is exhausted, also stop here. S(p) has not evaluated t.
7. Otherwise persist state_valid=false, then forward_step(t,p,H). Existing per-layer contexts restore serialized memory, decode, serialize state and are freed on the original schedule.
8. Layer saves overwrite shared mapped files sequentially. Later-layer, projection or serialization failure can leave mixed old/new blobs.
9. After successful forward and valid-metadata save, replace coordinator history with H+[t], position p+1 and pending_token=-1.

Prefill uses the same advance helper but has no generated-text emission. A prefill checkpoint also needs the exact prompt token list and the next prefill cursor; early prefill steps do not necessarily compute final logits. The historical two-token and later native non-thinking prompt protocols must not be mixed.

| Failure window | What remains valid today |
| --- | --- |
| Selection/conversion/UTF-8 validation before output | Prior live state is not mutated by that operation; helper records error and refuses continuation. No durable complete checkpoint is established. |
| Output-intent save fails | No new output; in-memory output invalid. Prior on-disk metadata may be stale. |
| Partial stdout write or crash before output acknowledgment | Some bytes may be visible. No reliable delivery offset; no safe automatic replay. |
| Output succeeds, subsequent metadata save fails | Text may already be visible; output invalidated, no automatic continuation. |
| Evaluation-start marker save fails | Evaluate is not called; coordinator invalidated conservatively. |
| During any layer update or final projection | Mapped files may be mixed. Old coordinator position is not a usable rollback image. |
| Forward succeeds, metadata commit fails | Physical state may be advanced; old coordinator position remains and state is invalid. |
| Process death or power loss | No durable all-layer checkpoint; page writeback and metadata persistence need not coincide. |

The existing writer fsyncs only its temporary progress file and renames it. It does not fsync every layer file or the containing directory. Rename supplies atomic name replacement, not a multi-file data transaction. Directory synchronization is separately required for durable directory entries. These are the filesystem primitives used by the proposed protocol. [Linux fsync documentation](https://man7.org/linux/man-pages/man2/fsync.2.html), [Linux rename documentation](https://man7.org/linux/man-pages/man2/rename.2.html).

The CLI currently requires a new state directory and has no checkpoint resume loader. QIES092's 35 model-free cases and synthetic restart fixture do not establish real engine recovery.

## 3. Transaction states and invariants

A transaction key is (run_uuid, generation_id, input_position, token_sequence). A committed checkpoint C(p) identifies S(p), H, phase/cursor and any required next-selection logits. A durable output decision D(k) identifies a selected token and exact output bytes, derived from a specific C(p). **D(k) is not proof that t has been evaluated.**

Each record carries both an evaluation phase and a delivery phase. The five names below describe observable boundaries; they must not be encoded as a single irreversible ordinal. For the recommended streaming policy, EMITTED may precede EVALUATING and COMMITTED.

| State | Authoritative position/state | Recoverable checkpoint | Permitted output | Failure/restart rule |
| --- | --- | --- | --- | --- |
| PREPARED | p, S(p), H from C(p); t selected but not evaluated | C(p) plus durable D(k), including t, piece, suffix and output offsets | Nothing until D(k) durability succeeds | Recover C(p); use the recorded decision, not a fresh selection. If D(k) is incomplete, ignore its torn tail and do not claim it existed. |
| EVALUATING | Committed position remains p. Candidate S*(p+1) is private and never authoritative | C(p) and D(k) remain intact | Streaming bytes may already have been attempted; do not emit them again automatically | Abandon candidate on failure; restore C(p), evaluate the recorded pending t once in recovery. No continuation using mixed candidate layers. |
| COMMITTED | p+1, H+[t], S(p+1), required logits and metadata fully persisted and published | C(p+1), with parent C(p) retained | Policy A may now emit; policy B never re-emits a previously attempted record automatically | Recover C(p+1). No re-evaluation of t. Commit is durable only after the publication directory fsync succeeds. An interrupted/failed fsync has an uncertain outcome resolved by recovery, never guessed in-process. |
| EMITTED | Does not itself change p. In B, p may still be the source position; in A it may already be p+1 | The corresponding C(p) or C(p+1), plus D(k) and delivery evidence | Exact journal bytes only, in sequence; never fabricate a terminal acknowledgment | Local write return is not durable external delivery. On restart mark stdout delivery ambiguous and use explicit replay/skip policy or an acknowledged sink. |
| INVALID | Candidate has no authority; never advance the committed position based on a flag alone | Last fully verified published checkpoint, if one exists | No further live output/evaluation | Fail closed; preserve candidate/error evidence. Restart through verified recovery only. If no valid checkpoint exists, report unrecoverable and require a fresh run. |

Required invariants:

- p=len(H); for each committed generation, child position is parent position+1 for an evaluated input. Output-only/EOS/cancel receipts do not increment p.
- Generated-token sequence, evaluated generated-token prefix and optional pending token are separate fields. There is at most one pending generated token; it must equal the last selected generated ID and be derived from the named committed source checkpoint.
- Never evaluate EOG. Never add an evaluation solely to finish a token budget. A clean token-limit outcome can have C(p)+D(k) with pending t and no C(p+1).
- A failed candidate cannot overwrite published blobs. Progress metadata cannot convert a partial candidate into a checkpoint.
- Restore exact H for PLE/indexer history dependencies, exact full hybrid blobs for recurrent/KV/indexer state, and the exact boundary metadata needed for continuation. Layer-to-layer transient HC handoffs are not restart points; checkpoint only a completed whole-forward boundary.
- Native state bytes, arithmetic, layer/context ownership, token scheduling, n_ctx and model flags stay unchanged. The state-store adapter owns file versions; the coordinator owns positions/journal receipts. No persistent llama_context redesign.

Conceptual paths:

```text
B: C(p) -> PREPARED D(k) -> output intent -> EMITTED/ambiguous
                                  -> [cap/cancel: retain pending t]
                                  -> EVALUATING -> COMMITTED C(p+1)
A: C(p) -> PREPARED -> EVALUATING -> COMMITTED -> EMITTED
Any candidate/write error -> INVALID -> verified recovery or stop
```

For B, an output error stops evaluation as today; durable D(k) still preserves the intended text, but delivery ambiguity must be resolved before resuming. Cancellation before preparation writes only a stop receipt. During evaluation it is cooperative: finish and publish safely if possible, or invalidate the candidate; do not relabel a partial write as a clean cancellation. Guard escalation can leave only the previous checkpoint recoverable.

## 4. Output consistency: comparison and recommendation

| Property | A: evaluate/commit before emit | B: journal decision before current-order emit |
| --- | --- | --- |
| Additional memory | One bounded piece/suffix plus checkpoint/hash buffers | Same, plus bounded journal framing; no accumulated token-text heap |
| Storage | Immutable checkpoints plus delivery records | Same checkpoints, plus exact token/piece/output decision records |
| Serialization cost | Durable full state publication before displaying t | Small decision fsync before display; full checkpoint at the original evaluation boundary |
| First/subsequent display latency | Includes t's next forward and checkpoint; potentially substantial | Includes prior-state publication and decision fsync, not t's own forward |
| Crash semantics | State may be committed but output delivery unknown | Output may have been delivered while t remains pending; journal distinguishes this from evaluated state |
| Complexity | Simpler order for evaluated tokens, but still needs output ambiguity handling | Two linked state/output positions and pending-token recovery |
| Scheduling compatibility | Strictly evaluated-state-first changes emission order and evaluates the final capped token unless special-cased | Preserves existing emission/evaluation order, EOG and pending cap behavior |

**Recommend B.** A variant that merely commits a selected decision against S(p) before emission is valid but is decision journaling, not evaluated-state-first commitment. Do not call that variant C(p+1).

The durable journal is the authoritative ordered text transcript. Before stdout, persist a DELIVERY_INTENT naming the decision, byte range and sink epoch. After a successful local write, persist LOCAL_WRITE_COMPLETE; its name must not imply consumer acknowledgment. Crash between write and receipt, or after receipt while a downstream buffer is lost, is still ambiguous.

Default restart behavior: no automatic replay to ordinary stdout. Report the ambiguous range on stderr; require an explicit replay-from-offset or skip-ambiguous-range choice for a new sink epoch. Replay is at-least-once and can duplicate visible bytes; skipping is at-most-once attempt and can omit visible bytes. Keep the complete journal transcript available regardless. Never merge bytes from divergent run/branch IDs. A consumer with stable (run,sequence,offset) deduplication and durable acknowledgments could offer stronger delivery semantics, but that is a separate protocol, not a property of stdout.

UTF-8 decisions store raw native piece bytes, old/new suffix and complete emitted bytes. Incomplete bytes remain pending across decisions/checkpoints; completed output offsets count bytes, not tokens. EOS with an unfinished suffix remains an error. Preserve the current CLI's incomplete-suffix-at-final-cap failure policy unless separately authorized. Newlines/status messages are separately classified output events, not generated token bytes.

## 5. Checkpoint storage and publication

Proposed private persistent root, same verified filesystem for all renames:

```text
run-<uuid>/
  identity.json
  writer.lock
  journal/segment-000001.bin
  candidates/g-<id>.tmp/
  generations/g-<id>/
    manifest.json
    state-layer-0.bin ... state-layer-47.bin
    next-logits.f32                 # only when required at this boundary
  commits/c-<id>.json               # publication receipt, immutable
  CURRENT                           # optional hint, never sole authority
  failures/                         # small diagnostic receipts
```

Single writer holds an exclusive lock throughout recovery and generation. Use exclusive creation, directory-relative operations, strict relative filenames, no symlink traversal and no overwrite of existing IDs. This is a dedicated new run root, never an original milestone state directory. A second writer refuses admission. An OS lock is not the durable commit record.

### Metadata and identity

A versioned manifest must include:

- Protocol schema and native state ABI/source digest; executable and loaded-library identities, build/ISA/context/thread configuration; sequence ID and full-state flags. Conservative version mismatch rejection until an explicit compatibility migration is qualified.
- Model shard lengths/SHA-256s, published/adapted index identities, architecture and tokenizer/template/config identities. Pin native non-thinking mode, exact prompt/formatted input and token IDs; disallow hidden diagnostic/test environment cuts in a normal resumable run.
- Run/generation/parent IDs and parent manifest hash; phase (prefill or generation), p, H, prompt cursor, generated IDs, pending ID, stop reason and UTF-8 suffix.
- Exactly 48 layer entries with layer number, native blob format/config identity, exact length and SHA-256. No missing/duplicate/mixed-generation entries.
- Required next-selection logits length, F32 endianness, vocabulary count and SHA-256, or explicit absence at a prefill boundary where the original schedule does not request logits. Resume must follow the recorded phase rather than reselecting from absent logits.
- Exact journal prefix boundary (segment, byte offset, sequence, chain hash), selected-decision references and output byte boundaries. S(p)'s checkpoint and output receipts can advance separately, but their ancestry must agree.

Use deterministic canonical manifest bytes; reject duplicate keys, unbounded lengths, malformed IDs, overflow, unknown required versions and unsafe paths. Hash the exact stored manifest, not an implementation-dependent reserialization. SHA-256 detects corruption against a trusted receipt; it is not authentication against an attacker replacing all files. The existing raw sequence API is insufficient to verify model identity by itself.

A journal frame has a fixed versioned little-endian header, type, bounded payload length, run/transaction/sequence/parent references, previous-record hash and SHA-256 of header+payload. Payload encodes token IDs, source checkpoint/logits identity, conversion bytes, UTF-8 transitions and output offsets using exact lengths. Specify a 2 MiB maximum frame for the current 1 MiB conversion bound, with no duplicated large payload fields inside that cap. Detect any corruption in the committed prefix; only an incomplete final uncommitted frame may be ignored. Preserve it and start a new linked segment rather than silently appending through damaged bytes.

### Publication ordering

Initialize and synchronize the new run root, identity file and required directories before the first receipt. Publish a genesis C(0) with empty history and 48 explicit uninitialized layer entries (not 48 fabricated native blobs). Only p=0 permits those entries; restore follows the existing no-state-at-position-zero rule. A post-evaluation checkpoint requires all 48 valid serialized blobs. Prefill transactions use the exact recorded prompt token rather than the greedy-output journal path, and preserve the original compute-logits decision for each position.

1. Begin with verified committed C(p). Never expose its layer objects as writable storage for the candidate.
2. Persist PREPARED decision and output intent before any B emission. Check every write/flush result. A selected decision is durable, but t is still pending, not evaluated-state committed.
3. At the existing evaluation call, create private candidate state files. Route the **already required** layer serialization buffers directly to candidate files; no second full-state capture or aggregate 48-layer heap. Restore each layer from C(p) using existing native state APIs and unchanged decode arguments.
4. After the entire original forward succeeds, finish required logits and metadata. Check complete writes and exact sizes; hash the serialized bytes with a bounded buffer and verify file contents in the prototype. Fsync every candidate state/logits file and the manifest. If a mapped writer is retained later, qualify its synchronization path explicitly; prefer checked file writes in the prototype.
5. Fsync the candidate directory, rename it to its unique generation name without replacement, and fsync both affected parent directories. Every reference must name an immutable file that is already durable. A generation directory alone is not a commit.
6. Create a publication receipt in commits using exclusive temporary creation; include generation/manifest hash, parent receipt hash, evaluated position and journal prefix reference. Fsync it, rename to a unique final receipt name, then fsync commits. **Only successful completion of this last directory sync authorizes the live COMMITTED acknowledgment and position advancement.**
7. Optional CURRENT updates are hints, written atomically and directory-synced. Recovery can reconstruct from publication receipts; a stale/missing hint cannot authorize a different state.

This follows the separate file/directory durability requirements documented by [fsync(2)](https://man7.org/linux/man-pages/man2/fsync.2.html); name publication uses [rename(2)](https://man7.org/linux/man-pages/man2/rename.2.html). Guarantees assume the local filesystem/device honor completed flushes; prototype crash tests do not prove power-loss behavior of the hardware.

### Recovery and last-known-good selection

Under the writer lock, validate identity first. Read bounded receipts and their parent/hash links; choose the newest **uniquely linked, fully verified published** checkpoint, never newest mtime or an orphan directory. Check every retained layer/logits hash and all required metadata before calling any native state reader. A receipt present after interruption around fsync can represent an uncertain commit that actually survived; accepting its fully verified generation on restart is allowed. Never retry that transaction in the failed live process without recovery arbitration.

An incomplete candidate or generation with no receipt is not authoritative. Preserve it for diagnosis and restore the last committed generation. Never mix individually readable layers across generations. A corrupt referenced journal prefix, conflicting child receipts or identity mismatch is a hard failure. If the newest published checkpoint is corrupt, report it explicitly; an older verified retained checkpoint is a recovery option only with an explicit rewind/replay plan, not a silent fallback presented as uninterrupted success.

Replay durable decisions after the selected checkpoint in order. If its pending t has not been evaluated, evaluate t exactly once before choosing another token, unless the recorded cap/cancellation requires remaining stopped. Do not replay text automatically. If a checkpoint already includes t, do not evaluate it again. Preserve separate stop receipts for EOG, token cap, cancellation and error; restarting does not implicitly increase a token budget or clear an error.

### Avoiding full-state copies

Crash consistency requires either preserving old bytes, durable undo information, or deterministic reconstruction from an older checkpoint. Metadata alone cannot protect in-place overwritten files. Initial design chooses immutable old bytes and replacement candidate blobs, using existing serialization output once. No copy of an entire state directory, writable hard links, cache dropping or whole-state anonymous snapshot is required.

All 48 opaque blobs may change per forward, so worst-case full serialized replacement remains necessary; page-level deltas cannot be assumed safe without a separate format/access audit. The current workspace reports btrfs on `/dev/mapper/root[/@home]`; filesystem COW/reflinks are not adopted as application transaction guarantees. Reflink/snapshot optimization, delta logs and sparse checkpoints are deferred.

Keep two verified generations and room for one candidate as a planning bound, approximately 3 x 5.31 = 15.93 GiB of logical state using QIES083's historical size. This is an estimate, not a measured current checkpoint cost. QIES094 must not delete prior artifacts or implement production GC. Its small fixtures have a hard capacity cap and stop before exhaustion. Real integration needs separately qualified retention: never retire a referenced generation/journal prefix, and durably publish retention decisions before removing only owned obsolete objects. Until that exists, stop at the capacity bound rather than accumulate unbounded checkpoints.

## 6. Failure and recovery matrix for the proposed design

| Crash/failure point | Durable authority | Required recovery result |
| --- | --- | --- |
| Before PREPARED durability | C(p) | Ignore incomplete decision; no authorized new output |
| After PREPARED, before output | C(p)+D(k) | Pending decision preserved; output may be delivered under chosen sink policy |
| During output / before or after local completion receipt | C(p)+D(k) | Report delivery uncertainty; no automatic terminal replay |
| After emitted token at cap/cancel | C(p)+pending D(k)+stop receipt | Stop at same budget/boundary; never invent S(p+1) |
| Before first layer or after any of 48 layer writes | C(p) | Ignore private candidate, restore all layers from C(p); no mixed-layer state |
| After all state writes but before required logits/manifest | C(p) | Candidate incomplete, not committed |
| After generation rename, before commit receipt | C(p) | Orphan generation ignored; no position increment |
| During receipt publication or directory sync | C(p) or fully valid surviving C(p+1) | Resolve by verified receipt on restart; never assume failed sync means old state won |
| After durable receipt, before in-memory position update | C(p+1) | Recover p+1; do not re-evaluate t |
| After commit, before CURRENT update | C(p+1) | Ignore stale hint; discover valid receipt |
| Corrupt/mismatched state, logits, model or version | No acceptable candidate under that identity | Reject; report last verified retained alternative and explicit rewind requirement |
| ENOSPC, EIO, short write, failed fsync/close | Prior verified commit remains immutable | Stop with failure; preserve evidence; no output after failed required journal durability |

Neither IMMUTABLE filenames nor checksum validity alone imply semantic correctness of native state. Exact real-model continuation remains a separate qualification gate.

## 7. Performance and memory tradeoffs

All numbers here are planning estimates, not QIES093 measurements:

- Current historical serialized set: approximately 5.31 GiB. Per-evaluated-token durable checkpoints can write and hash approximately that amount even though stdout chunks are tiny. At sustained device throughput R GiB/s, the write-only lower bound is S/R seconds; hash verification/readback and fsync add cost. No throughput value for this SSD is assumed.
- The 26-token prompt plus answer evaluation would represent about 27 state publications, roughly 143.4 GiB logical writes at constant 5.31 GiB/set. Actual early-state sizes may differ. Forced writeback can be substantially more expensive than baseline dirty-page coalescing.
- A full 248,320-element F32 next-logits vector is 993,280 bytes, about 0.947 MiB. Store only the vector required for each retained checkpoint; these are continuation state, not unnecessary diagnostic captures. No new projection/decode is performed to manufacture missing prefix logits.
- Target incremental anonymous memory for the prototype: <=4 MiB beyond its fixture/state input, using <=256 KiB hashing/I/O scratch and one <=2 MiB journal frame. This is an acceptance target to measure, not a proven engine budget. Do not create another copy of an existing serialization vector.
- Old and new file-backed pages can coexist and raise cgroup memory even if anonymous RSS stays flat. Bound outstanding candidate writes, synchronize at safe boundaries, unmap unused old read views only when no graph/context references them, and measure RSS/PSS, cgroup file/dirty/writeback/anon, MemAvailable and I/O. No madvise/allocator/guard adjustment is part of this design.
- Current n_ctx=256 also bounds journal tokens; it is not unlimited context generation. At the worst 1 MiB piece bound, raw token pieces can approach 256 MiB before framing. Enforce byte quotas and actual disk reserve as well as token limits. No unbounded in-memory transcript.

If per-step checkpoint I/O is unacceptable, a later design may retain a coarser immutable checkpoint plus deterministic replay journal. That can preserve correct recovery but changes the recovery work/commit horizon: intervening tokens must not be called durably state-committed. Do not silently introduce it as a faster implementation of the same per-step guarantee.

## 8. QIES094: minimal implementation and validation task

**Implement a standalone, model-free transactional fixture store and journal, exclusively as experimental validation infrastructure on this development branch. Do not wire it into the production CLI yet.** Reuse the QIES091/QIES092 simulated transition/output tests; retain their original evidence and all guards.

1. Define bounded protocol structures, immutable generation writer, exact-size/hash verifier and receipt-based recovery selector. Use tiny representative layer blobs, including distinct attention/recurrent/indexer-tagged bytes and history/UTF-8/pending/logits fixtures. Native state remains opaque; synthetic tags do not prove real state coverage.
2. Provide an injected filesystem abstraction and explicit fault IDs before/after each write, file sync, close, directory sync, rename, journal acknowledgment and coordinator update. Distinguish nondurable writes from durable operations in a deterministic crash-storage model.
3. Add process-kill tests on owned temporary SSD fixtures at every transaction boundary, including every layer index. Process death is supplemental; the storage model must also simulate loss of unsynced file data/directory entries, reordered persistence permitted by the protocol and torn final records. No reboot, global cache reset or real model.
4. Test strict parsing, identity/version/model mismatches, malformed lengths, duplicate/missing layers, swapped files, changed hashes, invalid positions/pending tokens, damaged journal prefixes, conflicting receipts, writer exclusion and interrupted publication. Inject ENOSPC/EIO/EINTR/short writes and failed final syncs.
5. Assert recovery returns exactly the last permitted complete generation or fails explicitly. Compare every recovered synthetic byte, history/position, next logits, token decisions and UTF-8 transitions to the uninterrupted reference. Assert evaluated position never changes merely because text was selected/emitted.
6. Test clean EOS at multiple positions; cap with one pending token; cancellation before/after output and during evaluation; repeated restart before/after commit; partial output; ambiguous-delivery replay and skip policies. Supported acknowledged-sink tests may deduplicate IDs, but terminal tests must report uncertainty rather than assert exactly once.
7. Measure incremental anonymous/PSS/peak RSS, cgroup totals/file cache, MemAvailable, write/read bytes and publication latency over repeated fixture cycles. Enforce <=4 MiB incremental anonymous target and bounded object/journal counts; test capacity refusal. No change to resource guards to make a test pass. Prototype supports no destructive GC.
8. Publish protocol fixtures, deterministic seeds/fault IDs, per-case recovery receipts and SHA-256 manifests. Stop on corruption, drift, output-policy violation or memory growth. No automatic commit or production integration.

Passing QIES094 would qualify the storage protocol under its tested failure model, not the QUORTEX adapter. A later separately authorized integration must preserve exact restore/save bytes and decode scheduling, qualify native checkpoint restart against the independent stock reference and original baseline, and satisfy bit-exact 48-layer/consecutive-logit/state-continuity checks. The QIES089 stock candidate is still not runtime-qualified; existing incomplete goldens/guard-adapter prerequisites remain blocked.

No inference or new gate attempt was needed for QIES093. QIES092's last recorded 3.79 GiB MemAvailable was below the unchanged 6 GiB entry requirement; it is historical evidence, not a current measurement or launch authorization. Any later inference requires fresh unchanged admission, all swap/disk/cgroup/PSI/runtime checks and a qualified launcher. A larger dedicated CPU validation host remains unverified.

## 9. Final decision

**GO for QIES094's bounded, non-inference protocol prototype and fault-injection tests. NO-GO for production adoption, exactly-once stdout claims, automatic reuse of invalid state or real checkpoint recovery claims.** B preserves decode scheduling while making output decisions durable; immutable checkpoint publication is required before any evaluated state is called committed. Its likely SSD cost and file-cache pressure require measurement before integration.

No production code, tests, allocator behavior, inference mathematics, state ownership, memory guards or system settings were modified. No inference, QIES086 application, commit, push or merge occurred.
