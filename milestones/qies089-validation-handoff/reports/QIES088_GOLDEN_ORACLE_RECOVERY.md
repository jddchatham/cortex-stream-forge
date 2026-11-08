# QIES-088 - Golden oracle recovery and validation readiness

2026-11-07. **NO-GO for inference or QIES-086 acceptance in the present desktop session.** Original full golden fixtures and the historically verified stock executable were not recovered. The published stock source was identified by its recorded hash and compiled successfully into a new, explicitly unqualified oracle candidate. Current desktop memory fails the unchanged 6 GiB MemAvailable gate. No model evaluation, fixture regeneration, guard modification or production change occurred.

## 1. Fixture archaeology and provenance

QIES-087 searched the project/work/download locations, three Git worktrees, all 17 locally reachable commits, relevant remote milestone/multi-token/continuous-generation/first-answer branch heads, and the additional lean-lifecycle remote tree. No tensor golden blobs were present in those tracked trees; no unreachable Git objects, published GitHub releases or public Actions artifacts supplied the originals. Its search evidence is preserved and was reused rather than re-downloading identical branch trees.

This investigation extended the search through readable `/home/user`, `/mnt`, `/media` and `/opt` files, including build directories previously skipped, archived-name candidates and likely local executable identities. Exactly two relevant tensor files were found: the existing QIES-081 answer/EOG selection arrays. Thirty-two likely local ELF files were hashed; none matches the historical stock executable digest. The sole archive candidate was a Thorium browser package, not a capture archive. No original capture directory, additional stock output or original reference bundle was found. No new external fixture volume was available.

This is a bounded readable-storage search, not proof about inaccessible/unmounted media or every unrelated remote branch's history. `/.snapshots` is root-only and was not entered or reconfigured. No recovery operation modified old files, Git objects or snapshots.

### Available verified subset

The two QIES-081 files remain in place under `qies075-data/outputs/20261107-083141-efe63add`:

| File | SHA256 | Contents |
|---|---|---|
| logits-select-0.bin | bb796a4131b858e2a692ab8e7faf2dedc14fcc269272c056bfece80e52dd0cdf | 993,280 bytes, 248,320 F32; historical argmax 19 (`4`) |
| logits-select-1.bin | 6d82281115bf3d3a48606bb4f4915c7f2fd252f2c48343306123932820549fdf | 993,280 bytes, 248,320 F32; historical argmax 248046 (native EOG) |

Hashes match the protected manifest. These are established QUORTEX baseline outputs, not independent stock fixtures for every layer/prefix. The 48 persistent state files and their size/mtime inventory also remain intact. They do not by themselves establish checkpoint/resume correctness.

The published `docs/QIES072_NONTHINKING_FIRST_ANSWER.md` explicitly says original stock and corrected QUORTEX run outputs were retained locally and not published. Its historical claim of 28 exact comparisons is evidence of that recorded run, not a replacement for absent numerical files. Known hashes do not reconstruct missing bytes.

### Missing fixture sets

The existing milestone suites expect stock `hc_combine-0.bin` through `hc_combine-47.bin` for their exact one/two-token protocols, full stock logits at consecutive positions, and corresponding streamed captures. Their default original root is `/home/user/quortex-diag-20261103`, absent here. QIES-072 additionally requires the complete non-thinking prompt-prefix and answer/EOG stock oracle, not just the older one/two-token milestone inputs. Checkpoint/retry expected states and outputs also need an authoritative oracle/harness.

Every recovered bundle must carry trusted complete-file hashes, lengths/shapes/dtypes, model/index identity, token/position/sequence metadata, context/backend/compiler/library configuration and original capture provenance. No complete original per-layer reference-hash manifest was recovered. Do not substitute QIES-086 outputs or rename the two selection arrays to fill those gaps.

## 2. Independent stock oracle: what exists and what does not

Published source:

`cortex-stream-forge-qies074/runtime-flashnext-layer-exp/examples/flashnext-stock-two-layer.cpp`

| Identity | Finding |
|---|---|
| Published base commit | 8fac93bb2bfc4b2d7069c4044f2f71cd25f80388 |
| Published stock source SHA256 | e169159b26d68409c178bc8fbe496fc5582e240c8c9ff6804e78ac2385c8b35f - matches file |
| Historical stock executable SHA256 | eda8ceefc94f296a66d217b79d0aba64317b5f24295b822e086e7d3736169fb2 - not recovered |
| Older source-overlay note | 4faa9bd2... - only a partial digest is published; exact overlay unavailable |
| New candidate executable SHA256 | 6a29bf69f5976e9f4bd0e6df8e351336e4b52ca93d492d7e66bd82ee8cb22963 |

The stock source has five versions in reachable Git history. All were hashed; none starts with the older note's `4faa9bd2` prefix. That provenance gap remains unresolved. The new executable is **not** the historical executable and is not asserted historically verified.

New build location:

`qies088-investigation/stock-build/stock-reference-candidate`

It was compiled without source changes using the recorded CPU-only stock target compile command and the existing verified baseline shared libraries. Build commands, source/binary hashes and warnings are preserved in `logs/qies088/stock-build-identity.json` and `stock-build.log`. A **no-arguments** invocation returned usage/exit 2 before backend or model initialization, as audited in main(). This proves compilation/linkage, not numerical correctness. Do not use `--help` as a harmless stock-runner probe: this runner accepts a single model filename and has no such early help option.

### Source-level independence and limitations

Despite its historical two-layer name, this runner loads the normal stock graph without streamed-layer cuts. It clears the relevant CORTEX streaming/segment overrides, sets n_gpu_layers=0, uses n_ctx=256 and n_batch=1 for the prompt case, and keeps the same context across consecutive tokens. It does not call the QIES-086 transaction coordinator or read the removed staged-state member. Thus it offers a genuinely separate full-graph execution path, using the shared baseline numerical engine rather than experimental CLI output as the oracle.

The runner's prompt branch checks native non-thinking-template support, renders the audited text, tokenizes once, decodes each token at its explicit position with sequence ID 0, synchronizes, and exports complete vocabulary logits. `STOCK_PROMPT_MAX_TOKENS` must be **16** for the full answer/EOG protocol; the default is only one generated token and can omit post-answer evaluation. Its non-prompt path starts with token 248045 and can follow greedy tokens for older milestone protocols. Those are distinct fixtures.

Historical stock build settings/dependent-library identities were not completely recovered. The report specifies a preserved stock configuration, while the new build uses current verified QUORTEX libraries with REPACK=ON. Earlier ON/OFF equivalence reports do not establish the entire historical stock binary identity. Normal-stock loader/mmap/repack memory behavior must be audited before a low-RAM launch; the QUORTEX 1 MiB expert arena is not automatically a stock full-graph weight-memory bound. Stock context thread counts use library defaults, so OMP_NUM_THREADS=1 alone must not be described as fixing every backend context thread. Record actual defaults/configuration in any new oracle manifest.

The published runner does not implement cross-process checkpoint/resume testing. A harness using the existing state serializer/restorer must validate those semantics separately; persisting a file is insufficient.

### Exact common model and prompt identity

Use the same three Qwen3.8-Flash-Next-Uncensored-IQ4_XS shards, with the already-established SHA256 values:

```text
00001-of-00003: 28a99b125cb905fc3bdc06baf6266a56b0cbd1b27e833b9142acc2017b673704
00002-of-00003: 2a309e0b112fde96ba3bcba5a6b58cc05e5df7bb7fad5a990eaa51df335b0e43
00003-of-00003: ebc43c58e2eaeba1d5bdf62c8cb1f0eac198c4dc01941f771921edeebf574bc3
```

Full filenames have prefix `Qwen3.8-Flash-Next-Uncensored-IQ4_XS-`. Weights were not re-downloaded or rehashed in this task. Published tensor index SHA256 is `bc78052954dc3ab6d37ad189a2643c68cc8ad6ecfab5a6bb76361dc0b94f22c7`; the verified local model-directory adaptation is `e6766dd25a3942216bcac35c4e6731c370836622464b7abb8627d08e8e0f0242`. The stock full loader reads GGUFs directly; the index identifies the matching streamed protocol.

Prompt: `What is 2 + 2? Answer with only the number.` Audited 112-byte rendered prompt:

```text
<|im_start|>user
What is 2 + 2? Answer with only the number.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Expected 26 input token IDs:

```text
248045,846,198,3710,369,220,17,478,220,17,30,21134,440,1132,279,1324,13,248046,198,248045,74455,198,248068,271,248069,271
```

Historical first answer ID 19, then EOG 248046; maximum generation 16. Every logits vector must contain all 248,320 little-endian F32 elements (993,280 bytes), not a common-length prefix or just argmax. No complete expected logit values beyond the two retained arrays can be recovered from token IDs alone.

### Can stock regenerate references?

**Conditionally yes in principle; not validation-ready here.** Prefer recovering the historical binary plus its dependency/configuration bundle from the original machine. Otherwise qualify the identified stock source/build as a new reference protocol and record all identities. Do not describe new outputs as recovered originals.

Before generating anything:

1. Freeze the stock source, executable, all loaded library hashes, build flags, CPU/ISA, thread settings, model hashes and token/context configuration. Resolve or explicitly carry the historical provenance limitation.
2. Use a clean environment without inherited CORTEX overrides. For the clean prompt reference, use STOCK_NO_DIAG=1, the exact STOCK_PROMPT, STOCK_PROMPT_MAX_TOKENS=16, a unique STOCK_PROMPT_LOGITS_DIR, OMP_NUM_THREADS=1 and the exact first shard. All launches must go through the unchanged safety policy after fresh passing preflight; this configuration description is not an unguarded launch command.
3. Capture clean stock prefix logits and post-answer logits, then verify the two retained QIES-081 full-vector hashes as independent historical anchors. Any disagreement stops qualification; matching those anchors is necessary but not sufficient for 48-layer/state acceptance.
4. Obtain per-layer captures in a separately audited stock diagnostic pass, and require its complete logits to match the clean pass byte-for-byte at every relevant position. Certify diagnostic capture independence before using those layer outputs as golden.
5. Test repeated explicit token inputs, full hybrid state serialization/restoration, checkpoint continuation and rollback/retry against uninterrupted reference outputs. Preserve KV, DeltaNet, indexer, PLE and HC/token-history semantics; do not assume byte equality of synthetic state files proves them.
6. Repeat the reference run under identical conditions; preserve exact token logs, full array lengths and bytes, checkpoint/state manifests, executable/library/guard identity and resource telemetry. Hash all completed artifacts. Label the bundle **NEW_STOCK_REFERENCE**, with a new ID/date and the qualification results, never ORIGINAL_QIES072.

No regeneration or qualification inference was performed.

## 3. Why desktop MemAvailable is near 3.8 GiB

Fresh `/proc`/PSS census (`logs/qies088/memory-census.json`), separate from later compilation:

| Host metric | MiB |
|---|---:|
| MemTotal | 7,811.54 |
| MemAvailable | 3,909.47 (3.818 GiB) |
| MemFree | 616.38 |
| Anonymous pages | 1,776.42 |
| Cached | 4,647.77 |
| Shared/tmpfs pages | 946.09 |
| Active / inactive file pages | 397.40 / 3,307.82 |
| Unevictable | 709.23 |
| Mlocked | 0.05 |
| Reclaimable / unreclaimable slab | 97.15 / 128.86 |
| Kernel stacks / page tables | 9.38 / 26.36 |
| Dirty / writeback | 0.86 / 0 |

These rows overlap. In particular Cached includes tmpfs/shmem, PSS includes pages also classified by host/cgroup accounting, and Unevictable is not an independent heap to add again. MemTotal minus MemAvailable is not a precise anonymous-allocation total. The older 3.756 GiB observation and today's 3.818 GiB differ with ordinary workload timing; there is no fixed unexplained 3.87 GiB allocation.

Disjoint process-name groups, readable smaps PSS:

| Group | PSS MiB | Private resident MiB | Anonymous PSS MiB | File PSS MiB |
|---|---:|---:|---:|---:|
| Browser | 765.9 | 564.3 | 507.9 | 254.3 |
| PandaCode/WebKit | 822.9 | 763.1 | 674.4 | 147.1 |
| Codex | 256.3 | 256.3 | 87.8 | 168.5 |
| Named desktop processes | 416.4 | 390.1 | 329.9 | 85.9 |
| Other readable processes | 183.3 | 155.0 | 122.2 | 60.5 |

Readable PSS totals 2,444.8 MiB, apportioned for shared mappings rather than summed RSS. These are name-based groups, not complete application ancestry. Largest readable consumers included WebKitWebProcess PID 5190 at 647.9 MiB PSS, Thorium PID 5257 at 365.0 MiB, quickshell PID 1024 at 272.3 MiB and Codex PID 9883 at 231.2 MiB. PIDs and sizes are transient.

PSS was unavailable for 170 process entries, mostly kernel/root-owned entries. Some nonzero-RSS daemons (including Ollama at about 35.8 MiB RSS) could not be apportioned. Do not add their raw RSS to the readable PSS total or infer that a resident model is hidden in those values. No unrelated process was terminated. No GGUF or state-layer mapping was found in **readable process maps**; inaccessible maps prevent a system-wide absolute statement. No QUORTEX inference process was launched. Existing model/state file-cache residency cannot be attributed by the cache total alone, and no model files were faulted into RAM to investigate it.

User-slice cgroup memory.current was 6,158.8 MiB: anon 1,728.2 MiB, file 4,017.2 MiB, including shmem 944.5 MiB. Approximately 709.18 MiB of host unevictable pages were charged to this user slice. That localizes the charge broadly but does not identify its exact application/object owner or prove it will all be freed on logout. Mlocked is much smaller; do not equate the whole unevictable value with mlock. Per-process PSS cannot account for every kernel-held shared/graphics object. A definitive per-object owner attribution was not established.

The system slice charged about 320.6 MiB. Parent/user-manager charges overlap and were not added. Clean cache dominates file charging, while dirty/writeback is tiny. This is consistent with significant normal desktop overhead and reclaimable caches, not evidence that QUORTEX's 91 GiB model is allocated in RAM or that the state cache is an anonymous leak.

Both swap devices remain unchanged and logically unused in the census: 7.63 GiB zram plus 7.63 GiB disk swap, approximately 15.26 GiB free total. Zram data/allocated counters were tiny. Memory PSI full/some avg10 was 0%; no sustained pressure stall was observed in this short snapshot. Swap capacity does not count toward the RAM admission gate.

A later unchanged guard-only check measured 4,083,666,944 bytes (3.803 GiB) available and returned resources_ready=false. The required threshold is **6,442,450,944 bytes of MemAvailable**, not physical RAM or virtual mapped size. The guard's swap/disk/PSI and ancestor checks remain additional requirements.

## 4. Reversible readiness and recommended environment

Saving work and closing unneeded browser/PandaCode sessions manually would release meaningful private anonymous memory. File-backed PSS is already partly reclaimable, so summed PSS is not a prediction of increased MemAvailable. The approximately 2.18 GiB gap at the census cannot be certified closed by adding application PSS to the current number. Unknown shared/unevictable ownership adds uncertainty.

The strongest measured route on this laptop remains a **user-controlled headless TTY session**, following the existing QIES-079/080 recovery instructions. The previous successful guarded preflight recorded 6.565 GiB MemAvailable. That demonstrates a historical reversible route without permanent OS changes; it is not today's admission result. Save all work, leave a recovery console available, stop the graphical runtime only under user control, then repeat the unchanged memory measurement and complete guard preflight. Restore SDDM if the check fails. No application, service, display manager, cgroup, swap/zram setting or kernel option was changed here.

For independent stock reference generation, prefer the original oracle machine with its verified executable/dependency tree, or a dedicated CPU host with substantially more RAM headroom and adequate persistent SSD capacity. A 16 GiB-or-larger host is a planning preference, not a proved stock requirement or permission to weaken any guard. The approximately 91 GiB GGUF payload, state/checkpoints/captures and reserve must fit; the stock loader's actual memory behavior must be established first. Passing the 6 GiB entry test alone does not guarantee stock or QUORTEX completion.

The existing guard CLI pins baseline QUORTEX. Any stock/candidate adapter must preserve the existing run_guard defaults, ancestor checks, tree monitoring, OOM/PSI rules, cleanup and failure semantics, and must be tested with harmless processes before inference. No such adapter was introduced or model launch authorized by this report's candidate build. The full QIES-086 validation still needs the separately reviewed checkpoint/rollback harness and phase-specific anonymous/PSS/cgroup/host telemetry.

## 5. Remaining prerequisites and decision

1. Recover and authenticate original oracle artifacts from the owner/original storage, **or** explicitly qualify and reproducibly hash the new stock reference protocol above. Never use experimental results as golden.
2. Establish complete layer/prefix/consecutive-token reference coverage and checkpoint/rollback expectations, with exact model/token/configuration identities.
3. Pass the unchanged resource gate in the actual intended execution session, and preserve all runtime/ancestor limits and storage reserves. Current desktop session is NO-GO.
4. Use the isolated NumPy environment restored in QIES-087, unchanged strict comparison semantics, exact complete-file lengths and byte comparisons. Token agreement alone is insufficient.
5. Only after oracle, guard integration and resource prerequisites pass, run the QIES-086 baseline/candidate correctness checks and at least three comparable trials per build. Measure actual anonymous RSS/PSS, cgroup/host pressure, save-time/multi-token peaks and throughput; do not infer savings from source deletion or vector microbenchmarks.

**Final decision: NO-GO now.** A conditional safe route exists through qualified independent stock regeneration on a suitable CPU host and fresh guarded headless validation, but neither recovered full fixtures nor new qualified oracle outputs exist yet. No inference was performed, no memory saving or new numerical equivalence is claimed, and QIES-086 remains uncommitted/unmerged.

Evidence: `qies075-data/logs/qies088/` contains the extended file/executable search, stock source history/digests, new stock build identity/commands/usage result, detailed memory census, admission check, worktree inventory and protected baseline verification. `qies088-investigation/` contains only the read-only census diagnostic and isolated stock build. The original baseline, QIES-086 patch/binaries, model/state files, numerical references and guards are preserved.
