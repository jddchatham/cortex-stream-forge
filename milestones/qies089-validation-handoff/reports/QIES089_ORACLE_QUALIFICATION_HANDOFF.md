# QIES-089 - Oracle qualification and validation handoff

Date: 2026-11-08. **GO for portable handoff and offline build/tool review. NO-GO for runtime oracle qualification, QIES-086 acceptance or merging.** No inference, current-desktop memory-gate retry, system/guard modification, commit or merge occurred.

## Delivered package

- Directory: `qies089-handoff/`
- Portable archive: `qies089-handoff.tar.gz`
- Archive SHA256: `bdd2803102039e2cf98c447c40f884842bcebe4c99585756887198bedbd8c22f`
- Entry point: `qies089-handoff/README.md`
- File identities: `manifest.json` and `SHA256SUMS` inside the package.

The package contains 34 files: a complete 17-commit Git bundle for the published branch, the exact uncommitted QIES-086 patch, two small historical baseline selection arrays, unchanged guards, historical reports, model/input identities, acceptance states, portable build-only tooling and a strict complete-file byte comparator. No GGUF weights, multi-gigabyte state directory, local virtual environment, host-specific binaries or dependencies are copied into the archive.

The source bundle is self-contained: it restores commit `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`, branch `feature/flashnext-first-text-generation`. Git bundle verification, fresh clone and patch-applicability checks passed. The archive was extracted to a separate directory and every shipped checksum passed again. Integrity hashes protect transport; they do not make unqualified outputs authoritative.

## Oracle provenance and non-inference qualification

| Item | Verified identity / status |
|---|---|
| Published stock source | examples/flashnext-stock-two-layer.cpp at the published commit |
| Stock source SHA256 | e169159b26d68409c178bc8fbe496fc5582e240c8c9ff6804e78ac2385c8b35f |
| Rebuilt stock candidate SHA256 | 6a29bf69f5976e9f4bd0e6df8e351336e4b52ca93d492d7e66bd82ee8cb22963 |
| Historical stock executable SHA256 | eda8ceefc94f296a66d217b79d0aba64317b5f24295b822e086e7d3736169fb2 - missing |
| Older stock source overlay | Only prefix 4faa9bd2... reported; exact overlay not recovered |
| Baseline CLI SHA256 | 329ebd647b04a2d6464461ddd94fc07a2691a948b2918d320ac1eb3703f9c101 |
| Controlled QIES-086 candidate SHA256 | 952eb2e92a657fba693b27dc9bd9fd55a86c0003848a0cd2453adc25d92f70e2 |
| Candidate patch SHA256 | 50c0475c7cfd731b4655e8de44651c997b4f69a15fe023902325c7ea0a3737cd |

The candidate is base commit plus two source deletions: the unread staged_state member and its copy assignment. No candidate commit was created. The required rollback owner, state serializers, context behavior and persistent handoff code are unchanged. Exact patch/source identity is checked before the portable helper compiles anything.

### Compiler and configuration

The current reproduction used GCC 16.2.1 20260810 and CMake 4.4.4, Release `-O3 -DNDEBUG`; CPU defines include GGML_USE_CPU/GGML_SHARED/LLAMA_SHARED. GGML_CUDA=OFF, GGML_CPU_REPACK=ON; examples ON and tools/server/app/tests OFF. The shared CPU library uses `-std=gnu++17 -fPIC -march=native -fopenmp` plus the recorded warning flags. The original compile/link commands and four dependent-library hashes are shipped as evidence. Baseline, experiment and rebuilt stock are linked against that **same** verified engine-library set.

Host: x86_64 Intel Core i5-7200U, two cores/four threads. Native ISA selection, ABI, library RPATH and compiler differences prevent treating these ELF identities as portable runtime guarantees. The package therefore ships source and build instructions, not runnable original-host binaries. Rebuild on the receiving host and record its identities; different bytes there are not automatically a numerical mismatch or proof of equivalence.

The freshly built baseline, candidate and stock reproduced all three known executable hashes locally. The helper does not run those executables. QIES-088's stock no-argument usage check is historical startup evidence; no new stock invocation or model load was needed here.

### Architecture compatibility and independence

Source support is for the exact QWEN4EXP/Qwen3.8 Flash-Next IQ4_XS model: 48 layers, 2560 hidden dimensions, four HC streams, 512 experts/top-10, recurrent layers interleaved with full attention at 3,7,...47 and the established PLE/indexer state. The source runner sets n_gpu_layers=0 and executes the normal full stock graph, clearing relevant streamed-layer/segment overrides. It is independent of the changed staged-state transaction coordinator, while sharing the pinned baseline numerical engine.

This is static compatibility, not a successful model load or runtime qualification. The rebuilt stock executable is not the historical binary. Original stock dependent libraries, complete build configuration and older source overlay remain incompletely identified. Historical REPACK ON/OFF agreement is reported evidence, not an independently rerun qualification of this build. Actual stock context/backend thread defaults and normal loader/mmap/repack allocation must be audited and recorded; OMP_NUM_THREADS=1 and QUORTEX's 1 MiB arena are not sufficient descriptions of stock memory behavior.

**Qualification ceiling reached without inference:** source identity, source-path independence, static architecture support, reproducible compilation and dependent-library identity. Runtime equivalence, deterministic reference output, full-layer capture validity, continuity and recovery remain blocked.

## Historical evidence versus recovered numerical files

The following original reports are included unmodified and hashed:

| Historical report | Reported result | Independently recovered coverage |
|---|---|---|
| FULL_48_LAYER_REPORT.md, 2026-11-03 | Token 248045 at position 0; 48 layer outputs and 248,320 logits exact; argmax 846, reported logit 21.0668354; REPACK ON/OFF agreement | Report and source history only; raw stock layer/logit fixture set absent |
| MULTITOKEN_2TOKEN_REPORT.md, 2026-11-04 | Inputs 248045 then 846; all 48 layers exact at both positions; next argmax 198, reported logit 26.4632492; state handoff and determinism checks | Report and state inventory only; raw stock two-token fixture set absent |
| MULTITOKEN_STATE_INVENTORY.md | KV/cell/token metadata, DeltaNet/conv state, QSA indexer, PLE predecessor handling and restore mechanism documented | Source-supported historical audit; no new dynamic resume validation |
| QIES072_NONTHINKING_FIRST_ANSWER.md | Exact 26-token non-thinking prompt, answer 19 (`4`), then native EOG248046; 28 complete logits comparisons exact | Report plus two independent-laptop QUORTEX baseline selection arrays; not all original stock prefix/layer vectors |

Historical memory numbers, including wrapper-sized RSS figures in early milestone notes, are not used to change the current resource gate. The archived reports' claims and comparison methods are preserved as reported, not promoted to newly measured results.

The two copied anchors are exactly 993,280 bytes each:

- anchors/logits-select-0.bin: `bb796a4131b858e2a692ab8e7faf2dedc14fcc269272c056bfece80e52dd0cdf`
- anchors/logits-select-1.bin: `6d82281115bf3d3a48606bb4f4915c7f2fd252f2c48343306123932820549fdf`

They remain historical baseline anchors, not candidate-produced or complete independent golden data. The full original oracle fixtures are still missing. Source/model/token identities cannot reconstruct their bytes.

## Portable validation contents and instructions

The package README supplies commands and protocols for each required area:

1. **Exact revisions/builds:** clone the Git bundle into a new persistent RUNROOT, create a separate candidate worktree at the same commit, apply only the shipped patch, configure the documented CPU-only CMake build, and invoke tools/build_pair.py. It recompiles the two CLIs and stock source against common baseline libraries and records all identities. Its output directory must be new. The package's pandacode-run.sh is a build-only wrapper requiring explicit source/output directories.
2. **Baseline/patch identity:** complete source history, exact unified patch, source/executable/library hashes and previous build metadata. Existing verified binary paths are never overwritten.
3. **Model/tokenizer identity:** protocol/inputs.json contains all three complete model SHA256s, published index identity and architecture/shape constraints. The embedded GGUF tokenizer is pinned transitively by those model hashes; no replacement tokenizer is permitted. Verify any receiving machine's weights before use.
4. **Deterministic inputs:** exact 112-byte native non-thinking formatted prompt, its SHA256, all 26 token IDs, sequence/context/batch settings, maximum 16 generated tokens, expected answer/EOG IDs, and the distinct older two-token protocol. Actual context thread defaults must be recorded.
5. **Oracle qualification:** use an independently reviewed stock build with frozen identities, clean environment and unchanged guarded admission. Verify the two retained full-vector anchors, repeat for determinism and separately prove diagnostic captures do not change clean stock logits. A build cannot mark a reference QUALIFIED.
6. **Golden generation:** recover authentic originals if available; otherwise label all new output NEW_STOCK_REFERENCE with a new identity/date. Capture complete stock layers and logits, finite/shape checks, tokens/positions, configuration, resource result and per-file hashes. Never use experimental output as golden.
7. **Exact comparison:** tools/compare_exact.py requires qualified-manifest status, identity/receipt fields, complete coverage, correct lengths/reference hashes and exact bytes. It rejects signed-zero changes, truncation, extra bytes, duplicate/missing IDs and same-file aliases. For the full prompt it requires 1,324 files: 48 layer handoffs at positions 0..26 plus 28 logits comparisons. The separate two-token case requires 98 files. This checker does not certify a caller's provenance statements; review the actual qualification receipt/logs before use. The supplied empty template remains BLOCKED and fails closed.
8. **Checkpoint/resume:** a detailed serialize/restore/uninterrupted-versus-resumed protocol covers hybrid state, token/position metadata, coordinator history/HC handoff as appropriate, multiple further tokens, rollback/retry, I/O failures and incomplete checkpoints. Neither current CLI has a cross-process resume switch. The required harness remains missing, not replaced with an invented flag or synthetic-state pass.
9. **Memory/performance:** at least three trials per build, alternating order and common dependencies/inputs/guard conditions; fresh state directories; anonymous RSS/PSS, peak tree RSS, cgroup/host pressure, transaction/save/multi-token peaks, I/O/faults, wall time and throughput. Capture-correctness and performance workloads are separate because --capture-dir adds prefix output projections. Sampling/observer qualification, uncertainty and acceptance criteria are explicit. Six retained state sets alone need about 31.9 GiB logical, plus captures/checkpoints/logs and reserves.
10. **Rollback:** stop selecting the candidate, preserve the untouched baseline and all output evidence; reverse-applicability check for the two-line patch is provided. No deletion, commit, merge, automatic service action or automatic reverse patch is performed.

### Guard portability and immutable policy

The original guard and memory measurement scripts are included byte-for-byte. Deploy them at the intended RUNROOT, alongside qies075-data and the baseline checkout: they derive ROOT from their file location. Do not execute them under the package's nested guards directory and assume its paths are a deployment. Create a local model-index copy with only model_dir adapted, preserving the published original and recording the adapted identity. Rebuild moved binaries rather than relying on original-host absolute RPATH.

A target must provide compatible Linux/systemd-user/cgroup-v2 accounting and ancestor headroom. Existing guard admission remains **MemAvailable >=6 GiB**, free swap >=4 GiB, SSD >=30 GiB and PSI full avg10 <1%; existing runtime floors, RSS ceiling, deadline, OOM/PSI and cleanup semantics are unchanged. The guard CLI pins baseline; a separately reviewed stock/candidate adapter must call the same run_guard/default policy and preserve all checks. That adapter is not shipped as an assumed-valid launcher. On the intended host, validate it with harmless children in a genuinely passing environment; no lowering of policies is part of this handoff.

No current-desktop memory check was repeated. Target admission is explicitly NOT OBSERVED. Historical headless success does not authorize launch on a new session or machine.

## Non-inference package validation

- Git bundle complete-history verification: PASS.
- Fresh restore at the recorded commit and exact patch --check: PASS.
- Build-only helper: PASS; baseline/candidate/stock hashes reproduced against common libraries.
- Final helper source/patch preconditions and exclusive-output behavior: checked; existing output correctly rejected.
- Strict comparator: six synthetic byte tests plus required-coverage counts passed. These were synthetic tooling tests, not generated model golden files.
- BLOCKED oracle manifest: correctly refused; no false pass with missing fixtures.
- Python AST and POSIX build-wrapper syntax: PASS.
- All packaged checksums before and after archive extraction: PASS.
- Original protected executable/library/guard/index/reference hashes and all 48 state sizes/mtimes: PASS. QIES-086's exact patch is unchanged.

Test logs and local reproduction objects remain in `qies089-checks/`; only small identity/test evidence is included in the handoff. No model data or inference launch was involved.

## Gates and remaining blockers

`protocol/acceptance-gates.json` is the explicit status record. Passed: static source identity and non-inference build reproduction. Blocked: receiving-host model/admission identity, runtime oracle qualification, complete original/new golden data, reviewed guard adapter, all layer/logit comparisons, multi-token continuity, checkpoint/resume and rollback/invalid-access harnesses, phase/PSS observer, three trials per build and memory/performance acceptance.

A target host must satisfy the unchanged gate before **any** inference. A larger-memory dedicated CPU environment or the original verified reference environment is preferable for stock qualification; a fresh user-controlled headless laptop session is conditional on measured admission and sufficient stock-loader headroom. Merely passing the gate is not proof of completion or numerical correctness.

**Decision:** handoff is ready for transfer/review and reproducible offline building. Oracle remains SOURCE/BUILD-QUALIFIED ONLY, NOT RUNTIME-QUALIFIED. QIES-086 remains NO-GO for acceptance/merge until every missing runtime gate passes. No memory-saving, throughput or new bit-exact inference claim is made.
