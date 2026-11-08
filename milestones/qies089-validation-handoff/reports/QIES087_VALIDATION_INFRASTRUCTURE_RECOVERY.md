# QIES-087 - Validation infrastructure recovery

Date: 2026-11-07. **NO-GO for QIES-086 inference validation.** NumPy and reproducible non-inference builds are recovered. Authoritative full golden fixtures remain unavailable, and the current desktop fails the unchanged 6 GiB memory admission gate. No inference was attempted or golden data regenerated.

## 1. Exact RAM gate and origin

The gate was proposed in `QIES075_NEW_LAPTOP_INFERENCE_READINESS.md:92` as a conservative new-laptop experimental admission policy, using the QIES-072 reference footprint as context. QIES-077 adopted it and tested the guard. It is **not** a proved intrinsic model minimum, not a requirement encoded in QUORTEX tensor mathematics, and not a test of physical MemTotal.

Implementation in `qies077-guard.py`:

- `G = 1024**3`; `Policy.entry_ram = 6*G` at line 24.
- `host()` at lines 58-65 reads `/proc/meminfo`, converts kB to bytes and returns `mem['MemAvailable']` as `available`.
- `run_guard()` checks available against policy.entry_ram before launching its child, around line 116.
- CLI `--check` and the CLI launch path separately enforce the literal 6*G requirement, around lines 235-239.
- The exact accepted boundary is **MemAvailable >= 6,442,450,944 bytes**. Total physical RAM is approximately 7.63 GiB; swap does not satisfy this comparison. MemAvailable already estimates reclaimable memory, so adding cached-file RSS to it would double-count headroom.

The QIES-079 measurement and QIES-080 launcher preserve this screen. The original guard SHA256 remains `ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2`.

Current resource-only check (`logs/qies087/entry-check.json`):

| Metric | Bytes / observation |
|---|---:|
| MemAvailable | 4,033,200,128 = 3.756 GiB |
| Required available | 6,442,450,944 = 6 GiB |
| SwapFree | 16,386,322,432 = 15.261 GiB |
| SSD available | 114,392,690,688 = 106.536 GiB |
| Memory full PSI avg10 | 0.0% |
| Exit / readiness | 1 / false |

This is a preflight refusal, not a runtime safety breach. SDDM and the graphical session are currently active. No applications or services were stopped.

### Which tests need it?

Full-model baseline/candidate generation and all real-model QIES-086 comparisons must retain the original admission and runtime safety policy. The present evidence does not establish a lower-memory model configuration for the changed chat transaction path. A shorter token count alone is not an independently validated lower-memory configuration.

File hashing, source/ownership audits, syntax checks, compilation, NumPy comparisons of existing files and CLI `--help` do not evaluate model layers and can run without passing an inference admission check. They do not bypass it: they never enter an inference launch path.

QIES-083/084/085 validated bounded file/vector diagnostics under separate 512 MiB zero-swap limits with 3 GiB diagnostic admission and 2 GiB host reserve. Those experiments do **not** validate smaller model inference. The existing QIES-077 harmless guard test harness intentionally uses fixture-only policy overrides and creates fixed historical log paths; it was not rerun, no thresholds were altered, and its old 15-test result was not presented as fresh QIES-087 validation.

### Can the laptop pass without system configuration changes?

Historically yes, after user-controlled graphical shutdown. The actual QIES-080/081 preflight monitor recorded **7,048,679,424 bytes = 6.565 GiB MemAvailable**, before the successful independent run. QIES-080 also records the user's reversible TTY/SDDM recovery test. This is evidence that this laptop previously passed the unchanged gate without permanent configuration changes, not a guarantee of today's headless availability.

The supported route is to save work, use the already-tested local TTY, temporarily stop the graphical session as documented in QIES-079/080, and repeat the unchanged measurement and full guard preflight. That is a user-controlled future action. This investigation neither stopped the desktop nor enabled SSH, changed boot targets, swap/zram, kernel settings or services. If a fresh headless check fails, stop; do not lower the gate or infer a pass from historical measurements.

## 2. Golden fixture recovery

### Search performed

- Readable files under `/home/user` (including project/work/download locations), `/mnt`, `/media`, `/opt` and the package cache; no attached external fixture volume was identified. Common dependency/build/cache trees were excluded from filename traversal; Git contents were examined separately.
- Original and experimental worktrees, local branches and remote-tracking refs.
- All **17 reachable local commits**, inspecting full tracked trees for layer outputs, stock logits, golden files, binary arrays and archives/checksum manifests. No matching golden-data blobs were found. Read-only fsck reported no unreachable objects to recover.
- Remote refs: the relevant 48-layer, multi-token, continuous-generation and first-answer branch heads correspond to commits already inspected locally. The additional remote `optimization/lean-lifecycle` tree was inspected through GitHub: 3,046 entries, not truncated, no matching golden tensor/archive entries.
- GitHub releases: empty list. Public Actions artifacts: total_count=0. The connector did not support the repository-wide artifacts endpoint, so that read used the public GitHub API directly. No artifacts or source branches were modified.

Search coverage is not a claim to have examined every unrelated remote branch's historical objects or inaccessible/unmounted storage. One denied path was a package-manager temporary download directory, recorded in `filesystem-search.json`. No readable archive candidate was found in the selected local scan. The complete search scope/results are retained in the evidence directory.

The [published QIES-072 report](https://github.com/jddchatham/cortex-stream-forge/blob/8fac93bb2bfc4b2d7069c4044f2f71cd25f80388/docs/QIES072_NONTHINKING_FIRST_ANSWER.md) explicitly describes stock and corrected QUORTEX run outputs as retained locally and not published. The reports/scripts are present; those statements are not substitutes for the numerical fixtures.

### Available authoritative local subset

The original QIES-081 selection arrays were found in place and verified against the protected QIES-082 hashes. They were neither moved nor rewritten:

| File under qies075-data/outputs/20261107-083141-efe63add | SHA256 | Length / interpretation |
|---|---|---|
| logits-select-0.bin | bb796a4131b858e2a692ab8e7faf2dedc14fcc269272c056bfece80e52dd0cdf | 993,280 bytes, 248,320 finite little-endian F32, argmax 19 |
| logits-select-1.bin | 6d82281115bf3d3a48606bb4f4915c7f2fd252f2c48343306123932820549fdf | 993,280 bytes, 248,320 finite little-endian F32, argmax 248046 |

These are historically verified **baseline** outputs, not fresh experimental outputs or a complete stock oracle. The 48 original persistent state files also remain intact, with protected size/mtime inventory matching. Such snapshots alone do not supply a resume oracle, per-layer golden output or every earlier prompt-position state.

### Missing bundle and provenance

Full validation still needs the original stock per-layer captures and consecutive-token logits used by each existing suite, including:

- Stock layer outputs `hc_combine-0.bin` through `hc_combine-47.bin` for the suite's exact token/position protocols. The scripts refer to directories such as dump-stock-off2, dump-stockref and dump-stock2tok beneath `/home/user/quortex-diag-20261103`.
- Full stock logits for both positions in the two-token suite, plus QIES-072's complete prompt-prefix/answer/EOG oracle vectors for the non-thinking prompt protocol. One-token milestone and 26-token QIES-072 prompt fixtures must not be mixed.
- Checkpoint/resume and rollback/retry expected serialized state/output evidence, or an independently audited harness/oracle for those checks.
- A trusted manifest of complete-file SHA256, lengths/shapes/dtypes, model shard hashes, index identity, token sequence/positions/sequence IDs, context configuration, stock source/binary identity and capture protocol. Per-layer original hashes were not recoverable here; a new untrusted candidate file cannot become golden merely by hashing it.

The published QIES-072 note itself records a stock source-overlay provenance limitation; preserve that qualification when recovering the original bundle. Request an owner-provided original-machine/archive export with its original manifest, then verify it before comparison. Do not regenerate golden data from QIES-086 or populate missing files with the two local selection arrays.

**Affected numerical validation remains stopped.** Missing-input diagnostics below do not constitute passing or mismatching tensor comparisons.

## 3. NumPy recovery

The existing comparison scripts import NumPy but do not pin a version. Repository requirements that pin numpy~=2.2.6 belong to converters/benchmarks/GUI tooling, not a specific golden-regression environment. No original regression Python/NumPy lock was found. Only system Python **3.14.8** was available; no compatible local environment/wheel/package was found by the recorded search.

Created the isolated environment:

```text
/home/user/PandaCode Bots/meatball-84ccea00/qies087-validation/venv
Python 3.14.8
NumPy 2.3.4
```

[NumPy 2.3.4 on PyPI](https://pypi.org/project/numpy/2.3.4/) provides the matching cp314 Linux x86_64 wheel. The official PyPI JSON metadata, URL and digest are saved in `numpy-pypi-metadata.json`. Selected wheel:

```text
numpy-2.3.4-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
SHA256 d5e081bc082825f8b139f9e9fe42942cb4054524598aaeb177ff476cc76d09d2
```

The venv was created with `python3 -m venv --without-pip`; uv installed only NumPy from the public PyPI index using a binary wheel, with configuration disabled and a workspace-local cache. No converter, Torch, GPU dependency or system pip/package installation was performed. A hash-pinned requirement for this platform is saved at `qies087-validation/requirements-numpy.txt`. This selected compatible environment is not claimed to recreate an unknown original NumPy version.

**Tool bootstrap disclosure:** invoking the pre-existing mise uv shim for its version caused mise to automatically install uv 0.13.7 under `/home/user/.local/share/mise/installs/uv/`. Subsequent installation used that actual binary. This was an additional user-local tool/cache change outside the dedicated validation directory; no system Python/global packages or OS settings were changed. The installed NumPy imports from the venv, system-site packages are excluded, and system Python still reports no NumPy installation.

Use the isolated interpreter directly; activation is unnecessary:

```sh
cd '/home/user/PandaCode Bots/meatball-84ccea00'
./qies087-validation/venv/bin/python -I -B qies087-validation/non_inference_checks.py
```

This checks existing files and emits fresh diagnostic logs; it does not launch inference. NumPy identity/difference tests passed, including rejection of a one-ULP float change. The model scripts' np.array_equal behavior remains unchanged. Full byte/length checks are still required for the bit-exact acceptance standard; numerical equality alone does not distinguish every representation, such as signed zero.

## 4. Non-inference validation results

| Check | Result |
|---|---|
| All 12 protected executable/library/script/index/selection-array hashes | PASS |
| All 48 baseline state sizes/mtimes | PASS |
| Baseline executable identity | PASS: 329ebd647b04a2d6464461ddd94fc07a2691a948b2918d320ac1eb3703f9c101 |
| Fresh baseline CLI rebuild with verified common libraries | PASS, same executable SHA256 |
| Fresh QIES-086 candidate CLI rebuild with verified common libraries | PASS: 952eb2e92a657fba693b27dc9bd9fd55a86c0003848a0cd2453adc25d92f70e2, identical to QIES-086 candidate |
| Fresh CLI --help startup/linkage | PASS for both; audited early exit before model initialization |
| Exact source difference | PASS: only staged_state member and copy assignment deleted |
| Ownership/destruction audit | PASS statically; required rollback owner retained, no staged-state reader/export found |
| Serialization/persistent-state code | Unchanged by exact source comparison |
| Python AST and launcher/build-shell syntax | PASS |
| git diff --check | PASS |
| Isolated NumPy import / comparison smoke checks | PASS |
| Existing 48-layer comparator | Import recovered; 49 MISSING entries, exit 1; BLOCKED |
| Existing two-token comparator | Import recovered; 98 MISSING entries, exit 1; BLOCKED |
| Real 48-layer arithmetic, continuity, rollback/resume, sanitizer model tests | NOT RUN |
| Inference memory savings, save-time peak, throughput | NOT MEASURED |

Fresh objects/executables are under `qies087-validation/rebuild/`, using QIES-086's recorded compile/link commands with only new output paths. Both use each checkout's matching header roots and the same verified baseline shared libraries. The originals and QIES-086 worktree/build files were not overwritten. Commands, compiler warnings and identity results are logged.

Static ownership finding: txc.staged_state's only chat-source references were its declaration and assignment from result.state_before. The transaction-scoped vector previously allocated/copied and destructed at function return. No rollback, checkpoint, graph or diagnostic read was found. result.state_before remains alive through rollback and caller save; current context and persistent state store remain authoritative. Removing the unused allocation has not been dynamically validated for inference, exception/recovery behavior or numerical equivalence here.

The untouched comparator invocations used the isolated interpreter, qies086 tag and 48 layers. They now reach file lookup and correctly stop on missing data. No tolerance was weakened or candidate-produced oracle substituted. The original CLI still has no cross-process resume command; an appropriate audited harness remains necessary for that required acceptance case.

## 5. Exact remaining prerequisites

1. Recover the trusted original fixture bundle and provenance described above, and verify complete file hashes, lengths and protocol metadata. Until then, stop affected golden tests. This task did not create a replacement oracle.
2. Obtain a fresh user-controlled headless TTY measurement that passes the unchanged 6 GiB MemAvailable, 4 GiB swap-free, 30 GiB disk-free and PSI admission checks. Preserve the existing runtime reserves, ancestor cgroup checks, OOM/pressure monitoring and process-tree cleanup. Restore the graphical session using the established recovery instructions if preflight fails.
3. Use the reproduced baseline/candidate pair with common verified libraries and the same IQ4_XS model/index, exact prompt, native non-thinking behavior, 16-token maximum, loader cache, 1 MiB expert arena and OMP_NUM_THREADS=1. Keep unique SSD-backed state/output/log directories for every trial.
4. Validate an adapter that submits either binary to the existing run_guard with its default policy; its CLI is intentionally pinned to baseline. Do not overwrite the baseline binary or introduce policy overrides to run the candidate.
5. Audit and run small changed-path rollback/retry and serializer/checkpoint-resume tests with the authoritative oracle, then the full golden comparisons and three comparable guarded trials per build. Add reviewed external/diagnostic memory telemetry for anonymous RSS/PSS, cgroup/host pressure and transaction/save peaks without modifying numerical operations. Stop at any mismatch, corruption or guard violation.
6. Recheck space for at least six retained state directories (approximately 31.9 GiB logical), logs/captures/checkpoint copies and the original reserve. Judge memory and performance only from those actual paired trials.

**Final recommendation: NO-GO for inference now; GO for fixture transfer and user-controlled preflight preparation.** NumPy is no longer a blocker, baseline/candidate build reproduction is established, and the two-line experiment remains isolated and uncommitted. Golden coverage and current RAM admission remain blockers. No inference, runtime memory-saving claim, numerical-equivalence claim, commit, merge, guard change or production behavior change occurred.

## Evidence directory

`qies075-data/logs/qies087/` contains filesystem/Git/remote fixture searches, release/artifact results, official NumPy metadata/install log, exact rebuild commands/output/hashes, fresh guard admission result, non-inference check results and comparator missing-input logs. `qies087-validation/` contains the isolated environment/cache, pinned requirement, read-only validation script and fresh CLI build artifacts. Existing reports, state/model files and QIES-086 artifacts remain preserved.
