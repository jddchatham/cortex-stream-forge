# QIES106–112A preservation

This archive preserves the QIES106–111 experimental CPU streaming sources and
QIES112A stability evidence. It does not merge them into production. QIES112's
unfinished generation prototype is separately labelled in `qies112/STATUS.md`.
The archive parent is the frozen QIES105 snapshot:
`14b3f7902864b02e1a84e5e6168a3cf99087e670`.

## Verified historical result

Three fresh QIES111 80-layer/two-position trials completed **960 native
subprocesses with exit status0 and12,480 byte-exact capture pairs**. Each trial has
320 process receipts and4,160 capture hashes. Every role/position/layer identity,
cleanup result, unchanged resource limit and OOM delta is represented in
`evidence/trial-N-processes.jsonl`. Capture SHA256 records identify the actual
historical data, not newly generated golden outputs.

The older QIES111 acceptance also remains explicit:5,148 captures and1,584
matching projection-weight hashes across its2/4/8/80-layer and boundary campaigns.
The additional QIES112A four-position extension has8,320 captures across positions
0–3; it reuses the third trial's first two positions. It is not a fourth fresh
trial or autoregressive language generation.

The historical alleged inference crash remains **INCONCLUSIVE**: its executable,
command and failure record were not recovered. Old desktop SIGBUS evidence does
not identify a QUORTEX engine cause. Synthetic stability is a PASS within the
reported fixed-capacity, repeated-synthetic-profile configuration, not real-weight
or production qualification. See the root QIES112A report.

## Verify this preservation without inference

```sh
python3 -I -B qies112a/reproduce.py
```

This reads source/package hashes, all960 process receipts and all12,480 capture
records. It executes no GGML arithmetic, downloads nothing and changes no guards.
`PACKAGE-SHA256.jsonl` is the authoritative published-file manifest. Historical
source/result manifests are retained under `evidence/historical`; paths and
private telemetry were redacted, so their original-file hashes are historical
identities, not replacements for the published-file manifest.

## Rebuild and reproduce on suitable Linux hardware

Requirements: Linux x86-64, C++17 compiler, CMake, make, OpenSSL development
headers, OpenMP, Python3.12+, working bubblewrap user namespaces, sufficient
persistent disk/swap and the original model-free resource headroom. Historical
workers used GCC16.2.1, `-std=c++17 -O2`, one numerical CPU thread, native Release
GGML SDKs. Full sanitized compiler commands and library/binary identities are
preserved. Native instruction selection can differ across machines; never relax
comparison criteria to conceal a changed result.

Build the inherited pinned CPU SDK from source. This fetches stock **source
only**, not models; offline `--stock-source` is also supported. No NumPy is needed
for the SDK `build` command or the QIES111 streaming test.

```sh
unset LD_LIBRARY_PATH
python3 -I -B qies105/replay.py build --jobs 2 --out /persistent/qies112a-sdk
export QIES106_SDK_STATE=/persistent/qies112a-sdk/state.json
python3 -I -B qies112a/reproduce.py --run --trials 3 --four-positions \
  --out /persistent/qies112a-new-results
```

Use fresh output/checkouts; old outputs are never deleted. The wrapper invokes
unchanged QIES111 campaign/transaction/launch APIs, the preserved diagnostic fault
and interruption functions, and the original sanitizer source harness. Only
historical location/discovery bindings are relocated; numerical code, ordering,
comparisons and resource enforcement are unchanged. The optional `--run` path is
a prepared source reproduction procedure; this preservation task validates the
non-inference verification path and does **not** rerun numerical workloads.

The SDK source revisions, CPU-only options and source-archive checksum are in
`qies105/README.md`. Stock revision:
`240d68fcfe0c2a1f298c8e18ca181368b1914259`; baseline engine ancestor:
`8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. The source builder verifies all4,084
stock source hashes. Rebuilt ELF hashes may differ; historical original ELF
hashes are provenance records, not promises about a new build.

Real execution remains locked: the unchanged6GiB MemAvailable requirement and
QIES104 unconditional unqualified-runtime refusal still apply. Synthetic workers
retain the existing separate model-free monitoring and sealed no-model namespace.
No model weights are needed or mounted. Private CPU SDK build products must not
be exposed as unguarded real-model runtime interfaces.

## Publication scope and privacy

Numerical source and original test-harness files are preserved byte-for-byte.
Reports/JSON evidence have path substitutions and removal of hostnames, machine/
boot IDs, process IDs, raw cgroup paths and file device/inode identities. No core
dump or raw system journal is published. `PROVENANCE.json` distinguishes original
hashes, published hashes and transformations. Reports describe the original local
diagnostics; raw machine logs remain local.

Excluded: GGUFs, model shards, compiled executables/shared libraries, raw generated
state/tensor captures, build/cache directories and the old nested preservation
tarballs that contain ELF files. Their source and relevant hashes are retained.
Original historical launchers remain available as source, but some require those
unshipped binary archives or old machine-local layouts; use the relocatable entry
point above for the80-layer stability reproduction.

No existing branch/tag, production source, memory guard or frozen milestone was
changed. New archive refs preserve experimental work; they do not qualify or
install a production engine.

Two preserved historical build scripts (`qies111/build.py`, `qies112/build.py`)
have an extra blank line at EOF. Their bytes are intentionally unchanged;
Git's optional blank-at-EOF warning is not a source-integrity failure.
