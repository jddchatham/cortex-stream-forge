# QIES110 reproduction

Experimental, CPU-only, storage adapter plus **synthetic** layer validation. The
real execution boundary remains unconditionally locked by unchanged QIES104.
No command here authorizes real-weight arithmetic.

From this worktree root, with the preserved QIES105 source-built SDK available:

```sh
env -u LD_LIBRARY_PATH python3 -I -B qies109/run.py
env -u LD_LIBRARY_PATH ./pandacode-run.sh
# Or the short, independently compared synthetic smoke test:
env -u LD_LIBRARY_PATH ./pandacode-run.sh --smoke
```

`qies110/build.py` accepts `QIES106_SDK_STATE` pointing to the QIES105 build state
JSON; otherwise it uses the preserved sibling `qies105-publication` SDK. Source
and library SHA-256 checks are mandatory. Private numerical workers run only in
the sealed bwrap namespace, with one CPU thread. They cannot access the GGUF.
A ~546 MB synthetic raw fixture is generated once on the persistent SSD using
bounded memory, then mounted read-only. The independent numerical worker still
generates its own weights procedurally.

Run data-only and sanitizer tests **after the build completes**:

```sh
env -u LD_LIBRARY_PATH python3 -I -B qies110/storage_tests.py
env -u LD_LIBRARY_PATH python3 -I -B qies110/sanitizer_tests.py
```

For a separately preserved storage run, set `QIES110_STORAGE_RESULTS` to a new
workspace-relative results-directory name (e.g. `storage-results-final`).

Those two commands refuse to replace an existing results directory. Preserve
existing artifacts and repeat in a fresh isolated worktree when required.
`run.py --resume PATH` resumes an interrupted numerical campaign only when all
build/source identities still match. It verifies cached output/checkpoint hashes.
Never delete or edit a failed result to turn a failure into a pass.

Files:

- `range_source.h`: shared bounded read/row source; no compute/backend APIs.
- `adapter.py`: QIES101 full-descriptor validation and source-plan creation.
- `stream.cpp`: QIES109 kernels consuming the row source; REAL sources rejected.
- `reference.cpp`: preserved independent stock-GGML synthetic graph oracle.
- `fixture_writer.cpp`: bounded synthetic raw-file generator.
- `storage.cpp`: no-arithmetic hashing/row-schedule validator.
- `reference_hash.cpp`: independent stock GGUF parser and bounded byte reader.
- `adapter_faults.cpp`, `sanitizer_tests.py`: synthetic read/lifetime faults.
- `run.py`, `build.py`: sealed numerical campaign and pinned SDK builds.
- `storage-results-final/summary.json`: independent real-byte storage results.
- `sanitizer-results-final/summary.json`: instrumented data-path test results.
- `measurements.json`, `final-integrity.json`, `SOURCE-SHA256.json`: evidence.

Real data requires the exact external model and the QIES102 verification receipt.
The current receipt is stat-bound and selected tensors are freshly hashed;
QIES110 does not claim a new whole-model SHA-256 pass. On another machine or any
identity change, rehash the complete model before creating a new trusted plan.

A manifest checksum is an expected identity, not an implicit verification on
every row read. Full tensor hashing is performed by the storage validator. Real
numerical use remains blocked pending independent execution and live guard
qualification; the adapter is not a production loader or a hostile-input sandbox.
