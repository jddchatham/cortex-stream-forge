# QIES101 — Bounded Real-Tensor Extraction

## Decision

**GO for QIES102 bounded execution-harness integration and qualification work. NO-GO for real-model inference now.** All selected raw tensor extractions agree exactly with an independent stock GGUF path, and fault tests pass. This qualifies byte extraction, not Ricursive numerical execution, a real-model oracle, or the live guard adapter.

Worktree: `/home/user/PandaCode Bots/meatball-84ccea00/qies098-worktree`; branch `experimental/ripple-65b-q4`; HEAD `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. No commits, merges, pushes, production/FlashNext changes, QIES086 application, GPU use, system changes or resource-guard changes occurred. No real tensor arithmetic or inference was executed.

## Input and provenance verification

A **fresh complete SHA-256 pass** read all **40,426,079,712 bytes** using 1 MiB reads. Expected and actual SHA-256:

`9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc`

The published value was independently retrieved from the [pinned Hugging Face repository metadata](https://huggingface.co/api/models/AtomicChat/Ripple-65B-Preview-GGUF/revision/df550a7178a87bbfb6ee2bdbd37603c553d536ee?blobs=true), revision `df550a7178a87bbfb6ee2bdbd37603c553d536ee`. The exact filename, LFS size and SHA agree. Device/inode/length/mtime/ctime matched before and after hashing. Evidence: `qies101/published-model-metadata.json`, `model-hash-result.json`, `model-sha256.txt`. The hash pass took 149.91 seconds. The GGUF remains at `../qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf` and was opened read-only.

**All 1,203 descriptors pass three-way comparison:** the existing validated QIES098 index, the unchanged original QUORTEX descriptor parser, and an independent stock GGUF parser. Names, absolute offsets, byte lengths, type IDs and canonical dimensions match. Stock exposes four dimensions padded with unit dimensions; comparison preserves the index's rank and checks its padded GGML dimensions. The original parser's native one-/two-dimensional fields are checked separately. No descriptor was inferred from extracted bytes. Evidence: `descriptor-verification.json`, `baseline-descriptors.tsv`, `reference-descriptors.tsv`.

Independent upstream source remains llama.cpp b12167, commit `240d68fcfe0c2a1f298c8e18ca181368b1914259`. All 4,084 stock source hashes and QIES100 executable/library identities were reverified. The original descriptor checker matches its QIES098 SHA-256 `4f3d4444e7524d2c9254e81aadb38199084a0b9afb685a57b1dcd47a554b644e`.

The new standalone reference reader links only the verified stock GGML-base library, SHA-256 `f34fb8f64ea4efe5aa82894c3eaaa8a0cee38e63e0c31feafc1ecdb8afc3e519`; it does not link QUORTEX, libllama, or a GPU backend. Compiler is GCC 16.2.1; reader flags are C++17, `-O2 -Wall -Wextra`, with the existing Release stock library. Full build commands, executable/source/library hashes and linkage are in `reference-build-identity.json`, `reference-provenance.json` and `reference-ldd.log`. Current reference reader SHA-256 is `364b292c44071dc693916494b6bc94deb9a3df2714ed675a28602bdae0feb35d`.

The Git working tree already contained untracked QIES098–100 reports/tools and the maintained Run Script. QIES101 adds its own untracked tools, results and report. Tracked and staged diffs remain empty. `git-status-before.txt`, `git-status-after.txt`, `git-head.txt` and `preservation-checks.json` record this state. HEAD alone does not identify these uncommitted experimental sources; use the QIES101 source/evidence hash manifest. QIES099/QIES100 artifacts remain hash-identical except the intentionally maintained root Run Script, whose previous contents are preserved as `qies101/qies100-pandacode-run.sh`.

## Source changes and extraction contract

All new code is under `qies101/`:

- `hash_model.py`: pinned publication metadata lookup and fresh bounded full-file hashing.
- `extract.py`: strict index validation and `Extractor`, an indexed, read-only `pread` range reader.
- `reference_extract.cpp`: independently parsed stock GGUF ranges, descriptor export and streamed raw extraction.
- `build_reference.py`: pinned CPU reference-reader build.
- `verify_descriptors.py`: reproducible 1,203-descriptor cross-check and selection inventory.
- `run_index_extract.py`: verified-identity indexed extraction driver and isolated process RSS measurement.
- `compare.py`: serial independent extractions and complete byte-by-byte comparison.
- `test_extract.py`, `test_reference_faults.py`: fault tests on synthetic files.
- `verify_saved.py`: repeatable bounded verification of preserved extraction artifacts.

The root `pandacode-run.sh` now rebuilds only the tiny range reader, verifies metadata, runs synthetic faults and checks saved byte comparisons. It never runs tensor arithmetic, rehashes the entire model, re-extracts weights, or launches inference.

The index reader caps JSON input at 2 MiB, rejects duplicate JSON keys, bounds tensor counts and names, checks supported types and quantized block/row sizes, rejects invalid integer types and signed-offset overflow, validates shape products, requires alignment and consistent absolute/relative offsets and shard identity, checks every interval against file size, and rejects overlaps before reading any tensor payload. The GGUF data start is the previously validated 3,615,712 bytes, with 32-byte alignment. The real-model driver additionally requires the published SHA receipt and unchanged file identity.

Each requested tensor is copied unchanged using reads of at most 1 MiB. Partial reads and writes are completed; premature EOF and I/O failures fail the operation. No tensor conversion, dequantization, whole-file mapping or tensor-sized allocation occurs. Each output is created privately, synchronized and published with an exclusive hard link; existing outputs and preexisting partial files are preserved. Failure removes only the newly created partial file. Descriptor/file handles close on all tested failures. File identity is checked against both the opened descriptor and the named path before/after extraction, including inode replacement, timestamp/size changes and symlink refusal.

The reference receives **names only**, never QUORTEX offsets, lengths or shapes. It calls stock `gguf_init_from_file` with no tensor-data context, resolves offsets/types/dimensions/sizes through stock GGUF APIs, independently checks all ranges and overlaps, and reads the selected raw bytes through its own 1 MiB buffer. `/proc/self/fd/N` binds stock metadata parsing to the already opened read-only inode; final descriptor/path checks reject source replacement. Stock metadata allocation is measured on the verified model; this does not claim a universal adversarial-input memory bound for upstream's GGUF parser. Raw extraction buffers are bounded independently of tensor size.

## Real extracted inventory and byte comparisons

| Tensor | GGML shape | Type | Bytes | Result |
|---|---|---|---:|---|
| `blk.0.attn_q.weight` | [5120, 10240] | q4_K | 29,491,200 | Exact |
| `blk.0.ffn_down.weight` | [40960, 5120] | q5_K | 144,179,200 | Exact |
| `blk.40.attn_k.weight` | [5120, 1024] | q8_0 | 5,570,560 | Exact |
| `blk.40.ffn_gate.weight` | [5120, 40960] | q4_K | 117,964,800 | Exact |
| `blk.79.attn_output.weight` | [10240, 5120] | q6_K | 43,008,000 | Exact |
| `blk.79.ffn_up.weight` | [5120, 40960] | q5_K | 144,179,200 | Exact |
| `token_embd.weight` | [5120, 100352] | iq4_xs | 272,957,440 | Exact |
| `output_norm.weight` | [5120] | f32 | 20,480 | Exact |

**Eight complete tensors; 757,370,880 bytes per path; every byte matches.** This includes early layer 0, middle layer 40 and final layer 79, both attention and FFN weights, and all five quantization formats plus F32. Norm and embedding samples also exercise tiny and large payload sizes. No sampled-prefix-only comparison was accepted.

Full per-tensor SHA-256 values, offsets, types, dimensions and lengths are in `byte-comparison.json`; matching SHA-256 values supplement direct byte comparison rather than replace it. Exact raw outputs are preserved in `qies101/artifacts/index/` and `qies101/artifacts/stock/`. Together they consume 1,514,741,760 logical bytes (about 1.41 GiB), not two copies of the full GGUF. These large generated artifacts are not source files and must be excluded from any future Git preservation commit unless separately authorized.

The indexed path's logical tensor reads total exactly 757,370,880 bytes. The stock path additionally reads its own GGUF metadata. Linux filesystem readahead/writeback/cache may perform different physical I/O; user-space range counts are not physical SSD traffic measurements. The separate required full-model SHA pass deliberately read the entire file for integrity verification.

## Memory and execution evidence

| Process | Peak RSS | Scope |
|---|---:|---|
| Full-model SHA verifier | 27,472 KiB | Separate streaming integrity pass |
| Index extraction driver | 20,112 KiB | All eight actual tensors |
| Independent stock extraction | 16,940 KiB | All eight actual tensors |
| Direct byte comparator | 20,564 KiB | Complete extracted files |
| Synthetic index fault suite | 21,324 KiB | Fault cases only |

These are per-process Linux `ru_maxrss` measurements, not sums and not system/cgroup memory estimates. Requests are ≤1 MiB, but Python may briefly retain old/new read objects; the claim is bounded streaming, not a one-MiB whole-process cap. The largest tensor is 272,957,440 bytes and was never retained as a complete in-memory buffer.

The index extraction took 4.15 seconds; direct comparison took 5.95 seconds in this run. The prior full-file hash and filesystem cache affect these timings; no inference-speed or cold-storage performance claim follows. File-cache residency and buffered output pages are not included in process RSS, so these numbers do not imply equivalent system-wide RAM recovery.

## Fault tests

**38 index-path tests pass**, including malformed/duplicate/missing descriptors; invalid or boolean type/alignment values; wrong quantization/length/shard; unaligned rows and offsets; negative, overflowing or outside-file ranges; dimension overflow; overlaps; duplicate JSON keys and metadata budget; missing names; truncated reads/files; recoverable short reads/writes; injected read/write failures; no-overwrite and preexisting-partial preservation; identity changes before/during reading; inode replacement; source symlink refusal; bounded multi-chunk requests; and descriptor/temp-file cleanup without leaks.

**Four independent stock-path tests pass:** missing tensor, missing file, preservation of an existing output, and rejection of a truncated synthetic GGUF. Tests use a small copied synthetic fixture; the real GGUF and original artifacts are never modified. Results: `fault-results.json`, `reference-fault-results.json`, corresponding logs and `final-run-script.log`.

Extraction publication is not a crash-consistent checkpoint protocol. Directory metadata is not presented as a durable generation transaction; unexpected interruption can leave completed earlier outputs or a partial artifact, which must not be trusted without successful receipt/hash verification. No original state files are touched.

## Reproduction and runtime readiness

For the existing experimental package:

```sh
cd '/home/user/PandaCode Bots/meatball-84ccea00/qies098-worktree'
./pandacode-run.sh
```

The fresh complete integrity check can be explicitly repeated using `env -u LD_LIBRARY_PATH python3 -I -B qies101/hash_model.py`. Descriptor checks can be repeated with `qies101/verify_descriptors.py`. The one-shot `compare.py` deliberately requires new output directories and refuses to overwrite the preserved extraction run; use `verify_saved.py` for routine verification. It is not an inference launcher.

At the readiness snapshot, MemAvailable is **3.453 GiB**, free swap **14.262 GiB**, and SSD free **60.849 GiB**. The original **6 GiB MemAvailable admission gate fails**. A 30 GiB disk reserve was checked before planning both extracted copies and by the index driver during its writes. The mount is recorded in `mount.txt`. No resource-guard policy was changed or bypassed for inference. Bounded raw file I/O is not model inference and did not invoke the inference guard.

QIES102's smallest execution comparison remains **one real layer, one position, starting with layer 0**. Before it can run, require all of:

1. Fresh unchanged ≥6 GiB MemAvailable admission plus the original swap/disk/pressure and ancestor checks.
2. Independently qualified bounded stock real-layer execution, not merely this stock byte extractor. Preserve QIES100's matched ordinary CPU-buffer configuration (`use_extra_bufts=false`), one thread, non-flash attention and F16 KV.
3. Qualified live guard adapter, including successful harmless-child execution, descendants, failure and cleanup under unchanged thresholds. It remains pending; no qualification receipt was fabricated.
4. Matching model/tensor/input/binary/library identities, with strict range/shape/type validation before numerical execution.
5. Identical input/scaled `x0`, context, mask, KV layout and position, plus intermediate/final-output and canonical KV comparisons against independently generated stock artifacts.

The extraction API may now supply raw active-layer tensor bytes to an isolated test harness. It does not provide a real numerical oracle, persistent KV integration, checkpoint continuity, long-context/window rollover, layer scheduling validation or full-model correctness. Do not infer numerical equivalence from raw-byte equality.

**Final recommendation: GO for QIES102 integration/qualification development; NO-GO for actual real-layer inference until every prerequisite above passes.**
