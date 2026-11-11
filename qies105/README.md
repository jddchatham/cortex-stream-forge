# QIES105 source-only Ripple validation handoff

This package freezes QIES098–104. It does **not** enable real-weight arithmetic.
QIES104 native and Python admission deliberately reject real roles even above
6 GiB MemAvailable. Do not edit that lock or run retired QIES099–103 workers.
A future reviewed implementation and positive qualification are required before
one real layer can execute. More RAM alone is insufficient.

## Identity and preservation

- Archive branch: `archive/ripple65b-qies104`.
- Annotated tag: `ripple65b-qies104-frozen` (peels to the preservation commit).
- Unmodified engine ancestor: `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`.
- Original experimental branch: `experimental/ripple-65b-q4`; its HEAD was the
  same ancestor, with QIES098–104 material untracked. HEAD alone was not its identity.
- `frozen-source-manifest.json` preserves every selected historical source/report/
  evidence file by hash. Historical binary hashes describe the former laptop,
  not hashes promised for a rebuild. `guards/qies077-guard.py` is byte-identical
  to the original guard, SHA256
  `ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2`.
- Complete original baseline source is inherited in Git. No QIES086 patch, recovery
  prototype merge, FlashNext tracked source modification, or inference occurred.
- No weights, compiled workers, model tensors, SDK binaries, build caches or
  private credential files are shipped. Retired legacy executable artifacts are
  absent. Their source/permission/hash records are retained, **not restored**.

## Rebuild and run only synthetic tests

Requirements: Linux x86-64, C++17 compiler, CMake, make, OpenSSL development
headers/library, Python >=3.12, bubblewrap with working unprivileged namespaces,
OpenMP, and isolated NumPy. CPU-only; no GPU dependency. Build with at most `-j2`
(`--jobs 1` is supported). Allow several GiB for source/build/synthetic output.
Use a new output directory for each selftest; previous artifacts are never cleaned.
The test driver fails rather than falling back to an unsealed execution.

```sh
# Run from an extracted package or checkout of the annotated tag.
unset LD_LIBRARY_PATH
python3 -m venv /path/outside-repo/qies105-venv
/path/outside-repo/qies105-venv/bin/python -m pip install --require-hashes \
  -r qies105/requirements-numpy.txt
python3 -I -B qies105/replay.py all --jobs 2 \
  --out /path/on-ssd/qies105-fresh-build \
  --numpy-python /path/outside-repo/qies105-venv/bin/python
```

The requirements hash identifies the tested CPython 3.14 Linux x86-64 wheel.
Use that Python environment for this exact lock; an incompatible interpreter
must not silently substitute a wheel. The installed package version was 2.3.4.
The source-only replay fetches the stock source, **not the model**:
`https://codeload.github.com/ggml-org/llama.cpp/tar.gz/240d68fcfe0c2a1f298c8e18ca181368b1914259`.
Source archive SHA256:
`b7c50c28d1c25942fdbca447326a692657238cd87beeea45797264b02bc64b4f`.
All 4,084 stock source file hashes are checked against the frozen manifest.
Offline use: add `--stock-source /path/to/extracted/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259`.

Both CPU SDKs are built from source, rather than restored ELF backups. Baseline
GGML 0.25.3 and stock GGML 0.29.0 are **different**, so CPU kernel identity is not
claimed. Single CPU thread, ordinary buffers (`use_extra_bufts=false`), nonflash
attention, F16 KV, matching input positions and ordering are required. Native CPU
instructions follow the build machine; new binaries may have different hashes.
Compile/configure commands and generated worker hashes are saved with results.

The portable replay regenerates bounded synthetic five-block and one-block GGUFs
using the original generators. It runs the frozen 14-group layer and 16-check
adapter bodies, the original 38 extraction faults and four independent extraction faults, fresh independent stock
captures and zero-tolerance comparisons (360 F32 + 360 quantized candidate taps,
72 + 72 stock one-block taps), plus allocation-trap/negative-admission tests.
All numerical workers require QIES104 admission and a synthetic-only namespace.
Real models/home directories are not mounted. Build products are never a release
package. Do not expose the private stock SDK as an unguarded runtime interface.

Historical path/hash/retired-ELF assertions and the QIES103 characterization of
then-existing entry gaps are preserved as **historical evidence**, not renamed
as new qualification. The portable test matrix reports exactly its own counts;
it does not claim the old 40-case matrix ran unchanged on another filesystem.
Original scripts remain byte-identical and retain their machine-specific paths.

## Model identity and download, on the validation host only

AtomicChat/Ripple-65B-Preview-GGUF, revision
`df550a7178a87bbfb6ee2bdbd37603c553d536ee`; Apache-2.0, public/non-gated
according to frozen QIES098 metadata. Recheck access/licensing before acquisition.
Exact file `Ripple-65B-Preview-AD-Q4_K_M.gguf`, **40,426,079,712 bytes**, SHA256:
`9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc`.
No model is downloaded by this package or this task.

```sh
mkdir -p "$HOME/quortex-ripple/models"
cd "$HOME/quortex-ripple/models"
# Stop if the final file or a foreign partial already exists; never overwrite it.
# Check disk first: model + temporary/copies + 30 GiB entry reserve must fit.
curl --fail --location --continue-at - \
  --output Ripple-65B-Preview-AD-Q4_K_M.gguf.partial \
  'https://huggingface.co/AtomicChat/Ripple-65B-Preview-GGUF/resolve/df550a7178a87bbfb6ee2bdbd37603c553d536ee/Ripple-65B-Preview-AD-Q4_K_M.gguf'
printf '%s  %s\n' \
  9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc \
  Ripple-65B-Preview-AD-Q4_K_M.gguf.partial | sha256sum --check --strict
# Publish only after exact size/hash verification, and only if final path absent.
# A resumable .partial avoids an additional full-sized duplicate.
```

The archived `qies098-checks/ripple-tensor-index.json` contains 1,203 descriptors;
`qies101/selected-tensors.json` identifies eight original byte-comparison tensors.
`qies102/bundle-manifest.json` identifies all 15 layer-0 tensors and SHA256s.
These manifests contain historical absolute paths: rebind read-only paths and
recompute stat identities on the receiving host; do not reuse inode receipts.
Independent stock extraction source is `qies101/reference_extract.cpp`; it uses
upstream GGUF parsing, not the QUORTEX index parser. Repeat descriptor and byte
comparisons from this verified model before arithmetic. Its data-only purpose
does not qualify a numerical reference.

## Unchanged resource policy

```sh
python3 -I -B qies105/resources.py --storage /verified/persistent/ssd/run-root
python3 -I -B qies105/replay.py inspect
```

The first command is read-only and exits nonzero when original entry requirements
are unmet. It does not launch children or issue qualification receipts. The second
always reports real execution disabled. Never invoke the packaged original guard's
FlashNext-specific `main()` as a Ripple launcher. Use only its unchanged
`run_guard(command, unique_logdir)` API in a future reviewed adapter, default
`Policy()`, **no overrides**. It discovers the actual user-manager app.slice ancestors.

Original gates: MemAvailable >=6 GiB, SwapFree >=4 GiB, filesystem free >=30 GiB,
full memory PSI avg10 <1. Finite cgroup ancestor memory.max/high require >=7 GiB
headroom, finite swap.max >=2 GiB. Existing address-space/RSS rlimits must be
recorded and inherited; unlimited historical values are not newly invented hard
caps. Runtime stops: MemAvailable <1 GiB, SwapFree <2 GiB, disk <20 GiB,
process-tree RSS >=7 GiB, pressure full avg10 >=10 for three samples, OOM event,
monitor error or four-hour timeout. Whole-cgroup TERM then KILL cleanup, one-second
sampling, two-second grace. No sysctl, swap, cgroup or rlimit changes are authorized.

For positive live qualification on a naturally passing host, review the unchanged
original `run_guard` and run **harmless** normal, failing, descendant, interrupt and
monitor-failure children under its default policy. Verify actual user systemd/cgroup
membership, all descendants cleaned, nonzero outcomes for failures, ancestor OOM
accounting and no inference launched. Preserve each result and receipt SHA256 bound
to exact adapter, guard and worker hashes. Current artifact has no such positive
qualification. Do not create a receipt merely because preflight passes.

## Real layer-0 contract and required future work

`qies103/inputs/contract.json` is a schema/input contract, **not golden output**.
Regenerate deterministic inputs with `qies103/contract.py` in a new adapted path,
preserving its calculations and refusing overwrite. External already-scaled F32LE
input and x0 are identical `[5120,1]`: `(i % 17 - 8)/16`. Position 0, seq 0, next
position 1, one token; sliding causal mask `[0,-inf,...]` over 256 slots; window
1024; 80 Q heads /8 KV heads, dimension128, FFN40960; initial physical K/V F16 zero,
logical length0. No embedding lookup/token generation. Fifteen exact quantized
layer-0 weights, total **545,752,068 bytes**. Reference source is independent stock
`src/models/ricursive.cpp` from the pinned revision. Its tensor loader must parse
GGUF independently. Do not call the QUORTEX layer as the oracle.

A real-width bounded stock observer and canonical KV export are **not implemented/
qualified in this frozen release**. Its stock engine accepts only 256-hidden
synthetic one-block weights. Before any real execution, a separately reviewed
implementation must provide independent bounded extraction/execution and all
contract outputs, preserve operator order and ordinary CPU profile, then qualify
shared admission/live guard. No environment flag/receipt unlocks QIES104.
Do not run stock llama.cpp on the whole 40.4 GB model as a shortcut on an 8 GB host.

Only after all prerequisites and unchanged resource gates pass, run reference
and candidate sequentially, each under the original guard. Supply identical
input/x0, empty prior KV, dtype/context/masks, weight identities and backend flags.
Capture 18 named intermediates, final layer-output, key-written and value-written
(21 schemas in contract). Compare every shape/dtype/byte count and every byte,
zero tolerance; reject missing/nonfinite outputs; layer-output must equal l_out.
Compare only initialized/written KV ranges and exact next-position/state header.
A mismatch, guard violation, incomplete output, or unknown identity is NO-GO.
Store new independent reference captures/hashes separately; never relabel them
as recovered historical goldens. Existing contract hashes for outputs are null.

Layer weight residency is ~520.47 MiB; staging can temporarily coexist with CPU
weight storage; graph/quantization buffers and stock loader costs remain real-
weight **estimates**, not measured peaks. A simple full-model stock reference is
unsuitable. Historical synthetic memory measurements do not predict real width.
Reserve storage for model, paired extracted weights (~1.02 GiB), captures, source/
builds and logs in addition to original 30 GiB entry reserve. No full state copies
or 80-layer model residency are authorized. Full-model runtime is unmeasured.

## Rollback and release verification

Keep the original FlashNext and active Ripple worktrees untouched. Stop test
processes normally; retain logs. To abandon this release, switch to a separate
verified `8fac93...` worktree; never reset/clean the active experiments and never
restore retired executables' permissions. Do not merge the archive into production.

```sh
git ls-remote origin refs/heads/archive/ripple65b-qies104 \
  refs/tags/ripple65b-qies104-frozen 'refs/tags/ripple65b-qies104-frozen^{}'
git rev-parse archive/ripple65b-qies104 ripple65b-qies104-frozen \
  'ripple65b-qies104-frozen^{}'
```

The branch commit and peeled tag commit must match. The annotated tag object has
its own different ID. Package SHA256 is recorded in the annotated tag message and
final handoff report; it is external to the archive it hashes, avoiding self-hash
circularity. GO for frozen model-free handoff; **NO-GO for real-weight execution**.
