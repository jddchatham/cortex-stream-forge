# QIES-089 frozen validation handoff

This is a preservation milestone, not an accepted optimization release.

## Verified engine baseline

The engine source is unchanged from published QIES-072 commit
`8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. The independently built baseline
completed the historical QIES-080/081 CPU inference. No new inference or
memory-guard modification was performed to create this archive.

`development/continuous-generation` starts directly from that verified commit,
not from this archival documentation commit or an applied QIES-086 patch.
That is the engine baseline intended for QIES-090 development.

## Unvalidated experimental material

`handoff/source/qies086.patch` preserves the exact QIES-086 two-line staged-state
removal. It is NOT applied to this branch's engine source. Its SHA256 is
`50c0475c7cfd731b4655e8de44651c997b4f69a15fe023902325c7ea0a3737cd`.

The QIES-089 handoff and reports are preserved byte-for-byte. Runtime oracle
qualification, full golden fixtures, checkpoint/resume and rollback validation,
repeated memory/performance measurements, and optimization acceptance remain
BLOCKED. Source/build reproducibility is not numerical-equivalence evidence.
Do not merge or apply the experiment to production based on this archive.

The two small anchor arrays are historical baseline selection outputs, not a
complete independent stock oracle. No experimental output is substituted for
reference data.

## Deliberate exclusions and recoverability

Model weights, executables, shared libraries, object files, build directories,
virtual environments, caches, credentials and unrelated working-tree changes
are excluded. The generated source Git bundle and compressed transfer archive
are also excluded: their substantive source history and small handoff files are
already recoverable from Git. Their exact original hashes and sizes remain in
`PRESERVATION_MANIFEST.json`; the original generated files remain local.

The original handoff `manifest.json` and `SHA256SUMS` are intentionally unchanged
and describe the complete original transfer package, including its omitted
source bundle. Use `PRESERVATION_MANIFEST.json` to verify this tracked subset.
This checkout is not claimed to reproduce the original tarball byte-for-byte.
To make a new portable package, reconstruct its source bundle from the verified
base commit, include the tracked handoff files, and record a new package
manifest/archive identity rather than relabeling it as the original tarball.

Annotated tag: `qies089-frozen`.
Archive branch: `archive/qies089-validation-handoff`.
No history rewrite, force push, production merge or main-branch update is part
of this milestone.
