# Ripple QIES112A stable synthetic archive

This snapshot adds QIES106–111 experimental source and tests, the explicitly
unfinished QIES112 prototype, and the QIES112A diagnostic evidence to frozen
QIES105 commit14b3f7902864b02e1a84e5e6168a3cf99087e670.

QIES112A:3 fresh80-layer/two-position trials;960 successful native subprocesses;
12,480 byte-exact capture pairs. Four-position continuity, bounded exception
faults, selected caller sanitizers, controlled termination/retry and sliding/
global boundary tests also passed. The historical inference crash remains
unattributed. These results do not qualify real weights or production operation.

Read `qies112a/README.md` for source rebuild and preservation verification.
No model weights, compiled binaries, core dumps, raw private machine telemetry
or large tensor/state captures are distributed. Original numerical code and
resource guards are unchanged. Historical manifests with redacted paths are
explicitly distinguished from the published-file SHA256 manifest.

New refs: `archive/ripple65b-qies112a-stable` and annotated
`ripple65b-qies112a-stable`. They are archival experimental refs, not production
branches. Prior milestones and branches are preserved.
