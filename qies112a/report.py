"""Summarize preserved diagnostics; never launches or changes inference code."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent;W=R.parent
p=json.loads((R/'tested-provenance.json').read_text());series=Path(json.loads((R/'active.json').read_text())['root']);trials=[]
for f in sorted(series.glob('trial-*/verified.json')):trials.append(json.loads(f.read_text()))
done=(series/'SUCCESS.json').exists();failures=list(series.glob('FAILURE-*.json'))
verdict='PENDING' if not done and not failures else ('FAIL / STOPPED' if failures else 'PASS for tested synthetic configuration')
text=f'''# QIES112A — QUORTEX crash and stability diagnostic

Updated: {datetime.datetime.now().astimezone().isoformat()}

**Historical inference crash: INCONCLUSIVE / not attributed.**
**QIES111 synthetic stability: {verdict}.** Completed fresh trials: {len(trials)}/3.

## What is and is not identified

No available record identifies the executable, command, PID, timestamp or signal of the alleged historical **inference-program** crash. It must not be assumed to be the QIES111 synthetic layer worker.

The broader core inventory does identify `/usr/bin/pandacode` [historical process], [historical thread] (`tokio-rt-worker`), on November6 at21:14:45 CST: signal7 SIGBUS, `BUS_ADRERR`, command `pandacode`. WebKit network/web processes also have SIGBUS dumps shortly afterward. The Thorium browser has a separate November7 SIGTRAP dump. These are desktop/browser programs, not FlashNext or Ripple inference executables, and predate QIES111. There is no established correlation to the later user-reported interruption. Their underlying cause is not diagnosed here. No unrelated core memory was extracted.

The known real FlashNext executable remains `cortex-stream-forge-qies074/build-quortex/bin/quortex`, SHA256 `329ebd647b04a2d6464461ddd94fc07a2691a948b2918d320ac1eb3703f9c101`, source commit `8fac93bb2bfc4b2d7069c4044f2f71cd25f80388`. QIES081 records its QIES080 run returning0 after answer4, EOG248046 and1326.023seconds. That is evidence of successful completion, not a recorded inference crash. The current executable hash still matches. It was not launched for this diagnostic.

A separate, identified QIES111 prototype startup failure (`20261111-173515-232345df`) returned ordinary exit1 with `activation input length`. The published QIES111 report attributes it to a missing input bind, fixed before the currently preserved artifact was accepted. Its stdout/stderr/result are copied read-only under `known-startup-failure/`. This does not identify a signal crash or justify changing the currently tested executable.

The unfinished QIES112 generation experiment is preserved and suspended. Its one accepted synthetic prediction, interrupted controller sessions and partial stacks are not automatically classified as numerical-program crashes. No QIES112 feature development is part of this mission.

Evidence preserved before execution: `qies112a-diagnostic/tested-provenance.json`, `historical-candidates.json`, `qies112-snapshot.json`, `cores-before.log`, `kernel-before.log` and `desktop-core-metadata.local`. The kernel query reports no matching accessible OOM/kill/segfault entries; absence of logs does not exclude an unrecorded historical event.

## Exact binaries under test

| Role | Existing executable | SHA256 |
|---|---|---|
'''
for role,d in p['executables'].items():text+=f"| {role} | `{d['path']}` | `{d['sha256']}` |\n"
text+=f'''
These are the **already-preserved QIES111** binaries, not rebuilt replacements. Their three GGML library hashes, sources and synthetic input hashes were checked before testing. Host mode600 and sealed-namespace dispatch are retained. Native workers have no real GGUF mounted.

Source HEAD: `{p['HEAD']}` plus preserved, uncommitted QIES106–111 experimental sources identified by the original source manifest. HEAD alone is not a claim that these workers are in the archive commit. Stock source: `{p['stock']}`. Compiler: `{p['compiler'].splitlines()[0]}`. Worker commands retain C++17 `-O2 -Wall -Wextra`; SDK libraries retain their original native CPU configuration. One numerical CPU thread, no GPU. Full build commands/library hashes are in `tested-provenance.json` and original `qies111/build/manifest.json`.

## Reproduction and safeguards

```sh
env -u LD_LIBRARY_PATH python3 -I -B qies112a-diagnostic/run.py --trials 3 --four-tokens
```

The diagnostic driver calls the unchanged QIES111 `campaign`, `transaction` and `launch` APIs. Each new trial runs all80 layers for positions0/1 in both candidate and independent reference. Every trial must produce4160 byte-exact captures matching the original QIES111 SHA256 map, preserve separate layer KV identities and complete320 numerical subprocesses successfully.

The actual commands are preserved per job in `result.json`, with stdout/stderr and process-tree/cgroup monitor logs. A representative native invocation is `/lib64/ld-linux-x86-64.so.2 /workers/candidate --synthetic 0 256 0 seed contract-v1 none` inside the unchanged bwrap namespace, marker, CPU environment, read-only synthetic weights and prior-state mounts. The reference uses `/workers/reference`. The host binary is copied into neither a new file nor a new build: its original path is bound read-only.

No real-weight entry is enabled. The existing real6GiB MemAvailable gate and unconditional unqualified-runtime lock remain unchanged. The separately established synthetic policies remain1.5/1.75GiB entry,1GiB running floor,2GiB swap reserve,20GiB disk reserve,256/640MiB worker RSS ceilings,120second per-worker timeout and ancestor OOM-event monitoring. No cgroup, address-space, swap, zram or OS policy was modified. Shared desktop cgroup peaks are not isolated worker accounting.

## Fresh trial results

| Trial | Exact captures | Native subprocesses | Exit | Signal | Streaming peak RSS/PSS MiB | Reference peak RSS/PSS MiB |
|---:|---:|---:|---:|---|---:|---:|
'''
for t in trials:
 c=t['peak_subtree_bytes']['candidate'];r=t['peak_subtree_bytes']['reference'];text+=f"| {t['trial']} | {t['captures']} | {t['job_count']} | 0 | none | {c['rss']/2**20:.2f}/{c['pss']/2**20:.2f} | {r['rss']/2**20:.2f}/{r['pss']/2**20:.2f} |\n"
text+='''
RSS/PSS are sampled subprocess-tree peaks, including namespace/loader and potentially pre-exec inherited pages; RSS may double-count shared mappings. Native high-water RSS is also retained separately. Components and unrelated peaks are not summed into a total-system claim. Per-layer completion, KV state/header/checksum, capture hashes, input identities, zero tracked ownership and zero open adapter files are checked. Allocator-retained pages can remain after tracked cleanup; zero modeled ownership is not zero RSS.

## Stress, faults, sanitizers and interruption
'''
for label,path in [('Original bounded faults',series/'bounded-faults/summary.json'),('Preserved sanitizer build',series/'sanitizer/summary.json'),('Interrupted execution and retry',series/'interruption/summary.json'),('Sliding/global boundary',series/'boundary/layers-5-start-1023/summary.json'),('Four-token extension',series/'trial-3/layers-80-start-0/four-token-summary.json')]:
 text+=f"- {label}: "+('completed; see `'+str(path.relative_to(W))+'`.' if path.exists() else 'not yet completed.')+'\n'
text+='''
Fault injection uses existing exception hooks at allocation/compute/serialization boundaries. It does not force actual system OOM or change allocator behavior. Malformed/truncated/cross-layer/wrong-position state is tested only with diagnostic copies or mismatched requests. Expected refusal/fault exit1 must not be reported as successful inference.

Sanitizer testing explicitly uses the existing separate `qies111/sanitizer-build/candidate` identified by its preserved hash, never replaces the ordinary candidate. ASan/UBSan/leak detection cover instrumented caller/state/adapter code; prebuilt GGML libraries are not fully instrumented. Passing these cases is not proof that all library paths are bug-free.

The initial diagnostic wrapper was deliberately interrupted with SIGINT after a static audit found that its invalid-layer case would bind a nonexistent fixture directory before reaching native bounds validation. The corrected wrapper instead binds valid layer0 fixtures and passes invalid layer80 to the native CLI. This change affected only diagnostic routing, not the executable, source, guards or math being diagnosed. The first partial run, source snapshot, exit130 and correction reason are preserved in `harness-interruption.json`, `run-initial.py` and the initial series directory; it is not counted as a completed trial or an unexplained inference crash.

## Attribution and recommended action

No QUORTEX root cause is established from historical evidence. Do not assign OOM, segmentation fault, invalid tensor access, KV corruption, resource-limit enforcement or supervision failure merely from an interrupted assistant session. Recorded desktop SIGBUS signals are distinct evidence, with no demonstrated inference correlation.

No inference-engine fix is justified yet, and none has been applied. For the next **actual** failure, preserve the exact executable SHA256, full guarded command, timestamp/PID, exit/signal, stdout/stderr and matching target/ancestor OOM counters. If a matching core exists, inspect it using the matching executable/debug identity. If only PandaCode disappears, diagnose its app/core separately; do not weaken QUORTEX's guards or modify its arithmetic as a workaround.

Synthetic stability results apply to this full-width, fixed256-slot, repeated-synthetic-profile configuration. They do not qualify real Ripple/FlashNext inference, arbitrary context lengths, full-state power-loss recovery or a long-lived persistent process. Four-token extension and a seeded1023/1024 sliding/global boundary are reported separately from the repeated two-token trials. No real-weight arithmetic, production changes, frozen-ref changes, commits, merges or pushes are authorized or performed.
'''
(W/'QIES112A_QUORTEX_CRASH_DIAGNOSTIC.md').write_text(text)
print(verdict,'completed trials',len(trials))
