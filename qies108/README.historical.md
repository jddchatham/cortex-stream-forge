# QIES108 bounded synthetic Ricursive attention

No model weights are opened. Real roles still enter the unchanged QIES104 real
admission and remain locked. This new interface accepts only four built-in
procedural cases and small checksummed synthetic KV checkpoints.

```sh
./pandacode-run.sh --smoke
./pandacode-run.sh
# Interrupted campaign: verify original source/binary identities and reuse ONLY
# complete per-job receipts; incomplete attempts are preserved under new names.
./pandacode-run.sh --resume qies108/runs/CAMPAIGN
```

A fresh QIES107 reproduction in this lineage must first pass its 63 comparisons,
100 faults and six refusals with hashes matching the preserved original:
`python3 -I -B qies107/run.py`. No original milestone is modified.

The candidate streams 128 output rows through verified GGML CPU matvecs. The
independent stock observer builds one conventional graph and holds all projection
weights. Shared code is input/state/ownership infrastructure only, not the
candidate's numerical graph or streaming schedule. Stock source is pinned at
240d68fcfe0c2a1f298c8e18ca181368b1914259. CPU SDKs/flags follow QIES106–107; no
libllama/GPU dependency, NumPy, model download or global install is used.

Head dimensions are128, Q heads80, KV heads8; hidden5120; Q/gate width10240;
K/V width1024. Layer0 uses sliding RoPE; layer4 uses global NoPE. The actual
SWA window is1024. Caches use F16 and are bounded to256 or1280 slots.

Cases `swa-start` and `global-start` start empty at position0 and continue to1.
Cases `swa-boundary` and `global-boundary` use explicitly synthetic prefilled
history at1023, then execute1024/1025. They are not 1,023 evaluated prior tokens.
Reference/candidate run in separate fresh processes for every token. Each new
checkpoint includes fixed identity/version/layer/slots/position, F16 K/V, and
SHA256. The input is mounted read-only and is never overwritten. Successful
outputs are published in an exclusive capture directory. Only after byte-exact
comparison may the host use that next checkpoint. This is a synthetic test ABI,
not a production multi-model checkpoint format or crash-recovery qualification.

Q8_0 uses32-value blocks: the attention factory correctly uses packed size for
a256-value template (eight Q8 blocks). Eighteen independent whole-row comparisons
with canaries test Q4_K/Q8_0/Q6_K template expansion before attention arithmetic.
The factory only accelerates synthetic input construction, never real weights.

Sealed namespaces expose private CPU libraries and at most one verified small KV
input, never the model/home directories. Worker files have no host execute bits
and namespace-only RPATH. All projection/read/state sizes are bounded; no GGUF
loader exists. State reads reject bad length/identity/checksum before execution.
Existing real-model guards, cgroups, rlimits, OS settings and allocator behavior
remain unchanged. Additional model-free monitoring is reused from QIES107.

Evidence: `runs/CAMPAIGN/summary.json`, `attention-result.json` per accepted job,
worker telemetry, monitor logs, exact capture bytes, mask checks and checkpoint
hashes. A missing completion receipt or partial directory is never success.
