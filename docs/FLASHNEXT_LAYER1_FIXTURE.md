# Flash-Next layer-1 fixture contract (v1)

This is a non-executing contract for the first supervised real transaction.
The fixture root contains `manifest.json`, `hc_input.bin` (little-endian F32,
shape `[2560,4,1]`, exactly 40,960 bytes), and `sequence_state.bin` (the
opaque payload returned by `llama_state_seq_get_data` for sequence 0). Every
payload is listed in `files` with exact length and SHA-256. The manifest also
records model-shard identities, physical layer, position, sequence/slot
metadata, PLE row selections and the boundary
`physical_layer_1_pre_moe`.

The native bridge is disabled unless `CORTEX_FLASHNEXT_FIXTURE_DIR` is set. It
then reads the HC payload, checks its exact size, and uses it instead of the
historical synthetic `sin()` activation. A fixture may either provide the
opaque `sequence_state.bin` payload, or explicitly set
`CORTEX_FLASHNEXT_FIXTURE_INITIAL_STATE=fresh_zero`; the latter relies only on
the runtime's documented new-context initialization and does not fabricate an
opaque state blob. Position and sequence can be
overridden with `CORTEX_FLASHNEXT_FIXTURE_POSITION` and
`CORTEX_FLASHNEXT_FIXTURE_SEQUENCE`. Schema and SHA-256 validation must be run
with `flashnext_fixture.py` before launch. PLE row records are validated
metadata; the existing bounded PLE provider remains the owner of actual row
reads, so this v1 bridge does not claim a separate PLE injection path.

Token IDs are recorded explicitly as unused for externally supplied embeddings.
Attention/KV is recorded as `NOT_CONSUMED` only when the graph evidence for the
selected boundary establishes that status; otherwise it must remain
`UNVERIFIED`. No output or reference result is included here.

## Synthetic validation

```sh
python3 runtime-flashnext-layer-exp/tools/test_flashnext_fixture.py
python3 runtime-flashnext-layer-exp/tools/test_flashnext_fixture.py
```

## Proposed supervised command (not executed)

The fixture must be validated first. The command is intentionally a template
until the manifest contains real captured state and reference output:

```sh
python3 runtime-flashnext-layer-exp/tools/flashnext_fixture.py /path/to/fixture
systemd-run --user --scope --wait \
  -p MemoryMax=2147483648 -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=5s 195s \
  env CORTEX_FLASHNEXT_LAYER_START=1 \
      CORTEX_FLASHNEXT_FIXTURE_DIR=/path/to/fixture \
      /path/to/llama-flashnext-owned-layer \
      /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf \
      /home/user/bitnet-cortex/cortex-stream-forge/flashnext-tensor-index.json
```

The launcher must verify the actual child cgroup values (`memory.max=2147483648`
and `memory.swap.max=0`) before model loading, record `memory.peak`,
`memory.events`, process RSS, disk bytes read, latency, and compare the captured
pre-MoE output to an independently produced reference. No inference has been
run as part of this fixture milestone.
