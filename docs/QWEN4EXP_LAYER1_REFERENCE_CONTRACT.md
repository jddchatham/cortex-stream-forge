# Qwen4Exp layer-1 reference contract

This is a non-executing contract for a future supervised comparison. It does
not authorize inference and contains no model payloads.

## Execution identity

- Stage: physical layer 1, pre-MoE boundary
- Input shape: `[2560,4,1]`, 10,240 F32 values, 40,960 bytes
- Graph output boundary: after layer-1 Graph A/router preparation and before
  sparse-MoE, final normalization, vocabulary projection, or decoding
- Preferred implementation: the existing owned-layer transaction engine
- Generic segment executable: diagnostic/plan fallback only
- Source revision: record the exact executable commit at test time

## Required byte-identical inputs

The future test must provide and hash all of the following before execution:

| Dependency | Required evidence |
|---|---|
| GGUF shards | SHA-256 for every shard used by the executable |
| Executable | SHA-256 and source commit |
| External HC input | 40,960 raw bytes and SHA-256 |
| Recurrent state | Every consumed initial payload, shape/type/range and SHA-256 |
| Attention/KV state | Consumed payloads and SHA-256, or explicit NOT_CONSUMED evidence |
| PLE input | Token/row IDs, row offsets, row hashes, and position metadata |
| Sequence metadata | Sequence ID, token position, batch and recurrent indices |

Values are intentionally **UNRECORDED** until captured from the same
transaction contract. Native output hashes from another invocation are not
accepted as substitutes.

## Comparison rule

The reference must use the same model shards, input bytes, state bytes,
positions, and pre-MoE boundary. Capture the complete output as canonical
F32 bytes and report shape, byte count, SHA-256, first differing element,
maximum absolute error, maximum relative error, and RMS error. A finite output
or matching repeated run is not mathematical equivalence.

## Resource gate and proposed command

The future run must be launched through the verified actual-child gate with:

```text
MemoryMax=2147483648
MemorySwapMax=0
wall timeout=195 seconds
```

Payload command:

```text
llama-qwen38-lowres-segment-test <FIRST_GGUF_SHARD> 1 2
```

Required measurements are child cgroup `memory.max`, `memory.swap.max`,
`memory.peak` when retained, `memory.events`, process RSS/PSS, swap usage,
bytes read, elapsed time, exit status, output hash, and numerical comparison.

No model execution is part of this contract change.
