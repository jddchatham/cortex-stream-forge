# Flash-Next owned transaction checkpoint

`CORTEX FLASHNEXT OWNED TRANSACTION: FROZEN`

This checkpoint records the verified in-process layer transaction in
`runtime-flashnext-layer-exp` before bounded stock-reference work.
It does not claim stock architectural layer correctness.

## Verified transaction

- pre-MoE hash: `34259a1e69a780eb`
- selected MoE hash: `822f120d0c7f2369`
- shared hash: `63055fc5061b323b`
- complete MoE hash: `c3485d69e3535a65`
- Segment-B hash: `0f82e252619bb73e`
- output shape: `[2560,4,1]`
- dynamic selected IDs: `136,77,317,257,226,142,175,414,373,131`
- selected expert bytes: `26,624,000`
- unselected expert reads: `0`
- Tensor Window: `16,384` bytes
- selected-expert pread calls: `1,650`
- same transaction instance: yes
- diagnostic input dependencies: none

## Model and residency

- model fingerprint: Qwen4Exp Flash-Next model at
  `/home/user/models/qwen38-flashnext-abliterated/`
- global buffer: `337,715,200` bytes
- layer buffer: `50,238,336` bytes
- CPU compute buffer: `6.43 MiB`
- selected tensors: `23`
- guarded allocation over 1 GiB: none
- `per_layer_token_embd.weight` resident: no
- full model resident: no
- all experts resident: no
- OOM pressure: none
- peak RSS: not measured validly; no value is inferred here

Required low-residency mode includes:

```text
CORTEX_PREAD_ARENA=1
CORTEX_LOW_RESIDENCY_LAYER=0
CORTEX_FLASHNEXT_ROUTER_ONLY=1
CORTEX_FLASHNEXT_SEGMENT_A=1
CORTEX_FLASHNEXT_SEGMENT_A_ROUTER=1
```

## Source and executable fingerprints

Build executable:

`runtime-flashnext-layer-exp-build/bin/llama-flashnext-owned-layer`

```text
abdea5b173b72278839bc08284aded23dfe9d8ef8f2205b9ba2aafe5f7d4c88c  executable
5357f9139a4f972f8984492b40ac0e32c0f901002eb80e0bc0582713989aeb56  examples/flashnext-owned-layer.cpp
985f9dc7fd1f7588f3eaab1865d6c5e3b3cdc153c79c74a8edc5e88881da0e0b  src/flashnext-layer-transaction.h
b1b99c0009e5dd26be9fff0a823adaaa37c9e89d9fb9ae0f1b5cf6a8df0e3ab6  src/flashnext-selected-moe.cpp
ff813bcd48b9bcb41cc7f17c2be251ead20e0975628d0bc49ddb0581defa63c2  src/flashnext-shared-expert.cpp
e622a36d02496b079420c472866c618a45a8e72f180dd177cd25d52c96ab4332  src/flashnext-segment-b.cpp
24a70f857cf7400650260123fe34511d2108cdc3386bf3ca4b746d988ef9b24e  src/llama-model.cpp
d2c58c7ea7504f2b643e19ab4bd761768b828251364facbbca9f5ccbdd6fefb4  src/models/qwen4exp.cpp
```

The historical sparse-MoE and Tensor-Window trees remain untouched.
