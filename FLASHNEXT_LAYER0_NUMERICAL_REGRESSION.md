# Flash-Next layer-0 numerical regression

This checkpoint compares the independent complete stock Qwen4Exp layer-0
oracle with the recovered Cortex owned-transaction runner. Stock and Cortex
run sequentially with the same model shards, tokens `248045,248046`, positions
`0,1`, sequence `0`, and one CPU thread.

The authoritative recurrent-state comparison is the **position-1 pre-update**
boundary: the state immediately consumed by the position-1 decode. Opaque
post-position-0 serialized blobs are not used as the semantic state oracle.

Measured result:

- position-1 `r_l[0]`: exact
- position-1 `s_l[0]`: exact
- position-1 router IDs: `226,144,215,483,239,419,504,302,35,136`
- stock output hash: `88fa3cc7aeb0930b`
- Cortex output hash: `13465c8ba3343b29`
- max absolute error: `1.4901161e-08`
- RMS error: `5.7019197e-10`
- unselected expert reads: `0`

Acceptance is numerical, not bitwise: FP32 outputs are accepted when the
documented max-absolute and RMS measurements remain at or below the recorded
values and routing/state inputs are exact. Relative error is reported with a
`1e-30` denominator floor and is diagnostic only for near-zero elements.

The layer-0 comparison is frozen pending extension of the same boundary
capture scheme to layers 0–3.
