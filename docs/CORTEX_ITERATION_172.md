# Cortex Iteration 172 — Down Dot-to-Destination and Oracle Layout Proof

## Preservation

- Branch: `cortex-iteration-172`
- Iteration 171 state and unrelated dirty changes preserved.
- No reset, clean, stash, merge, commit, or push.
- `git diff --check`: PASS.

## Explicit layouts

The captured runtime down tensor has logical shape `[2560,10,9,1]` and F32
strides `[4,10240,102400,921600]` bytes. Its physical linear index is:

```text
runtime_index(channel, slot, token) = channel + slot*2560 + token*25600
```

The independent oracle stores rows as `[token, slot, channel]`:

```text
oracle_index(token, slot, channel) = (token*10 + slot)*2560 + channel
```

For token 1, physical expert 257 is selected slot 9:

```text
token-1 IDs = [309,252,453,106,209,59,488,496,169,257]
runtime index(channel=0, slot=9, token=1) = 48640
oracle index(token=1, slot=9, channel=0) = 48640
```

The formulas agree at this coordinate; no blind flattened comparison was used
for the target proof.

## Four-way target comparison

Target: layer 4, down, token 1, expert 257, slot 9, output channel 0.

| Value | Result |
|---|---:|
| Independent scalar IQ4_NL/Q8_0 dot | `0.006443967811833318` |
| Keyed CPU dot accumulator | `0.006443969905376434` |
| Runtime stored element at index 48640 | `0.006443969905376434` |
| Existing oracle element at index 48640 | `0.0002231794933322817` |

CPU-versus-scalar absolute error: `2.09354e-09`.

CPU-versus-runtime stored error: `0` at F32 representation.

The runtime destination placement is therefore proven correct for the target
dot. The oracle is the first divergent boundary for this coordinate.

The keyed CPU record is `/tmp/cortex-iter170-l4-p2-t1-e257-r0.bin`; its CPU
weight capture was corrected to use `wdata`, the actual vec-dot row pointer.
The Q8_0 activation remains the byte-identical 680-byte capture from Iteration
169.

## Fresh full oracle

The current nine-token oracle was rerun after regenerating layer-4 fixtures:

```text
CORTEX_L2_REF gate n=57600 max_abs=5.24521e-06 rms=4.10268e-07 mismatches=0
CORTEX_L2_REF up   n=57600 max_abs=4.29153e-06 rms=3.47447e-07 mismatches=0
CORTEX_L2_REF down n=230400 max_abs=3.27526 rms=0.0792954 mismatches=229698 first=0
CORTEX_L2_REF aggregate n=23040 max_abs=1.05916 rms=0.0629292 mismatches=22918 first=0
```

Gate/up remain PASS. The down/aggregate FAIL is not accepted as runtime
failure because the target coordinate independently proves the oracle value
does not equal the canonical-row scalar or stored runtime result.

## Status

| Check | Status |
|---|---|
| Runtime/oracle axis formulas | PASS for target coordinate |
| Token/expert-slot mapping | PASS for token 1 expert 257 |
| CPU dot to runtime destination | PASS |
| CPU dot versus independent scalar | PASS (`2.09e-09`) |
| Existing oracle down value | FAIL at target coordinate |
| Full down oracle | FAIL / oracle-side issue unresolved |
| Routed aggregate | NOT TESTED as a trusted comparison |
| Gate/up oracle | PASS |
| Provider regression | PASS |
| Full layer-4 equivalence | NOT YET ACCEPTED |
| Full-model streaming | NOT TESTED |

## Next action

Audit the oracle's layer-4 down implementation with a target-row trace of
resolved expert offset, decoded first block, quantized activation, and scalar
result. The runtime down kernel and placement should remain unchanged until
the oracle emits `0.0064439678` for this exact tuple. Then rerun all 90
token/expert down slices and routed aggregation.
