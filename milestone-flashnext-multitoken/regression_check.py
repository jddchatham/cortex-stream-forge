#!/usr/bin/env python3
"""QUORTEX bit-exact regression: chain streaming vs stock oracle.

Usage: regression_check.py [chain_dir_on chain_dir_off stock_dir captures_on captures_off] [n_layers]
Defaults: dump-chain-on dump-chain-off dump-stock-off2 captures-chain-on captures-chain-off, n=2.
Stock layer-L output = dump-stock-off2/hc_combine-L.bin (== diag_hc_streams-(L+1) for L<47).
Exits 0 only if every compared tensor is bit-exact.
"""
import numpy as np, os, sys

D = '/home/user/quortex-diag-20261103'
on_dump, off_dump, stock, cap_on, cap_off = sys.argv[1:6] if len(sys.argv) >= 6 else \
    ['dump-chain-on', 'dump-chain-off', 'dump-stock-off2', 'captures-chain-on', 'captures-chain-off']
n_layers = int(sys.argv[6]) if len(sys.argv) >= 7 else 2
P = lambda *p: os.path.join(D, *p)
fail = 0

def load(p):
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None

def cmp(label, a, b, required=True):
    global fail
    if a is None or b is None:
        print(f'{label}: MISSING (a={"ok" if a is not None else "missing"}, b={"ok" if b is not None else "missing"})')
        if required: fail += 1
        return
    if np.array_equal(a, b):
        print(f'{label}: EXACT (n={a.size})')
    else:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print(f'{label}: DIFF max={d.max():.6g} rms={np.sqrt(np.mean(d**2)):.6g}')
        fail += 1

# per-layer outputs vs stock hc_combine-L
for L in range(n_layers):
    chain = load(P(cap_on, f'cortex_layer{L}_output_p0.bin'))
    ref = load(P(stock, f'hc_combine-{L}.bin'))
    cmp(f'layer{L} output vs stock hc_combine-{L}', chain, ref)

# ON vs OFF exactness on key dumps
for f in ['diag_hc_streams-0', 'diag_ple_embd-0', 'linear_attn_qkvz_input-1',
          'linear_attn_qkv_mixed-1', 'conv_input-1', 'diag_tx_inject-0', 'diag_tx_residual-0']:
    cmp(f'on/off {f}', load(P(on_dump, f + '.bin')), load(P(off_dump, f + '.bin')))
cmp('layer1 out on/off', load(P(cap_on, 'cortex_layer1_output_p0.bin')),
    load(P(cap_off, 'cortex_layer1_output_p0.bin')))

print('RESULT:', 'PASS' if fail == 0 else f'FAIL ({fail} mismatches)')
sys.exit(0 if fail == 0 else 1)
