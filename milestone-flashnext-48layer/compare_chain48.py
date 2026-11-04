#!/usr/bin/env python3
"""Compare CHAIN_0_N per-layer outputs and logits against the stock oracle.
Usage: compare_chain48.py <tag> <n_layers> [stock_dump_dir]
Stock layer-L output = <stock>/hc_combine-L.bin. For N=48 also compares logits.
"""
import numpy as np, os, sys
D = '/home/user/quortex-diag-20261103'
tag, n = sys.argv[1], int(sys.argv[2])
stock = sys.argv[3] if len(sys.argv) > 3 else 'dump-stock-off2'
cap = os.path.join(D, f'captures-{tag}')
fail = 0
for L in range(n):
    a_path = os.path.join(cap, f'cortex_layer{L}_output_p0.bin')
    b_path = os.path.join(D, stock, f'hc_combine-{L}.bin')
    a = np.fromfile(a_path, dtype=np.float32) if os.path.exists(a_path) else None
    b = np.fromfile(b_path, dtype=np.float32) if os.path.exists(b_path) else None
    if a is None or b is None:
        print(f'layer{L}: MISSING'); fail += 1; continue
    if np.array_equal(a, b):
        print(f'layer{L}: EXACT (n={a.size})')
    else:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print(f'layer{L}: DIFF max={d.max():.6g} rms={np.sqrt(np.mean(d**2)):.6g} first={int(np.argmax(d>0))}')
        fail += 1
if n >= 48:
    lg = os.path.join(cap, 'cortex_logits_p0.bin')
    ref = os.path.join(D, 'stock_logits_p0.bin')
    if os.path.exists(lg) and os.path.exists(ref):
        a = np.fromfile(lg, dtype=np.float32); b = np.fromfile(ref, dtype=np.float32)
        nmax = min(a.size, b.size)
        if np.array_equal(a[:nmax], b[:nmax]):
            print(f'logits: EXACT (n={nmax})')
        else:
            d = np.abs(a[:nmax].astype(np.float64) - b[:nmax].astype(np.float64))
            print(f'logits: DIFF max={d.max():.6g} rms={np.sqrt(np.mean(d**2)):.6g} first={int(np.argmax(d>0))}')
            fail += 1
    else:
        print('logits: MISSING'); fail += 1
print('RESULT:', 'PASS' if fail == 0 else f'FAIL ({fail})')
sys.exit(0 if fail == 0 else 1)
