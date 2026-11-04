#!/usr/bin/env python3
"""Two-token chain vs stock oracle comparison.
Usage: compare_multitoken.py <tag> <n_layers>
Pos-0 refs: dump-stockref (1-token stock run) + stock_logits_p0.bin
Pos-1 refs: dump-stock2tok (2-token stock run; dumps hold pos-1 values) + <STOCK2TOK_P1>
"""
import numpy as np, os, sys
D = '/home/user/quortex-diag-20261103'
tag, n = sys.argv[1], int(sys.argv[2])
cap = os.path.join(D, f'captures-{tag}')
p1_ref = os.environ.get('STOCK2TOK_P1', os.path.join(D, 'stock2tok_logits_p0.bin.p1'))
fail = 0

def cmp(label, a_path, b_path):
    global fail
    a = np.fromfile(a_path, dtype=np.float32) if os.path.exists(a_path) else None
    b = np.fromfile(b_path, dtype=np.float32) if os.path.exists(b_path) else None
    if a is None or b is None:
        print(f'{label}: MISSING'); fail += 1; return
    if np.array_equal(a, b):
        print(f'{label}: EXACT (n={a.size})')
    else:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print(f'{label}: DIFF max={d.max():.6g} rms={np.sqrt(np.mean(d**2)):.6g} first={int(np.argmax(d > 0))}')
        fail += 1

for L in range(n):
    cmp(f'layer{L} pos0', os.path.join(cap, f'cortex_layer{L}_output_p0.bin'),
        os.path.join(D, 'dump-stockref', f'hc_combine-{L}.bin'))
for L in range(n):
    cmp(f'layer{L} pos1', os.path.join(cap, f'cortex_layer{L}_output_p1_p0.bin'),
        os.path.join(D, 'dump-stock2tok', f'hc_combine-{L}.bin'))
if n >= 48:
    cmp('logits pos0', os.path.join(cap, 'cortex_logits_p0.bin'), os.path.join(D, 'stock_logits_p0.bin'))
    cmp('logits pos1', os.path.join(cap, 'cortex_logits_p1_p0.bin'), p1_ref)
print('RESULT:', 'PASS' if fail == 0 else f'FAIL ({fail})')
sys.exit(0 if fail == 0 else 1)
