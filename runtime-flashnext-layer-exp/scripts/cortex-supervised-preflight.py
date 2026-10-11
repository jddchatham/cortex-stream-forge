#!/usr/bin/env python3
"""Fail-closed preflight/telemetry helper for bounded Cortex diagnostics."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def cgroup_v2_path() -> Path:
    rel = None
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        fields = line.split(':', 2)
        if len(fields) == 3 and fields[0] == '0':
            rel = fields[2]
            break
    if rel is None:
        raise RuntimeError('cgroup_v2_membership_missing')
    mount = None
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        if ' - cgroup2 ' not in line:
            continue
        left, right = line.split(' - ', 1)
        lf = left.split()
        rf = right.split()
        if len(lf) >= 5 and rf and rf[0] == 'cgroup2':
            mount = Path(lf[4])
            break
    if mount is None:
        raise RuntimeError('cgroup_v2_mount_missing')
    path = mount / rel.lstrip('/')
    if not path.is_dir():
        raise RuntimeError('cgroup_path_missing:' + str(path))
    return path


def read_required(path: Path, name: str) -> str:
    p = path / name
    try:
        value = p.read_text().strip()
    except OSError as exc:
        raise RuntimeError(f'cgroup_file_unreadable:{p}:{exc}')
    if not value:
        raise RuntimeError('cgroup_file_empty:' + str(p))
    return value


def limits(path: Path) -> dict:
    memory = read_required(path, 'memory.max')
    swap = read_required(path, 'memory.swap.max')
    current = read_required(path, 'memory.current')
    peak = read_required(path, 'memory.peak')
    events = read_required(path, 'memory.events')
    if memory != '2147483648' or swap != '0':
        raise RuntimeError(f'cgroup_limits_mismatch:memory.max={memory}:swap.max={swap}')
    if not current.isdigit() or not peak.isdigit():
        raise RuntimeError('cgroup_memory_value_malformed')
    parsed_events = {}
    for line in events.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdigit():
            raise RuntimeError('cgroup_events_malformed')
        parsed_events[parts[0]] = int(parts[1])
    return {'path': str(path), 'memory.max': int(memory), 'memory.swap.max': int(swap),
            'memory.current': int(current), 'memory.peak': int(peak), 'memory.events': parsed_events}


def preflight(env: dict, estimated_global: int, estimated_layer: int, budget: int) -> dict:
    required = {
        'CORTEX_FLASHNEXT_EXTERNAL_EMBEDDING': '1',
        'CORTEX_FLASHNEXT_LAYER_START': '0',
        'CORTEX_FLASHNEXT_LAYER_END': '1',
        'CORTEX_FLASHNEXT_SSM_CONV_OPERAND_DIAGNOSTICS': '1',
    }
    for key, value in required.items():
        if env.get(key) != value:
            raise RuntimeError(f'missing_or_invalid_{key}')
    if env.get('CORTEX_QWEN35_SEGMENT_START') or env.get('CORTEX_QWEN35_SEGMENT_END'):
        raise RuntimeError('legacy_layer_selection_present')
    if estimated_global < 0 or estimated_layer < 0 or estimated_global + estimated_layer > budget:
        raise RuntimeError('bounded_residency_estimate_exceeds_budget')
    cg = cgroup_v2_path()
    return limits(cg) | {'estimated_global_bytes': estimated_global, 'estimated_layer_bytes': estimated_layer}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--estimated-global-bytes', type=int, default=0)
    ap.add_argument('--estimated-layer-bytes', type=int, default=0)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    try:
        result = preflight(dict(os.environ), args.estimated_global_bytes, args.estimated_layer_bytes, 2147483648)
        result['status'] = 'PASS'
        print(json.dumps(result, sort_keys=True) if args.json else result)
        return 0
    except RuntimeError as exc:
        print(json.dumps({'status': 'FAIL', 'reason': str(exc)}) if args.json else f'PREFLIGHT_FAIL {exc}', file=sys.stderr)
        return 78


if __name__ == '__main__':
    raise SystemExit(main())
