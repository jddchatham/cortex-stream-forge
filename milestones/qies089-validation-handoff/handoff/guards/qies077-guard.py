#!/usr/bin/env python3
"""External monitor for a transient user service; never changes QUORTEX source."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

G = 1024**3
ROOT = Path(__file__).resolve().parent
BASE = ROOT/'qies075-data'
CGROOT = Path('/sys/fs/cgroup')

class GuardAbort(Exception):
    pass

@dataclass
class Policy:
    entry_ram: int = 6*G
    entry_swap: int = 4*G
    entry_disk: int = 30*G
    min_ram: int = G
    min_swap: int = 2*G
    min_disk: int = 20*G
    max_rss: int = 7*G
    max_seconds: float = 4*3600
    interval: float = 1
    grace: float = 2


def ctl(*args):
    return subprocess.run(['systemctl', '--user', *args], text=True,
                          capture_output=True, timeout=10, check=True).stdout


def properties(unit, allow_missing=False):
    names = ['LoadState', 'ActiveState', 'SubState', 'Result', 'ControlGroup', 'ExecMainCode', 'ExecMainStatus']
    r = subprocess.run(['systemctl', '--user', 'show', unit, *['--property='+n for n in names]], text=True, capture_output=True, timeout=10)
    values = dict(x.split('=', 1) for x in r.stdout.splitlines() if '=' in x)
    if r.returncode and not (allow_missing and values.get('LoadState') == 'not-found'):
        raise GuardAbort('cannot inspect unit: '+r.stderr.strip())
    return values


def counters(path):
    return dict((k, int(v)) for k, v in (x.split() for x in (path/'memory.events').read_text().splitlines()))


def oom(ev):
    return sum(ev.get(k, 0) for k in ('oom', 'oom_kill', 'oom_group_kill'))


def host(base):
    mem = {k: int(v.split()[0])*1024 for k, v in
           (x.split(':', 1) for x in Path('/proc/meminfo').read_text().splitlines())}
    psi = next(x for x in Path('/proc/pressure/memory').read_text().splitlines() if x.startswith('full '))
    psi = float(dict(x.split('=') for x in psi.split()[1:])['avg10'])
    disk = os.statvfs(base)
    return dict(available=mem['MemAvailable'], swap_free=mem['SwapFree'],
                disk_free=disk.f_bavail*disk.f_frsize, psi=psi)


def members(cg):
    if not cg.exists():
        return []
    pids = set()
    for f in cg.rglob('cgroup.procs'):
        try:
            pids.update(int(x) for x in f.read_text().split())
        except FileNotFoundError:
            continue
    return sorted(pids)


def tree_sample(cg):
    data = dict(pids=[], rss=0, process_swap=0)
    for pid in members(cg):
        try:
            status = Path(f'/proc/{pid}/status').read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        values = dict((k, int(v.split()[0])*1024) for k, v in
                      (x.split(':', 1) for x in status.splitlines() if x.startswith(('VmRSS:', 'VmSwap:'))))
        data['pids'].append(pid)
        data['rss'] += values.get('VmRSS', 0)
        data['process_swap'] += values.get('VmSwap', 0)
    data.update(cgroup_memory=int((cg/'memory.current').read_text()),
                cgroup_swap=int((cg/'memory.swap.current').read_text()), events=counters(cg))
    return data


def run_guard(command, logdir, policy=None, sample_hook=None):
    """Hooks/threshold overrides are for harmless validation only; CLI uses fixed defaults."""
    policy = policy or Policy()
    logdir = Path(logdir)
    logdir.mkdir(mode=0o700, parents=False, exist_ok=False)
    unit = 'quortex-guard-'+uuid.uuid4().hex+'.service'
    cg = None
    started = False
    result = dict(ok=False, reason='not_started', unit=unit, cleanup=False)
    old_handlers = {}
    def emit(event, **values):
        with (logdir/'monitor.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(), event=event, **values))+'\n')
    def interrupt(signum, frame):
        raise GuardAbort(f'interrupted by signal {signum}')
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupt)
        h = host(logdir)
        if (h['available'] < policy.entry_ram or h['swap_free'] < policy.entry_swap
                or h['disk_free'] < policy.entry_disk or h['psi'] >= 1):
            raise GuardAbort('resource preflight failed')
        # Discover the actual service ancestors via the user manager's app.slice.
        manager = dict(x.split('=', 1) for x in ctl('show', 'app.slice', '-p', 'ControlGroup').splitlines())
        parent = CGROOT/manager['ControlGroup'].lstrip('/')
        ancestors = [p for p in [parent, *parent.parents] if (p/'memory.events').exists()]
        if not ancestors:
            raise GuardAbort('no cgroup memory accounting')
        baseline = {p: counters(p) for p in ancestors}
        for p in ancestors:
            for name, current, needed in [('memory.max', 'memory.current', 7*G),
                                          ('memory.high', 'memory.current', 7*G),
                                          ('memory.swap.max', 'memory.swap.current', 2*G)]:
                value = (p/name).read_text().strip()
                if value != 'max' and int(value)-int((p/current).read_text()) < needed:
                    raise GuardAbort(f'insufficient ancestor headroom: {p}/{name}')
        emit('preflight', host=h, unit=unit, command=command)
        props = ['Slice=app.slice', 'Type=exec', 'RemainAfterExit=yes', 'MemoryAccounting=yes', 'KillMode=control-group',
                 f'TimeoutStopSec={policy.grace}s', 'SendSIGKILL=yes', 'OOMPolicy=stop',
                 f'RuntimeMaxSec={policy.max_seconds+10}s', 'WorkingDirectory='+str(ROOT),
                 'StandardOutput=append:'+str(logdir/'execution.log'), 'StandardError=inherit']
        # Mark attempted before the RPC, so a timeout/interruption still cleans up the named unit.
        started = True
        launch = ['systemd-run', '--user', '--quiet', '--unit='+unit]
        subprocess.run(launch+['--property='+p for p in props]+command, check=True,
                       capture_output=True, text=True, timeout=15)
        began = time.monotonic()
        high_psi = 0
        while True:
            state = properties(unit)
            if state['Result'] not in ('success', ''):
                raise GuardAbort('unit failure: '+state['Result'])
            relative = state['ControlGroup']
            if relative:
                actual = CGROOT/relative.lstrip('/')
                if not actual.is_relative_to(parent) or actual == parent:
                    raise GuardAbort('unexpected cgroup location')
                if cg is not None and cg != actual:
                    raise GuardAbort('unit cgroup changed')
                cg = actual
            sample = host(logdir)
            if cg is not None and cg.exists():
                try:
                    sample.update(tree_sample(cg))
                except FileNotFoundError:
                    # The last process can exit during /proc/cgroup reads. Recheck the unit.
                    if properties(unit)['SubState'] != 'exited':
                        raise
            if sample_hook is not None:
                sample = sample_hook(sample)
            for p, before in baseline.items():
                if oom(counters(p)) > oom(before):
                    raise GuardAbort('ancestor OOM event')
            high_psi = high_psi+1 if sample['psi'] >= 10 else 0
            emit('sample', state=state, **sample)
            if oom(sample.get('events', {})):
                raise GuardAbort('target cgroup OOM event')
            reasons = []
            for key, minimum in [('available', policy.min_ram), ('swap_free', policy.min_swap),
                                 ('disk_free', policy.min_disk)]:
                if sample[key] < minimum:
                    reasons.append(key)
            if sample.get('rss', 0) >= policy.max_rss:
                reasons.append('tree_rss')
            if high_psi >= 3:
                reasons.append('memory_pressure')
            if time.monotonic()-began >= policy.max_seconds:
                reasons.append('timeout')
            if reasons:
                raise GuardAbort('guard threshold: '+','.join(reasons))
            if state['SubState'] == 'exited':
                # A successful main PID must not hide leftover detached descendants.
                if cg is not None and members(cg):
                    raise GuardAbort('main exited with descendants still running')
                if state['ExecMainCode'] != '1' or state['ExecMainStatus'] != '0':
                    raise GuardAbort('nonzero/signal main exit')
                result.update(ok=True, reason='child exited zero; no descendants')
                break
            if state['ActiveState'] in ('failed', 'inactive'):
                raise GuardAbort('unexpected inactive/failed unit')
            if not relative or not cg.exists():
                raise GuardAbort('active unit cgroup unavailable')
            time.sleep(policy.interval)
    except BaseException as exc:
        result.update(ok=False, reason=f'{type(exc).__name__}: {exc}')
    finally:
        # Cleanup does not depend on any log write succeeding.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if started:
                ctl('stop', unit)  # systemd TERM -> bounded wait -> KILL across the cgroup.
                result['cleanup'] = (not cg or not members(cg)) and properties(unit, allow_missing=True)['ActiveState'] in ('inactive', 'failed')
                if not result['cleanup']:
                    raise GuardAbort('cgroup cleanup incomplete')
                subprocess.run(['systemctl', '--user', 'reset-failed', unit], capture_output=True, timeout=10)
            else:
                result['cleanup'] = True
        except BaseException as exc:
            result.update(ok=False, cleanup=False, reason=result['reason']+'; cleanup: '+str(exc))
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        try:
            emit('result', **result)
            with (logdir/'result.json').open('x') as f:
                json.dump(result, f, indent=2)
        except BaseException as exc:
            result.update(ok=False, reason=result['reason']+'; result logging failed: '+str(exc))
        print(json.dumps(result), flush=True)
    return 0 if result['ok'] and result['cleanup'] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='resource check only; no model or unit execution')
    args = parser.parse_args()
    h = host(BASE)
    if args.check:
        print(json.dumps(dict(host=h, entry_ram_required=6*G, resources_ready=(h['available'] >= 6*G
              and h['swap_free'] >= 4*G and h['disk_free'] >= 30*G and h['psi'] < 1)), indent=2))
        return 0 if h['available'] >= 6*G and h['swap_free'] >= 4*G and h['disk_free'] >= 30*G and h['psi'] < 1 else 1
    # Fail before any inference if the published machine resource entry gate is unmet.
    if h['available'] < 6*G or h['swap_free'] < 4*G or h['disk_free'] < 30*G or h['psi'] >= 1:
        print('Resource preflight failed; no inference started.', file=sys.stderr)
        return 1
    repo = ROOT/'cortex-stream-forge-qies074'
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != '8fac93bb2bfc4b2d7069c4044f2f71cd25f80388':
        raise GuardAbort('wrong source commit')
    original = json.loads((repo/'flashnext-tensor-index.json').read_text())
    local = BASE/'models/flashnext-tensor-index.local.json'
    data = json.loads(local.read_text())
    assert data['model_dir'] == str(BASE/'models')
    assert {k:v for k,v in data.items() if k!='model_dir'} == {k:v for k,v in original.items() if k!='model_dir'}
    for i, size in [(1,44766155936),(2,44735995008),(3,7971004256)]:
        assert (BASE/f'models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-{i:05d}-of-00003.gguf').stat().st_size == size
    runid = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    state, output = BASE/'state'/runid, BASE/'outputs'/runid
    assert not state.exists() and not output.exists()
    command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin', 'HOME='+str(Path.home()), 'LANG=C.UTF-8',
               'CORTEX_FLASHNEXT_LOADER_CACHE=1', 'CORTEX_FLASHNEXT_EXPERT_ARENA_BYTES=1048576',
               'CORTEX_FLASHNEXT_LAYER_RANGE=1', 'OMP_NUM_THREADS=1', str(repo/'build-quortex/bin/quortex'),
               '--model', str(BASE/'models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf'),
               '--tensor-index', str(local), '--prompt', 'What is 2 + 2? Answer with only the number.',
               '--max-tokens', '16', '--state-dir', str(state), '--logits-dir', str(output)]
    return run_guard(command, BASE/'logs'/runid)

if __name__ == '__main__':
    sys.exit(main())
