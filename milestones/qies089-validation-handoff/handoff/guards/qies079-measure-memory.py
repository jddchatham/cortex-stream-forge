#!/usr/bin/env python3
"""Read-only resource measurement plus exclusive persistent logs; never launches inference."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'qies075-data'
GIB = 1024**3
MEMORY_GATE = 6*GIB
EXPECTED_GUARD_SHA256 = 'ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2'


def memory_pass(available):
    return available >= MEMORY_GATE


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return dict(argv=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(argv=args, returncode=None, stdout='', stderr=type(exc).__name__+': '+str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='measurement', help='description only; does not establish headless state')
    parser.add_argument('--self-test', action='store_true', help='test exact gate boundaries without measuring or writing logs')
    args = parser.parse_args()
    if args.self_test:
        cases = [(0,False),(MEMORY_GATE-1,False),(MEMORY_GATE,True),(MEMORY_GATE+1,True)]
        if not all(memory_pass(n)==expected for n,expected in cases):
            raise RuntimeError('boundary test failed')
        print('SELF_TEST: PASS (4 boundary cases; no headless measurement or inference)')
        return 0
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', args.label):
        parser.error('label must be 1-40 letters, digits, underscores or hyphens')
    logbase = BASE/'logs/qies079-memory'
    logbase.mkdir(mode=0o700, exist_ok=True)
    run = logbase/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+args.label+'-'+uuid.uuid4().hex[:8])
    run.mkdir(mode=0o700, exist_ok=False)
    try:
        free = command(['/usr/bin/free','-h'])
        swap = command(['/usr/bin/swapon','--show','--bytes'])
        memtext = Path('/proc/meminfo').read_text()
        mem = {k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in memtext.splitlines())}
        pressure = Path('/proc/pressure/memory').read_text()
        full = next(x for x in pressure.splitlines() if x.startswith('full '))
        psi = float(dict(x.split('=') for x in full.split()[1:])['avg10'])
        disk = os.statvfs(BASE)
        disk_free = disk.f_bavail*disk.f_frsize
        zram = {}
        for dev in sorted(Path('/sys/block').glob('zram*')):
            zram[dev.name] = {n:(dev/n).read_text().strip() for n in ['disksize','mm_stat','comp_algorithm'] if (dev/n).exists()}
        dm = command(['/usr/bin/systemctl','show','display-manager.service','-p','ActiveState','--value'])
        graphical = command(['/usr/bin/systemctl','--user','show','graphical-session.target','-p','ActiveState','--value'])
        sessions = command(['/usr/bin/loginctl','list-sessions','--no-pager'])
        terminal = command(['/usr/bin/tty'])
        mount = command(['/usr/bin/findmnt','-T',str(BASE),'-n','-o','SOURCE,FSTYPE,TARGET'])
        guard_sha = hashlib.sha256((ROOT/'qies077-guard.py').read_bytes()).hexdigest()
        errors = []
        if guard_sha != EXPECTED_GUARD_SHA256:
            errors.append('validated QIES-077 guard identity changed; comparison requires review')
        if free['returncode'] != 0 or swap['returncode'] != 0:
            errors.append('free/swapon acquisition failed')
        headless = ('OBSERVED' if dm['returncode']==0 and graphical['returncode']==0
                    and dm['stdout'].strip()=='inactive' and graphical['stdout'].strip()=='inactive'
                    else 'NOT_CONFIRMED')
        checks = dict(memory_6_gib=memory_pass(mem['MemAvailable']), swap_free_4_gib=mem['SwapFree']>=4*GIB,
                      disk_free_30_gib=disk_free>=30*GIB, psi_full_avg10_below_1=psi<1)
        memory_result = 'ERROR' if errors else ('PASS' if checks['memory_6_gib'] else 'FAIL')
        screen = 'ERROR' if errors else ('PASS' if all(checks.values()) else 'FAIL')
        result = dict(timestamp=datetime.datetime.now().astimezone().isoformat(), label=args.label,
                      memory_gate=memory_result, entry_screen=screen, headless_status=headless,
                      available_bytes=mem['MemAvailable'], required_bytes=MEMORY_GATE,
                      available_gib=mem['MemAvailable']/GIB, swap_total_bytes=mem['SwapTotal'],
                      swap_free_bytes=mem['SwapFree'], disk_free_bytes=disk_free, psi_full_avg10=psi,
                      checks=checks, guard_sha256=guard_sha, zram=zram, errors=errors,
                      commands=dict(free=free,swap=swap,display_manager=dm,graphical_target=graphical,
                                    sessions=sessions,terminal=terminal,mount=mount))
        # A headless label is never accepted as evidence that graphical shutdown occurred.
        with (run/'result.json').open('x') as f:
            json.dump(result,f,indent=2)
        chunks = [f'MEMORY_GATE: {memory_result}',f'ENTRY_SCREEN: {screen}',f'HEADLESS_STATUS: {headless}',
                  f'MemAvailable: {mem["MemAvailable"]} bytes ({mem["MemAvailable"]/GIB:.3f} GiB)',
                  f'Required: {MEMORY_GATE} bytes (6 GiB)', 'This measurement does not launch inference.',
                  '\n/proc/meminfo\n'+memtext, '\n/proc/pressure/memory\n'+pressure,
                  '\nZram sysfs\n'+json.dumps(zram,indent=2)]
        for name,rec in result['commands'].items():
            chunks.append('\n'+name+': '+' '.join(rec['argv'])+'\nexit='+str(rec['returncode'])+'\n'+rec['stdout']+rec['stderr'])
        with (run/'measurement.txt').open('x') as f:
            f.write('\n'.join(chunks)+'\n')
        print(f'MEMORY_GATE: {memory_result} ({mem["MemAvailable"]/GIB:.3f} GiB available; requires 6 GiB)')
        print(f'ENTRY_SCREEN: {screen}; HEADLESS_STATUS: {headless}')
        print('LOG_DIRECTORY: '+str(run))
        print('No inference launched. Restore the graphical login after the test.')
        return 2 if errors else (0 if screen=='PASS' else 1)
    except Exception as exc:
        msg = 'MEASUREMENT_ERROR: '+type(exc).__name__+': '+str(exc)
        try:
            with (run/'error.txt').open('x') as f:
                f.write(msg+'\n')
        except OSError:
            pass
        print(msg+'\nLOG_DIRECTORY: '+str(run),file=sys.stderr)
        return 2

if __name__ == '__main__':
    sys.exit(main())
