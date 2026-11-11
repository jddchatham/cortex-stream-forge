"""Shared Python boundary: synthetic namespace or locked real execution."""
from pathlib import Path
import os
G=1024**3
def legacy_build_only():
 if os.environ.get('QIES104_SYNTHETIC')!='1' or not Path('/qies104.synthetic').is_file():
  raise SystemExit('QIES104 BLOCKED: legacy worker/build entry retired; use the sealed QIES104 synthetic launcher')
def real(role):
 values={k:int(v.split()[0])*1024 for k,v in (row.split(':',1) for row in Path('/proc/meminfo').read_text().splitlines())}
 if values['MemAvailable']<6*G:raise SystemExit(f'QIES104 BLOCKED: {role}: unchanged 6 GiB MemAvailable gate; no model allocation')
 raise SystemExit(f'QIES104 BLOCKED: {role}: real execution locked pending live qualification; no model allocation')
if __name__=='__main__':
 import sys
 real(sys.argv[1] if len(sys.argv)>1 else 'missing role')
