"""Validate the already-built closure release; no real-weight execution."""
from pathlib import Path
import subprocess,json,hashlib,os,shutil
R=Path(__file__).resolve().parent;env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
for row in json.loads((R/'builds.json').read_text()):assert hashlib.sha256((R/row['name']).read_bytes()).hexdigest()==row['sha256']
for script in ('test_closure.py','test_active.py','regress.py','status.py','reference_contract.py'):
 p=subprocess.run(['/usr/bin/python3','-I','-B',str(R/script)],env=env,check=True)
print('PASS corrected full validation; allocation detector positive control verified; no real-weight arithmetic')
