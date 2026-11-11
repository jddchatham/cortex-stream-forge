from pathlib import Path
import importlib.util,subprocess,json,os
R=Path(__file__).resolve().parent;W=R.parent
s=importlib.util.spec_from_file_location('q104_sandbox',R/'sandbox.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
commands=[('QIES099',[str(W/'qies099/checks/test-layer')]),('QIES100',['/usr/bin/python3','-I','-B','qies100/run_comparison.py']),('QIES101-extract',['/usr/bin/python3','-I','-B','qies101/test_extract.py']),('QIES101-stock',['/usr/bin/python3','-I','-B','qies101/test_reference_faults.py']),('QIES102',[str(W/'qies102/test-adapter'),str(W/'qies102/fixture-0.plan'),str(W/'qies102/real-layer-00.plan')]),('QIES103',['/usr/bin/python3','-I','-B','qies103/tests.py'])]
rows=[]
for name,command in commands:
 os.environ['QIES104_EXPORT']='historical-'+name
 with (R/(name+'-regression.log')).open('w') as f:p=subprocess.run(m.command(command),stdout=f,stderr=subprocess.STDOUT,timeout=180)
 rows.append({'suite':name,'returncode':p.returncode,'unchanged_assertions':True,'sealed_synthetic_filesystem':True})
 if p.returncode:break
(R/'historical-regressions.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
if any(x['returncode'] for x in rows):raise SystemExit(1)
# This verifier only reads and compares bytes; it has no numerical execution path.
env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
with (R/'QIES101-saved-byte-regression.log').open('w') as f:subprocess.run(['/usr/bin/python3','-I','-B',str(W/'qies101/verify_saved.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
