from pathlib import Path
import subprocess,os,json,hashlib,importlib.util,struct
R=Path(__file__).resolve().parent;W=R.parent;env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
def run(cmd,log):
 with (R/log).open('w') as f:subprocess.run(list(map(str,cmd)),env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
# Reconfirm original eight actual tensor comparisons without altering their receipts.
run(['/usr/bin/python3','-I','-B',W/'qies101/verify_saved.py'],'qies101-reconfirmation.log')
counts={}
for label,stockfolder in [('fixture','stock-captures'),('fixture-quantized','stock-captures-quantized-plain')]:
 out=R/('captures-'+label);run([R/'probe','--synthetic',R/(label+'-'),out],label+'-run.log');rows=[]
 for p in sorted(out.glob('*.bin')):
  q=W/'qies100'/stockfolder/p.name
  if not q.exists() and p.name.endswith('-4.bin'):q=q.with_name(p.name.replace('Qcur_normed','Qcur').replace('Kcur_normed','Kcur'))
  a=p.read_bytes();b=q.read_bytes();assert a==b,p.name
  rows.append({'name':p.name,'reference':str(q),'sha256':hashlib.sha256(a).hexdigest(),'bytes':len(a),'byte_exact':True})
 assert len(rows)==360;counts[label]=len(rows);(R/(label+'-comparison.json')).write_text(json.dumps(rows,indent=2)+'\n')
run([R/'test-adapter',R/'fixture-0.plan',R/'real-layer-00.plan'],'adapter-tests.log')
run([R/'probe','--storage',R/'real-layer-00.plan'],'real-storage.tsv')
s=importlib.util.spec_from_file_location('q102_runtime',R/'runtime.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
try:m.qualifications(None,None)
except ValueError:pass
else:raise AssertionError('missing qualifications accepted')
r=subprocess.run(['/usr/bin/python3','-I','-B',str(R/'runtime.py'),'--check'],env=env,text=True,capture_output=True);assert r.returncode==1
(R/'runtime-readiness.json').write_text(r.stdout);status=json.loads(r.stdout);assert not status['qualified_reference_and_guard'] and status['real_numerical_execution'] is False
(R/'test-summary.json').write_text(json.dumps({'synthetic_byte_exact':counts,'qies101_saved_comparisons_pass':True,'real_weight_storage_only':True,'real_weight_arithmetic':False,'missing_runtime_qualification_rejected':True},indent=2)+'\n')
print('PASS 720 synthetic intermediate comparisons, adapter failures, actual quantized storage roundtrip, unchanged runtime blocking')
