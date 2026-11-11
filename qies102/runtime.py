"""Prepared original-policy launcher; missing qualifications always block execution."""
from pathlib import Path
import argparse,importlib.util,hashlib,json,os,sys,struct
R=Path(__file__).resolve().parent;W=R.parent
GUARD='ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2'
MODEL='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb',buffering=0) as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def guard():
 p=W.parent/'qies077-guard.py'
 if sha(p)!=GUARD:raise ValueError('guard identity changed')
 return module('q102_original_guard',p)
def qualifications(reference,receipt):
 if reference is None or receipt is None:raise ValueError('real stock reference and live guard qualification absent')
 ref=json.loads(reference.read_text());old=module('q102_q100_contract',W/'qies100/guard_adapter.py');old.validate_reference_files(ref)
 old99=module('q102_q099_contract',W/'qies099/launch_real.py');old99.validate_reference(reference,0)
 if ref.get('use_extra_bufts') is not False or ref.get('position')!=0 or ref.get('previous_kv')!='-':raise ValueError('first comparison requires matched plain CPU, layer 0, position 0, empty prior KV')
 q=json.loads(receipt.read_text());binary=R/'probe'
 if q.get('status')!='QUALIFIED_HARMLESS_CHILDREN' or q.get('guard_sha256')!=GUARD or q.get('launcher_sha256')!=sha(__file__) or q.get('binary_sha256')!=sha(binary):raise ValueError('live adapter qualification identity mismatch')
 expected={'normal','failure','descendants','interrupt','monitor_failure'}
 if set(q.get('trials',{}))!=expected:raise ValueError('missing live qualification trials')
 for name,item in q['trials'].items():
  if sha(item['path'])!=item['sha256']:raise ValueError('qualification evidence hash')
  data=json.loads(Path(item['path']).read_text())
  if data.get('cleanup') is not True or data.get('ok') is not (name=='normal'):raise ValueError('live qualification trial result')
 return ref

def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--reference',type=Path);p.add_argument('--guard-qualification',type=Path);p.add_argument('--run-root',type=Path);a=p.parse_args()
 g=guard();h=g.host(R);ready=h['available']>=6*g.G and h['swap_free']>=4*g.G and h['disk_free']>=30*g.G and h['psi']<1
 ref=None;error=None
 try:ref=qualifications(a.reference,a.guard_qualification)
 except (OSError,ValueError,KeyError,TypeError) as e:error=str(e)
 status={'host':h,'entry_pass':ready,'qualified_reference_and_guard':ref is not None,'blocker':error,'real_numerical_execution':False,'guard_policy_unchanged':True}
 if a.check:print(json.dumps(status,indent=2));return 0 if ready and ref else 1
 if not ready or ref is None:raise SystemExit('BLOCKED: '+str(status))
 if a.run_root is None or not a.run_root.is_dir():raise ValueError('pre-existing persistent SSD run root required')
 run=a.run_root.resolve();model=W.parent/'qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf'
 if run==Path('/tmp') or Path('/tmp') in run.parents or run.stat().st_dev!=model.stat().st_dev:raise ValueError('run root must use verified persistent SSD')
 build=json.loads((R/'build-identity.json').read_text());binary=R/'probe'
 if sha(binary)!=build['binary_sha256']:raise ValueError('candidate identity changed')
 m=module('q102_index_identity',W/'qies101/extract.py');receipt=json.loads((R/'model-hash-result.json').read_text());bundle=json.loads((R/'bundle-manifest.json').read_text())
 if receipt['actual']!=MODEL or bundle['model_sha256']!=MODEL or m.identity(model.stat())!=receipt['identity']:raise ValueError('model identity')
 if sha(R/'real-layer-00.plan')!=bundle['plan_sha256']:raise ValueError('bundle plan changed')
 for t in bundle['tensors']:
  if m.identity(Path(t['path']).stat())!=t['identity'] or sha(t['path'])!=t['sha256']:raise ValueError('tensor identity')
 command=['/usr/bin/env','-i','PATH=/usr/bin:/bin','HOME='+str(Path.home()),'LANG=C.UTF-8','OMP_NUM_THREADS=1','QIES102_QUALIFIED_GUARDED_RUN=1',str(binary),'--real',str(R/'real-layer-00.plan'),str(Path(ref['input']['path']).resolve()),str(Path(ref['x0']['path']).resolve()),'-',str(run/'layer-output')]
 code=g.run_guard(command,run/'guard-log') # Original Policy(), no overrides.
 if code:return code
 result=json.loads((run/'guard-log/result.json').read_text())
 if result.get('ok') is not True or result.get('cleanup') is not True:raise ValueError('guard result not successful/clean')
 out=run/'layer-output';state=(out/'next.kv').read_bytes();header=struct.unpack('<24Q',state[:192]);count=128*8*256
 if header[11]!=1 or header[12]!=count or len(state)!=192+count*4:raise ValueError('KV position/length mismatch')
 values={name:sha(out/name) for name in ref['outputs'] if name.endswith('.f32')}
 values['key-written.f16']=hashlib.sha256(state[192:192+128*8*2]).hexdigest();values['value-written.f16']=hashlib.sha256(state[192+count*2:192+count*2+128*8*2]).hexdigest()
 equal=values==ref['outputs'];(run/'comparison.json').write_text(json.dumps({'all_bytes_equal':equal,'candidate':values,'reference':ref['outputs']},indent=2)+'\n');return 0 if equal else 1
if __name__=='__main__':sys.exit(main())
