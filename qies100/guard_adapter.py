"""Fail-closed qualification adapter. No policy, resource-limit or cgroup mutations."""
from pathlib import Path
import importlib.util,json,hashlib,sys,resource,os
ROOT=Path(__file__).resolve().parent.parent
GUARD=ROOT.parent/'qies077-guard.py'
GUARD_SHA='ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2'
G=1024**3
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_guard():
 if sha(GUARD)!=GUARD_SHA:raise ValueError('original guard identity mismatch')
 s=importlib.util.spec_from_file_location('qies100_unchanged_guard',GUARD);g=importlib.util.module_from_spec(s);sys.modules[s.name]=g;s.loader.exec_module(g);return g
def entry(h):
 # Pure predicate mirrors original fixed entry rules, for model-free boundary tests.
 return h['available']>=6*G and h['swap_free']>=4*G and h['disk_free']>=30*G and h['psi']<1

def inspect_limits():
 rel=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
 cg=Path('/sys/fs/cgroup')/rel.lstrip('/');rows=[]
 for p in (cg,*cg.parents):
  if p==Path('/sys'):break
  row={'path':str(p)}
  for name in ('memory.max','memory.high','memory.swap.max','memory.current'):
   f=p/name
   if f.exists():row[name]=f.read_text().strip()
  if len(row)>1:rows.append(row)
 return {'rlimit_as':resource.getrlimit(resource.RLIMIT_AS),'cgroup_v2':Path('/sys/fs/cgroup/cgroup.controllers').exists(),'ancestors':rows}

def checks():
 g=load_guard();h=g.host(ROOT.parent/'qies098-data');limits=inspect_limits()
 return {'guard_sha256':GUARD_SHA,'host':h,'entry_pass':entry(h),'limits':limits,'reference_qualified':False,'real_guard_adapter_qualified':False,'ready':False,'inference_executed':False,'blockers':['independent reference not real-model qualified','live passing-gate guard/cgroup/address-space enforcement qualification pending']}

def run_harmless_rejection(logdir):
 g=load_guard()
 if entry(g.host(ROOT.parent/'qies098-data')):raise ValueError('This rejection-only test requires a naturally failing host; no admission override')
 code=g.run_guard(['/usr/bin/true'],Path(logdir))
 result=json.loads((Path(logdir)/'result.json').read_text())
 if code==0 or result.get('ok') or result.get('reason')!='GuardAbort: resource preflight failed':raise ValueError('original guard did not fail closed')
 return result

def validate_reference_files(ref):
 """Identity checks are necessary evidence; they do not confer numerical qualification."""
 if ref.get('status')!='RUNTIME_QUALIFIED_RICURSIVE_LAYER':raise ValueError('Reference not runtime qualified')
 if ref.get('source_revision')!='240d68fcfe0c2a1f298c8e18ca181368b1914259':raise ValueError('Wrong stock source')
 for key in ('binary','source_manifest'):
  item=ref[key]
  if not Path(item['path']).is_file() or sha(item['path'])!=item['sha256']:raise ValueError('Reference artifact identity mismatch')
 if not ref.get('libraries'):raise ValueError('No reference library identities')
 for item in ref['libraries']:
  if sha(item['path'])!=item['sha256']:raise ValueError('Reference library mismatch')
 if ref.get('backend')!='CPU' or ref.get('threads')!=1 or ref.get('flash_attention') is not False:raise ValueError('Reference execution mismatch')
 if ref.get('model_sha256')!='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc':raise ValueError('Wrong model')
 if ref.get('layer') not in (0,4) or ref.get('context')!=256 or ref.get('kv_type')!='F16':raise ValueError('Reference layout mismatch')
 return True

def real_command(*args):
 # No qualification receipt is issued by this milestone. This is an intentional interlock.
 raise RuntimeError('BLOCKED: stock real-layer extraction and live unchanged-policy guard qualification incomplete')
if __name__=='__main__':
 print(json.dumps(checks(),indent=2));sys.exit(1)
