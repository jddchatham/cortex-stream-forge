"""QIES111 isolated synthetic numerical failures and fail-closed entry tests."""
from pathlib import Path
import sys,importlib.util,json,shutil
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('q111',R/'run.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
campaign=Path(sys.argv[1]).resolve();source=campaign/'layers-2-start-0';O=R/'fault-results';O.mkdir(exist_ok=False);tests=[]
for fault in ('after-backend','qk.after-compute','kv.after-key-update','between-branches','state.after-serialization','capture.final.after','before-publish'):
 b=O/fault;b.mkdir();prior=b/'candidate-token-0.COMMITTED';shutil.copytree(source/prior.name,prior);old={p:a.sha(p) for p in prior.rglob('*') if p.is_file()}
 try:a.transaction(b,'candidate',2,256,1,prior,source/'embedding-1.f32',worker_fault=fault)
 except RuntimeError:pass
 else:raise AssertionError('injected fault accepted')
 assert not (b/'candidate-token-1.COMMITTED').exists()
 assert all(a.sha(p)==h for p,h in old.items())
 assert not list((b/'candidate-token-1.UNCOMMITTED').glob('*/captures-*'))
 tests.append({'name':fault,'passed':True,'authoritative_state_unchanged':True})
# Startup interfaces reject both actors before numerical allocation.
for role in ('candidate','reference'):
 for label,args,marker in [('real',['--real'],True),('no-marker',None,False)]:
  b=O/(role+'-'+label);b.mkdir();r=a.launch(b,role,0,256,0,None,source/'embedding-0.f32',source/'embedding-0.f32',status=1,marker=marker,argv=args)
  assert r['worker_records'][-1]['peak_weight_bytes']==0;tests.append({'name':role+'-'+label,'passed':True})
b=O/'cross-layer-kv';b.mkdir();r=a.launch(b,'candidate',1,256,1,source/'candidate-token-0.COMMITTED/layer-0/captures-0/state.kv',source/'embedding-1.f32',source/'embedding-1.f32',status=1);assert r['worker_records'][-1]['peak_weight_bytes']==0;tests.append({'name':'cross-layer-kv','passed':True})
# Corrupt only the binding of a synthetic plan. No real GGUF is visible.
oldns=a.m.monitor.namespace
def wrong_plan(role,out,args,marker=True):
 cmd=oldns(role,out,args,marker);correct=str(R/'fixtures/layer-1/manifest');return [str(R/'fixtures/layer-0/manifest') if v==correct else v for v in cmd]
a.m.monitor.namespace=wrong_plan
b=O/'wrong-weight-layer';b.mkdir();r=a.launch(b,'candidate',1,256,0,None,source/'embedding-0.f32',source/'embedding-0.f32',status=1);assert r['worker_records'][-1]['peak_weight_bytes']==0;tests.append({'name':'wrong-weight-layer','passed':True});a.m.monitor.namespace=oldns
(O/'summary.json').write_text(json.dumps({'passed':len(tests),'tests':tests,'synthetic_only':True},indent=2)+'\n');print('PASS numerical/entry faults:',len(tests))
