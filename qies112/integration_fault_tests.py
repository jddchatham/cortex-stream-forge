"""Invoke the real generation-step control using simulated stacks/endpoints.
No model or numerical GGML execution; numerical evidence is recorded separately.
"""
from pathlib import Path
import sys,importlib.util,json,struct,hashlib
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));s=importlib.util.spec_from_file_location('g112',R/'generate.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
O=R/'integration-fault-results';O.mkdir(exist_ok=False);tests=[];active=None
original_publish=g.publish_decision

def fake_endpoint(folder,label,role,op,token,input_path=None,**kw):
 job=folder/label;caps=job/'captures-0';caps.mkdir(parents=True,exist_ok=True)
 if op=='embedding':(caps/'embedding.f32').write_bytes(bytes(5120*4));(caps/'embedding-unscaled.f32').write_bytes(bytes(5120*4))
 else:
  if active=='after-stack':raise RuntimeError('after-stack')
  logits=[0.]*g.V;logits[7]=1.;(caps/'logits.f32').write_bytes(struct.pack('<100352f',*logits));(caps/'final-norm.f32').write_bytes(bytes(5120*4));(caps/'selected.i32').write_bytes(struct.pack('<i',7))
  if active=='after-head':raise RuntimeError('after-head')
 (job/'HASHES.json').write_text(json.dumps({p.name:g.sha(p) for p in caps.iterdir()}));(job/'REQUEST.json').write_text(json.dumps({'role':role,'token':token,'mock':True}))
 if active=='after-embedding' and op=='embedding' and role=='reference':raise RuntimeError('after-embedding')
 return {}
def fake_roles(root,pos,embeddings,priors):
 if active=='before-stack':raise RuntimeError('before-stack')
 for role in ('candidate','reference'):
  cp=root/'stack'/f'{role}-token-{pos}.COMMITTED';(cp/'layer-79/captures-0').mkdir(parents=True,exist_ok=True);(cp/'COMMIT.json').write_text('{}');(cp/'layer-79/captures-0/final.f32').write_bytes(bytes(5120*4))
 return {'aggregate_concurrent_peak_bytes':{},'statuses':[0,0]}
def fake_publish(root,pos,data):
 if active=='before-publication':raise RuntimeError('before-publication')
 return original_publish(root,pos,data)
g.endpoint=fake_endpoint;g.run_roles=fake_roles;g.publish_decision=fake_publish;g.a.comparisons=lambda *args:2080;g.checked_decision=lambda root,pos:json.loads((root/f'decision-{pos}.json').read_text())
for boundary in ('after-embedding','before-stack','after-stack','after-head','before-publication'):
 root=O/boundary;root.mkdir();(root/'IDENTITY.json').write_text('{}');(root/'decision-0.json').write_text('{"prior":true}');before=g.sha(root/'decision-0.json');active=boundary
 try:g.step(root,1,{'candidate':g.BOS,'reference':g.BOS},{'candidate':None,'reference':None})
 except RuntimeError:pass
 else:raise AssertionError('failure did not propagate')
 assert not (root/'decision-1.json').exists() and g.sha(root/'decision-0.json')==before
 active=None;d=g.step(root,1,{'candidate':g.BOS,'reference':g.BOS},{'candidate':None,'reference':None});assert d['evaluated_count']==2 and d['selected_ids']['candidate']==7 and d['predicted_token_pending_evaluation'];assert g.sha(root/'decision-0.json')==before
 tests.append({'boundary':boundary,'failure_not_promoted':True,'retry_completed':True,'previous_decision_unchanged':True})
(O/'summary.json').write_text(json.dumps({'passed':len(tests),'tests':tests,'simulated_stack_only':True,'real_generation_step_control':True},indent=2)+'\n');print('PASS actual step-control fault/retry:',len(tests))
