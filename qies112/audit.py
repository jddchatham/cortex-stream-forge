"""Read-only independent artifact audit. No numerical inference or tolerance."""
from pathlib import Path
import json,struct,hashlib,sys,math,importlib.util,argparse
R=Path(__file__).resolve().parent;W=R.parent
s=importlib.util.spec_from_file_location('generation112',R/'generate.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
ap=argparse.ArgumentParser();ap.add_argument('--tokens',type=int,required=True);ap.add_argument('--root',type=Path);args=ap.parse_args();root=args.root or Path(json.loads((R/'active-generation.json').read_text())['root'])
assert json.loads((root/'IDENTITY.json').read_text())==g.identities()
rows=[];expected={'candidate':g.BOS,'reference':g.BOS};total=0;weight_count=0
for pos in range(args.tokens):
 d=g.checked_decision(root,pos);assert d['input_ids']==expected and d['selected_ids']['candidate']==d['selected_ids']['reference'];stage=root/f'step-{pos}';count=0;weights=0;role_results={}
 for role in ('candidate','reference'):
  cp=Path(d['checkpoints'][role]);embedding=stage/(role+'-embedding')/'captures-0/embedding.f32';original_hash=g.sha(embedding)
  raw=(stage/(role+'-embedding')/'captures-0/embedding-unscaled.f32').read_bytes();unscaled=struct.unpack('<5120f',raw);scaled=struct.pack('<5120f',*[x*71.5541763305664 for x in unscaled]);assert scaled==embedding.read_bytes(),'embedding scale mismatch'
  logits_path=stage/(role+'-head')/'captures-0/logits.f32';logits=struct.unpack('<100352f',logits_path.read_bytes());assert all(math.isfinite(v) for v in logits);selected=max(range(100352),key=logits.__getitem__);assert selected==d['selected_ids'][role]
  job_results=[]
  for layer in range(80):
   job=cp/f'layer-{layer}';request=json.loads((job/'INPUT.json').read_text());res=json.loads((job/'result.json').read_text());assert res['status']==0 and not res['reasons'] and res['rlimits_unchanged'];assert res['worker_records'][-1]['owned_resources']==0
   assert request['layer']==layer and request['position']==pos and request['slots']==256 and request['original_sha256']==original_hash
   inp=embedding if layer==0 else cp/f'layer-{layer-1}/captures-0/final.f32';assert request['input_sha256']==g.sha(inp)
   prev=root/'stack'/f'{role}-token-{pos-1}.COMMITTED/layer-{layer}/captures-0/state.kv' if pos else None;assert request['previous']==(g.sha(prev) if prev else None)
   assert all(v.get('adapter_open_files_after_cleanup',0)==0 for v in res['worker_records']);job_results.append(res)
  head_request=json.loads((stage/(role+'-head')/'REQUEST.json').read_text());assert head_request['input_sha256']==g.sha(cp/'layer-79/captures-0/final.f32') and head_request['token']==expected[role]
  embed_request=json.loads((stage/(role+'-embedding')/'REQUEST.json').read_text());assert embed_request['token']==expected[role]
  role_results[role]=job_results
 for layer in range(80):
  a=Path(d['checkpoints']['candidate'])/f'layer-{layer}/captures-0';b=Path(d['checkpoints']['reference'])/f'layer-{layer}/captures-0'
  assert {p.name for p in a.iterdir()}=={p.name for p in b.iterdir()}
  for p in a.iterdir():assert p.read_bytes()==(b/p.name).read_bytes(),('capture mismatch',pos,layer,p.name);count+=1
  inventories=[]
  for role in ('candidate','reference'):
   inventories.append({x['weight']:(x['sha256'],x['bytes']) for x in role_results[role][layer]['worker_records'] if 'weight' in x})
  assert len(inventories[0])==8 and inventories[0]==inventories[1];weights+=8
  expected_ffn=144179200 if layer<15 or layer>=65 else 117964800;assert all(inventories[0][name][1]==expected_ffn for name in ('ffn-gate','ffn-up','ffn-down'))
  mask=struct.unpack('<256f',(a/'mask.f32').read_bytes());assert list(mask)==[0. if i<=pos and (layer%5==4 or pos-i<1024) else -float('inf') for i in range(256)]
  probability=struct.unpack('<20480f',(a/'probabilities.f32').read_bytes());assert all(probability[h*256+i]==0. for h in range(80) for i in range(256) if mask[i]==-float('inf'))
  if layer%5==4:assert (a/'q-current.f32').read_bytes()==(a/'q-norm.f32').read_bytes()
  elif pos:assert (a/'q-current.f32').read_bytes()!=(a/'q-norm.f32').read_bytes()
 for part in ('embedding','head'):
  ca=stage/('candidate-'+part);re=stage/('reference-'+part)
  for p in (ca/'captures-0').iterdir():assert p.read_bytes()==(re/'captures-0'/p.name).read_bytes();count+=1
  ci={x['weight']:(x['sha256'],x['bytes']) for x in json.loads((ca/'result.json').read_text())['worker_records'] if 'weight' in x};ri={x['weight']:(x['sha256'],x['bytes']) for x in json.loads((re/'result.json').read_text())['worker_records'] if 'weight' in x};assert len(ci)==1 and ci==ri;weights+=1
 assert count==2085 and weights==642
 result_files=list(stage.glob('parallel-result-*.json'));assert result_files
 parallel=json.loads(sorted(result_files)[-1].read_text());assert parallel['statuses']==[0,0] and not parallel['safety_reasons']
 peaks=dict(parallel['aggregate_concurrent_peak_bytes']);host=dict(parallel['host_minima_bytes']);endpoint_times=[]
 for role in ('candidate','reference'):
  for part in ('embedding','head'):
   res=json.loads((stage/(role+'-'+part)/'result.json').read_text());assert res['status']==0 and not res['reasons'] and res['worker_records'][-1]['owned_resources']==0
   for k,v in res['aggregate_concurrent_peak_bytes'].items():peaks[k]=max(peaks[k],v)
   for k,v in res.get('host_minima_bytes',{}).items():host[k]=min(host[k],v)
   endpoint_times.extend(dict(x,role=role) for x in res['worker_records'] if 'compute_seconds' in x)
 rows.append({'position':pos,'input_ids':expected,'selected_ids':d['selected_ids'],'capture_comparisons':count,'packed_weight_hash_comparisons':weights,'elapsed_seconds_recorded':d['elapsed_seconds'],'parallel_stack_seconds_latest_attempt':parallel['elapsed_seconds'],'concurrent_peak_bytes':peaks,'host_minima_bytes':host,'endpoint_timings':endpoint_times,'persistent_checkpoint_bytes_per_branch':80*(128+256*4096),'eos':d['eos']})
 expected=d['selected_ids'];total+=count;weight_count+=weights
out=R/f'audit-{args.tokens}.json';assert not out.exists();out.write_text(json.dumps({'success':True,'root':str(root),'tokens':args.tokens,'selected_sequence':[r['selected_ids']['candidate'] for r in rows],'capture_comparisons':total,'packed_weight_hash_comparisons':weight_count,'rows':rows,'last_prediction_pending_evaluation':True,'real_weights':False,'synthetic_weight_profiles_repeated':True},indent=2)+'\n');print('PASS independent generation audit:',args.tokens,'predictions;',total,'exact captures;',weight_count,'packed-weight hashes')
