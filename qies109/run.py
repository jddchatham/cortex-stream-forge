"""QIES109: complete bounded synthetic layer, independent CPU oracle, immutable KV inputs."""
from pathlib import Path
import importlib.util,json,hashlib,subprocess,os,datetime,uuid,struct,argparse,shutil
R=Path(__file__).resolve().parent;W=R.parent;B=R/'build';env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
spec=importlib.util.spec_from_file_location('q108_monitor',W/'qies107/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.R=R;m.B=B;m.monitor.R=R;m.monitor.B=B
original_namespace=m.monitor.namespace;previous=None

def namespace(role,out,args,marker=True):
 cmd=original_namespace(role,out,args,marker);i=cmd.index('--');extra=['--dir','/state']
 if previous is not None:extra+=['--ro-bind',str(previous),'/state/previous.kv']
 return cmd[:i]+extra+cmd[i:]
m.monitor.namespace=namespace

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def baseline():
 protected=json.loads((R/'preservation.json').read_text());old=Path(protected['original_worktree'])
 assert all(sha(old/n)==h for n,h in protected['original_files'].items())
 fbase=old/'qies107/runs/20261111-130001-e209c0f6';f=json.loads((fbase/'summary.json').read_text())
 for q in f['comparisons']:
  path=fbase/q['case']/f"captures-{q['repeat']}"/(q['tap']+'.f32');refname=q['case'].replace('candidate','reference') if q['case']!='candidate-q5-repeat3' else 'reference-q5-v0'
  ref=fbase/refname/'captures-0'/(q['tap']+'.f32');assert sha(path)==q['sha256']==sha(ref)
 abase=old/'qies108/runs/20261111-130947-6dfa5b92';a=json.loads((abase/'summary.json').read_text())
 for q in a['comparisons']:assert sha(abase/q['candidate']/'captures-0'/q['tap'])==q['sha256']==sha(abase/q['reference']/'captures-0'/q['tap'])
 assert len(f['comparisons'])==63 and len(a['comparisons'])==228 and len(f['faults_passed'])==100 and len(a['faults'])==105
 for suite in (f,a):
  for j in suite['runs']:
   assert not j['reasons'] and j['rlimits_unchanged'] and j['worker_records'][-1]['owned_resources']==0
   if j['label'].startswith('fault-'):assert j['status']==1 and j['worker_records'][-1]['fault_seen']
 return {'FFN_captures':63,'FFN_faults':100,'attention_captures':228,'attention_faults':105,'actual_saved_bytes_reverified':True,'original_sources_unchanged':True}

def job(root,label,role,case,mode='seed',fault='none',state=None,status=0,marker=True,argv=None):
 global previous
 previous=state;oldhash=sha(state) if state else None
 out=root/label;cached=out/'attention-result.json'
 if cached.exists():
  result=json.loads(cached.read_text());assert result['worker_sha256']==sha(B/role) and result['input_checkpoint_sha256']==oldhash and result['status']==status and not result['reasons'] and result['rlimits_unchanged']
  assert result['worker_records'][-1]['owned_resources']==0
  assert result['command']==namespace(role,out,argv or ['--synthetic',case,mode,fault],marker)
  if status:assert not list(out.glob('captures-*')) and not list(out.glob('staging-*'))
  else:assert result['checkpoint_sha256']==sha(out/'captures-0/state.kv')
  return result
 if out.exists():out.rename(root/(label+'.interrupted-'+uuid.uuid4().hex[:8]))
 result=m.job(root,label,role,argv or ['--synthetic',case,mode,fault],status=status,marker=marker)
 assert not state or sha(state)==oldhash
 if status and state:assert any(v.get('authoritative_state_unchanged') for v in result['worker_records'])
 result['input_checkpoint_sha256']=oldhash;result['input_checkpoint_unchanged']=True;result['worker_sha256']=sha(B/role)
 if status==0:
  new=root/label/'captures-0/state.kv';data=new.read_bytes();h=struct.unpack('<8Q',data[:64]);assert len(data)==128+h[5]*4 and h[5]==h[6]
  assert hashlib.sha256(data[:64]+data[128:]).hexdigest()==data[64:128].decode()
  records=[d for d in result['worker_records'] if 'state_phase'in d];assert len(records)==2 and records[1]['position']==records[0]['position']+1==h[4]
  result['next_position']=h[4];result['checkpoint_sha256']=sha(new)
 (root/label/'attention-result.json').write_text(json.dumps(result,indent=2)+'\n');return result

def compare(root,c,r):
 cw={x['weight']:(x['sha256'],x['bytes']) for x in c['worker_records'] if 'weight'in x};rw={x['weight']:(x['sha256'],x['bytes']) for x in r['worker_records'] if 'weight'in x};assert len(cw)==8 and cw==rw
 a=root/c['label']/'captures-0';b=root/r['label']/'captures-0';assert {p.name for p in a.iterdir()}=={p.name for p in b.iterdir()};rows=[]
 for p in sorted(a.iterdir()):
  q=b/p.name;av=p.read_bytes();bv=q.read_bytes()
  if av!=bv:
   info={'candidate':c['label'],'reference':r['label'],'tap':p.name,'candidate_sha':sha(p),'reference_sha':sha(q),'candidate_bytes':len(av),'reference_bytes':len(bv)}
   if p.suffix=='.f32' and len(av)==len(bv):
    x=struct.unpack('<'+str(len(av)//4)+'f',av);y=struct.unpack('<'+str(len(bv)//4)+'f',bv);info.update(different=sum(i!=j for i,j in zip(x,y)),max_abs=max(abs(i-j) for i,j in zip(x,y)))
   (root/'mismatch.json').write_text(json.dumps(info,indent=2));raise AssertionError(('BYTE MISMATCH; NO TOLERANCE RELAXATION',info))
  rows.append({'candidate':c['label'],'reference':r['label'],'tap':p.name,'bytes':len(av),'byte_exact':True,'sha256':sha(p)})
 return rows

def masks(root,record,case,pos):
 p=root/record['label']/'captures-0';v=struct.unpack('<'+str((p/'mask.f32').stat().st_size//4)+'f',(p/'mask.f32').read_bytes());expected=[0. if i<=pos and (case.startswith('global') or pos-i<1024) else -float('inf') for i in range(len(v))];assert list(v)==expected
 scores=struct.unpack('<'+str((p/'probabilities.f32').stat().st_size//4)+'f',(p/'probabilities.f32').read_bytes());slots=len(v)
 assert all(scores[h*slots+i]==0 for h in range(80) for i,mask in enumerate(v) if mask==-float('inf'))
 if case.startswith('global'):
  assert (p/'q-current.f32').read_bytes()==(p/'q-norm.f32').read_bytes() and (p/'k-current.f32').read_bytes()==(p/'k-norm.f32').read_bytes()
 elif pos>0:assert (p/'q-current.f32').read_bytes()!=(p/'q-norm.f32').read_bytes()
 return {'case':case,'position':pos,'visible_keys':sum(x==0 for x in v),'masked_probabilities_exact_zero':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',type=Path);a=ap.parse_args();base=baseline()
 if a.resume:
  root=a.resume.resolve();root.relative_to((R/'runs').resolve());saved=json.loads((root/'build-manifest.json').read_text())
  assert saved==json.loads((B/'manifest.json').read_text())
  assert all(sha(W/n)==h for n,h in saved['source_sha256'].items())
  assert all(sha(B/role)==v['sha256'] for role,v in saved['workers'].items())
  assert not (root/'summary.json').exists(),'Campaign already completed; results preserved'
 else:
  subprocess.run(['/usr/bin/python3','-I','-B',R/'build.py'],env=env,check=True)
  root=R/'runs'/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
  (root/'build-manifest.json').write_bytes((B/'manifest.json').read_bytes())
 records=[];rows=[];coverage=[];accepted={}
 fixture=root/'fixture-validation/result.json'
 if fixture.exists():
  previous_fixture=json.loads(fixture.read_text());assert previous_fixture['status']==0 and previous_fixture['worker_records'][-1]['owned_resources']==0
  assert sum(v.get('three_rows_exact',False) for v in previous_fixture['worker_records'])==12
 else:previous_fixture=m.job(root,'fixture-validation','fixture-check',[])
 records.append(previous_fixture)
 for case,steps,start in ([('swa-start',1,0)] if a.smoke else [('swa-start',2,0),('global-start',2,0),('swa-boundary',3,1023),('global-boundary',3,1023)]):
  cp=rp=None
  for step in range(steps):
   mode='seed' if step==0 else 'resume'
   ref=job(root,f'reference-{case}-{step}','reference',case,mode,state=rp);records.append(ref)
   cand=job(root,f'candidate-{case}-{step}','candidate',case,mode,state=cp);records.append(cand);rows+=compare(root,cand,ref);coverage.append(masks(root,cand,case,start+step));accepted[(case,step)]=(cand,ref)
   cp=root/cand['label']/'captures-0/state.kv';rp=root/ref['label']/'captures-0/state.kv'
 if a.smoke:
  (root/'smoke.json').write_text(json.dumps({'comparisons':rows,'coverage':coverage,'runs':records},indent=2));print('PASS full-layer smoke:',root,flush=True);return
 # Deterministic reload of the same authoritative checkpoint must reproduce all
 # output/state bytes; input checkpoint is never written by the worker.
 prior=root/accepted[('swa-start',0)][0]['label']/'captures-0/state.kv'
 again=job(root,'candidate-repeated-continuation','candidate','swa-start','resume',state=prior);records.append(again);rows+=compare(root,again,accepted[('swa-start',1)][1])
 faultlist=[l.strip().strip('\",') for l in (R/'faults.inc').read_text().splitlines()]
 for fault in faultlist:
  print('Checking fault '+fault,flush=True)
  r=job(root,'fault-'+fault,'candidate','swa-start','resume',fault,state=prior,status=1);records.append(r);assert r['worker_records'][-1]['fault_seen'] is True
 # After failures the unchanged checkpoint still resumes to the exact oracle.
 retried=job(root,'candidate-after-failures','candidate','swa-start','resume',state=prior);records.append(retried);rows+=compare(root,retried,accepted[('swa-start',1)][1])
 # Malformed copies only: the accepted checkpoint itself is never changed.
 for label,data in [('truncated',prior.read_bytes()[:-2]),('corrupted',prior.read_bytes()[:129]+bytes([prior.read_bytes()[129]^1])+prior.read_bytes()[130:])]:
  f=root/(label+'.kv')
  if f.exists():assert f.read_bytes()==data
  else:f.write_bytes(data)
  r=job(root,'reject-state-'+label,'candidate','swa-start','resume',state=f,status=1);records.append(r)
 for role in ('candidate','reference'):
  records.append(job(root,'reject-real-'+role,role,'swa-start',status=1,argv=['--real']))
  records.append(job(root,'reject-no-marker-'+role,role,'swa-start',status=1,marker=False))
 result={'success':True,'baseline':base,'build':json.loads((B/'manifest.json').read_text()),'results_root':str(root),'comparisons':rows,'mask_and_rope_checks':coverage,'faults':faultlist,'runs':records,'continuation_uses_disk_checkpoints_and_fresh_processes':True,'boundary_seed_is_synthetic_history_not_1023_evaluated_tokens':True,'no_real_weights':True,'complete_synthetic_layer':True}
 (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS full layer:',len(rows),'byte-exact captures;',len(faultlist),'faults;',root/'summary.json',flush=True)
if __name__=='__main__':main()
