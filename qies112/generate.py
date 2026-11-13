"""Synthetic autoregression with an outer durable decision boundary.
Layer-stack publication alone never authorizes token feedback.
"""
from pathlib import Path
import sys,os,json,struct,subprocess,time,datetime,uuid,fcntl,hashlib,importlib.util,argparse
R=Path(__file__).resolve().parent;W=R.parent;sys.path.insert(0,str(R));from support import endpoint,compare,Sampler,sha,m,ENV
s=importlib.util.spec_from_file_location('q111_generation',W/'qies111/run.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
V=100352;BOS=100257;EOS=100265
CODE=['generate.py','support.py','role.py','io_candidate.cpp','io_reference.cpp','io_fixture.h']
def identities():
 for folder in (R/'build',W/'qies111/build'):
  manifest=json.loads((folder/'manifest.json').read_text())
  assert all(sha(W/n)==h for n,h in manifest['source_sha256'].items()),'build source identity'
  assert all(sha(folder/k)==v['sha256'] for k,v in manifest['workers'].items()),'executable identity'
 return {'version':1,'source':{n:sha(R/n) for n in CODE},'endpoint_build':sha(R/'build/manifest.json'),'layer_build':sha(W/'qies111/build/manifest.json'),'layer_controller':sha(W/'qies111/run.py'),'architecture':{'width':5120,'vocab':V,'layers':80,'embedding_type':'IQ4_XS','output_type':'Q6_K','untied':True,'embedding_scale':71.5541763305664},'synthetic_only':True}
def durable(path,data):a.jsonwrite(path,data)
def publish_decision(root,pos,data):
 target=root/f'decision-{pos}.json';tmp=root/(f'decision-{pos}.PENDING-'+uuid.uuid4().hex);durable(tmp,data);os.link(tmp,target);a.syncdir(root);tmp.unlink();a.syncdir(root)
def checked_decision(root,pos):
 d=json.loads((root/f'decision-{pos}.json').read_text());assert d['position']==pos and d['evaluated_count']==pos+1 and d['identity_sha256']==sha(root/'IDENTITY.json') and d['selected_ids']['candidate']==d['selected_ids']['reference'];assert d['previous_decision_sha256']==(sha(root/f'decision-{pos-1}.json') if pos else None)
 for role in ('candidate','reference'):
  cp=Path(d['checkpoints'][role]);a.verified(cp,80,256,pos+1);assert sha(cp/'COMMIT.json')==d['checkpoint_sha256'][role]
  stage=root/f'step-{pos}'
  for part in ('embedding','head'):
   job=stage/(role+'-'+part)
   assert sha(job/'REQUEST.json')==d['endpoint_request_sha256'][role][part]
   assert sha(job/'HASHES.json')==d['endpoint_hash_manifest_sha256'][role][part]
   for name,h in json.loads((job/'HASHES.json').read_text()).items():assert sha(job/'captures-0'/name)==h
   for row in d['comparisons']['embedding' if part=='embedding' else 'head']:assert sha(job/'captures-0'/row['file'])==row['sha256']
  assert sha(stage/(role+'-head')/'captures-0/logits.f32')==d['logits_sha256']
  selected=struct.unpack('<i',(stage/(role+'-head')/'captures-0/selected.i32').read_bytes())[0];assert selected==d['selected_ids'][role] and 0<=selected<V
 return d

def run_roles(root,pos,embeddings,priors):
 step=root/f'step-{pos}';children=[];logs=[];cg0=m.monitor.cgroups();start=time.monotonic();reasons=[]
 try:
  for role in ('candidate','reference'):
   request=step/(role+'-request.json');body={'synthetic_only':True,'role':role,'layers':80,'position':pos,'base':str(root/'stack'),'prior':str(priors[role]) if priors[role] else None,'embedding':str(embeddings[role])}
   if request.exists():assert json.loads(request.read_text())==body
   else:durable(request,body)
   log=(step/(role+'-layers-'+str(time.time_ns())+'.log')).open('x');logs.append(log);children.append(subprocess.Popen(['/usr/bin/python3','-I','-B',R/'role.py',request,str(os.getpid())],env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
  with Sampler(step/('parallel-aggregate-'+str(time.time_ns())+'.jsonl')) as sampler:
   while any(p.poll() is None for p in children):
    h=m.monitor.host()
    if h['MemAvailable']<1024**3 or h['SwapFree']<2*1024**3 or h['disk_free']<20*1024**3:reasons.append('existing synthetic system reserve')
    if sampler.error:raise RuntimeError(sampler.error)
    if sampler.peaks['tree_rss']>1024**3:reasons.append('additional 1 GiB generation-tree cap')
    if time.monotonic()-start>1200:reasons.append('additional 1200-second stack-step timeout')
    for path,stats in m.monitor.cgroups().items():
     if path in cg0 and any(stats['memory.events'].get(k,0)>cg0[path]['memory.events'].get(k,0) for k in ('oom','oom_kill','oom_group_kill')):reasons.append('ancestor OOM event')
    if any(p.poll() not in (None,0) for p in children):reasons.append('synthetic role failure')
    if reasons:raise RuntimeError(str(reasons))
    time.sleep(.1)
   assert all(p.wait()==0 for p in children),'branch exit failure'
  result={'statuses':[p.returncode for p in children],'elapsed_seconds':time.monotonic()-start,'aggregate_concurrent_peak_bytes':sampler.peaks,'samples':sampler.samples,'host_minima_bytes':sampler.minima,'safety_reasons':reasons,'cgroup_before':cg0,'cgroup_after':m.monitor.cgroups(),'shared_cgroup_not_worker_isolation':True}
  metrics=step/('parallel-result-'+str(time.time_ns())+'.json');durable(metrics,result);return result
 finally:
  for p in children:
   if p.poll() is None:m.monitor.terminate(p)
  for log in logs:log.close()

def step(root,pos,input_ids,priors):
 start=time.monotonic();folder=root/f'step-{pos}';folder.mkdir(exist_ok=True);embeddings={}
 for role in ('candidate','reference'):
  endpoint(folder,role+'-embedding',role,'embedding',input_ids[role]);embeddings[role]=folder/(role+'-embedding')/'captures-0/embedding.f32'
 emcomparisons=compare(folder/'candidate-embedding',folder/'reference-embedding');parallel=run_roles(root,pos,embeddings,priors)
 checkpoints={role:root/'stack'/f'{role}-token-{pos}.COMMITTED' for role in ('candidate','reference')}
 comp_path=folder/('layer-comparisons-'+str(time.time_ns())+'.jsonl')
 with comp_path.open('x') as output:count=a.comparisons(checkpoints['candidate'],checkpoints['reference'],80,256,pos,output)
 assert count==2080
 selected={};norm_comparisons=[]
 for role in ('candidate','reference'):
  input_path=checkpoints[role]/'layer-79/captures-0/final.f32';endpoint(folder,role+'-head',role,'head',input_ids[role],input_path)
  selected[role]=struct.unpack('<i',(folder/(role+'-head')/'captures-0/selected.i32').read_bytes())[0]
 norm_comparisons=compare(folder/'candidate-head',folder/'reference-head');assert selected['candidate']==selected['reference']
 # Independent Python greedy check, including finite values and lowest-ID ties.
 for role in ('candidate','reference'):
  logits=struct.unpack('<100352f',(folder/(role+'-head')/'captures-0/logits.f32').read_bytes());assert all(__import__('math').isfinite(x) for x in logits);assert max(range(V),key=logits.__getitem__)==selected[role]
 d={'version':1,'position':pos,'evaluated_count':pos+1,'input_ids':input_ids,'selected_ids':selected,'predicted_token_pending_evaluation':True,'eos':selected['candidate']==EOS,'identity_sha256':sha(root/'IDENTITY.json'),'previous_decision_sha256':sha(root/f'decision-{pos-1}.json') if pos else None,'checkpoints':{k:str(v) for k,v in checkpoints.items()},'checkpoint_sha256':{k:sha(v/'COMMIT.json') for k,v in checkpoints.items()},'logits_sha256':sha(folder/'candidate-head/captures-0/logits.f32'),'endpoint_request_sha256':{role:{part:sha(folder/(role+'-'+part)/'REQUEST.json') for part in ('embedding','head')} for role in ('candidate','reference')},'endpoint_hash_manifest_sha256':{role:{part:sha(folder/(role+'-'+part)/'HASHES.json') for part in ('embedding','head')} for role in ('candidate','reference')},'comparisons':{'layers':count,'embedding':emcomparisons,'head':norm_comparisons},'elapsed_seconds':time.monotonic()-start,'aggregate_concurrent_peak_bytes':parallel['aggregate_concurrent_peak_bytes'],'synthetic_only':True}
 for role in ('candidate','reference'):
  for part in ('embedding','head'):
   job=folder/(role+'-'+part)
   for name in ('REQUEST.json','HASHES.json'):
    fd=os.open(job/name,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
   a.syncdir(job/'captures-0');a.syncdir(job)
 a.syncdir(folder)
 publish_decision(root,pos,d);checked_decision(root,pos);print('ACCEPTED',pos,'input',input_ids,'selected',selected,'seconds',d['elapsed_seconds'],flush=True);return d

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tokens',type=int,choices=[2,4,8,16],default=2);ap.add_argument('--resume',type=Path);args=ap.parse_args()
 base=json.loads((R/'baseline-active.json').read_text());assert (Path(base['root'])/'SUCCESS.json').exists(),'QIES111 replay must finish before full generation'
 assert json.loads((R/'baseline-verification.json').read_text())['success']
 assert json.loads((R/'endpoint-results/summary.json').read_text())['passed']
 assert json.loads((R/'sanitizer-results/summary.json').read_text())['passed']==7
 assert json.loads((R/'integration-fault-results/summary.json').read_text())['passed']==5
 assert json.loads((R/'parallel-control-final/summary.json').read_text())['passed']
 assert json.loads((R/'termination-result-final/summary.json').read_text())['passed']
 for folder,n in [('contract-results-q112',31),('fault-results',13),('recovery-results',2),('sanitizer-numerical-results',4)]:assert json.loads((W/'qies111'/folder/'summary.json').read_text())['passed']==n
 if args.resume:
  root=args.resume.resolve();root.relative_to((R/'runs').resolve());assert json.loads((root/'IDENTITY.json').read_text())==identities(),'generation source/build identity mismatch'
 else:
  root=R/'runs'/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True);(root/'stack').mkdir();durable(root/'IDENTITY.json',identities());(R/'active-generation.json').write_text(json.dumps({'root':str(root)},indent=2)+'\n')
 lock=(root/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);input_ids={k:BOS for k in ('candidate','reference')};priors={k:None for k in input_ids};sequence=[];stop='token_limit'
 for pos in range(args.tokens):
  if (root/f'decision-{pos}.json').exists():d=checked_decision(root,pos);assert d['input_ids']==input_ids
  else:
   if (root/'CANCEL').exists():stop='cancelled_before_next_transaction';break
   d=step(root,pos,input_ids,priors)
  sequence.append(d['selected_ids']['candidate']);input_ids=d['selected_ids'];priors={k:Path(v) for k,v in d['checkpoints'].items()}
  if d['eos']:stop='EOS';break
  if len(sequence) in (2,4,8,16):
   p=root/f'MILESTONE-{len(sequence)}.json'
   if not p.exists():durable(p,{'success':True,'predicted_token_ids':sequence,'evaluated_inputs':len(sequence),'last_prediction_not_evaluated':True,'next_input_ids':input_ids,'real_weights':False})
 result={'predicted_token_ids':sequence,'evaluated_inputs':len(sequence),'requested_tokens':args.tokens,'stop':stop,'root':str(root),'real_weights':False}
 durable(root/('RESULT-'+str(time.time_ns())+'.json'),result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
