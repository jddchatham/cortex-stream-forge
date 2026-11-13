"""Read-only qualification of preserved QIES111 binaries; no rebuild or repair.
Only synthetic actors are called, through their existing admission/monitoring.
"""
from pathlib import Path
import importlib.util,json,subprocess,os,sys,time,datetime,uuid,struct,traceback,argparse,signal
R=Path(__file__).resolve().parent;W=R.parent;O=W/'qies111-worktree';S=O/'qies111';B=S/'build'
s=importlib.util.spec_from_file_location('preserved_q111',S/'run.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)

def write(p,d):a.jsonwrite(p,d)
def identities():
 p=json.loads((R/'tested-provenance.json').read_text())
 for role,d in p['executables'].items():
  assert a.sha(Path(d['path']))==d['sha256'],'diagnosed executable changed'
  for n,h in d['build']['libraries'].items():assert a.sha(B/(role+'-libs')/n)==h
 manifest=json.loads((B/'manifest.json').read_text());assert all(a.sha(O/n)==h for n,h in manifest['source_sha256'].items())
 assert a.sha(S/'SOURCE-SHA256.json')==p['source_manifest_sha256']
 return p

def collect(base):
 jobs=[];peaks={role:{k:0 for k in ('rss','pss','anon','file_pss')} for role in ('candidate','reference')};native={role:0 for role in peaks};minimum=None
 for path in sorted(base.glob('*COMMITTED/layer-*/result.json')):
  d=json.loads(path.read_text());assert d['status']==0 and not d['reasons'] and d['rlimits_unchanged'];role=d['role']
  assert d['worker_records'][-1]['owned_resources']==0
  assert all(v.get('adapter_open_files_after_cleanup',0)==0 for v in d['worker_records'])
  for k,v in d['peak_tree_sampled'].items():peaks[role][k]=max(peaks[role][k],v)
  native[role]=max(native[role],max(v.get('peak_rss_kib',0) for v in d['worker_records']))
  minimum=d['minimum_system_MemAvailable'] if minimum is None else min(minimum,d['minimum_system_MemAvailable'])
  jobs.append({'result':str(path),'status':d['status'],'role':role,'signal':None,'elapsed_seconds':d['elapsed_seconds']})
 return {'jobs':jobs,'job_count':len(jobs),'peak_subtree_bytes':peaks,'native_high_water_rss_kib':native,'minimum_host_MemAvailable_bytes':minimum,'no_recorded_safety_violations':True}

def compare_original(base):
 old=S/'runs/20261111-174023-e367d945/layers-80-start-0'
 expected={(d['layer'],d['token_position'],d['tap']):d['sha256'] for d in map(json.loads,(old/'comparisons.jsonl').read_text().splitlines())}
 actual={(d['layer'],d['token_position'],d['tap']):d['sha256'] for d in map(json.loads,(base/'comparisons.jsonl').read_text().splitlines())};assert expected==actual and len(actual)==4160
 for (layer,pos,tap),h in actual.items():
  for role in ('candidate','reference'):assert a.sha(base/f'{role}-token-{pos}.COMMITTED/layer-{layer}/captures-0'/tap)==h
 return 4160

def faults(series,base):
 folder=series/'bounded-faults';folder.mkdir(exist_ok=False);prior=base/'candidate-token-0.COMMITTED/layer-0/captures-0/state.kv';embedding=base/'embedding-1.f32';old=a.sha(prior);cases=[]
 names=[x.strip().strip('\",') for x in (S/'faults.inc').read_text().splitlines()]
 for name in names:
  out=folder/name;out.mkdir();r=a.launch(out,'candidate',0,256,1,prior,embedding,embedding,fault=name,status=1)
  assert a.sha(prior)==old and r['worker_records'][-1]['fault_seen']
  assert not list(out.glob('*/captures-*')) and not list(out.glob('*/staging-*'))
  assert any(v.get('authoritative_state_unchanged') for v in r['worker_records'])
  cases.append({'fault':name,'exit_status':r['status'],'signal':None,'passed':True})
 # Invalid state/dimension cases mutate only diagnostic copies.
 for name,blob in [('truncated',prior.read_bytes()[:-1]),('corrupted',prior.read_bytes()[:129]+bytes([prior.read_bytes()[129]^1])+prior.read_bytes()[130:])]:
  f=folder/(name+'.kv');f.write_bytes(blob);out=folder/name;out.mkdir();r=a.launch(out,'candidate',0,256,1,f,embedding,embedding,status=1);cases.append({'fault':name,'exit_status':r['status'],'passed':True})
 for name,layer,pos,cp in [('cross-layer',1,1,prior),('wrong-position',0,2,prior),('invalid-layer',80,1,None)]:
  out=folder/name;out.mkdir();r=a.launch(out,'candidate',layer,256,pos,cp,embedding,embedding,status=1);cases.append({'fault':name,'exit_status':r['status'],'passed':True})
 write(folder/'summary.json',{'passed':True,'original_hooks':len(names),'state_contract_refusals':5,'cases':cases,'real_malloc_OOM_forced':False,'failure_model':'original bounded exception hooks at allocation/execution boundaries','source_checkpoint_unchanged':a.sha(prior)==old});print('PASS bounded faults',len(cases),flush=True)

def sanitizer(series,base):
 # Explicitly distinct, already-preserved instrumented build, never replace candidate.
 d=json.loads((S/'sanitizer-numerical-results/summary.json').read_text());sb=S/'sanitizer-build';assert a.sha(sb/'candidate')==d['worker_sha256']
 spec=importlib.util.spec_from_file_location('q111_instrumented',S/'run.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);q.B=sb;q.m.B=sb;q.m.monitor.B=sb
 ns=q.m.monitor.namespace
 def wrapped(role,out,args,marker=True):
  c=ns(role,out,args,marker);i=c.index('--');return c[:i]+['--setenv','ASAN_OPTIONS','detect_leaks=1:halt_on_error=1','--setenv','UBSAN_OPTIONS','halt_on_error=1:print_stacktrace=1']+c[i:]
 q.m.monitor.namespace=wrapped;folder=series/'sanitizer';folder.mkdir(exist_ok=False);rows=[]
 for name,fault,status in [('continuation','none',0),('kv-update','kv.after-value-update',1),('serialization','state.after-serialization',1),('publication','before-publish',1)]:
  out=folder/name;out.mkdir();r=q.launch(out,'candidate',0,256,1,base/'candidate-token-0.COMMITTED/layer-0/captures-0/state.kv',base/'embedding-1.f32',base/'embedding-1.f32',fault=fault,status=status)
  err=(out/'layer-0/stderr.log').read_text();assert 'ERROR: AddressSanitizer' not in err and 'runtime error:' not in err and 'LeakSanitizer' not in err
  if status==0:
   expected=base/'reference-token-1.COMMITTED/layer-0/captures-0'
   for p in (out/'layer-0/captures-0').iterdir():assert p.read_bytes()==(expected/p.name).read_bytes()
  rows.append({'name':name,'exit_status':r['status'],'passed':True})
 write(folder/'summary.json',{'passed':True,'instrumented_executable':str(sb/'candidate'),'sha256':d['worker_sha256'],'cases':rows,'scope':d['scope'],'leak_detection':True,'no_rebuild':True});print('PASS preserved sanitizer build',flush=True)

def interruption(series,base):
 # Terminate one synthetic layer after it reaches the Q projection. Existing
 # process-tree cleanup is called, and no production file is touched.
 folder=series/'interruption';folder.mkdir(exist_ok=False);out=folder/'layer-0';out.mkdir();prior=base/'candidate-token-0.COMMITTED/layer-0/captures-0/state.kv';emb=base/'embedding-1.f32';h=a.sha(prior)
 a.bindings=[(emb,'/state/input.f32'),(emb,'/state/original.f32'),(prior,'/state/previous.kv'),(S/'fixtures/layer-0/manifest','/fixtures/manifest'),(S/'fixtures/0/tensors.bin','/fixtures/tensors.bin')]
 args=['--synthetic','0','256','1','resume','contract-v1','none'];cmd=a.namespace('candidate',out,args);before=a.m.monitor.host();assert before['MemAvailable']>=1.5*1024**3 and before['SwapFree']>=2*1024**3 and before['disk_free']>=20*1024**3
 with (out/'stdout.jsonl').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
  p=subprocess.Popen(cmd,env=a.ENV,stdout=stdout,stderr=stderr,start_new_session=True);start=time.monotonic();trigger=False;peak=0
  try:
   while p.poll() is None:
    peak=max(peak,sum(x.get('Rss',0) for x in a.m.monitor.tree_memory(p.pid)));assert peak<=256*1024**2 and time.monotonic()-start<30
    if 'q.weights-live' in (out/'stdout.jsonl').read_text():trigger=True;break
    time.sleep(.005)
   assert trigger,'termination phase not reached';desc=a.m.monitor.subtree(p.pid);a.m.monitor.terminate(p)
  finally:
   if p.poll() is None:a.m.monitor.terminate(p)
 assert p.returncode!=0 and a.sha(prior)==h and not (out/'captures-0').exists()
 deadline=time.monotonic()+2;live=[]
 while True:
  live=[]
  for pid in desc:
   try:
    state=Path(f'/proc/{pid}/stat').read_text().split()[2]
    if state!='Z':live.append((pid,state))
   except FileNotFoundError:pass
  if not live or time.monotonic()>deadline:break
  time.sleep(.01)
 assert not live,'live descendants after cleanup'
 # Retry from the exact same prior checkpoint under unchanged admission.
 retry=folder/'retry';retry.mkdir();r=a.launch(retry,'candidate',0,256,1,prior,emb,emb)
 for f in (retry/'layer-0/captures-0').iterdir():assert f.read_bytes()==(base/'reference-token-1.COMMITTED/layer-0/captures-0'/f.name).read_bytes()
 write(folder/'summary.json',{'passed':True,'command':cmd,'intentional_termination':True,'exit_status':p.returncode,'signal':-p.returncode if p.returncode<0 else None,'peak_subtree_rss_bytes':peak,'source_unchanged':True,'no_published_partial_capture':True,'no_live_descendants':True,'retry_byte_exact':True,'uncommitted_disk_evidence_preserved':True});print('PASS interrupted cleanup and exact restart',flush=True)

def extend(series,base):
 prev={role:base/f'{role}-token-1.COMMITTED' for role in ('candidate','reference')};count=0
 with (base/'extension-comparisons.jsonl').open('x') as f:
  for pos in (2,3):
   emb=base/f'embedding-{pos}.f32';emb.write_bytes(struct.pack('<5120f',*[float((i*13+pos*7+7)%257-128)/128 for i in range(5120)]))
   for role in ('candidate','reference'):prev[role]=a.transaction(base,role,80,256,pos,prev[role],emb)
   count+=a.comparisons(prev['candidate'],prev['reference'],80,256,pos,f)
 write(base/'four-token-summary.json',{'success':True,'positions':[0,1,2,3],'additional_captures':count,'total_captures':4160+count,'no_binary_changes':True});print('PASS four-token synthetic stress',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trials',type=int,default=3,choices=[1,2,3]);ap.add_argument('--resume',type=Path);ap.add_argument('--four-tokens',action='store_true');args=ap.parse_args();p=identities();a.baseline()
 if args.resume:series=args.resume.resolve();series.relative_to((R/'runs').resolve())
 else:
  series=R/'runs'/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]);series.mkdir(parents=True);write(series/'PROVENANCE.json',p);(R/'active.json').write_text(json.dumps({'root':str(series)},indent=2)+'\n')
 assert json.loads((series/'PROVENANCE.json').read_text())==p
 try:
  for number in range(1,args.trials+1):
   trial=series/f'trial-{number}';trial.mkdir(exist_ok=True);summary=trial/'verified.json'
   if summary.exists():compare_original(trial/'layers-80-start-0');continue
   print('START trial',number,flush=True);started=time.monotonic();base=a.campaign(trial,80);captures=compare_original(base);d=collect(base);assert d['job_count']==320
   write(summary,dict(d,trial=number,exit_status=0,signal=None,captures=captures,elapsed_seconds_latest_attempt=time.monotonic()-started,original_capture_hashes_match=True));identities();print('PASS trial',number,'4160 captures',flush=True)
   if number==1:
    faults(series,base);sanitizer(series,base);interruption(series,base)
    a.campaign(series/'boundary',5,1023)
  if args.four_tokens:extend(series,series/f'trial-{args.trials}/layers-80-start-0')
  identities();write(series/'SUCCESS.json',{'success':True,'trials':args.trials,'four_tokens':args.four_tokens,'no_rebuild':True,'real_weights':False});print('DIAGNOSTIC COMPLETE',series,flush=True)
 except BaseException as e:
  diagnostic={'success':False,'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'stopped_on_first_unexplained_failure':True}
  path=series/('FAILURE-'+str(time.time_ns())+'.json');path.write_text(json.dumps(diagnostic,indent=2)+'\n');raise
if __name__=='__main__':main()
