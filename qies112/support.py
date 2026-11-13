from pathlib import Path
import importlib.util,json,hashlib,os,time,threading
R=Path(__file__).resolve().parent;W=R.parent;B=R/'build'
s=importlib.util.spec_from_file_location('q107_endpoint',W/'qies107/run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.R=R;m.B=B;m.monitor.R=R;m.monitor.B=B
original=m.monitor.namespace;bindings=[]
def namespace(role,out,args,marker=True):
 cmd=original(role,out,args,marker);i=cmd.index('--');extra=['--dir','/state']
 for source,target in bindings:extra+=['--ro-bind',str(source),target]
 return cmd[:i]+extra+cmd[i:]
m.monitor.namespace=namespace
ENV=os.environ.copy();ENV.pop('LD_LIBRARY_PATH',None)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
class Sampler:
 def __init__(self,path):self.path=Path(path);self.done=threading.Event();self.peaks={k:0 for k in ('tree_rss','tree_pss','controller_rss','controller_pss')};self.samples=0;self.ready=threading.Event();self.error=None;self.minima={k:None for k in ("MemAvailable","SwapFree","disk_free")}
 def __enter__(self):self.t=threading.Thread(target=self.loop,daemon=False);self.t.start();assert self.ready.wait(5),"sampler initialization timeout";assert not self.error,self.error;return self
 def loop(self):
  try:self._loop()
  except BaseException as e:self.error=repr(e);self.ready.set();self.done.set()
 def _loop(self):
  with self.path.open('x') as f:
   while not self.done.is_set():
    rows=m.monitor.tree_memory(os.getpid());own=next((r for r in rows if r['pid']==os.getpid()),{})
    current={'tree_rss':sum(r.get('Rss',0) for r in rows),'tree_pss':sum(r.get('Pss',0) for r in rows),'controller_rss':own.get('Rss',0),'controller_pss':own.get('Pss',0)}
    for k,v in current.items():self.peaks[k]=max(self.peaks[k],v)
    host=m.monitor.host()
    for k in self.minima:self.minima[k]=host[k] if self.minima[k] is None else min(self.minima[k],host[k])
    current['host']=host
    f.write(json.dumps(dict(current,monotonic=time.monotonic()))+'\n');f.flush();self.samples+=1;self.ready.set();self.done.wait(.05)
 def __exit__(self,*args):self.done.set();self.t.join();assert self.samples>0 and not self.error,"aggregate sampler produced no samples or failed"
def endpoint(root,label,role,op,token,input_path=None,fault='none',status=0,marker=True,argv=None):
 global bindings
 root=Path(root);bindings=[(input_path,'/state/input.f32')] if input_path else [];out=root/label
 request={'role':role,'op':op,'token':token,'fault':fault,'input_sha256':sha(input_path) if input_path else None,'worker_sha256':sha(B/role),'marker':marker,'argv':argv}
 if (out/'REQUEST.json').exists() and (out/'HASHES.json').exists():
  assert json.loads((out/'REQUEST.json').read_text())==request
  for n,h in json.loads((out/'HASHES.json').read_text()).items():assert sha(out/'captures-0'/n)==h
  return json.loads((out/'result.json').read_text())
 if out.exists():out.rename(root/(label+'.interrupted-'+str(time.time_ns())))
 with Sampler(root/(label+'-aggregate-'+str(time.time_ns())+'.jsonl')) as sample:result=m.job(root,label,role,argv or ['--synthetic',op,str(token),fault],status=status,marker=marker)
 result['aggregate_concurrent_peak_bytes']=sample.peaks;result['aggregate_samples']=sample.samples;result['host_minima_bytes']=sample.minima
 assert all(not r.get('real_weights',False) for r in result['worker_records'])
 if status==0:
  (out/'REQUEST.json').write_text(json.dumps(request,indent=2)+'\n');(out/'HASHES.json').write_text(json.dumps({p.name:sha(p) for p in (out/'captures-0').iterdir()},indent=2)+'\n')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

def compare(a,b):
 a=Path(a)/'captures-0';b=Path(b)/'captures-0';assert {p.name for p in a.iterdir()}=={p.name for p in b.iterdir()}
 rows=[]
 for p in sorted(a.iterdir()):
  assert p.read_bytes()==(b/p.name).read_bytes(),('BYTE-EXACT MISMATCH',p,b/p.name)
  rows.append({'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
 return rows
