"""Read-only provenance, resource snapshot and entrypoint audit."""
from pathlib import Path
import json,hashlib,subprocess,os,resource,importlib.util,sys
R=Path(__file__).resolve().parent;W=R.parent
GUARD='ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_guard():
 p=W.parent/'qies077-guard.py'
 if sha(p)!=GUARD:raise ValueError('original guard changed')
 s=importlib.util.spec_from_file_location('q103_original_guard',p);g=importlib.util.module_from_spec(s);sys.modules[s.name]=g;s.loader.exec_module(g);return g
def audit():
 env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
 branch=subprocess.check_output(['git','branch','--show-current'],cwd=W,env=env,text=True).strip();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,env=env,text=True).strip()
 assert branch=='experimental/ripple-65b-q4' and head=='8fac93bb2bfc4b2d7069c4044f2f71cd25f80388'
 assert not subprocess.check_output(['git','diff'],cwd=W,env=env) and not subprocess.check_output(['git','diff','--cached'],cwd=W,env=env)
 for name in ('qies099','qies100','qies101','qies102'):
  m=json.loads((W/name/'SHA256-MANIFEST.json').read_text())
  assert all(sha(W/path)==value for path,value in m.items() if path!='pandacode-run.sh'),name
 stock=W/'qies100/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259';source=json.loads((W/'qies100/stock-source-sha256.json').read_text());assert all(sha(stock/p)==h for p,h in source.items())
 binaries=json.loads((W/'qies100/executable-library-sha256.json').read_text());assert all(sha(W/p)==h for p,h in binaries.items())
 receipt=json.loads((W/'qies102/model-hash-result.json').read_text());model=W.parent/'qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf';st=model.stat();assert [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]==receipt['identity']
 bundle=json.loads((W/'qies102/bundle-manifest.json').read_text());assert bundle['all_bytes_exact'] and len(bundle['tensors'])==15
 for q in bundle['tensors']:
  h=hashlib.sha256()
  with Path(q['path']).open('rb') as f:
   while b:=f.read(1024*1024):h.update(b)
  assert h.hexdigest()==q['sha256'] and Path(q['path']).stat().st_size==q['bytes']
 g=load_guard();host=g.host(R);policy=g.Policy();cgroup=Path('/proc/self/cgroup').read_text();rel=next(x.split(':',2)[2] for x in cgroup.splitlines() if x.startswith('0::'));cg=Path('/sys/fs/cgroup')/rel.lstrip('/');limits=[]
 for path in (cg,*cg.parents):
  if path==Path('/sys'):break
  row={'path':str(path)}
  for name in ('memory.max','memory.high','memory.swap.max','memory.current','memory.swap.current'):
   p=path/name
   if p.exists():row[name]=p.read_text().strip()
  if len(row)>1:limits.append(row)
 return {'branch':branch,'HEAD':head,'tracked_staged_diffs_empty':True,'prior_artifact_manifests_intact':True,'stock_source_files_verified':len(source),'stock_binaries_libraries_verified':True,'model_hash_receipt_verified':True,'new_complete_GGUF_hash_this_task':False,'model_sha256':receipt['actual'],'fifteen_tensor_hashes_verified':True,'weight_bytes':sum(q['bytes'] for q in bundle['tensors']),'host':host,'policy':vars(policy),'rlimit_as':resource.getrlimit(resource.RLIMIT_AS),'rlimit_rss':resource.getrlimit(resource.RLIMIT_RSS),'cgroup':cgroup,'ancestor_limits':limits,'reference_real_execution_qualified':False,'live_guard_positive_paths_qualified':False,'alternate_entrypoints_guard_complete':False,'real_weight_arithmetic_executed':False,'guard_sha256':GUARD}
if __name__=='__main__':
 result=audit();(R/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS source/library/model/tensor identities; readiness remains NO-GO')
