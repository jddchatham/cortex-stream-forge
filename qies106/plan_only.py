"""Compile/run only no-allocation GGML plans in the same sealed synthetic domain."""
from pathlib import Path
import subprocess,os,json,hashlib,sys,datetime
R=Path(__file__).resolve().parent;B=R/'build';W=R.parent
sys.path.insert(0,str(R));from run import namespace
manifest=json.loads((B/'build-manifest.json').read_text());info=manifest['workers']['candidate'];env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
libs=B/'candidate-libs';target=B/'planner-libs'
if not target.exists():target.symlink_to('candidate-libs',target_is_directory=True)
command=['c++','-std=c++17','-O2','-Wall','-Wextra','-I'+str(R),'-I'+str(W/'qies104'),'-I'+str(W/'runtime-flashnext-layer-exp/ggml/include'),str(R/'plan.cpp'),str(W/'qies104/admission.cpp'),*[str(libs/n) for n in ('libggml.so','libggml-cpu.so','libggml-base.so')],'-lcrypto','-Wl,-rpath,/synthetic-sdk','-o',str(B/'planner')]
with (B/'planner-build.log').open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
(B/'planner').chmod(0o600)
out=R/'runs'/('planning-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir(exist_ok=False)
p=subprocess.run(namespace('planner',out,[]),env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=30)
(out/'plans.jsonl').write_text(p.stdout);(out/'stderr.log').write_text(p.stderr)
if p.returncode:raise RuntimeError(p.stderr)
rows=[json.loads(s) for s in p.stdout.splitlines()];assert sum(r.get('stage')=='attention' for r in rows)==7
assert all(not r.get('tensor_buffers_allocated',False) and not r.get('arithmetic_executed',False) for r in rows)
(out/'identity.json').write_text(json.dumps({'command':command,'source_sha256':hashlib.sha256((R/'plan.cpp').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((B/'planner').read_bytes()).hexdigest(),'no_alloc_API':'ggml_gallocr_reserve_n_size(no_alloc=true); no reserve/alloc_graph/compute calls','input_metadata_only':True},indent=2))
print('PASS metadata-only plans:',out)
