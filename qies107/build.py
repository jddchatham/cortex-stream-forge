"""CPU-only fixed synthetic FFN workers. Frozen sources/guards are not edited."""
from pathlib import Path
import os,subprocess,json,hashlib,shutil
R=Path(__file__).resolve().parent;W=R.parent;B=R/'build';B.mkdir(exist_ok=True);env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
state=Path(os.environ.get('QIES106_SDK_STATE',str(W.parent/'qies105-publication/.qies105-build/state.json')));sdk=json.loads(state.read_text());stock=Path(sdk['stock_source'])
assert sdk['stock']=='240d68fcfe0c2a1f298c8e18ca181368b1914259'
assert all(sha(stock/p)==h for p,h in json.loads((W/'qies100/stock-source-sha256.json').read_text()).items())
index=json.loads((W/'qies098-checks/ripple-tensor-index.json').read_text());ts={q['name']:q for q in index['tensors']}
for layer,typ in ((0,13),(15,12)):
 for name,shape in [('gate',[5120,40960]),('up',[5120,40960]),('down',[40960,5120])]:assert ts[f'blk.{layer}.ffn_{name}.weight']['shape']==shape and ts[f'blk.{layer}.ffn_{name}.weight']['type']==typ
meta=json.loads((W/'qies098-checks/gguf-inspection.json').read_text())['metadata'];assert meta['ricursive.swiglu_clamp']==20.0
manifest={'parent':subprocess.check_output(['git','-C',W,'rev-parse','HEAD'],env=env,text=True).strip(),'stock_source':sdk['stock'],'compiler':subprocess.check_output(['c++','--version'],env=env,text=True),'profiles':{'q5':{'verified_real_descriptor_layer':0,'type':13},'q4':{'verified_real_descriptor_layer':15,'type':12}},'reference_source_sha256':sha(stock/'src/models/ricursive.cpp'),'model_bytes_read':0,'threads':1,'buffer_profile':'ordinary CPU; no extra buffer/repack selection','workers':{},'source_sha256':{str(p.relative_to(W)):sha(p) for p in R.iterdir() if p.suffix in ('.cpp','.h','.inc','.py')}}
for role,key,source in [('candidate','cpu_libs','stream_ffn.cpp'),('reference','stock_libs','reference_ffn.cpp')]:
 libs=B/(role+'-libs');libs.mkdir(exist_ok=True);origin=Path(sdk[key]);hashes={}
 for name in ('libggml.so','libggml-cpu.so','libggml-base.so'):
  actual=(origin/name).resolve();dest=libs/actual.name
  if not dest.exists():shutil.copyfile(actual,dest)
  assert sha(dest)==sha(actual);hashes[actual.name]=sha(dest)
  for alias in (name,name+'.0'):
   if not (libs/alias).exists():(libs/alias).symlink_to(actual.name)
 headers=W/'runtime-flashnext-layer-exp/ggml/include' if role=='candidate' else stock/'ggml/include'
 cmd=['c++','-std=c++17','-O2','-Wall','-Wextra','-I'+str(R),'-I'+str(W/'qies104'),'-I'+str(headers),str(R/source),str(W/'qies104/admission.cpp'),*[str(libs/n) for n in ('libggml.so','libggml-cpu.so','libggml-base.so')],'-lcrypto','-Wl,-rpath,/synthetic-sdk','-o',str(B/role)]
 with (B/(role+'-build.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (B/role).chmod(0o600);manifest['workers'][role]={'command':cmd,'sha256':sha(B/role),'libraries':hashes,'headers':str(headers),'no_llama_library':True}
(B/'marker').write_text('QIES104_SEALED_SYNTHETIC_DOMAIN_V1\n');(B/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built independent stock whole-FFN and streamed FFN workers, fixed procedural synthetic input only')
