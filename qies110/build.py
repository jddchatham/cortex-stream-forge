from pathlib import Path
import subprocess,os,json,hashlib,shutil
R=Path(__file__).resolve().parent;W=R.parent;B=R/'build';B.mkdir(exist_ok=True);env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sdk=json.loads(Path(os.environ.get('QIES106_SDK_STATE',str(W.parent/'qies105-publication/.qies105-build/state.json'))).read_text());stock=Path(sdk['stock_source'])
assert all(sha(stock/p)==h for p,h in json.loads((W/'qies100/stock-source-sha256.json').read_text()).items())
index={q['name']:q for q in json.loads((W/'qies098-checks/ripple-tensor-index.json').read_text())['tensors']}
for layer in (0,4):
 for name,typ,shape in [('q',12,[5120,10240]),('k',8,[5120,1024]),('v',8,[5120,1024]),('gate',12,[5120,10240]),('output',14,[10240,5120])]:assert (index[f'blk.{layer}.attn_{name}.weight']['type'],index[f'blk.{layer}.attn_{name}.weight']['shape'])==(typ,shape)
for layer in (0,4):
 for name,shape in [('gate',[5120,40960]),('up',[5120,40960]),('down',[40960,5120])]:assert index[f'blk.{layer}.ffn_{name}.weight']['type']==13 and index[f'blk.{layer}.ffn_{name}.weight']['shape']==shape
meta=json.loads((W/'qies098-checks/gguf-inspection.json').read_text())['metadata'];assert meta['ricursive.attention.sliding_window']==1024 and meta['ricursive.attention.sliding_window_pattern'][0] and not meta['ricursive.attention.sliding_window_pattern'][4]
manifest={'parent':subprocess.check_output(['git','-C',W,'rev-parse','HEAD'],env=env,text=True).strip(),'stock':sdk['stock'],'reference_ricursive_source_sha256':sha(stock/'src/models/ricursive.cpp'),'compiler':subprocess.check_output(['c++','--version'],env=env,text=True),'threads':1,'source_sha256':{str(p.relative_to(W)):sha(p) for p in list(R.glob('*.cpp'))+list(R.glob('*.h'))+list(R.glob('*.inc'))+[W/'qies107/support.h',W/'qies107/fixture.h',W/'qies107/telemetry.h',W/'qies104/admission.cpp']},'workers':{}}
for role,key,source in [('candidate','cpu_libs','stream.cpp'),('reference','stock_libs','reference.cpp'),('fixture-check','cpu_libs','fixture_check.cpp'),('fixture-writer','cpu_libs','fixture_writer.cpp')]:
 libs=B/(role+'-libs');libs.mkdir(exist_ok=True);origin=Path(sdk[key]);hashes={}
 for name in ('libggml.so','libggml-cpu.so','libggml-base.so'):
  p=(origin/name).resolve();dest=libs/p.name
  if not dest.exists():shutil.copyfile(p,dest)
  assert sha(p)==sha(dest);hashes[p.name]=sha(dest)
  for alias in (name,name+'.0'):
   if not (libs/alias).exists():(libs/alias).symlink_to(p.name)
 headers=W/'runtime-flashnext-layer-exp/ggml/include' if role!='reference' else stock/'ggml/include'
 cmd=['c++','-std=c++17','-O2','-Wall','-Wextra','-I'+str(R),'-I'+str(W/'qies107'),'-I'+str(W/'qies104'),'-I'+str(headers),str(R/source),str(W/'qies104/admission.cpp'),*[str(libs/n) for n in ('libggml.so','libggml-cpu.so','libggml-base.so')],'-lcrypto','-Wl,-rpath,/synthetic-sdk','-o',str(B/role)]
 with (B/(role+'-build.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (B/role).chmod(0o600);manifest['workers'][role]={'command':cmd,'libraries':hashes,'sha256':sha(B/role)}
(B/'marker').write_text('QIES104_SEALED_SYNTHETIC_DOMAIN_V1\n');(B/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Built adapter-backed synthetic layer and storage fixture writer; real execution locked')

# Data-only tools link GGML base/parser, never the CPU compute backend or llama.
for name,source,stock_headers in [('storage','storage.cpp',False),('reference-hash','reference_hash.cpp',True)]:
 headers=stock/'ggml/include' if stock_headers else W/'runtime-flashnext-layer-exp/ggml/include'
 base=Path(sdk['stock_libs'] if stock_headers else sdk['cpu_libs'])
 command=['c++','-std=c++17','-O2','-Wall','-Wextra','-I'+str(R),'-I'+str(W/'qies107'),'-I'+str(headers),str(R/source),str(base/'libggml-base.so'),'-lcrypto','-Wl,-rpath,'+str(base),'-o',str(B/name)]
 with (B/(name+'-build.log')).open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 manifest['workers'][name]={'command':command,'sha256':sha(B/name),'data_only':True}
(B/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
