"""Portable QIES104 model-free replay. No real execution enablement or model download."""
from pathlib import Path
import argparse,subprocess,os,json,hashlib,tarfile,urllib.request,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
BASE='8fac93bb2bfc4b2d7069c4044f2f71cd25f80388';STOCK='240d68fcfe0c2a1f298c8e18ca181368b1914259'
ARCHIVE_SHA='b7c50c28d1c25942fdbca447326a692657238cd87beeea45797264b02bc64b4f'
ENV=os.environ.copy();ENV.pop('LD_LIBRARY_PATH',None)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def run(cmd,log,**kw):
 log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('w') as f:subprocess.run(list(map(str,cmd)),env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True,**kw)
def save(p,data):Path(p).write_text(json.dumps(data,indent=2)+'\n')
def build(out,stock_source=None,jobs=2):
 out.mkdir(parents=True,exist_ok=True);state={'root':str(ROOT),'out':str(out),'baseline':BASE,'stock':STOCK,'real_enabled':False}
 if stock_source:
  source=stock_source.resolve()
 else:
  source=out/'vendor'/('llama.cpp-'+STOCK)
  if not source.exists():
   archive=out/'stock-source.tar.gz'
   if not archive.exists():
    with urllib.request.urlopen('https://codeload.github.com/ggml-org/llama.cpp/tar.gz/'+STOCK,timeout=90) as f,archive.open('wb') as g:shutil.copyfileobj(f,g,1024*1024)
   if sha(archive)!=ARCHIVE_SHA:raise ValueError('stock source archive identity mismatch')
   source.parent.mkdir(exist_ok=True)
   with tarfile.open(archive) as t:t.extractall(source.parent,filter='data')
 manifest=json.loads((ROOT/'qies100/stock-source-sha256.json').read_text())
 if not all(sha(source/p)==h for p,h in manifest.items()):raise ValueError('stock source-file provenance mismatch')
 state['stock_source']=str(source)
 common=['-DCMAKE_BUILD_TYPE=Release','-DGGML_CPU=ON','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DGGML_OPENCL=OFF','-DGGML_CPU_REPACK=ON','-DGGML_NATIVE=ON','-DGGML_BLAS=OFF','-DLLAMA_BUILD_COMMON=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=OFF','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_BUILD_APP=OFF','-DLLAMA_BUILD_TESTS=OFF']
 for name,src,target in [('baseline',ROOT/'runtime-flashnext-layer-exp','ggml'),('stock',source,'llama')]:
  dest=out/(name+'-build')
  # Resume is allowed for compiler output only; this is not a model cache.
  run(['cmake','-S',src,'-B',dest,*common],out/(name+'-configure.log'),timeout=180)
  run(['cmake','--build',dest,'--target',target,'-j'+str(jobs)],out/(name+'-build.log'),timeout=3600)
 state['cpu_libs']=str(out/'baseline-build/bin');state['stock_libs']=str(out/'stock-build/bin')
 state['compiler']=subprocess.check_output(['c++','--version'],env=ENV,text=True)
 save(out/'state.json',state);print('Built exact baseline CPU and independent stock sources; no model evaluated',flush=True)
 return state

def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','selftest','all','inspect'));p.add_argument('--out',type=Path,default=ROOT/'.qies105-build');p.add_argument('--stock-source',type=Path);p.add_argument('--numpy-python',type=Path,default=Path(sys.executable));p.add_argument('--jobs',type=int,choices=(1,2),default=2);return p.parse_args()
if __name__=='__main__':
 a=parse();out=a.out.resolve()
 if a.command in ('build','all'):build(out,a.stock_source,a.jobs)
 if a.command in ('selftest','all'):
  sys.path.insert(0,str(Path(__file__).resolve().parent));from portable_tests import selftest
  selftest(ROOT,out,a.numpy_python.absolute())
 if a.command=='inspect':
  print(json.dumps({'real_execution_enabled':False,'model_downloaded':False,'remaining':['naturally passing original entry/ancestor checks','positive live guard qualification','reviewed real-admission protocol','bounded real reference worker and canonical KV export']},indent=2))
