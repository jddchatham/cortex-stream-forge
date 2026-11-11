from pathlib import Path as _q104_Path
import sys as _q104_sys
_q104_sys.path.insert(0,str(_q104_Path(__file__).resolve().parents[1]/'qies104'))
from gate import legacy_build_only as _q104_gate
_q104_gate()
"""Compile independently linked probes and compare only synthetic fixtures."""
from pathlib import Path
import subprocess,os,json,hashlib,struct,math
R=Path(__file__).resolve().parent;W=R.parent;S=next(R.glob('llama.cpp-*'));B=R/'stock-build/bin';env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
old=W.parent/'cortex-stream-forge-qies074/build-quortex/bin';oldh=W/'runtime-flashnext-layer-exp/ggml/include'
commands=[]
def compile(name,source,includes,libraries,libpath,extra=[]):
 cmd=['c++','-std=c++17','-O2','-Wall','-Wextra',*['-I'+str(x) for x in includes],*map(str,source),'-Wl,-rpath,'+str(libpath),*map(str,libraries),*extra,'-o',str(R/name)];commands.append(cmd)
 subprocess.run(cmd,env=env,check=True)
stocklibs=[B/'libllama.so',B/'libggml.so',B/'libggml-cpu.so',B/'libggml-base.so']
compile('stock-probe',[R/'stock_probe.cpp'],[S/'include',S/'ggml/include'],stocklibs,B)
compile('stock-metadata',[R/'stock_metadata.cpp'],[S/'ggml/include'],stocklibs[1:],B)
compile('candidate-probe',[W/'qies099/ricursive-layer.cpp',R/'candidate_probe.cpp'],[W/'qies099',oldh],[old/n for n in ('libggml.so','libggml-cpu.so','libggml-base.so')],old)
(R/'probe-build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
def compare_case(label,fixture,plain=False):
 stock=R/('stock-captures'+label);candidate=R/('candidate-captures'+label)
 for name,cmd in [('stock',[R/'stock-probe',fixture/'synthetic.gguf',stock]),('candidate',[R/'candidate-probe',fixture,candidate])]:
  runenv=env.copy();runenv.pop('QIES100_REFERENCE_PLAIN',None)
  if name=='stock' and plain:runenv['QIES100_REFERENCE_PLAIN']='1'
  with (R/(name+label+'-run.log')).open('w') as f:subprocess.run(list(map(str,cmd)),env=runenv,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
 rows=[];missing=[];aliases=[]
 for p in sorted(candidate.glob('*.bin')):
  q=stock/p.name
  if not q.exists():
   # Global NoPE Q/K has the same pointer before and after build_rope; stock's later callback renames it.
   for a,b in [('Qcur_normed','Qcur'),('Kcur_normed','Kcur')]:
    if a in p.name and p.name.endswith('-4.bin') and (stock/p.name.replace(a,b)).exists():
     q=stock/p.name.replace(a,b);aliases.append({'candidate':p.name,'stock':q.name,'reason':'global NoPE identity alias'})
   if not q.exists():missing.append(p.name);continue
  a=p.read_bytes();b=q.read_bytes()
  if len(a)!=len(b):rows.append({'name':p.name,'same_size':False,'candidate_bytes':len(a),'stock_bytes':len(b)});continue
  av=struct.unpack('<'+'f'*(len(a)//4),a);bv=struct.unpack('<'+'f'*(len(b)//4),b)
  rows.append({'name':p.name,'same_size':True,'elements':len(av),'byte_exact':a==b,'max_abs':max((abs(x-y) for x,y in zip(av,bv)),default=0),'finite':all(math.isfinite(x) and math.isfinite(y) for x,y in zip(av,bv))})
 result={'synthetic_only':True,'comparisons':rows,'missing_stock_taps':missing,'documented_aliases':aliases,'all_byte_exact':len(rows)==360 and not missing and all(x.get('byte_exact',False) for x in rows),'real_model_executed':False}
 (R/('comparison'+label+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'case':label or 'F32','comparisons':len(rows),'missing':len(missing),'byte_exact':sum(x.get('byte_exact',False) for x in rows)},indent=2))
 return result
f32=compare_case('',R/'fixture')
repacked=compare_case('-quantized',R/'fixture-quantized')
plain=compare_case('-quantized-plain',R/'fixture-quantized',True)
assert f32['all_byte_exact'] and plain['all_byte_exact'], 'matched-path synthetic comparison failed'
print('PASS F32 and matched plain-CPU quantized synthetic exact-comparison gates; repacked path is diagnostic only')
