#!/usr/bin/env python3
"""Build only: freeze common engine libraries, compile two CLIs and stock source."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
BASE='8fac93bb2bfc4b2d7069c4044f2f71cd25f80388'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--baseline',type=Path,required=True);a.add_argument('--candidate',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args();b=x.baseline.resolve();c=x.candidate.resolve();out=x.output.resolve()
 for p in [b,c]:assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==BASE
 subprocess.run(['git','-C',str(b),'diff','--exit-code','HEAD','--','runtime-flashnext-layer-exp'],check=True,stdout=subprocess.DEVNULL)
 changed=subprocess.check_output(['git','-C',str(c),'diff','--name-only','HEAD'],text=True).splitlines();assert changed==['runtime-flashnext-layer-exp/examples/flashnext-chat.cpp']
 rel=Path('runtime-flashnext-layer-exp/examples/flashnext-chat.cpp');source=(b/rel).read_text();assert (c/rel).read_text()==source.replace('    std::vector<uint8_t> staged_state;\n','').replace('    txc.staged_state = result.state_before;\n','')
 assert sha(b/'runtime-flashnext-layer-exp/examples/flashnext-stock-two-layer.cpp')=='e169159b26d68409c178bc8fbe496fc5582e240c8c9ff6804e78ac2385c8b35f'
 out.mkdir(parents=True,exist_ok=False);(out/'bin').mkdir();entries=json.loads((b/'build-quortex/compile_commands.json').read_text());lib=b/'build-quortex/bin';names=['libllama.so.0.5.0','libggml.so.0.25.3','libggml-cpu.so.0.25.3','libggml-base.so.0.25.3'];results={'base_commit':BASE,'commands':[],'libraries':{n:sha(lib/n)for n in names},'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'binaries':{},'inference_executed':False}
 with (out/'build.log').open('x') as log:
  for label,root,file in [('baseline',b,'flashnext-chat.cpp'),('candidate',c,'flashnext-chat.cpp'),('stock',b,'flashnext-stock-two-layer.cpp')]:
   e=next(e for e in entries if e['file'].endswith('/examples/'+file));cmd=[s.replace(str(b/'runtime-flashnext-layer-exp'),str(root/'runtime-flashnext-layer-exp')) for s in shlex.split(e['command'])];obj=out/(label+'.o');cmd[cmd.index('-o')+1]=str(obj);results['commands'].append(cmd);subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
   exe=out/'bin'/label;cmd=['/usr/bin/c++','-O3','-DNDEBUG',str(obj),'-o',str(exe),'-Wl,-rpath,'+str(lib),*[str(lib/n)for n in names]];results['commands'].append(cmd);subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True);results['binaries'][label]={'sha256':sha(exe),'path':str(exe)}
 (out/'identity.json').write_text(json.dumps(results,indent=2));print('BUILD ONLY PASS; no binaries were executed')
if __name__=='__main__':main()
