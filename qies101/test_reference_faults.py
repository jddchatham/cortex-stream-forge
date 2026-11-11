"""Reference range-reader faults use copies of a tiny synthetic GGUF only."""
from pathlib import Path
import tempfile,shutil,subprocess,json,os
R=Path(__file__).resolve().parent;env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None);results=[]
with tempfile.TemporaryDirectory(dir=R,prefix='stock-faults-') as tmp:
 d=Path(tmp);model=d/'synthetic.gguf';shutil.copyfile(R.parent/'qies100/fixture/synthetic.gguf',model);out=d/'out';out.mkdir()
 def reject(label,path,name):
  p=subprocess.run([str(R/'reference-extract'),str(path),str(out),name],env=env,capture_output=True,text=True,timeout=20)
  assert p.returncode==1,(label,p.returncode,p.stderr);assert not list(out.glob('*.partial'));results.append({'name':label,'exit_status':p.returncode,'passed':True,'stderr':p.stderr.strip()})
 reject('missing_tensor',model,'not_present.weight')
 reject('missing_file',d/'missing.gguf','output_norm.weight')
 saved=out/'output_norm.weight.bin';saved.write_bytes(b'preserve')
 reject('existing_output_preserved',model,'output_norm.weight');assert saved.read_bytes()==b'preserve'
 with model.open('r+b') as f:f.truncate(model.stat().st_size-16)
 reject('truncated_synthetic_gguf',model,'output.weight')
(R/'reference-fault-results.json').write_text(json.dumps(results,indent=2)+'\n');print('PASS',len(results),'stock extraction fault tests; no real GGUF modified')
