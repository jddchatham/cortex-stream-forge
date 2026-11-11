"""Independent metadata cross-check before any range extraction."""
from pathlib import Path
import importlib.util,json,subprocess,os,hashlib
R=Path(__file__).resolve().parent;W=R.parent;env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
s=importlib.util.spec_from_file_location('q101_ext',R/'extract.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=m.read_json(W/'qies098-checks/ripple-tensor-index.json');receipt=m.read_json(R/'model-hash-result.json');model=W.parent/'qies098-data/models'/i['model_filename']
assert receipt['passed'] and receipt['actual']==i['model_sha256']=='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc' and m.identity(model.stat())==receipt['identity']
t=m.validate(i,receipt['bytes']);baseline=W/'qies098-checks/index-compat'
assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='4f3d4444e7524d2c9254e81aadb38199084a0b9afb685a57b1dcd47a554b644e'
for name,cmd in [('baseline',[baseline,W/'qies098-checks/ripple-tensor-index.json',W/'qies098-checks/tensor-names.txt']),('reference',[R/'reference-extract',model,'--describe'])]:
 with (R/(name+'-descriptors.tsv')).open('w') as f,(R/(name+'-describe-resources.log')).open('w') as e:subprocess.run(list(map(str,cmd)),env=env,stdout=f,stderr=e,check=True,timeout=30)
a=(R/'baseline-descriptors.tsv').read_text().splitlines();b=(R/'reference-descriptors.tsv').read_text().splitlines();assert len(a)==len(b)==len(t)==1203
for row in a:
 name,off,n,kind,x,y=row.split('\t');q=t[name];assert list(map(int,[off,n,kind,x,y]))==[q['offset'],q['bytes'],q['type'],q['shape'][0],q['shape'][1] if len(q['shape'])>1 else 0]
for row in b:
 name,off,n,kind,*dims=row.split('\t');q=t[name];assert list(map(int,[off,n,kind,*dims]))==[q['offset'],q['bytes'],q['type'],*q['shape'],*[1]*(4-len(q['shape']))]
names=['blk.0.attn_q.weight','blk.0.ffn_down.weight','blk.40.attn_k.weight','blk.40.ffn_gate.weight','blk.79.attn_output.weight','blk.79.ffn_up.weight','token_embd.weight','output_norm.weight'];selected=[t[n] for n in names];total=sum(x['bytes'] for x in selected);v=os.statvfs(R);available=v.f_bavail*v.f_frsize
assert available-2*total>=30*1024**3
(R/'selected-tensors.json').write_text(json.dumps(selected,indent=2)+'\n');(R/'descriptor-verification.json').write_text(json.dumps({'descriptor_count':1203,'baseline_parser_exact':True,'stock_parser_exact':True,'index_validation_exact':True,'selected_count':len(names),'total_selected_bytes':total,'projected_two_copies_bytes':2*total,'free_at_check':available,'reserve_bytes':30*1024**3},indent=2)+'\n')
print('PASS 1203 named descriptors: index, original baseline parser and independent stock GGUF parser')
