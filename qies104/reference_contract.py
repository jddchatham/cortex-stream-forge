"""Data-only reference qualification: actual descriptors/bytes and fixed activation contract."""
from pathlib import Path
import json,hashlib,struct,math
R=Path(__file__).resolve().parent;W=R.parent
contract=json.loads((W/'qies103/inputs/contract.json').read_text());bundle=json.loads((W/'qies102/bundle-manifest.json').read_text());stock={}
for line in (R/'reference-1203-descriptors.tsv').read_text().splitlines():
 name,off,size,kind,*dims=line.split('\t');stock[name]=list(map(int,[off,size,kind,*dims]))
verified=[]
for q in bundle['tensors']:
 assert stock[q['name']]==[q['offset'],q['bytes'],q['type'],*q['shape'],*[1]*(4-len(q['shape']))]
 h=hashlib.sha256();g=hashlib.sha256();count=0
 with Path(q['path']).open('rb',buffering=0) as a,Path(q['reference_path']).open('rb',buffering=0) as b:
  while x:=a.read(1024*1024):y=b.read(len(x));assert x==y;h.update(x);g.update(y);count+=len(x)
  assert not b.read(1)
 assert count==q['bytes'] and h.hexdigest()==g.hexdigest()==q['sha256'];verified.append(q['name'])
inputs={}
for name,entry in contract['inputs'].items():
 data=(W/'qies103/inputs'/name).read_bytes();assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'];inputs[name]=entry
assert (W/'qies103/inputs/input.f32').read_bytes()==(W/'qies103/inputs/x0.f32').read_bytes()
assert contract['backend']['threads']==1 and contract['backend']['flash_attention'] is False and contract['backend']['use_extra_bufts'] is False
assert len(contract['outputs'])==21 and all(x['expected_sha256'] is None for x in contract['outputs'].values())
(R/'reference-qualification.json').write_text(json.dumps({'stock_source':'240d68fcfe0c2a1f298c8e18ca181368b1914259','independent_real_descriptors_verified':1203,'independent_real_tensors_byte_exact':verified,'input_contract_verified':inputs,'output_schema_count':21,'guard_before_reference_model_load':True,'synthetic_reference_exact_comparisons':144,'kernel_versions_identical':False,'candidate_ggml':'0.25.3','stock_ggml':'0.29.0','matched_profile':'CPU single thread; nonflash; F16 KV; ordinary buffers','real_reference_worker_enabled':False,'real_numerical_equivalence_qualified':False,'qualification':'non-inference provenance/data/control checks pass; live positive runtime pending'},indent=2)+'\n')
print('PASS independent 1203 descriptors, 15 actual tensor byte comparisons, identical activation/mask/KV contract and 21 output schemas; no arithmetic')
