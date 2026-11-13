"""Reproduce QIES111 unchanged APIs; replace only historical artifact lookup,
not resource admission. All numerical workers use the unchanged monitor/guards.
"""
from pathlib import Path
import importlib.util,subprocess,json,hashlib,datetime,uuid
R=Path(__file__).resolve().parent;W=R.parent
p=json.loads((R/'preservation.json').read_text());original=Path(p['original'])
assert all(hashlib.sha256((original/n).read_bytes()).hexdigest()==h for n,h in p['original_source_manifest'].items())
old=json.loads((original/'qies111/verified-results.json').read_text());assert sum(d['capture_comparisons'] for d in old['campaigns'])==5148 and sum(d['weight_hash_comparisons'] for d in old['campaigns'])==1584
for d in old['campaigns']:
 base=Path(d['root'])/f"layers-{d['layers']}-start-{d['positions'][0]}"
 for line in (base/'comparisons.jsonl').read_text().splitlines():
  row=json.loads(line)
  for role in ('candidate','reference'):
   q=base/f"{role}-token-{row['token_position']}.COMMITTED/layer-{row['layer']}/captures-0"/row['tap'];assert hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256']
spec=importlib.util.spec_from_file_location('q111',W/'qies111/run.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
subprocess.run(['/usr/bin/python3','-I','-B',W/'qies111/build.py'],env=a.ENV,check=True);a.fixtures()
root=W/'qies111/runs'/('q112-replay-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]);root.mkdir(parents=True)
a.jsonwrite(root/'provenance.json',{'preserved':p,'source_manifest':json.loads((W/'qies111/build/manifest.json').read_text()),'guard_unchanged':True})
(R/'baseline-active.json').write_text(json.dumps({'root':str(root)},indent=2)+'\n')
a.campaign(root,2);a.campaign(root,5,1023);a.campaign(root,80)
a.jsonwrite(root/'SUCCESS.json',{'success':True,'root':str(root),'layers':[2,5,80],'boundary':True,'real_weight_arithmetic':False});print('PASS Q111 replay:',root,flush=True)
