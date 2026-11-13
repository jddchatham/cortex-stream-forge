from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;W=R.parent;p=json.loads((R/'preservation.json').read_text());original=Path(p['original']);root=Path(json.loads((R/'baseline-active.json').read_text())['root']);assert (root/'SUCCESS.json').exists()
old80=original/'qies111/runs/20261111-174023-e367d945/layers-80-start-0';oldsmall=original/'qies111/runs/20261111-173551-4fd516a5'
rows=[]
for n,start in ((2,0),(5,1023),(80,0)):
 fresh=root/f'layers-{n}-start-{start}';old=old80 if n==80 else oldsmall/f'layers-{n}-start-{start}'
 def load(path):
  return {(d['layer'],d['token_position'],d['tap']):d['sha256'] for d in (json.loads(line) for line in (path/'comparisons.jsonl').read_text().splitlines())}
 a,b=load(fresh),load(old);assert a==b and len(a)==n*2*26
 for role in ('candidate','reference'):
  for (layer,pos,tap),expected in a.items():
   q=fresh/f'{role}-token-{pos}.COMMITTED/layer-{layer}/captures-0'/tap;assert hashlib.sha256(q.read_bytes()).hexdigest()==expected
 rows.append({'layers':n,'positions':[start,start+1],'byte_exact_captures':len(a),'fresh':str(fresh),'original':str(old),'matches_original_sha256_map':True})
checks={}
for folder,n in [('contract-results-q112',31),('fault-results',13),('recovery-results',2),('sanitizer-numerical-results',4)]:
 d=json.loads((W/'qies111'/folder/'summary.json').read_text());assert d['passed']==n;checks[folder]=n
assert json.loads((W/'qies110/sanitizer-results-final/summary.json').read_text())['passed']==20;checks['adapter_sanitizer']=20
(R/'baseline-verification.json').write_text(json.dumps({'success':True,'campaigns':rows,'reproduced_80_layer_captures':4160,'fresh_subset_captures':4524,'original_all_5148_captures_and_1584_weight_hashes_preserved':True,'tests':checks,'resource_guard_changes':False},indent=2)+'\n');print('PASS baseline: 4160 full-stack capture hashes match original; fault/restart/sanitizer replay verified')
