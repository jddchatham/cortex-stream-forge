from pathlib import Path
import sys,json,struct
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));from support import endpoint,compare
O=R/'endpoint-results';O.mkdir(exist_ok=False);tests=[]
for token in (0,100257,100351):
 for role in ('candidate','reference'):endpoint(O,f'{role}-embedding-{token}',role,'embedding',token)
 tests.append({'embedding_token':token,'comparisons':compare(O/f'candidate-embedding-{token}',O/f'reference-embedding-{token}')})
x=O/'input.f32';x.write_bytes(struct.pack('<5120f',*[float((i*19+7)%257-128)/128 for i in range(5120)]))
for role in ('candidate','reference'):endpoint(O,role+'-head',role,'head',100257,x)
tests.append({'head':True,'comparisons':compare(O/'candidate-head',O/'reference-head')})
for token in (-1,100352):
 for role in ('candidate','reference'):endpoint(O,f'{role}-bad-{token}',role,'embedding',token,status=1)
for fault in ('embedding.before-buffer','embedding.after-compute','head.before-buffer','head.after-compute','before-publish'):
 op='embedding' if fault.startswith('embedding') else 'head';endpoint(O,'fault-'+fault,'candidate',op,100257,x if op=='head' else None,fault=fault,status=1)
for role in ('candidate','reference'):
 endpoint(O,role+'-reject-real',role,'embedding',0,status=1,argv=['--real']);endpoint(O,role+'-reject-marker',role,'embedding',0,status=1,marker=False)
(O/'summary.json').write_text(json.dumps({'passed':True,'numerical_cases':tests,'invalid_token_refusals':4,'faults':5,'guard_refusals':4,'synthetic_only':True},indent=2)+'\n');print('PASS endpoint exactness, bounds, lifetime and guard tests')
