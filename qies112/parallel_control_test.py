"""Read-only reuse of completed baseline stacks to test two-branch supervision.
No new numerical layers execute; this is a controller test, not inference evidence.
"""
from pathlib import Path
import sys,json,importlib.util,os
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));s=importlib.util.spec_from_file_location('g112',R/'generate.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
source=Path(json.loads((R/'baseline-active.json').read_text())['root'])/'layers-80-start-0';O=R/os.environ.get('QIES112_PARALLEL_TEST','parallel-control-result');O.mkdir(exist_ok=False);(O/'step-0').mkdir();(O/'stack').mkdir()
for role in ('candidate','reference'):
 p=source/f'{role}-token-0.COMMITTED';assert p.exists();(O/'stack'/p.name).symlink_to(p,target_is_directory=True)
emb=source/'embedding-0.f32';result=g.run_roles(O,0,dict(candidate=emb,reference=emb),dict(candidate=None,reference=None));assert result['statuses']==[0,0] and not result['safety_reasons'] and result['samples']>0
(O/'summary.json').write_text(json.dumps({'passed':True,'result':result,'cached_stack_reuse_only':True,'real_weights':False},indent=2)+'\n');print('PASS parallel controller supervision and aggregate sampler')
