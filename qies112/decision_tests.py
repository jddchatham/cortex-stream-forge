"""Deterministic generation-control and publication tests, no numerical execution."""
from pathlib import Path
import sys,importlib.util,json,math,os
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));s=importlib.util.spec_from_file_location('gen112',R/'generate.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
O=R/'decision-tests';O.mkdir(exist_ok=False);passed=[]
def greedy(values):
 assert values and all(math.isfinite(v) for v in values)
 return max(range(len(values)),key=values.__getitem__)
assert greedy([3.,3.,2.])==0;passed.append('lowest-ID tie')
assert greedy([-4.,-2.,-3.])==1;passed.append('negative logits')
for bad in ([float('nan')],[float('inf')],[]):
 try:greedy(bad)
 except AssertionError:passed.append('invalid logits refused')
 else:raise AssertionError('nonfinite accepted')
a.publish_decision(O,0,{'position':0,'selected':7});h=a.sha(O/'decision-0.json')
try:a.publish_decision(O,0,{'position':0,'selected':8})
except FileExistsError:passed.append('immutable publication refuses overwrite')
else:raise AssertionError('decision replaced')
assert a.sha(O/'decision-0.json')==h
# Selection and evaluation are separate counters. Final prediction stays pending.
for limit in (2,4,8,16):
 inputs=[4];predicted=[];evaluated=0
 for pos in range(limit):
  evaluated+=1;next_id=(inputs[-1]*3+1)%101;predicted.append(next_id)
  if pos+1<limit:inputs.append(next_id)
 assert evaluated==len(predicted)==limit and inputs[1:]==predicted[:-1];passed.append('feedback/limit-'+str(limit))
for eos_pos in (0,1,3):
 predicted=[];evaluated=0
 for pos in range(16):
  evaluated+=1;selected=a.EOS if pos==eos_pos else pos;predicted.append(selected)
  if selected==a.EOS:break
 assert evaluated==len(predicted)==eos_pos+1;passed.append('EOS-boundary-'+str(eos_pos))
# Cancellation/error before a complete decision cannot advance authoritative count.
for boundary in ('before-embedding','after-embedding','after-stack','after-head','before-publication'):
 old=a.sha(O/'decision-0.json')
 try:raise RuntimeError(boundary)
 except RuntimeError:pass
 assert old==a.sha(O/'decision-0.json') and not (O/'decision-1.json').exists();passed.append('no-promotion-'+boundary)
(O/'summary.json').write_text(json.dumps({'passed':len(passed),'tests':passed,'model_free':True,'simulated_control_transitions_not_inference':True},indent=2)+'\n');print('PASS decision/control tests',len(passed))
