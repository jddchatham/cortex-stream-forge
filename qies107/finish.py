"""Finish the interrupted final campaign without overwriting any prior attempt.
Source/binary identities must match the recorded final build before any reuse.
Only the fixed QIES107 campaign is supported; this cannot load model data.
"""
from pathlib import Path
import importlib.util,json,hashlib,struct
R=Path(__file__).resolve().parent;W=R.parent
spec=importlib.util.spec_from_file_location('q107_final_campaign',R/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original=R/'runs/20261111-112705-03caed05';completion=R/'runs/20261111-112705-03caed05-completion';completion.mkdir(exist_ok=False)
build=json.loads((R/'build/build-manifest.json').read_text())
assert all(m.sha(W/p)==h for p,h in build['source_sha256'].items())
assert all(m.sha(R/'build'/role)==item['sha256'] for role,item in build['workers'].items())
baseline=m.baseline();original_incomplete=[str(p) for p in original.iterdir() if p.is_dir() and not (p/'result.json').exists()]
rows=[];jobs=[];coverage=[];records={}
for p in original.glob('*/result.json'):
 d=json.loads(p.read_text());assert not d['reasons'] and d['rlimits_unchanged'] and d['worker_records'][-1]['owned_resources']==0
 records[d['label']]=d
for quant in ('q5','q4'):
 for vector in (0,1,2):
  ref=records[f'reference-{quant}-v{vector}'];candidate=records[f'candidate-{quant}-v{vector}'];assert ref['status']==candidate['status']==0
  rows+=m.compare(original,candidate,ref);jobs.extend([ref,candidate])
  p=original/ref['label']/'captures-0';g=struct.unpack('<40960f',(p/'gate.f32').read_bytes());u=struct.unpack('<40960f',(p/'up.f32').read_bytes())
  c={'profile':quant,'vector':vector,'gate_above_20':sum(x>20 for x in g),'gate_below_minus20_not_lower_clamped':sum(x< -20 for x in g),'up_above_20':sum(x>20 for x in u),'up_below_minus20':sum(x< -20 for x in u)}
  if vector!=2:assert all(v>0 for k,v in c.items() if k not in ('profile','vector'))
  else:assert all(x==0 for x in struct.unpack('<5120f',(p/'final.f32').read_bytes()))
  coverage.append(c)
repeated=records['candidate-q5-repeat3'];assert repeated['status']==0;rows+=m.compare(original,repeated,records['reference-q5-v0'],3);jobs.append(repeated)
faults=[s.strip().strip('\",') for s in (R/'faults.inc').read_text().splitlines()];assert len(faults)==len(set(faults))==100
reused=13;retried=[]
for fault in faults:
 label='fault-'+fault
 if label in records:d=records[label];folder=original/label;reused+=1
 else:
  d=m.job(completion,label,'candidate',['--synthetic','q5','0',fault,'1'],status=1);folder=completion/label;retried.append(label)
 assert d['status']==1 and d['worker_records'][-1]['fault_seen'] is True and 'injected '+fault in (folder/'stderr.log').read_text()
 assert not list(folder.glob('captures-*')) and not list(folder.glob('staging-*'))
 d=dict(d,record_path=str(folder/'result.json'));jobs.append(d)
for role in ('candidate','reference'):
 good=['--synthetic','q5','0',*(['none','1'] if role=='candidate' else [])]
 for name,args,marker in [('real',['--real','missing'],True),('no-marker',good,False),('file-input',['--synthetic','/models/model.gguf','0',*(['none','1'] if role=='candidate' else [])],True)]:
  label='reject-'+role+'-'+name
  if label in records:d=records[label];reused+=1
  else:d=m.job(completion,label,role,args,status=1,marker=marker);retried.append(label)
  assert d['status']==1 and d['worker_records'][-1]['peak_weight_bytes']==0;jobs.append(d)
assert len(jobs)==119 and len(rows)==63
result={'success':True,'baseline_reproduction':baseline,'build':build,'results_root':str(original),'comparisons':rows,'clamp_coverage':coverage,'faults_passed':faults,'rejected_entry_cases':6,'runs':jobs,'real_model_access':False,'full_Ricursive_layer_claim':False,'existing_gate_changed':False,'whole_system_4GiB_qualification':False,'interruption_recovery':{'same_sources_and_binaries_verified':True,'accepted_existing_jobs':reused,'new_jobs':len(retried),'retried_labels':retried,'original_incomplete_attempts_preserved':original_incomplete,'completion_directory':str(completion),'note':'No incomplete/missing result accepted. Original attempt files unchanged; new attempts have separate directories. Numerical comparisons/clamp coverage/repeated weight identities rechecked.'}}
assert all(not j['reasons'] and j['rlimits_unchanged'] and j['worker_records'][-1]['owned_resources']==0 for j in jobs)
with (original/'summary.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
(completion/'resume-receipt.json').write_text(json.dumps(result['interruption_recovery'],indent=2)+'\n')
print('PASS final: 63 byte-exact taps, 100 faults, 6 rejections; reused',reused,'complete jobs; ran',len(retried),'new jobs; summary',original/'summary.json',flush=True)
