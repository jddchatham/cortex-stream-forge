"""Source-only preservation entry point. Default: verify evidence, no inference.
--run calls unchanged QIES111 synthetic APIs and their unchanged resource guards.
No model download, real-weight dispatch or production integration is provided.
"""
from pathlib import Path
import argparse,json,hashlib,importlib.util,subprocess,os,sys,datetime,uuid,shutil
R=Path(__file__).resolve().parent;W=R.parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify():
 acceptance=json.loads((R/'ACCEPTANCE.json').read_text());assert acceptance['native_processes']==960 and acceptance['exact_capture_pairs']==12480
 provenance=json.loads((R/'PROVENANCE.json').read_text())
 for row in provenance['source_files']:assert sha(W/row['path'])==row['published_sha256'],row['path']
 expected=None;processes=captures=0
 for trial in (1,2,3):
  jobs=[json.loads(x) for x in (R/'evidence'/f'trial-{trial}-processes.jsonl').read_text().splitlines()];assert len(jobs)==320
  assert {(j['role'],j['position'],j['layer']) for j in jobs}=={(role,pos,layer) for role in ('candidate','reference') for pos in (0,1) for layer in range(80)}
  for j in jobs:
   assert j['exit_status']==0 and j['termination_signal'] is None and j['rlimits_unchanged']
   assert j['owned_resources_after_cleanup']==j['adapter_open_files_after_cleanup']==0
   assert all(all(d[k]==0 for k in ('oom','oom_kill','oom_group_kill')) for d in j['oom_deltas'])
  rows=[json.loads(x) for x in (R/'evidence'/f'trial-{trial}-captures.jsonl').read_text().splitlines()];assert len(rows)==4160
  mapping={(d['token_position'],d['layer'],d['tap']):d['sha256'] for d in rows};assert len(mapping)==4160 and all(d['byte_exact'] for d in rows)
  if expected is None:expected=mapping
  else:assert expected==mapping,'historical repetitions differ'
  processes+=len(jobs);captures+=len(rows)
 for line in (R/'PACKAGE-SHA256.jsonl').read_text().splitlines():
  d=json.loads(line);assert sha(W/d['path'])==d['sha256'],d['path']
 print(f'PASS preservation evidence: {processes} exit-zero processes; {captures} exact capture pairs; source/package hashes valid',flush=True)
 return expected

def run(out,trials,four,expected):
 if 'QIES106_SDK_STATE' not in os.environ:raise RuntimeError('Build the pinned source-only SDK and export QIES106_SDK_STATE; see README')
 out=out.resolve();out.mkdir(parents=True,exist_ok=False)
 # The SDK is not an automatically enabled full-model oracle. Build only the
 # preserved private workers; their QIES104 synthetic/real boundaries are intact.
 env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
 subprocess.run(['/usr/bin/python3','-I','-B',W/'qies111/build.py'],env=env,check=True)
 s=importlib.util.spec_from_file_location('preserved111',W/'qies111/run.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a);a.fixtures()
 results=[]
 for number in range(1,trials+1):
  trial=out/f'trial-{number}';trial.mkdir();base=a.campaign(trial,80)
  got={(d['token_position'],d['layer'],d['tap']):d['sha256'] for d in map(json.loads,(base/'comparisons.jsonl').read_text().splitlines())};assert got==expected,'historical SHA mismatch; investigate platform/compiler/configuration without relaxing exactness'
  results.append({'trial':number,'byte_exact_captures':4160,'matches_historical_sha256':True})
 # Preserve the original diagnostic functions. Only their location bindings are
 # relocated; no numerical body, source file, tolerance or resource policy changes.
 legacy=out/'legacy-layout';legacy.mkdir();(legacy/'qies111-worktree').symlink_to(W,target_is_directory=True);driver=legacy/'qies112a-diagnostic';driver.mkdir();shutil.copyfile(R/'run.py',driver/'run.py')
 s=importlib.util.spec_from_file_location('diagnostic112a',driver/'run.py');d=importlib.util.module_from_spec(s);s.loader.exec_module(d);d.O=W;d.S=W/'qies111';d.B=d.S/'build';d.a=a
 first=out/'trial-1/layers-80-start-0';d.faults(out,first);d.interruption(out,first)
 (out/'boundary').mkdir();a.campaign(out/'boundary',5,1023)
 # This explicitly builds a separate sanitizer caller from the preserved source,
 # unlike historical QIES112A, which tested its already-preserved original ELF.
 subprocess.run(['/usr/bin/python3','-I','-B',W/'qies111/sanitizer_numerical.py',out/'trial-1'],env=env,check=True)
 if four:d.extend(out,out/f'trial-{trials}/layers-80-start-0')
 (out/'REPRODUCTION.json').write_text(json.dumps({'success':True,'trials':results,'four_positions':four,'new_build_not_original_ELF':True,'real_weights':False,'guards_unchanged':True},indent=2)+'\n')
 print('PASS source reproduction:',out,flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path);p.add_argument('--trials',type=int,choices=[1,2,3],default=3);p.add_argument('--four-positions',action='store_true');a=p.parse_args();expected=verify()
 if a.run:
  if a.out is None:p.error('--run requires a fresh persistent --out directory')
  run(a.out,a.trials,a.four_positions,expected)
if __name__=='__main__':main()
