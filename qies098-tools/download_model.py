"""Single exact-file HF acquisition, resumable temporary file and checksum publication."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time
r=Path(__file__).resolve().parent.parent;data=r.parent/'qies098-data';meta=data/'metadata';models=data/'models'
info=json.loads((meta/'model-api.json').read_text());revision=info['sha'];name='Ripple-65B-Preview-AD-Q4_K_M.gguf'
file=next(x for x in info['siblings'] if x['rfilename']==name);expected=file['lfs']['sha256'];size=file['size'];reserve=30*1024**3
final=models/name
if final.exists():
 raise SystemExit('Existing target preserved. Verify separately; no overwrite permitted.')
free=lambda:os.statvfs(models).f_bavail*os.statvfs(models).f_frsize
existing_partial=sum(p.stat().st_size for p in models.rglob('*.incomplete'))
assert free()>=max(0,size-existing_partial)+reserve,'Insufficient remaining-file + reserve capacity'
env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
env.update({'HF_HOME':str(data/'hf-home'),'HF_HUB_DISABLE_XET':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','HF_HUB_DOWNLOAD_TIMEOUT':'60','HF_HUB_ETAG_TIMEOUT':'30'})
cmd=[str(data/'hf-venv/bin/hf'),'download','AtomicChat/Ripple-65B-Preview-GGUF',name,'--revision',revision,'--local-dir',str(models),'--max-workers','1']
start=time.monotonic();last=0;peak=0
with (meta/'download-private.log').open('a') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while p.poll() is None:
  partial=list(models.rglob('*.incomplete'));done=sum(x.stat().st_size for x in partial)
  if final.exists():done=final.stat().st_size
  peak=max(peak,done)
  if free()<reserve or free()-max(0,size-done)<reserve:
   os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);raise SystemExit('Storage reserve protection stopped download; partial retained')
  now=time.monotonic()
  if now-last>=20:
   print(json.dumps({'download_bytes':done,'expected_bytes':size,'percent':round(done/size*100,2),'elapsed_seconds':round(now-start),'free_bytes':free()}),flush=True);last=now
  time.sleep(1)
 status=p.wait()
if status:
 (meta/'download-result.json').write_text(json.dumps({'status':'FAILED_RESUMABLE','exit_status':status,'revision':revision,'expected_sha256':expected,'expected_bytes':size},indent=2)+'\n')
 raise SystemExit('HF download failed; partial retained. Private log not printed to avoid signed-URL exposure.')
assert final.stat().st_size==size,'Wrong size; do not accept target'
# Quarantine until locally verified. Atomic rename makes no duplicate large copy.
quarantine=models/(name+'.verifying');assert not quarantine.exists();final.rename(quarantine)
print('Download complete; streaming SHA-256 verification started',flush=True)
h=hashlib.sha256()
with quarantine.open('rb') as f:
 for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
actual=h.hexdigest();assert actual==expected,'Checksum mismatch; quarantine preserved; not accepted'
quarantine.rename(final)
result={'status':'VERIFIED','repository':'AtomicChat/Ripple-65B-Preview-GGUF','revision':revision,'filename':name,'bytes':size,'sha256':actual,'expected_sha256':expected,'elapsed_seconds':time.monotonic()-start,'free_bytes':free(),'path':str(final),'huggingface_hub_version':'2.4.0','only_one_quantization_downloaded':True}
(meta/'download-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
