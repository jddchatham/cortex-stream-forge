"""Bounded complete-file integrity check; no tensor evaluation."""
from pathlib import Path
import hashlib,json,os,resource,time,urllib.request
R=Path(__file__).resolve().parent;W=R.parent
revision='df550a7178a87bbfb6ee2bdbd37603c553d536ee'
url='https://huggingface.co/api/models/AtomicChat/Ripple-65B-Preview-GGUF/revision/'+revision+'?blobs=true'
with urllib.request.urlopen(url,timeout=60) as f:meta=json.load(f)
(R/'published-model-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
name='Ripple-65B-Preview-AD-Q4_K_M.gguf';entry=next(x for x in meta['siblings'] if x['rfilename']==name)
expected=entry['lfs']['sha256'];assert expected=='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc'
p=W.parent/'qies098-data/models'/name
fd=os.open(p,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
def identity(st):return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
t=time.monotonic();h=hashlib.sha256();n=0
try:
 before=identity(os.fstat(fd))
 while b:=os.read(fd,1024*1024):h.update(b);n+=len(b)
 after=identity(os.fstat(fd));assert before==after and n==entry['size']==entry['lfs']['size']
finally:os.close(fd)
actual=h.hexdigest();assert actual==expected
(R/'model-sha256.txt').write_text(actual+'  '+str(p)+'\n')
(R/'model-hash-result.json').write_text(json.dumps({'published_url':url,'revision':revision,'expected':expected,'actual':actual,'bytes':n,'identity':before,'elapsed_seconds':time.monotonic()-t,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'chunk_bytes':1024*1024,'passed':True},indent=2)+'\n')
print('PASS fresh full SHA256',n,'bytes',actual,flush=True)
