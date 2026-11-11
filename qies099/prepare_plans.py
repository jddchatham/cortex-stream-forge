"""Metadata only; binds active-layer manifests to the freshly verified model inode."""
from pathlib import Path
import json,os
r=Path(__file__).resolve().parent.parent;c=r/'qies098-checks';out=r/'qies099/checks'
index=json.loads((c/'ripple-tensor-index.json').read_text());identity=json.loads((c/'model-file-identity.json').read_text())
sha='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc'
assert (out/'model-sha256.txt').read_text().split()[0]==sha==index['model_sha256']==identity['sha256']
path=Path(index['model_dir'])/index['model_filename'];s=path.stat()
assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==tuple(identity[k] for k in ['device','inode','bytes','mtime_ns','ctime_ns'])
for layer in range(80):
 tensors=[t for t in index['tensors'] if t['name'].startswith(f'blk.{layer}.')];assert len(tensors)==15
 text='QIES099-RICURSIVE-PLAN-1\n'+f'{layer} 256\n{path}\n{sha}\n'
 text+=' '.join(str(identity[k]) for k in ['device','inode','bytes','mtime_ns','ctime_ns'])+'\n'
 for t in tensors:
  shape=t['shape'];text+=f"{t['name']} {t['type']} {shape[0]} {shape[1] if len(shape)>1 else 1} {t['offset']} {t['bytes']}\n"
 (out/f'layer-{layer:02d}.plan').write_text(text)
print('Prepared all 80 layer manifests from verified metadata, no tensor reads')
