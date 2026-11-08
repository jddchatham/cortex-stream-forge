#!/usr/bin/env python3
"""Strict full-file comparison, not an oracle-provenance certifier or inference runner."""
import argparse,hashlib,json,os,struct,tempfile
from pathlib import Path
def digest(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pair(a,b,n,h):
 if os.path.samefile(a,b):raise ValueError('reference and actual are the same file')
 if a.stat().st_size!=n or b.stat().st_size!=n:raise ValueError('wrong complete file length')
 if digest(a)!=h:raise ValueError('reference hash mismatch')
 with a.open('rb')as f,b.open('rb')as g:
  while True:
   x=f.read(1024*1024);y=g.read(1024*1024)
   if x!=y:raise ValueError('byte mismatch')
   if not x:break
def required(case):
 ids={f'layer-{l:02d}-p{p:02d}':40960 for p in range(27 if case=='prompt' else 2)for l in range(48)}
 if case=='prompt':ids.update({f'prefix-{p:02d}':993280 for p in range(26)});ids.update({'select-answer':993280,'select-eog':993280})
 else:ids.update({'logits-p00':993280,'logits-p01':993280})
 return ids
def safe(root,name):
 p=(root/name).resolve()
 if not p.is_relative_to(root.resolve())or not p.is_file():raise ValueError('missing/escaping path')
 return p
def selftest():
 with tempfile.TemporaryDirectory()as d:
  a=Path(d)/'a';b=Path(d)/'b';a.write_bytes(struct.pack('<f',0.0));b.write_bytes(a.read_bytes());pair(a,b,4,digest(a));passed=1
  for payload,n,h in [(struct.pack('<f',-0.0),4,digest(a)),(b'',4,digest(a)),(a.read_bytes()+b'xxxx',4,digest(a)),(a.read_bytes(),4,'0'*64)]:
   b.write_bytes(payload)
   try:pair(a,b,n,h)
   except ValueError:passed+=1
   else:raise AssertionError('invalid comparison accepted')
  try:pair(a,a,4,digest(a))
  except ValueError:passed+=1
  else:raise AssertionError('same file accepted')
  assert len(required('prompt'))==1324 and len(required('two-token'))==98
  print('PASS',passed,'synthetic byte checks and mandatory coverage counts; no model fixtures')
def main():
 p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--case',choices=['prompt','two-token']);p.add_argument('--manifest',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--actual',type=Path);a=p.parse_args()
 if a.self_test:selftest();return
 if not all([a.case,a.manifest,a.reference,a.actual]):p.error('case, manifest, reference, actual required')
 m=json.loads(a.manifest.read_text())
 if m.get('oracle_status')!='QUALIFIED' or m.get('oracle_role')not in ['NEW_STOCK_REFERENCE','RECOVERED_ORIGINAL']:raise ValueError('oracle is not qualified; comparison blocked')
 if not m.get('qualification_receipt_sha256')or not m.get('oracle_executable_sha256'):raise ValueError('missing provenance receipt/identity')
 entries=m.get('pairs',[]);d={e['id']:e for e in entries};need=required(a.case)
 if len(d)!=len(entries)or set(d)!=set(need):raise ValueError('missing, duplicate or unexpected coverage IDs')
 for k,n in need.items():
  e=d[k]
  if e['bytes']!=n:raise ValueError('wrong expected shape/length')
  pair(safe(a.reference,e['reference']),safe(a.actual,e['actual']),n,e['sha256'])
 print('PASS',len(need),'complete files byte-identical. Manifest provenance must be independently reviewed.')
if __name__=='__main__':main()
