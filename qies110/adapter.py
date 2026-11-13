"""Trusted catalog/manifest preparation using the unchanged QIES101 validator.
Data-only: this module cannot run GGML arithmetic or load whole tensor payloads.
"""
from pathlib import Path
import importlib.util,json,os,hashlib
R=Path(__file__).resolve().parent;W=R.parent
s=importlib.util.spec_from_file_location('q101_readonly',W/'qies101/extract.py');q101=importlib.util.module_from_spec(s);s.loader.exec_module(q101)
MODEL_SHA='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc'
def catalog():
 index=q101.read_json(W/'qies098-checks/ripple-tensor-index.json');assert index['model_sha256']==MODEL_SHA
 return index,q101.validate(index,40426079712)
def layer(number):
 if type(number)is not int or not 0<=number<80:raise ValueError('layer identity')
 index,tensors=catalog();rows=[dict(t) for n,t in tensors.items() if n.startswith(f'blk.{number}.')]
 if len(rows)!=15:raise ValueError('layer tensor count')
 return rows
def manifest(path,model,number,hashes,mode='REAL',virtual=None):
 rows=layer(number);identity=q101.identity(Path(model).stat());lines=['QIES110-SOURCE-1',mode,str(number),json.dumps(str(virtual or Path(model).resolve()))+' '+' '.join(map(str,identity)),'15']
 for t in rows:
  shape=t['shape'];name=t['name'].split('.',2)[2]
  lines.append(' '.join([json.dumps(name),str(t['type']),str(shape[0]),str(shape[1] if len(shape)>1 else 1),str(t['offset']),str(t['bytes']),hashes[name]]))
 Path(path).write_text('\n'.join(lines)+'\n')
 return identity
