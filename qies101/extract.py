"""Bounded read-only index-based tensor extraction, not an inference path."""
from pathlib import Path
import hashlib,json,os,stat,uuid
MAX=(1<<63)-1
CHUNK=1024*1024
TYPES={0:(1,4,'f32'),8:(32,34,'q8_0'),12:(256,144,'q4_K'),13:(256,176,'q5_K'),14:(256,210,'q6_K'),23:(256,136,'iq4_xs')}
def number(n):
 if type(n) is not int or n<0 or n>MAX:raise ValueError('integer/offset overflow')
 return n
def identity(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def no_duplicates(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate JSON key')
  d[k]=v
 return d
def read_json(path):
 with open(path,'rb') as f:b=f.read(2*1024*1024+1)
 if len(b)>2*1024*1024:raise ValueError('index metadata budget')
 return json.loads(b,object_pairs_hook=no_duplicates)
def validate(index,size,start=3615712,alignment=32,expected_count=1203):
 number(size);number(start);number(alignment)
 if alignment<=0 or alignment>4096 or alignment&(alignment-1) or start%alignment:raise ValueError('data alignment')
 if index.get('schema')!='QIES098_RICURSIVE_READONLY_INDEX_v1' or index.get('architecture')!='ricursive':raise ValueError('index identity')
 name=index['model_filename']
 if not isinstance(name,str) or Path(name).name!=name:raise ValueError('unsafe model name')
 tensors=index['tensors']
 if not isinstance(tensors,list) or len(tensors)!=expected_count or expected_count>10000:raise ValueError('tensor count')
 out={};ranges=[]
 for t in tensors:
  n=t['name'];shape=t['shape'];kind=t['type']
  if not isinstance(n,str) or not n or len(n)>256 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.' for c in n) or n in out:raise ValueError('tensor name/duplicate')
  if type(kind) is not int or kind not in TYPES:raise ValueError('unsupported quantization')
  if not isinstance(shape,list) or not 1<=len(shape)<=4:raise ValueError('tensor dimensions')
  block,unit,label=TYPES[kind];elements=1
  for dim in shape:
   number(dim)
   if dim==0 or dim>2**32 or elements>MAX//dim:raise ValueError('shape overflow')
   elements*=dim
  if shape[0]%block or elements//block>MAX//unit:raise ValueError('quantized row/size overflow')
  length=elements//block*unit;offset=number(t['offset']);relative=number(t['relative_offset'])
  if relative>MAX-start or start+relative!=offset or relative%alignment or t.get('shard')!=name:raise ValueError('offset/shard mismatch')
  if number(t['bytes'])!=length or t.get('quantization')!=label:raise ValueError('tensor type/length mismatch')
  if offset>size or length>size-offset:raise ValueError('tensor outside file')
  ranges.append((offset,offset+length));out[n]=dict(t)
 ranges.sort()
 if any(a[1]>b[0] for a,b in zip(ranges,ranges[1:])):raise ValueError('overlapping descriptors')
 return out
class Extractor:
 def __init__(self,path,index,expected_identity,**validation):
  self.fd=-1;self.path=Path(path);self.expected=list(expected_identity);self.bytes_read=0
  self.tensors=validate(index,self.expected[2],**validation)
  try:
   self.fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
   if not stat.S_ISREG(os.fstat(self.fd).st_mode):raise ValueError('regular file required')
   self.check()
  except BaseException:self.close();raise
 def check(self):
  if identity(os.fstat(self.fd))!=self.expected or identity(os.stat(self.path,follow_symlinks=False))!=self.expected:raise ValueError('file identity changed')
 def close(self):
  if self.fd>=0:os.close(self.fd);self.fd=-1
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def extract(self,name,destination,reader=os.pread,writer=os.write,hook=None):
  t=self.tensors[name];self.check();dest=Path(destination)
  if dest.exists() or dest.is_symlink():raise FileExistsError(dest)
  temporary=dest.with_name(dest.name+'.partial-'+uuid.uuid4().hex);out=-1;created=False
  try:
   out=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|os.O_NOFOLLOW,0o600);created=True
   done=0;h=hashlib.sha256()
   while done<t['bytes']:
    wanted=min(CHUNK,t['bytes']-done);b=reader(self.fd,wanted,t['offset']+done)
    if not b or len(b)>wanted:raise EOFError('truncated/invalid read')
    self.bytes_read+=len(b);h.update(b);view=memoryview(b)
    while view:
     n=writer(out,view)
     if n<=0 or n>len(view):raise OSError('short/failed write')
     view=view[n:]
    done+=len(b)
    if hook:hook(done)
   os.fsync(out);self.check();os.close(out);out=-1
   os.link(temporary,dest,follow_symlinks=False) # Exclusive publication, never overwrite.
   return dict(t,sha256=h.hexdigest(),extracted_bytes=done,output=str(dest))
  finally:
   if out>=0:os.close(out)
   if created:temporary.unlink(missing_ok=True)
