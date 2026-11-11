"""Tiny header-only fixtures. No actual model tensor reads or inference."""
from pathlib import Path
import struct,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from inspect_gguf import inspect
ROOT=Path(__file__).resolve().parent.parent
LIB=ROOT.parent/'cortex-stream-forge-qies074/build-quortex/bin/libggml-base.so.0.25.3'

def string(x):
 b=x.encode();return struct.pack('<Q',len(b))+b

def fixture(offset=0,shape=(32,4),kind=0,alignment=32):
 meta=string('general.architecture')+struct.pack('<I',8)+string('ricursive')
 meta+=string('general.alignment')+struct.pack('<II',4,alignment)
 head=b'GGUF'+struct.pack('<IQQ',3,1,2)+meta
 return head+string('blk.0.fixture')+struct.pack('<I',len(shape))+b''.join(struct.pack('<Q',x) for x in shape)+struct.pack('<IQ',kind,offset)

class Tests(unittest.TestCase):
 def run_fixture(self,blob,expected_size=4096):
  with tempfile.TemporaryDirectory(prefix='metadata-',dir=ROOT/'qies098-checks') as d:
   p=Path(d)/'header-only.gguf';p.write_bytes(blob)
   return inspect(p,expected_size,LIB)
 def test_header(self):
  d=self.run_fixture(fixture());self.assertEqual(d['tensor_count'],1);self.assertEqual(d['tensors'][0]['bytes'],512)
  self.assertFalse(d['tensor_payload_read'])
 def test_invalid(self):
  for b in (b'FAIL'+fixture()[4:],fixture(offset=4096),fixture(offset=1),fixture(shape=(0,4)),fixture(kind=999),fixture(alignment=0)):
   with self.subTest():self.assertRaises((ValueError,EOFError),self.run_fixture,b)
 def test_partial(self):
  self.assertRaises(EOFError,self.run_fixture,fixture()[:20])

if __name__=='__main__':unittest.main()
