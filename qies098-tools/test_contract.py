"""Tests metadata bounds and admission refusal, never model tensors or decode."""
from pathlib import Path
import json,sys,unittest,copy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from layer_contract import plan,admission,GIB

class Tests(unittest.TestCase):
 def test_layout(self):
  d=json.loads(next(p for p in [Path(__file__).resolve().parent.parent/'qies098-checks/gguf-inspection.json',Path(__file__).resolve().parent.parent/'qies098-checks/gguf-prefix-provisional.json'] if p.exists()).read_text())
  a=plan(d,0);b=plan(d,4)
  self.assertEqual(a['attention'],'sliding');self.assertEqual(b['attention'],'global')
  self.assertEqual(len(a['tensors']),15);self.assertLess(a['weight_bytes'],GIB)
  self.assertEqual(a['kv_f16_payload_bytes'],1048576)
  self.assertRaises(ValueError,plan,d,80);self.assertRaises(ValueError,plan,d,0,1024)
 def test_bad_shape(self):
  d=json.loads(next(p for p in [Path(__file__).resolve().parent.parent/'qies098-checks/gguf-inspection.json',Path(__file__).resolve().parent.parent/'qies098-checks/gguf-prefix-provisional.json'] if p.exists()).read_text())
  changed=copy.deepcopy(d);changed['metadata']['ricursive.embedding_length']=4096
  self.assertRaises(ValueError,plan,changed)
  changed=copy.deepcopy(d)
  next(t for t in changed['tensors'] if t['name']=='blk.0.attn_q.weight')['shape']=[5120,5120]
  self.assertRaises(ValueError,plan,changed)
 def test_gate(self):
  self.assertFalse(admission(6*GIB-1,True,True,True)[0])
  self.assertFalse(admission(6*GIB,False,True,True)[0])
  self.assertFalse(admission(6*GIB,True,False,True)[0])
  self.assertFalse(admission(6*GIB,True,True,False)[0])
  self.assertTrue(admission(6*GIB,True,True,True)[0])

if __name__=='__main__':unittest.main()
