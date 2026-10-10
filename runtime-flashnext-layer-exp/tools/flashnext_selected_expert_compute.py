#!/usr/bin/env python3
"""Run the isolated normal-ggml per-expert executor for the frozen route."""
import json, os, struct, subprocess, tempfile

N = 2560
def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('index'); ap.add_argument('hidden');
    ap.add_argument('--router-artifact', required=True, help='Segment-A/router f32 artifact')
    args=ap.parse_args()
    raw = struct.unpack('<248320f', open(args.router_artifact, 'rb').read())
    logits = raw[12804:12804+512]
    ids = sorted(range(512), key=lambda i: logits[i], reverse=True)[:10]
    m = max(logits)
    ex = [__import__('math').exp(logits[i] - m) for i in range(512)]
    z = sum(ex)
    # Qwen4Exp's selected routing distribution is the softmax over the full
    # expert router; only these dynamically selected entries are consumed.
    weights = [ex[i] / z for i in ids]
    selected_sum = sum(weights)
    weights = [w / selected_sum for w in weights]
    print('FLASHNEXT_DYNAMIC_ROUTE ids=' + ','.join(map(str, ids)))
    print('FLASHNEXT_DYNAMIC_ROUTE weights=' + ','.join(f'{w:.12g}' for w in weights))
    x=json.load(open(args.index)); ts={t['name']:t for t in x['tensors']}
    names=['blk.0.ffn_gate_exps.weight','blk.0.ffn_up_exps.weight','blk.0.ffn_down_exps.weight']
    shard=os.path.join(x['model_dir'],ts[names[0]]['shard'])
    exe=os.path.join(os.path.dirname(os.path.dirname(__file__)),'..','runtime-flashnext-exp-build','bin','llama-flashnext-single-expert')
    exe=os.path.abspath(exe)
    acc=[0.0]*N
    with tempfile.TemporaryDirectory(prefix='flashnext-experts-') as d:
        for eid,rw in zip(ids,weights):
            a=[]
            for n in names:
                t=ts[n]; b=t['bytes']//512; a += [str(t['offset']+eid*b),str(b)]
            out=os.path.join(d,f'{eid}.f32')
            c=subprocess.run([exe,shard,str(eid),*a,args.hidden,out],capture_output=True,text=True)
            if c.returncode: raise SystemExit(c.stderr)
            print(c.stdout.strip())
            vals=struct.unpack('<%df'%N,open(out,'rb').read())
            for i,v in enumerate(vals): acc[i]+=rw*v
    out='/tmp/flashnext-selected-moe.f32'; open(out,'wb').write(struct.pack('<%df'%N,*acc))
    h=1469598103934665603
    for v in acc:
        u=struct.unpack('<I',struct.pack('<f',v))[0]; h^=u; h=(h*1099511628211)&((1<<64)-1)
    print(f'FLASHNEXT_DYNAMIC_SPARSE_MOE output={out} hash={h:016x} selected={ids} weight_sum={sum(weights):.12g}')
if __name__=='__main__': main()
