"""Synthetic decoder-only MoE adapters reuse the kernels without any Mamba layers."""
import argparse
import json
import types
import numpy as np
import cupy as cp
from kindi.lightning import Engine
from kindi.lightning_batch import BatchOps
from kindi.moe import CudaOps
from kindi.packed import dequantize
p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();results=[]
for width,experts,topk,hidden in [(64,8,2,64),(128,32,4,192)]:
    rng=np.random.RandomState(width+experts);eng=Engine.__new__(Engine)
    s=types.SimpleNamespace(width=width,experts=experts,topk=topk,scale=2.5,eps=1e-5,
        vocab=128,qheads=4,kvheads=2,hd=width//4,blocks=['attention','moe','attention','moe'],layers=4)
    eng.spec=s;eng.ops=CudaOps();eng.cp=cp;eng.w={};eng.file=types.SimpleNamespace(tensors={});dense={};eng.fingerprint='synthetic'
    eng.states={};eng.histories={};eng.keys={};eng.values={};eng.ctx=64;eng.position=0;eng.tokens=[];eng.last_logits=None
    def add(name,shape,kind=42,value=None):
        if kind==0:
            raw=np.asarray(value if value is not None else rng.randn(*shape)*.05,dtype=np.float32);dec=raw
        else:
            bs,size={42:(64,18),8:(32,34)}[kind];count=int(np.prod(shape));raw=np.zeros((count//bs,size),np.uint8)
            raw[:,:2]=np.tile(np.frombuffer(np.float16(.025).tobytes(),np.uint8),(len(raw),1))
            if kind==42:
                codes=rng.randint(0,3,(len(raw),16,4)).astype(np.uint8)
                raw[:,2:]=(codes[:,:,0]|(codes[:,:,1]<<2)|(codes[:,:,2]<<4)|(codes[:,:,3]<<6))
            else:raw[:,2:]=rng.randint(-8,9,(len(raw),32)).astype(np.int8).view(np.uint8)
            raw=raw.reshape(-1);dec=dequantize(raw,kind).reshape(shape)
        eng.w[name]=cp.asarray(raw);dense[name]=dec
        eng.file.tensors[name]=types.SimpleNamespace(name=name,shape=shape,kind=kind,nbytes=raw.nbytes)
    add('token_embd.weight',(s.vocab,width),8);add('output.weight',(s.vocab,width),8)
    add('output_norm.weight',(width,),0,np.ones(width))
    for layer,kind in enumerate(s.blocks):
        prefix='blk.%d.'%layer;add(prefix+'attn_norm.weight',(width,),0,np.ones(width))
        if kind=='attention':
            for name,n in [('attn_q',s.qheads),('attn_k',s.kvheads),('attn_v',s.kvheads)]:add(prefix+name+'.weight',(n*s.hd,width))
            add(prefix+'attn_output.weight',(width,width));eng.keys[layer]=cp.empty((64,s.kvheads,s.hd),cp.float32);eng.values[layer]=cp.empty_like(eng.keys[layer])
        else:
            add(prefix+'ffn_gate_inp.weight',(experts,width),0);add(prefix+'exp_probs_b.bias',(experts,),0)
            add(prefix+'ffn_up_exps.weight',(experts,hidden,width));add(prefix+'ffn_down_exps.weight',(experts,width,hidden))
            add(prefix+'ffn_up_shexp.weight',(hidden*2,width));add(prefix+'ffn_down_shexp.weight',(width,hidden*2))
    assert eng.memory_plan(64)['state']==0
    assert eng.memory_plan(64)['kv']==2*2*64*s.kvheads*s.hd*4
    keys={};values={}
    def norm(x,w):return (x.astype(np.float64)/np.sqrt(np.mean(x.astype(np.float64)**2)+s.eps)*w).astype(np.float32)
    def linear(name,x,expert=None):
        w=dense[name] if expert is None else dense[name][expert]
        return w.astype(np.float64).dot(x.astype(np.float64)).astype(np.float32)
    def reference(token):
        x=dense['token_embd.weight'][token].copy()
        for layer,kind in enumerate(s.blocks):
            prefix='blk.%d.'%layer;z=norm(x,dense[prefix+'attn_norm.weight'])
            if kind=='moe':
                logits=linear(prefix+'ffn_gate_inp.weight',z);scores=(1/(1+np.exp(-logits.astype(np.float64)))).astype(np.float32)
                ids=np.argsort(-(scores+dense[prefix+'exp_probs_b.bias']),kind='stable')[:topk]
                probs=scores[ids]/scores[ids].sum()*s.scale;y=np.zeros(width,np.float64)
                for e,prob in zip(ids,probs):
                    up=np.maximum(linear(prefix+'ffn_up_exps.weight',z,e),0)**2
                    y+=prob*linear(prefix+'ffn_down_exps.weight',up,e)
                up=np.maximum(linear(prefix+'ffn_up_shexp.weight',z),0)**2;y+=linear(prefix+'ffn_down_shexp.weight',up)
            else:
                q=linear(prefix+'attn_q.weight',z).reshape(s.qheads,s.hd)
                keys.setdefault(layer,[]).append(linear(prefix+'attn_k.weight',z).reshape(s.kvheads,s.hd))
                values.setdefault(layer,[]).append(linear(prefix+'attn_v.weight',z).reshape(s.kvheads,s.hd))
                k=np.stack(keys[layer]);v=np.stack(values[layer]);heads=[]
                for h in range(s.qheads):
                    score=k[:,h//2].astype(np.float64).dot(q[h])/np.sqrt(s.hd);pr=np.exp(score-score.max());pr/=pr.sum();heads.append(pr.dot(v[:,h//2]))
                y=linear(prefix+'attn_output.weight',np.asarray(heads,np.float32).reshape(-1))
            x=(x+y).astype(np.float32)
        return linear('output.weight',norm(x,dense['output_norm.weight']))
    tokens=rng.randint(0,s.vocab,17).tolist();maximum=0
    for token in tokens:
        actual=eng.forward(token).get();expected=reference(token)
        np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=5e-4);maximum=max(maximum,float(np.max(np.abs(actual-expected))))
    final=eng.last_logits.get();eng.reset();batch=BatchOps(eng)
    for start in range(0,len(tokens),7):batch.forward(tokens[start:start+7])
    np.testing.assert_array_equal(eng.last_logits.get(),final)
    snapshot=eng.snapshot()
    for update in ({'version':-1},{'fingerprint':'other'},{'position':17.0},{'tokens':[-1]*17},
                   {'logits':np.zeros(128,np.float64)},{'values':{}},{'keys':{0:np.zeros((17,2,s.hd),np.float32),2:np.zeros((17,2,s.hd),np.float64)}}):
        invalid=dict(snapshot);invalid.update(update)
        try:eng.restore(invalid);raise AssertionError('invalid snapshot accepted')
        except ValueError:pass
        np.testing.assert_array_equal(eng.last_logits.get(),final);assert eng.position==17
        for i,array in snapshot['keys'].items():np.testing.assert_array_equal(eng.keys[i][:17].get(),array)
    eng.forward(2);eng.forward(3);continuation=eng.last_logits.get();eng.restore(snapshot)
    eng.forward(2);eng.forward(3);np.testing.assert_array_equal(continuation,eng.last_logits.get())
    eng.reset();eng.prefill(tokens+[2,3],chunk_size=7);np.testing.assert_array_equal(continuation,eng.last_logits.get())
    results.append({'width':width,'experts':experts,'topk':topk,'intermediate':hidden,'mamba_layers':0,'max_abs_vs_numpy':maximum,'batch_bit_exact':True})
    del batch,eng;cp.get_default_memory_pool().free_all_blocks()
result={'passed':True,'cases':results}
if a.output:
    with open(a.output,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
