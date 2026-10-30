import argparse
import json
import numpy as np
from kindi.lightning import Engine
from kindi.lightning_batch import BatchOps
from kindi.lightning_tokenizer import Tok
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
eng=Engine(a.model,ctx=2048);b=BatchOps(eng);tok=Tok(a.model)
text=('Explain arithmetic and memory with worked examples. Compare reversible operations and verify each result. '
      'Привет, мир! Numbers: 17 + 25 = 42.\n')*600
tokens=tok.encode(tok.chat([{'role':'user','content':text}]))[:896]
reference={i:[] for i in range(eng.spec.layers)}
for step,token in enumerate(tokens):
    def trace(i,x,e):reference[i].append(x.get())
    eng.forward(token,logits=False,trace=trace if step>=768 else None)
reference={i:np.stack(v) for i,v in reference.items()};print('sequential reference ready',flush=True)
eng.reset();eng.prefill(tokens[:768],chunk_size=128);batch={};result={}
def trace(i,x,e):
    batch[i]=x.get();diff=np.max(np.abs(batch[i]-reference[i]),axis=1)
    if diff.max()<1e-3:return
    indices=np.flatnonzero(diff>=1e-3);t=int(indices[0]);kind=eng.spec.blocks[i]
    result.update(layer=i,kind=kind,position=768+t,token_offset=t,output_max_abs=float(diff[t]),affected_positions=(indices+768).tolist())
    if kind=='moe' and i>0:
        prefix='blk.%d.'%i;refx=eng.cp.asarray(reference[i-1][t]);actual=eng.cp.asarray(batch[i-1])
        rz=eng.ops.norm(refx,eng.w[prefix+'attn_norm.weight'],eng.spec.eps)
        bz=b.norm(actual,eng.w[prefix+'attn_norm.weight'])
        rl=eng.linear(prefix+'ffn_gate_inp.weight',rz);bl=b.linear(prefix+'ffn_gate_inp.weight',bz)
        ri,rw=eng.ops.router(rl,eng.w[prefix+'exp_probs_b.bias'],eng.spec.topk,eng.spec.scale)
        bi,bw=b.route(bl,eng.w[prefix+'exp_probs_b.bias'],eng.spec.topk,eng.spec.scale)
        rl=rl.get();bl=bl[t].get();bias=eng.w[prefix+'exp_probs_b.bias'].get()
        rs=1/(1+np.exp(-rl.astype(np.float64)))+bias;bs=1/(1+np.exp(-bl.astype(np.float64)))+bias
        candidates=np.argsort(-rs)[:eng.spec.topk+2]
        matrix=eng.w[prefix+'ffn_gate_inp.weight'].get().astype(np.float64)
        rr=matrix.dot(rz.get().astype(np.float64));br=matrix.dot(bz[t].get().astype(np.float64))
        rr=1/(1+np.exp(-rr))+bias;br=1/(1+np.exp(-br))+bias
        result.update(input_max_abs=float(np.max(np.abs(reference[i-1][t]-batch[i-1][t]))),
            router_logit_max_abs=float(np.max(np.abs(rl-bl))),reference_ids=ri.get().tolist(),batch_ids=bi[t].get().tolist(),
            candidates=[{'id':int(j),'reference_score':float(rs[j]),'batch_score':float(bs[j]),'fp64_ref':float(rr[j]),'fp64_batch':float(br[j])} for j in candidates],
            fp64_ref_ids=np.argsort(-rr)[:eng.spec.topk].tolist(),fp64_batch_ids=np.argsort(-br)[:eng.spec.topk].tolist())
    print(json.dumps(result,indent=2),flush=True)
    with open(a.output,'w') as f:json.dump(result,f,indent=2)
    raise StopIteration()
try:b.forward(tokens[768:],trace=trace)
except StopIteration:pass
assert result,'divergence not reproduced'
