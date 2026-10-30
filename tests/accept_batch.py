"""Full-model prefill differential; stores activations, not per-token recurrent states."""
import argparse
import json
import time
import numpy as np
from kindi.lightning import Engine
from kindi.lightning_batch import BatchOps
from kindi.lightning_tokenizer import Tok
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
p.add_argument('--length',type=int,default=129);p.add_argument('--chunks',default='1,7,17,64,128');a=p.parse_args()
eng=Engine(a.model,ctx=1024);b=BatchOps(eng);tok=Tok(a.model)
text=tok.chat([{'role':'user','content':('Explain how arithmetic and computer memory work. Привет, мир! Write clear examples.\n'*30)}])
tokens=tok.encode(text)[:a.length];acts={i:[] for i in range(eng.spec.layers)}
start=time.time()
for token in tokens:
    def trace(i,x,e):acts[i].append(x.get())
    logits=eng.forward(token,trace=trace)
reference=eng.snapshot();acts={i:np.stack(v) for i,v in acts.items()}
print(json.dumps({'sequential_s':time.time()-start,'tokens':len(tokens)}),flush=True)
results=[]
for chunk in map(int,a.chunks.split(',')):
    eng.reset();errors=[];start=time.time()
    for offset in range(0,len(tokens),chunk):
        current=tokens[offset:offset+chunk]
        def trace(i,x,e):
            expected=acts[i][offset:offset+len(current)];actual=x.get()
            np.testing.assert_allclose(actual,expected,atol=3e-3,rtol=2e-3,err_msg='chunk %d position %d layer %d'%(chunk,offset,i))
            errors.append(float(np.max(np.abs(actual-expected))))
        logits=b.forward(current,trace=trace)
    snapshot=eng.snapshot()
    for kind in ('states','histories','keys','values'):
        for i,array in reference[kind].items():np.testing.assert_allclose(snapshot[kind][i],array,atol=3e-3,rtol=3e-3,err_msg=kind+' layer '+str(i))
    np.testing.assert_allclose(snapshot['logits'],reference['logits'],atol=3e-3,rtol=2e-3)
    assert snapshot['logits'].argmax()==reference['logits'].argmax()
    result={'chunk':chunk,'seconds':time.time()-start,'max_activation_abs':max(errors),'logits_max_abs':float(np.max(np.abs(snapshot['logits']-reference['logits'])))}
    results.append(result);print(json.dumps(result),flush=True)
with open(a.output,'w') as f:json.dump({'passed':True,'tokens':len(tokens),'chunks':results},f,indent=2)
