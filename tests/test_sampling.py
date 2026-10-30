import argparse,json,time
import numpy as np
from kindi.lightning_server import sample
p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
class Logits(object):
    def __init__(self,x):self.x=x
    def get(self):return self.x
def reference(x,rng,temp,k,p):
    values=x.astype(np.float64)
    with np.errstate(over='ignore'):values=(values-values.max())/temp
    ids=np.argsort(-values,kind='stable')
    if k>0:ids=ids[:k]
    prob=np.exp(values[ids]-values[ids].max());prob/=prob.sum()
    if p<1:
        n=max(1,int(np.searchsorted(np.cumsum(prob),p))+1);ids=ids[:n];prob=prob[:n];prob/=prob.sum()
    return int(rng.choice(ids,p=prob))
generator=np.random.RandomState(5);checks=0
for x in (generator.randn(131072).astype(np.float32),np.zeros(256,np.float32),generator.randint(-3,4,256).astype(np.float32)):
    for k in (1,20,255,256,300,0):
        for temp in (.7,1e-308):
            for p in (.8,1.):
                first=np.random.RandomState(7);second=np.random.RandomState(7)
                for _ in range(4):
                    assert sample(Logits(x),first,temp,k,p)==reference(x,second,temp,k,p)
                    checks+=1
x=generator.randn(131072).astype(np.float32);times={}
for name,fn in [('full_sort',lambda r:reference(x,r,.7,20,.8)),('partial_selection',lambda r:sample(Logits(x),r,.7,20,.8))]:
    rng=np.random.RandomState(7);fn(rng);start=time.perf_counter()
    for _ in range(100):fn(rng)
    times[name]=(time.perf_counter()-start)/100*1000
result={'passed':True,'exact_seeded_comparisons':checks,'milliseconds_per_sample':times}
if a.output:
    with open(a.output,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
