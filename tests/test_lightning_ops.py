"""Independent NumPy equations vs the actual CUDA kernels, including Kepler tails."""
import argparse
import json
import time
import numpy as np
import cupy as cp
from kindi.moe import CudaOps
from kindi.packed import dequantize

parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
rng=np.random.RandomState(407)
ops=CudaOps();checks=[]
def check(name,actual,expected,atol=2e-5,rtol=2e-4):
    a=cp.asnumpy(actual) if isinstance(actual,cp.ndarray) else actual
    b=np.asarray(expected,dtype=np.float32)
    err=float(np.max(np.abs(a-b)))
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol,err_msg=name)
    checks.append({'name':name,'max_abs':err})

def packed(rows,cols,kind,experts=1):
    bs,size={42:(64,18),8:(32,34)}[kind]
    b=np.zeros((experts*rows*cols//bs,size),np.uint8)
    scales=(rng.uniform(.001,.1,len(b))).astype(np.float16)
    b[:,:2]=scales.view(np.uint8).reshape(-1,2)
    b[:,2:]=rng.randint(0,256,size=(len(b),size-2)).astype(np.uint8)
    return b.reshape(-1)

for kind in (42,8):
    for cols in (64,1856,2688,3712):
        rows=19;raw=packed(rows,cols,kind,3)
        dense=dequantize(raw,kind).reshape(3,rows,cols)
        x=rng.randn(cols).astype(np.float32)
        check('linear-%d-%d'%(kind,cols),ops.linear(cp.asarray(raw),cp.asarray(x),rows,cols,kind),dense[0].astype(np.float64).dot(x),atol=1e-4)
        ids=np.array([2,0],np.int32)
        xs=rng.randn(2,cols).astype(np.float32)
        check('expert-%d-%d'%(kind,cols),ops.linear(cp.asarray(raw),cp.asarray(xs),rows,cols,kind,cp.asarray(ids),2,True),
            np.stack([dense[e].astype(np.float64).dot(xs[i]) for i,e in enumerate(ids)]),atol=1e-4)

# Exhaustive packed-byte ordering, including code 3 (+2 scale).
raw=np.zeros((256,18),np.uint8);raw[:,:2]=np.float16(1).tobytes()[0],np.float16(1).tobytes()[1]
raw[:,2:]=np.arange(256,dtype=np.uint8)[:,None]
d=dequantize(raw.reshape(-1),42).reshape(256,64)
for b in range(256):
    expected=np.tile([((b//(4**i))%4)-1 for i in range(4)],16)
    assert np.array_equal(d[b],expected)
checks.append({'name':'all-256-Q2-bytes','max_abs':0})

x=rng.randn(4096).astype(np.float32);w=rng.randn(4096).astype(np.float32);g=rng.randn(4096).astype(np.float32)
gx=x.astype(np.float64)*g/(1+np.exp(-g.astype(np.float64)))
expected=(gx.reshape(8,512)/np.sqrt(np.mean(gx.reshape(8,512)**2,axis=1,keepdims=True)+1e-5)).reshape(-1)*w
check('gated-group-rms',ops.norm(cp.asarray(x),cp.asarray(w),1e-5,512,cp.asarray(g)),expected)
logits=rng.randn(128).astype(np.float32);bias=rng.randn(128).astype(np.float32)*.1
score=1/(1+np.exp(-logits.astype(np.float64)))
ids,weights=ops.router(cp.asarray(logits),cp.asarray(bias),6,2.5)
expected_ids=np.argsort(-(score+bias),kind='stable')[:6]
assert np.array_equal(ids.get(),expected_ids)
check('router-weights',weights,score[expected_ids]/score[expected_ids].sum()*2.5)
tie_ids,tie_weights=ops.router(cp.zeros(128,cp.float32),cp.zeros(128,cp.float32),6,2.5)
assert np.array_equal(tie_ids.get(),np.arange(6))
checks.append({'name':'router-tie','max_abs':0})

heads,hd,ss,groups,convk=4,8,128,2,4
inner=heads*hd;channels=inner+2*groups*ss
cw=rng.randn(channels,convk).astype(np.float32)*.1;cb=rng.randn(channels).astype(np.float32)*.01
hist=np.zeros((channels,convk-1),np.float32);gh=cp.asarray(hist)
state=np.zeros((heads,hd,ss),np.float32);gs=cp.asarray(state)
a=-np.exp(rng.randn(heads)).astype(np.float32);db=rng.randn(heads).astype(np.float32);skip=rng.randn(heads).astype(np.float32)
for step in range(97):
    v=rng.randn(channels).astype(np.float32)*.3;dt=rng.randn(heads).astype(np.float32)
    if step in (7,29,73):dt=np.array([-80,-20,20,80],np.float32)
    joined=np.concatenate((hist,v[:,None]),axis=1)
    cv=np.sum(joined.astype(np.float64)*cw,axis=1)+cb
    cv=(cv/(1+np.exp(-cv))).astype(np.float32)
    hist=joined[:,1:]
    gc=cp.empty(channels,cp.float32)
    ops.k['causal_conv'](((channels+255)//256,),(256,),(cp.asarray(v),cp.asarray(cw),cp.asarray(cb),gh,gc,np.int32(channels),np.int32(convk)))
    check('conv-%d'%step,gc,cv)
    check('conv-history-%d'%step,gh,hist,atol=0,rtol=0)
    delta=np.logaddexp(0,dt.astype(np.float64)+db)
    xx=cv[:inner].reshape(heads,hd)
    bb=np.repeat(cv[inner:inner+groups*ss].reshape(groups,ss),heads//groups,axis=0)
    cc=np.repeat(cv[inner+groups*ss:].reshape(groups,ss),heads//groups,axis=0)
    state=(np.exp(delta*a)[:,None,None]*state+delta[:,None,None]*xx[:,:,None]*bb[:,None,:]).astype(np.float32)
    expected=np.sum(state.astype(np.float64)*cc[:,None,:],axis=-1)+skip[:,None]*xx
    gy=cp.empty(inner,cp.float32)
    ops.k['ssm_step']((inner,),(128,),(gc,cp.asarray(dt),cp.asarray(db),cp.asarray(a),cp.asarray(skip),gs,gy,np.int32(heads),np.int32(hd),np.int32(ss),np.int32(groups)))
    check('ssm-output-%d'%step,gy,expected.reshape(-1))
    check('ssm-state-%d'%step,gs,state)

length,qheads,kvheads,dim=13,4,2,128
q=rng.randn(qheads,dim).astype(np.float32);k=rng.randn(length,kvheads,dim).astype(np.float32);v=rng.randn(length,kvheads,dim).astype(np.float32)
out=cp.empty((qheads,dim),cp.float32)
ops.k['attention']((qheads,),(128,),(cp.asarray(q),cp.asarray(k),cp.asarray(v),out,np.int32(length),np.int32(qheads),np.int32(kvheads),np.int32(dim)))
expected=[]
for h in range(qheads):
    score=k[:,h//2].astype(np.float64).dot(q[h])/np.sqrt(dim)
    prob=np.exp(score-score.max());prob/=prob.sum()
    expected.append(prob.dot(v[:,h//2]))
check('causal-GQA',out,np.stack(expected))
cp.cuda.Stream.null.synchronize()
result={'passed':True,'checks':checks,'device':str(cp.cuda.runtime.getDeviceProperties(0)['name']),
        'cupy':cp.__version__,'runtime':cp.cuda.runtime.runtimeGetVersion()}
if args.output:
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
print(json.dumps({'passed':True,'checks':len(checks),'device':result['device']}))
