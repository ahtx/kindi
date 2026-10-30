import argparse
import json
import types
import numpy as np
import cupy as cp
from kindi.moe import CudaOps
from kindi.lightning_batch import BatchOps
from kindi.packed import dequantize
p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
rng=np.random.RandomState(122);ops=CudaOps();checks=[]
eng=types.SimpleNamespace(ops=ops,cp=cp,spec=types.SimpleNamespace(eps=1e-5),w={},file=types.SimpleNamespace(tensors={}))
b=BatchOps(eng)
def check(name,value,expected,atol=1e-4):
    actual=cp.asnumpy(value);np.testing.assert_allclose(actual,expected,atol=atol,rtol=3e-4,err_msg=name)
    checks.append({'name':name,'max_abs':float(np.max(np.abs(actual-expected)))})
def weight(name,rows,cols,experts=1,kind=42):
    bs,size={42:(64,18),8:(32,34)}[kind]
    raw=rng.randint(0,256,(experts*rows*cols//bs,size)).astype(np.uint8)
    raw[:,:2]=(rng.uniform(.001,.05,len(raw)).astype(np.float16)).view(np.uint8).reshape(-1,2)
    shape=(rows,cols) if experts==1 else (experts,rows,cols)
    eng.w[name]=cp.asarray(raw);eng.file.tensors[name]=types.SimpleNamespace(shape=shape,kind=kind)
    return dequantize(raw.reshape(-1),kind).reshape(experts,rows,cols)
def check_tile16(label,name,x,dispatch=None,per_assignment=False):
    tensor=eng.file.tensors[name];rows,cols=tensor.shape[-2:]
    m=x.shape[0] if dispatch is None else dispatch['assignments']
    if tensor.kind==42:
        reference=cp.empty((m,rows),cp.float32)
        reference_x=cp.repeat(x,dispatch['topk'],axis=0) if dispatch is not None and not per_assignment else x
        ops.k['q2_mv'](((rows+3)//4,m),(128,),
            (eng.w[name],reference_x,reference,ops.null if dispatch is None else dispatch['ids'],
             np.int32(rows),np.int32(cols),np.int32(m),np.int32(True)))
    else:reference=ops.linear(eng.w[name],x,rows,cols,tensor.kind,selected=m,per_input=True).reshape(m,rows)
    attribute='dense_mode' if dispatch is None else 'expert_mode';previous=getattr(b,attribute)
    try:
        setattr(b,attribute,'tile16');actual=b.linear(name,x,dispatch,per_assignment).get()
    finally:setattr(b,attribute,previous)
    reference=reference.get()
    assert actual.shape==reference.shape and actual.dtype==reference.dtype,label+' shape/dtype'
    assert np.isfinite(actual).all() and np.isfinite(reference).all(),label+' non-finite output'
    np.testing.assert_array_equal(actual.view(np.uint8),reference.view(np.uint8),err_msg=label+' bytes (including signed zero)')
    checks.append({'name':label,'max_abs':0.,'byte_identical':True})

for kind in (42,8):
    dense=weight('dense',23,1856,kind=kind)
    for m in (1,7,8,17,256):
        x=rng.randn(m,1856).astype(np.float32)
        check('dense-%d-%d'%(kind,m),b.linear('dense',cp.asarray(x)),x.astype(np.float64).dot(dense[0].T),atol=3e-4)
        check_tile16('tile16-dense-%d-%d'%(kind,m),'dense',cp.asarray(x))
for m,experts,topk,skew in [(7,8,2,False),(17,16,6,False),(17,16,6,True),(129,128,6,False)]:
    logits=rng.randn(m,experts).astype(np.float32);bias=rng.randn(experts).astype(np.float32)*.1
    ids,weights=b.route(cp.asarray(logits),cp.asarray(bias),topk,2.5)
    scores=1/(1+np.exp(-logits.astype(np.float64)));expected=np.argsort(-(scores+bias),axis=1,kind='stable')[:,:topk]
    assert np.array_equal(ids.get(),expected)
    selected=np.take_along_axis(scores,expected,axis=1)
    check('routing-%d'%m,weights,selected/selected.sum(1,keepdims=True)*2.5)
    if skew:ids[:]=cp.arange(topk,dtype=cp.int32)
    dispatch=b.dispatch(ids,experts);order=dispatch['order'].get();cpu_ids=ids.get().reshape(-1)
    assert np.array_equal(np.sort(order),np.arange(m*topk))
    assert np.all(np.diff(cpu_ids[order])>=0)
    dense=weight('expert',19,64,experts);x=rng.randn(m,64).astype(np.float32)
    expected=np.stack([dense[e].astype(np.float64).dot(x[i//topk]) for i,e in enumerate(cpu_ids)])
    check('grouped-up-%d-%s'%(m,skew),b.linear('expert',cp.asarray(x),dispatch),expected)
    check_tile16('tile16-grouped-up-%d-%s'%(m,skew),'expert',cp.asarray(x),dispatch)
    xx=rng.randn(m*topk,64).astype(np.float32)
    expected=np.stack([dense[e].astype(np.float64).dot(xx[i]) for i,e in enumerate(cpu_ids)])
    check('grouped-down-%d-%s'%(m,skew),b.linear('expert',cp.asarray(xx),dispatch,True),expected)
    check_tile16('tile16-grouped-down-%d-%s'%(m,skew),'expert',cp.asarray(xx),dispatch,True)

# Force every job length 1..8, row tails and the 64-column tail in cols=1856.
weight('tile16-tails',19,1856,8)
ids=cp.asarray(np.repeat(np.arange(8,dtype=np.int32),np.arange(1,9)).reshape(-1,1))
dispatch=b.dispatch(ids,8);x=rng.randn(36,1856).astype(np.float32)
x[0].fill(0.);x[1].fill(-0.)
for per_assignment in (False,True):
    check_tile16('tile16-job-tails-%s'%per_assignment,'tile16-tails',cp.asarray(x),dispatch,per_assignment)
eng.w['tile16-f32']=cp.asarray(rng.randn(23,1856).astype(np.float32))
eng.file.tensors['tile16-f32']=types.SimpleNamespace(shape=(23,1856),kind=0)
check_tile16('tile16-f32-fallback','tile16-f32',cp.asarray(x[:9]))

# Float32 selection ties use lower expert ID; original probabilities determine weights.
for value in (-1000.,0.,1000.):
    ids,weights=b.route(cp.full((3,8),value,dtype=cp.float32),cp.zeros(8,cp.float32),2,2.5)
    np.testing.assert_array_equal(ids.get(),np.tile([0,1],(3,1)))
    check('router-extreme-%g'%value,weights,np.full((3,2),0 if value<0 else 1.25,dtype=np.float32),0)

m,inner,heads,hd,ss,groups,kernel=9,32,4,8,128,2,4
expert=cp.asarray(rng.randn(9,6,64).astype(np.float32));mix=cp.asarray(rng.randn(9,6).astype(np.float32))
shared=cp.asarray(rng.randn(9,64).astype(np.float32));xx=cp.asarray(rng.randn(9,64).astype(np.float32));expected=xx.copy()
for token in range(9):ops.add(expected[token],ops.combine(expert[token],mix[token],shared[token]))
ops.k['combine_batch'](((xx.size+255)//256,),(256,),(expert,mix,shared,xx,np.int32(9),np.int32(64),np.int32(6)))
assert np.array_equal(xx.get(),expected.get()),'shared-expert accumulation order changed'
check('combine-order-bit-exact',xx,expected.get(),0)
channels=inner+groups*ss*2;stride=inner+channels+heads
proj=rng.randn(m,stride).astype(np.float32)*.1;cw=rng.randn(channels,kernel).astype(np.float32)*.1;cb=rng.randn(channels).astype(np.float32)*.01
hist=rng.randn(channels,kernel-1).astype(np.float32)*.1;gh=cp.asarray(hist)
conv=[]
for token in proj:
    joined=np.concatenate([hist,token[inner:inner+channels,None]],axis=1)
    cv=np.sum(joined.astype(np.float64)*cw,axis=1)+cb;cv=cv/(1+np.exp(-cv));conv.append(cv)
    hist=joined[:,1:]
conv=np.asarray(conv,dtype=np.float32);gc=cp.empty((m,channels),cp.float32);gp=cp.asarray(proj)
ops.k['conv_chunk'](((channels+255)//256,),(256,),(gp,cp.asarray(cw),cp.asarray(cb),gh,gc,np.int32(m),np.int32(channels),np.int32(kernel),np.int32(stride),np.int32(inner)))
check('chunk-conv',gc,conv);check('chunk-history',gh,hist,0)
state=rng.randn(heads,hd,ss).astype(np.float32)*.1;gs=cp.asarray(state)
db=rng.randn(heads).astype(np.float32);aa=-np.exp(rng.randn(heads)).astype(np.float32);dd=rng.randn(heads).astype(np.float32);ys=[]
for i,cv in enumerate(conv):
    dt=np.logaddexp(0,proj[i,-heads:].astype(np.float64)+db)
    xx=cv[:inner].reshape(heads,hd);bb=np.repeat(cv[inner:inner+groups*ss].reshape(groups,ss),heads//groups,axis=0);cc=np.repeat(cv[inner+groups*ss:].reshape(groups,ss),heads//groups,axis=0)
    state=(np.exp(dt*aa)[:,None,None]*state+dt[:,None,None]*xx[:,:,None]*bb[:,None,:]).astype(np.float32)
    ys.append(np.sum(state.astype(np.float64)*cc[:,None,:],axis=-1)+dd[:,None]*xx)
gy=cp.empty((m,inner),cp.float32)
ops.k['ssm_chunk']((inner,),(128,),(gc,gp,cp.asarray(db),cp.asarray(aa),cp.asarray(dd),gs,gy,np.int32(m),np.int32(heads),np.int32(hd),np.int32(ss),np.int32(groups),np.int32(stride)))
check('chunk-ssm',gy,np.array(ys).reshape(m,inner));check('chunk-final-state',gs,state)
ww=rng.randn(inner).astype(np.float32)
gated=gy.get().astype(np.float64)*proj[:,:inner]/(1+np.exp(-proj[:,:inner].astype(np.float64)))
expected=(gated.reshape(m,groups,-1)/np.sqrt(np.mean(gated.reshape(m,groups,-1)**2,-1,keepdims=True)+1e-5)).reshape(m,inner)*ww
check('chunk-gated-norm',b.norm(gy,cp.asarray(ww),groups,gp),expected)
for past,m in [(0,1),(1,7),(127,9),(256,17),(2048,3)]:
    qh,kh,dim=4,2,128
    q=rng.randn(m,qh,dim).astype(np.float32);k=rng.randn(past+m,kh,dim).astype(np.float32);v=rng.randn(past+m,kh,dim).astype(np.float32)
    out=cp.empty_like(cp.asarray(q))
    ops.k['attention_tiled']((qh,m),(128,),(cp.asarray(q),cp.asarray(k),cp.asarray(v),out,np.int32(m),np.int32(past),np.int32(qh),np.int32(kh),np.int32(dim)))
    expected=np.empty_like(q)
    for token in range(m):
        length=past+token+1
        for h in range(qh):
            score=k[:length,h//2].astype(np.float64).dot(q[token,h])/np.sqrt(dim);pr=np.exp(score-score.max());pr/=pr.sum();expected[token,h]=pr.dot(v[:length,h//2])
    check('tiled-causal-attention-%d-%d'%(past,m),out,expected)
cp.cuda.Stream.null.synchronize()
result={'passed':True,'checks':checks,'device':str(cp.cuda.runtime.getDeviceProperties(0)['name'])}
if a.output:
    with open(a.output,'w') as f:json.dump(result,f,indent=2)
print(json.dumps({'passed':True,'checks':len(checks),'device':result['device']}))
