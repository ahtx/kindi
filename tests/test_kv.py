"""Independent storage/attention oracle, boundaries, extremes and write canaries."""
import argparse,json
import numpy as np
from kindi.moe import CudaOps
from kindi.kv import KVCacheOps
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--long',action='store_true');a=p.parse_args()
ops=CudaOps();cp=ops.cp;rng=np.random.RandomState(901);records=[]

def oracle(x,mode):
    if mode=='fp32':return x.copy(),x.copy()
    if mode=='fp16':return x.astype(np.float16),x.astype(np.float16).astype(np.float32)
    blocks=x.reshape(-1,32);scale=np.max(np.abs(blocks),axis=1).astype(np.float32)/np.float32(127)
    codes=np.rint(np.divide(blocks,scale[:,None],out=np.zeros_like(blocks),where=scale[:,None]!=0)).clip(-127,127).astype(np.int8)
    stored=scale.astype(np.float16);raw=np.empty((len(blocks),34),np.uint8)
    raw[:,:2]=stored.view(np.uint8).reshape(-1,2);raw[:,2:]=codes.view(np.uint8)
    return raw.reshape(x.shape[0],x.shape[1],x.shape[2]//32*34),(codes.astype(np.float32)*stored.astype(np.float32)[:,None]).reshape(x.shape)

for mode in ('fp32','fp16','q8_0'):
    kv=KVCacheOps(ops,mode)
    for n,dim in ((1,32),(7,64),(129,128)):
        x=rng.randn(n,2,dim).astype(np.float32);x[0,0,:32]=np.arange(32,dtype=np.float32)-15.5
        if n>1:x[1]=0
        stored,decoded=oracle(x,mode);dst=kv.allocate(n+2,2,dim);dst.fill(23)
        kv.store(dst,cp.asarray(x),1,2,dim);kv.check();actual=dst.get()
        np.testing.assert_array_equal(actual[1:-1],stored)
        assert np.all(actual[0]==23) and np.all(actual[-1]==23),'KV write crossed its range'
        records.append({'mode':mode,'test':'storage','n':n,'dim':dim,'bit_exact':True,'max_quant_error':float(np.max(np.abs(x-decoded)))})
    cases=[(0,1),(115,13),(127,3),(1023,3),(32767,1)]
    if a.long:cases += [(32767,2),(131071,1)]
    for past,m in cases:
        kh,qh,dim=2,4,128;n=past+m
        k=rng.randn(n,kh,dim).astype(np.float32)*.3;v=rng.randn(n,kh,dim).astype(np.float32)
        q=rng.randn(m,qh,dim).astype(np.float32)
        _,dk=oracle(k,mode);_,dv=oracle(v,mode)
        gk=kv.allocate(n,kh,dim);gv=kv.allocate(n,kh,dim)
        kv.store(gk,cp.asarray(k),0,kh,dim);kv.store(gv,cp.asarray(v),0,kh,dim);kv.check()
        out=kv.attention(cp.asarray(q),gk,gv,past,m,qh,kh,dim).get().reshape(m,qh,dim)
        expected=np.empty_like(out)
        for token in range(m):
            for h in range(qh):
                length=past+token+1;score=dk[:length,h//2].astype(np.float64).dot(q[token,h])/np.sqrt(dim)
                pr=np.exp(score-score.max());pr/=pr.sum();expected[token,h]=pr.dot(dv[:length,h//2])
        np.testing.assert_allclose(out,expected,atol=2e-5,rtol=3e-4)
        # Compare against unchanged FP32 attention using the independently decoded stored values.
        reference=cp.empty((m,qh*dim),cp.float32)
        ops.k['attention_tiled']((qh,m),(128,),(cp.asarray(q),cp.asarray(dk),cp.asarray(dv),reference,
            np.int32(m),np.int32(past),np.int32(qh),np.int32(kh),np.int32(dim)))
        np.testing.assert_array_equal(out,reference.get().reshape(m,qh,dim),err_msg='%s past=%d m=%d'%(mode,past,m))
        records.append({'mode':mode,'test':'attention','past':past,'m':m,'max_abs':float(np.max(np.abs(out-expected))),'decoded_fp32_bit_exact':True})
        del gk,gv,reference;cp.get_default_memory_pool().free_all_blocks()
    if mode!='fp32':
        for bad in (np.nan,np.inf,1e20):
            kv.reset_error();x=np.ones((1,2,128),np.float32);x[0,0,0]=bad;dst=kv.allocate(1,2,128)
            kv.store(dst,cp.asarray(x),0,2,128)
            try:kv.check();raise AssertionError('invalid storage value accepted')
            except FloatingPointError:pass
            records.append({'mode':mode,'test':'reject_nonfinite_or_overflow','value':str(bad)})
        kv.reset_error();kv.check()
with open(a.output,'w') as f:json.dump({'passed':True,'records':records,'device':str(cp.cuda.runtime.getDeviceProperties(0)['name'])},f,indent=2)
print(json.dumps({'passed':True,'checks':len(records)}),flush=True)
