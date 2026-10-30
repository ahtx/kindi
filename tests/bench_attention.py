"""Exact old/new attention comparison and timings at real model head geometry."""
import argparse,json,time
import numpy as np
from kindi.moe import CudaOps
from kindi.kv import KVCacheOps
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
ops=CudaOps();cp=ops.cp;rng=np.random.RandomState(317);records=[]
for mode in ('fp32','fp16','q8_0'):
    for past,m in ((0,1),(115,13),(8064,128),(32640,128),(32767,1)):
        n=past+m;kh,h,dim=2,32,128
        kv=KVCacheOps(ops,mode,'direct');k=kv.allocate(n,kh,dim);v=kv.allocate(n,kh,dim)
        q=cp.asarray(rng.randn(m,h,dim).astype('f')*.3)
        kv.store(k,cp.asarray(rng.randn(n,kh,dim).astype('f')*.3),0,kh,dim)
        kv.store(v,cp.asarray(rng.randn(n,kh,dim).astype('f')),0,kh,dim);kv.check()
        reference=None;timings={}
        for read in (('direct',) if mode=='fp32' else ('direct','expanded')):
            kv.read_mode=read
            for attention in ('tiled','warp','split'):
                kv.attention_mode=attention
                out=kv.attention(q,k,v,past,m,h,kh,dim);cp.cuda.Stream.null.synchronize()
                if reference is None:reference=out.get()
                else:np.testing.assert_array_equal(out.get(),reference,err_msg='%s %d %d %s %s'%(mode,past,m,read,attention))
                values=[]
                for _ in range(3):
                    start=cp.cuda.Event();end=cp.cuda.Event();start.record()
                    out=kv.attention(q,k,v,past,m,h,kh,dim)
                    end.record();end.synchronize();values.append(cp.cuda.get_elapsed_time(start,end))
                timings[read+'_'+attention]=float(np.median(values))
        record={'kv':mode,'past':past,'tokens':m,'bit_exact':True,'workspace_bytes':kv.split_workspace_bytes,'milliseconds':timings}
        records.append(record);print(json.dumps(record),flush=True)
        del k,v,q,kv,out;cp.get_default_memory_pool().free_all_blocks()
        with open(a.output,'w') as f:json.dump({'complete':False,'records':records},f,indent=2)
with open(a.output,'w') as f:json.dump({'complete':True,'passed':True,'records':records},f,indent=2)
