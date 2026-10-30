import argparse
import json
import numpy as np
import cupy as cp
from kindi.packed import PackedFile
from kindi.moe import CudaOps
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
f=PackedFile(a.model);ops=CudaOps();rng=np.random.RandomState(31);results=[]
for name in ['blk.0.ssm_in.weight','blk.0.ssm_out.weight','blk.1.ffn_up_exps.weight','blk.1.ffn_down_exps.weight','output.weight']:
    t=f.tensors[name];w=cp.asarray(f.raw(name));rows,cols=t.shape[-2:];selected=6 if len(t.shape)==3 else 1
    per_input='down_exps' in name;ids=cp.arange(selected,dtype=cp.int32) if selected>1 else None
    x=cp.asarray(rng.randn(selected if per_input else 1,cols).astype(np.float32));measure={};reference=None
    for profile in ('baseline','optimized'):
        ops.profile=profile
        for _ in range(3):y=ops.linear(w,x,rows,cols,t.kind,ids,selected,per_input)
        cp.cuda.Stream.null.synchronize()
        actual=y.get()
        if reference is None:reference=actual
        else:np.testing.assert_allclose(actual,reference,atol=2e-3,rtol=1e-3)
        samples=[]
        for _ in range(5):
            start,end=cp.cuda.Event(),cp.cuda.Event();start.record()
            for __ in range(30):y=ops.linear(w,x,rows,cols,t.kind,ids,selected,per_input)
            end.record();end.synchronize();samples.append(cp.cuda.get_elapsed_time(start,end)/30)
        measure[profile]={'median_ms':float(np.median(samples)),'samples_ms':samples}
    result={'tensor':name,'selected':selected,'profiles':measure,'speedup':measure['baseline']['median_ms']/measure['optimized']['median_ms']}
    results.append(result);print(json.dumps(result),flush=True)
    del w,x,y;cp.get_default_memory_pool().free_all_blocks()
with open(a.output,'w') as o:json.dump({'passed':True,'results':results,'device':str(cp.cuda.runtime.getDeviceProperties(0)['name'])},o,indent=2)
