import argparse,json
from types import SimpleNamespace
import numpy as np
from kindi.packed import PackedFile
from kindi.moe import CudaOps
from kindi.lightning_batch import BatchOps
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--tokens',type=int,default=128)
p.add_argument('--modes',default='tile8')
a=p.parse_args();f=PackedFile(a.model);ops=CudaOps();cp=ops.cp;rng=np.random.RandomState(127);records=[]
eng=SimpleNamespace(file=f,ops=ops,cp=cp,w={});batch=BatchOps(eng)
for name in ('blk.0.ssm_in.weight','blk.0.ssm_out.weight','blk.1.ffn_up_exps.weight','blk.1.ffn_down_exps.weight'):
    tensor=f.tensors[name];rows,cols=tensor.shape[-2:];experts=len(tensor.shape)==3;per_assignment='down_exps' in name
    eng.w[name]=cp.asarray(f.raw(name));m=a.tokens
    x=cp.asarray(rng.randn(m*(6 if per_assignment else 1),cols).astype(np.float32))
    dispatch=batch.dispatch(cp.asarray(np.stack([rng.permutation(128)[:6] for _ in range(m)]).astype(np.int32)),128) if experts else None
    reference=None;measure={}
    for mode in (['grouped_mv' if experts else 'batched_mv']+a.modes.split(',')):
        if experts:batch.expert_mode=mode
        else:batch.dense_mode=mode
        for _ in range(2):y=batch.linear(name,x,dispatch,per_assignment)
        actual=y.get()
        assert np.isfinite(actual).all(),'non-finite packed linear output'
        if reference is None:reference=actual
        else:
            assert actual.dtype==reference.dtype and actual.shape==reference.shape
            np.testing.assert_array_equal(actual.view(np.uint8),reference.view(np.uint8))
        samples=[]
        for _ in range(5):
            start,end=cp.cuda.Event(),cp.cuda.Event();start.record()
            for __ in range(5):y=batch.linear(name,x,dispatch,per_assignment)
            end.record();end.synchronize();samples.append(cp.cuda.get_elapsed_time(start,end)/5)
        measure[mode]={'median_ms':float(np.median(samples)),'samples':samples}
    records.append({'name':name,'tokens':m,'measure':measure,'bit_exact':True})
    print(json.dumps(records[-1]),flush=True)
    eng.w.clear();del x,y,dispatch;cp.get_default_memory_pool().free_all_blocks()
with open(a.output,'w') as out:json.dump({'passed':True,'records':records,'kernel':getattr(ops.k['q2_tile8'],'attributes',{}),
    'kernels':{name:getattr(ops.k[name],'attributes',{}) for name in ('q2_tile8','q2_tile16') if name in ops.k},
    'comparison':'finite same-shape/dtype uint8 byte equality'},out,indent=2)
