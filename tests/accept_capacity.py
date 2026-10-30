"""Capacity/boundary check, deliberately NOT a long-context quality benchmark."""
import argparse
import gc
import json
import numpy as np
import cupy as cp
from kindi.lightning import Engine
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args();records=[]
for ctx in (32768,131072):
    eng=Engine(a.model,ctx=ctx)
    for x in eng.keys.values():x.fill(0)
    for x in eng.values.values():x.fill(0)
    eng.position=ctx-1;eng.tokens=[1]*(ctx-1)
    logits=eng.forward(1);assert bool(cp.isfinite(logits).all());assert eng.position==ctx
    try:eng.forward(1);raise AssertionError('context overflow accepted')
    except OverflowError:pass
    eng.reset();eng.forward(1);assert bool(cp.isfinite(eng.last_logits).all())
    free,total=cp.cuda.runtime.memGetInfo()
    records.append({'ctx':ctx,'memory_plan':eng.memory,'gpu_used_bytes':total-free,'last_position_finite':True,
        'scope':'allocation and synthetic zero-history boundary; not semantic long-context acceptance'})
    del logits,eng,x;gc.collect();cp.get_default_memory_pool().free_all_blocks()
try:Engine(a.model,ctx=262144);raise AssertionError('oversized FP32 profile accepted on K6000')
except MemoryError as exc:records.append({'ctx':262144,'rejected_before_weight_upload':True,'reason':str(exc)})
result={'passed':True,'records':records}
with open(a.output,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
