import argparse
import json
import time
import numpy as np
from kindi.lightning import Engine
from kindi.lightning_tokenizer import Tok
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
p.add_argument('--lengths',default='1024,4096,8192');p.add_argument('--ctx',type=int,default=16384)
p.add_argument('--kv',choices=('fp32','fp16','q8_0'),default='fp32');p.add_argument('--reference-chunk',type=int,default=0)
p.add_argument('--chunk',type=int,default=128);p.add_argument('--exact',action='store_true');a=p.parse_args()
def assert_snapshot_bytes(actual,expected):
    assert set(actual)==set(expected),'snapshot fields differ'
    for name in ('version','fingerprint','position','tokens'):
        assert actual[name]==expected[name],'snapshot '+name+' differs'
    def array_bytes(value,reference,label):
        assert isinstance(value,np.ndarray) and isinstance(reference,np.ndarray),label+' is not an array'
        assert value.shape==reference.shape,label+' shape differs'
        assert value.dtype==reference.dtype,label+' dtype differs'
        if np.issubdtype(value.dtype,np.floating):
            assert np.isfinite(value).all() and np.isfinite(reference).all(),label+' contains non-finite values'
        np.testing.assert_array_equal(np.ascontiguousarray(value).view(np.uint8),
            np.ascontiguousarray(reference).view(np.uint8),err_msg=label+' bytes differ')
    for kind in ('states','histories','keys','values'):
        assert set(actual[kind])==set(expected[kind]),kind+' layer keys differ'
        for layer,value in expected[kind].items():array_bytes(actual[kind][layer],value,kind+' layer '+str(layer))
    array_bytes(actual['logits'],expected['logits'],'logits')

eng=Engine(a.model,ctx=a.ctx,kv_dtype=a.kv);tok=Tok(a.model);records=[]
last_progress=[time.time()]
def progress(done,total):
    now=time.time()
    if now-last_progress[0]>=30 or done==total:
        eng.synchronize();last_progress[0]=now
        value={'phase':phase,'processed':done,'total':total,'seconds':now-start,'kv':a.kv}
        with open(a.output+'.progress','w') as f:json.dump(value,f)
        print(json.dumps(value),flush=True)
text=('Explain arithmetic and memory with worked examples. Compare reversible operations and verify each result. '
      'Привет, мир! Numbers: 17 + 25 = 42.\n')*600
all_tokens=tok.encode(tok.chat([{'role':'user','content':text}]))
while len(all_tokens)<max(map(int,a.lengths.split(','))):
    text*=2;all_tokens=tok.encode(tok.chat([{'role':'user','content':text}]))
for n in map(int,a.lengths.split(',')):
    tokens=all_tokens[:n];assert len(tokens)==n
    eng.reset();phase='reference';start=time.time();eng.prefill(tokens,chunk_size=a.reference_chunk,progress=progress);ref=eng.snapshot();sequential=time.time()-start
    print(json.dumps({'length':n,'sequential_s':sequential}),flush=True)
    eng.reset();phase='candidate';start=time.time();eng.prefill(tokens,chunk_size=a.chunk,progress=progress);batched=eng.snapshot();batch_time=time.time()-start
    if a.exact:assert_snapshot_bytes(batched,ref)
    errors={}
    for kind in ('states','histories','keys','values'):
        errors[kind]=max([0.]+[float(np.max(np.abs(batched[kind][i].astype(np.float64)-array.astype(np.float64)))) for i,array in ref[kind].items()])
        for i,array in ref[kind].items():
            if not a.exact:np.testing.assert_allclose(batched[kind][i],array,atol=3e-3,rtol=3e-3,err_msg='%d %s %d'%(n,kind,i))
    np.testing.assert_allclose(batched['logits'],ref['logits'],atol=3e-3,rtol=2e-3)
    assert np.isfinite(batched['logits']).all()
    result={'length':n,'sequential_s':sequential,'batch_s':batch_time,'state_errors':errors,
        'logits_max_abs':float(np.max(np.abs(batched['logits']-ref['logits']))),'same_argmax':int(batched['logits'].argmax())==int(ref['logits'].argmax())}
    assert result['same_argmax'];records.append(result)
    with open(a.output,'w') as f:json.dump({'complete':False,'records':records,'kv':a.kv,'reference_chunk':a.reference_chunk,'chunk':a.chunk,
        'reference_scope':'full_independent_prefills','exact_required':a.exact,'exact_comparison':'snapshot_bytes' if a.exact else None},f,indent=2)
    print(json.dumps(result),flush=True)
with open(a.output,'w') as f:json.dump({'complete':True,'passed':True,'records':records,'kv':a.kv,'reference_chunk':a.reference_chunk,'chunk':a.chunk,
    'reference_scope':'full_independent_prefills','exact_required':a.exact,'exact_comparison':'snapshot_bytes' if a.exact else None,
    'numeric_profile':eng.numeric_profile},f,indent=2)
