"""Full checkpoint K6000 acceptance against frozen RTX/Torch-verified fixtures."""
import argparse
import json
import time
import numpy as np
from kindi.lightning import Engine
from kindi.lightning_tokenizer import Tok
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--fixture',required=True);p.add_argument('--output',required=True)
p.add_argument('--kv',choices=('fp32','fp16','q8_0'),default='fp32')
a=p.parse_args();start=time.time();engine=Engine(a.model,ctx=16384,kv_dtype=a.kv);load=time.time()-start
fixtures=np.load(a.fixture+'.npz');reference=json.load(open(a.fixture));records=[]
for step,token in enumerate(reference['tokens']):
    def trace(i,x,eng):
        actual=x.get();expected=fixtures['step%d_block%d'%(step,i)]
        np.testing.assert_allclose(actual,expected,atol=3e-3,rtol=2e-3,err_msg='step %d block %d'%(step,i))
        records.append({'step':step,'block':i,'max_abs':float(np.max(np.abs(actual-expected)))})
    logits=engine.forward(token,trace=trace).get();expected=fixtures['step%d_logits'%step]
    np.testing.assert_allclose(logits,expected,atol=3e-3,rtol=2e-3)
    assert logits.argmax()==expected.argmax()
    print(json.dumps({'step':step,'max_abs':float(np.max(np.abs(logits-expected))),'argmax':int(logits.argmax())}),flush=True)
snapshot=engine.snapshot();token=int(engine.last_logits.argmax())
first=engine.forward(token).get();engine.restore(snapshot);second=engine.forward(token).get()
assert np.array_equal(first,second)
tok=Tok(a.model);answers=[]
for prompt in ('Reply with only the number: 17 + 25 =','Name the capital of France in one word.'):
    engine.reset();tokens=tok.encode(tok.chat([{'role':'user','content':prompt}]))
    start=time.time();logits=engine.prefill(tokens);engine.cp.cuda.Stream.null.synchronize();prefill=time.time()-start
    ids=[];start=time.time()
    for i in range(48):
        token=int(logits.argmax())
        if token in tok.stop_ids:break
        ids.append(token);logits=engine.forward(token)
        assert bool(engine.cp.isfinite(logits).all())
    engine.cp.cuda.Stream.null.synchronize();decode=time.time()-start
    answer={'prompt':prompt,'prompt_tokens':len(tokens),'text':tok.decode(ids),'ids':ids,'prefill_s':prefill,'decode_s':decode,'decode_tps':len(ids)/decode}
    print(json.dumps(answer),flush=True);answers.append(answer)
free,total=engine.cp.cuda.runtime.memGetInfo()
result={'passed':True,'kv':a.kv,'device':str(engine.cp.cuda.runtime.getDeviceProperties(0)['name']),'source_sha256':engine.source_sha256,
    'load_s':load,'memory':engine.memory,'free_bytes':free,'total_bytes':total,'comparisons':records,'snapshot_bit_exact':True,'answers':answers}
with open(a.output,'w') as f:json.dump(result,f,indent=2)
print(json.dumps({'passed':True,'answers':len(answers),'blocks':len(records)}),flush=True)
