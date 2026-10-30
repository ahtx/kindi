"""One full 32K prefill, exact sequential tail, and retrieval at three depths.

The final 512 tokens compare old tiled/sequential attention with the candidate
from one identical prefix. This is explicitly not two independent full prefills.
"""
import argparse,json,time,hashlib
import numpy as np
from kindi.lightning import Engine
from kindi.lightning_tokenizer import Tok
try:
    from .bench_prefill_chunks import compare_snapshot
except ImportError:
    from bench_prefill_chunks import compare_snapshot

def compare_tail_snapshot(reference,candidate,kv):
    result=compare_snapshot(reference,candidate)
    scales=[]
    if kv=='q8_0':
        for label,snapshot in (('reference',reference),('candidate',candidate)):
            for group in ('keys','values'):
                for layer,values in snapshot[group].items():
                    valid=(values.dtype==np.uint8 and values.ndim==3
                           and values.shape[-1]>0 and values.shape[-1]%34==0)
                    if valid:
                        blocks=np.ascontiguousarray(values).reshape(-1,34)
                        valid=bool(np.isfinite(blocks[:,:2].copy().view(np.dtype('<f2'))).all())
                    entry={'path':'%s/%s/%s/q8_scales'%(label,group,layer),'passed':valid}
                    if not valid:entry['error']='invalid Q8_0 block layout or non-finite fp16 scale'
                    scales.append(entry)
    result['q8_scale_checks']=scales
    result['passed']=result['passed'] and all(entry['passed'] for entry in scales)
    result['policy']='same snapshot fields and metadata; identical layer keys, shape, dtype and raw uint8 bytes; all floats and Q8_0 scales finite; signed zero preserved'
    result['scope']='tail from one shared prefix; not two independent full prefills'
    return result

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
p.add_argument('--kv',choices=('fp32','fp16','q8_0'),default='fp32');p.add_argument('--ctx',type=int,default=32768)
a=p.parse_args();assert a.ctx<=32768,'current acceptance scope is at most 32K'
tok=Tok(a.model);facts=[('ALPHA','ZEK-4821'),('BETA','QUM-7395'),('GAMMA','VOP-2168')]
filler='Routine inspection found the storage equipment in working order. The temperature and humidity stayed within normal limits. No access code appears in this entry.\n'
def document(n):
    pieces=[];previous=0
    for depth,(label,code) in zip((.1,.5,.9),facts):
        stop=int(n*depth);pieces.append(filler*(stop-previous));previous=stop
        pieces.append('\nIMPORTANT RECORD: ARCHIVE-%s has access code %s.\n\n'%(label,code))
    pieces.append(filler*(n-previous));return ''.join(pieces)
def prompt(doc,label):return tok.encode(tok.chat([{'role':'user','content':'Read this document:\n'+doc+'\nReturn only the access code for ARCHIVE-'+label+'.'}]))
lo,hi=1,a.ctx
while lo<hi:
    mid=(lo+hi+1)//2
    if len(prompt(document(mid),'ALPHA'))<=a.ctx-100:lo=mid
    else:hi=mid-1
doc=document(lo);prompts=[prompt(doc,label) for label,code in facts]
assert min(map(len,prompts))>=a.ctx-256 and max(map(len,prompts))+48<=a.ctx
common=0
while len(set(x[common] for x in prompts))==1:common+=1
base=prompts[0][:common];split=max(1,len(base)-512)
eng=Engine(a.model,ctx=a.ctx,kv_dtype=a.kv);initial_attention=eng.kv.attention_mode
last=[time.time()];start=time.time()
def progress(done,total):
    now=time.time()
    if now-last[0]>=30 or done==total:
        last[0]=now;print(json.dumps({'phase':'context_prefix','done':done,'total':total,'seconds':now-start,'kv':a.kv}),flush=True)
eng.prefill(base[:split],chunk_size=128,progress=progress);prefix_s=time.time()-start;checkpoint=eng.snapshot()
eng.kv.attention_mode='tiled';start=time.time();eng.prefill(base[split:],chunk_size=0);reference=eng.snapshot();reference_s=time.time()-start
eng.restore(checkpoint);eng.kv.attention_mode=initial_attention;start=time.time();eng.prefill(base[split:],chunk_size=128);candidate=eng.snapshot();tail_s=time.time()-start
tail_comparison=compare_tail_snapshot(reference,candidate,a.kv)
assert reference['position']==candidate['position']==len(base),'tail did not consume the full common prefix'
assert reference['tokens']==candidate['tokens']==base,'tail snapshot tokens differ from the common prefix'
assert tail_comparison['passed'],json.dumps(tail_comparison,sort_keys=True)
del checkpoint,reference
results=[]
for (label,expected),tokens in zip(facts,prompts):
    eng.restore(candidate);logits=eng.prefill(tokens[common:],chunk_size=128);ids=[];start=time.time()
    for _ in range(48):
        token=int(logits.argmax())
        if token in tok.stop_ids:break
        ids.append(token);logits=eng.forward(token)
    answer=tok.decode(ids);duration=time.time()-start
    result={'label':label,'expected':expected,'answer':answer,'correct':expected in answer,'prompt_tokens':len(tokens),
        'generated_tokens':len(ids),'decode_s':duration,'decode_tps':len(ids)/duration if duration else None}
    results.append(result);print(json.dumps(result,ensure_ascii=True),flush=True)
free,total=eng.cp.cuda.runtime.memGetInfo()
report={'complete':True,'passed':all(x['correct'] for x in results),'kv':a.kv,'ctx':a.ctx,'prefix_tokens':common,
    'candidate_prefill_s':prefix_s+tail_s,'reference_tail_s':reference_s,'tail_tokens':len(base)-split,'tail_bit_exact':True,
    'reference_scope':'same prefix; final 512 tokens compare old tiled sequential path to candidate batched path',
    'tail_byte_comparison':tail_comparison,
    'retrieval':results,'memory_plan':eng.memory,'gpu_used_bytes':total-free,'numeric_profile':eng.numeric_profile,
    'document_sha256':hashlib.sha256(doc.encode()).hexdigest(),'facts_at_approximate_depths':[.1,.5,.9]}
with open(a.output,'w',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
assert report['passed'],'retrieval failed; inspect model answers separately from numerical checks'
