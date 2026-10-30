"""Finite-corpus KV regression budgets against native FP32, with fixed token prefixes.

Budgets are fixed before measurement. This is a regression sample, not a general
quality or long-context certification. Greedy answers are recorded separately.
"""
import argparse,gc,json,time
import numpy as np
from kindi.lightning import Engine
from kindi.lightning_tokenizer import Tok
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
budgets={'fp16':{'mean_kl':.001,'max_kl':.01,'rms_p95':.02,'argmax_match_min':.98},
         'q8_0':{'mean_kl':.03,'max_kl':.2,'rms_p95':.2,'argmax_match_min':.90}}
cases=[
 ('Compute 191 + 328. Give only the number.','519','519'),
 ('Сколько будет 17 умножить на 25? Ответь только числом.','425','425'),
 ('What is the capital of France? Answer with the city name only.','Paris','Paris'),
 ('Назови химический символ кислорода. Только символ.','O','O'),
 ('What does Python print: print(sum([2, 3, 5]))? Give only the output.','10','10'),
 ('Translate "доброе утро" to English. Give only the translation.','Good morning','good morning'),
 ('Explain why a cache can make repeated reads faster.','A cache keeps recently accessed data close to the processor, reducing the time needed to retrieve it.',''),
 ('Объясни разницу между оперативной памятью и диском в двух предложениях.','Оперативная память хранит данные работающих программ и быстро теряет их после отключения питания. Диск сохраняет файлы надолго.',''),
 ('Complete a Python function that adds two arguments: def add(a, b):','\n    return a + b',''),
 ('Return JSON with name Ada and age 36.','{"name": "Ada", "age": 36}',''),
 ('Summarize: Water evaporates, condenses into clouds, and returns to the ground as precipitation.','The water cycle moves water through evaporation, condensation, and precipitation.',''),
 ('Запись: код склада ALPHA равен ZEK-4821. Верни только этот код.','ZEK-4821','ZEK-4821')]
tok=Tok(a.model);reference=[];records=[];start=time.time()
for mode in ('fp32','fp16','q8_0'):
    eng=Engine(a.model,ctx=4096,kv_dtype=mode);cp=eng.cp;comparisons=[];answers=[]
    for index,(question,target,expected) in enumerate(cases):
        tokens=tok.encode(tok.chat([{'role':'user','content':question}]))
        continuation=tok.encode(target)[:24];eng.reset();logits=eng.prefill(tokens,chunk_size=128)
        for step,token in enumerate(continuation):
            values=logits.get()
            if mode=='fp32':reference.append(values.copy())
            else:
                base=reference[len(comparisons)].astype(np.float64);other=values.astype(np.float64)
                p0=base-base.max();p0-=np.log(np.exp(p0).sum());p1=other-other.max();p1-=np.log(np.exp(p1).sum())
                comparisons.append({'case':index,'step':step,'kl':max(0.,float(np.sum(np.exp(p0)*(p0-p1)))),
                    'rms':float(np.sqrt(np.mean((base-other)**2))),'same_argmax':int(base.argmax())==int(other.argmax())})
            logits=eng.forward(token)
        eng.reset();logits=eng.prefill(tokens,chunk_size=128);ids=[]
        for _ in range(48):
            token=int(logits.argmax());
            if token in tok.stop_ids:break
            ids.append(token);logits=eng.forward(token)
        answer=tok.decode(ids);answers.append({'question':question,'answer':answer,'expected_fragment':expected,
            'expected_found':expected.lower() in answer.lower() if expected else None})
    stats=None;passed=True
    if comparisons:
        stats={'mean_kl':float(np.mean([x['kl'] for x in comparisons])),'max_kl':max(x['kl'] for x in comparisons),
            'rms_p95':float(np.percentile([x['rms'] for x in comparisons],95)),
            'argmax_match':float(np.mean([x['same_argmax'] for x in comparisons]))}
        b=budgets[mode];passed=stats['mean_kl']<=b['mean_kl'] and stats['max_kl']<=b['max_kl'] and stats['rms_p95']<=b['rms_p95'] and stats['argmax_match']>=b['argmax_match_min']
    record={'mode':mode,'passed':passed,'stats':stats,'comparisons':comparisons,'answers':answers,'numeric_profile':eng.numeric_profile}
    records.append(record);print(json.dumps({'mode':mode,'passed':passed,'stats':stats}),flush=True)
    # Release buffers explicitly between profiles; construction frames can
    # outlive an engine variable in this validation runtime.
    eng.synchronize()
    for name in ('w','states','histories','keys','values'):getattr(eng,name).clear()
    eng.last_logits=None;eng.kv.scratch=None;eng.kv.split_scratch=None
    del logits,eng;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    with open(a.output,'w',encoding='utf-8') as f:json.dump({'complete':False,'budgets':budgets,'records':records},f,ensure_ascii=False,indent=2)
passed=all(x['passed'] for x in records)
with open(a.output,'w',encoding='utf-8') as f:json.dump({'complete':True,'passed':passed,'budgets':budgets,'records':records,'seconds':time.time()-start},f,ensure_ascii=False,indent=2)
assert passed,'KV regression budget exceeded; retain FP32 default and inspect report'
