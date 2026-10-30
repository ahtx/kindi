"""100 mixed requests, real in-flight cancellation, bounded cache and process memory."""
import argparse,configparser,json,os,subprocess,time,urllib.request
p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8095');p.add_argument('--config');p.add_argument('--output',required=True);a=p.parse_args()
key='lightning-acceptance-only'
if a.config:
    cfg=configparser.ConfigParser();cfg.read(a.config);key=cfg['kindi']['api_key']
def request(data):
    req=urllib.request.Request(a.base+'/v1/chat/completions',json.dumps(data).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+key})
    return urllib.request.urlopen(req,timeout=120)
def payload(text,**kwargs):
    data={'messages':[{'role':'user','content':text}],'temperature':0,'max_tokens':48};data.update(kwargs);return data
def call(data):
    with request(data) as r:return json.load(r)
def health():return json.load(urllib.request.urlopen(a.base+'/health',timeout=5))
def memory():
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=used_gpu_memory','--format=csv,noheader,nounits'],universal_newlines=True).strip()
    return {'gpu_process_mib':gpu,'cache_bytes':health()['prefix_cache_bytes']}
checks=[];records=[]
for phase in ('prefill','decode'):
    text=('Read this document carefully. A precise example explains arithmetic and memory. '*350) if phase=='prefill' else 'Write a very long detailed story about mathematics.'
    response=request(payload(text,stream=True,max_tokens=1000))
    for line in response:
        if line.startswith(b'data: '):break
    deadline=time.monotonic()+20
    while health()['phase']!=phase:
        if time.monotonic()>deadline:raise AssertionError('did not observe '+phase)
        time.sleep(.05)
    started=time.monotonic();response.close()
    answer=call(payload('Name the capital of France in one word.'))
    elapsed=time.monotonic()-started
    assert answer['choices'][0]['message']['content'].strip()=='Paris'
    assert elapsed<15,('cancellation too slow',phase,elapsed)
    checks.append({'cancelled_phase':phase,'recovery_seconds':elapsed})
templates=[
    lambda n:'Ответь только числом: %d + 25 ='%n,
    lambda n:'Return only valid JSON with one key named number and integer value %d.'%n,
    lambda n:'Write a Python function named add_%d that returns the sum of two arguments. Keep it short.'%n,
    lambda n:'Translate to Russian, briefly: The library contains %d books.'%n,
    lambda n:'Extract only the identifier from this sentence: Order AB-%d was delivered yesterday.'%n,
    lambda n:'Classify the sentiment in one word: I am happy with order number %d.'%n,
    lambda n:'Sort these numbers ascending, return a comma-separated list: %d, 2, 9.'%n,
    lambda n:'Summarize in one sentence: A server processed %d requests. It cached repeated prompts. All requests completed.'%n,
    lambda n:'Give one short sentence in Hebrew about a robot named R%d.'%n,
    lambda n:('Background: careful calculations should be checked, and a cache can reuse earlier results. '*12)+
        'The document identifier is DOC-%d. Reply with only that identifier.'%n]
for i in range(100):
    text=templates[i%10](i+31);started=time.monotonic();result=call(payload(text));choice=result['choices'][0]
    assert choice['finish_reason'] in ('stop','length','tool_calls')
    assert isinstance(choice['message']['content'],str) and '\ufffd' not in choice['message']['content']
    assert result['usage']['completion_tokens']<=48 and result['usage']['total_tokens']<=16384
    records.append({'case':i,'category':i%10,'seconds':time.monotonic()-started,'response':choice,'usage':result['usage']})
    if i%10==9:
        checks.append({'after_request':i+1,'memory':memory()});print(json.dumps(checks[-1]),flush=True)
        with open(a.output,'w') as f:json.dump({'complete':False,'records':records,'checks':checks},f,indent=2)
state=health();assert state['prefix_cache_bytes']<=512*1024**2 and not state['busy']
with open(a.output,'w') as f:json.dump({'complete':True,'passed':True,'request_count':100,'records':records,'checks':checks,'final_health':state},f,indent=2)
print(json.dumps({'passed':True,'requests':100,'cancellation':checks[:2]}),flush=True)
