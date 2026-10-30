"""Real HTTP acceptance inside the dedicated, network-isolated development container."""
import argparse
import configparser
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8095');p.add_argument('--config')
p.add_argument('--output',default='/workspace/evidence/api-acceptance.json');options=p.parse_args()
BASE=options.base;KEY='lightning-acceptance-only';records=[];cache_min=1
if options.config:
    cfg=configparser.ConfigParser();cfg.read(options.config);KEY=cfg['kindi']['api_key'];cache_min=int(cfg['kindi'].get('prefix_min','512'))
cache_background=('Reference note: answers should be concise and accurate. '*((cache_min+7)//8)) if cache_min>1 else ''
def request(payload,path='/v1/chat/completions',key=KEY):
    headers={'Content-Type':'application/json'}
    if key:headers['Authorization']='Bearer '+key
    req=urllib.request.Request(BASE+path,data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),headers=headers)
    return urllib.request.urlopen(req,timeout=120)
def call(payload):
    with request(payload) as response:return json.loads(response.read().decode('utf-8'))
def prompt(text,**kwargs):
    value={'messages':[{'role':'user','content':text}],'temperature':0,'max_tokens':96};value.update(kwargs);return value
def content(result):return result['choices'][0]['message'].get('content','')
def stream(payload):
    payload=dict(payload,stream=True,stream_options={'include_usage':True});text='';reason='';usage=None;calls=[];done=False
    with request(payload) as response:
        for raw in response:
            line=raw.decode('utf-8').strip()
            if not line.startswith('data: '):continue
            data=line[6:]
            if data=='[DONE]':done=True;break
            data=json.loads(data);assert 'error' not in data,data
            if data.get('usage'):usage=data['usage']
            delta=data['choices'][0]['delta'];text+=delta.get('content','');reason+=delta.get('reasoning_content','');calls+=delta.get('tool_calls',[])
    assert done and usage
    return text,reason,calls,usage
def record(name,detail=None):
    records.append({'name':name,'detail':detail});print(json.dumps(records[-1],ensure_ascii=True),flush=True)
    with open(options.output,'w') as f:json.dump({'complete':False,'records':records},f,indent=2)
server_health=json.loads(urllib.request.urlopen(BASE+'/health').read());assert server_health['backend']=='kindi_lightning'
try:request(prompt('hello'),key=None);raise AssertionError('missing authentication accepted')
except urllib.error.HTTPError as exc:assert exc.code==401
record('authentication and health')
arithmetic=call(prompt('Reply with only the number: 17 + 25 ='));assert content(arithmetic).strip()=='42',arithmetic
record('arithmetic',arithmetic)
paris=prompt(cache_background+'Name the capital of France in one word.')
answer=call(paris);assert content(answer).strip()=='Paris',answer
assert answer['usage']['prompt_tokens']>=cache_min
streamed=stream(paris);assert streamed[0]==content(answer);assert streamed[3]['prompt_tokens_details']['cached_tokens']>0
record('stream and prefix-cache parity',{'text':streamed[0],'usage':streamed[3]})
stopped=call(dict(paris,stop='ris'));assert content(stopped)=='Pa',stopped
assert stream(dict(paris,stop='ris'))[0]=='Pa'
record('stop inside one token')
ru=prompt('Ответь ровно одной строкой: Привет, мир!')
answer=call(ru);assert stream(ru)[0]==content(answer);assert '\ufffd' not in content(answer)
record('UTF-8 streaming parity',content(answer))
random=prompt(cache_background+'Invent four short names for imaginary cities.',temperature=.8,seed=42,max_tokens=64)
r1=call(random);r2=call(random);assert content(r1)==content(r2)
assert r2['usage']['prompt_tokens_details']['cached_tokens']>0
record('seeded sampling after prefix restore',content(r1))
thought=prompt('What is 17 + 25? Answer with the number.',enable_thinking=True,max_tokens=192)
r=call(thought);s=stream(thought)
assert s[0]==content(r) and s[1]==r['choices'][0]['message'].get('reasoning_content','')
assert s[1],r
record('thinking stream parity',r['choices'][0])
tools=[{'type':'function','function':{'name':'get_weather','description':'Get current weather for a city.','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
tool_req=prompt('Use get_weather to check current weather in Haifa. Pass city exactly as Haifa. Do not guess the weather.',tools=tools,max_tokens=192)
r=call(tool_req);message=r['choices'][0]['message'];calls=message.get('tool_calls',[])
assert calls and calls[0]['function']['name']=='get_weather',r
assert json.loads(calls[0]['function']['arguments'])['city']=='Haifa'
record('tool-call schema and parsing',message)
history=tool_req['messages']+[message,{'role':'tool','tool_call_id':calls[0]['id'],'content':'Temperature 22 degrees Celsius, clear sky.'},
    {'role':'user','content':'State the temperature in one sentence.'}]
r=call({'messages':history,'tools':tools,'temperature':0,'max_tokens':64});assert '22' in content(r),r
record('OpenAI string arguments and tool-result continuation',content(r))
for update in ({'max_tokens':server_health['ctx']},{'temperature':-1},{'seed':-1},{'stop':[3]},{'messages':[]},{'n':2},{'tool_choice':'required'},
               {'stream_options':{'include_usage':'yes'}},{'stream_options':[1]},{'parallel_tool_calls':False},
               {'max_tokens':1.5},{'seed':True},{'tools':[{'type':'function','function':{}}]},
               {'messages':[{'role':'user','content':[{'type':'image_url'}]}]}):
    bad=prompt('hello');bad.update(update)
    for streaming in (False,True):
        bad['stream']=streaming
        try:request(bad);raise AssertionError('invalid input accepted')
        except urllib.error.HTTPError as exc:assert exc.code==400,(bad,exc.code)
record('invalid inputs rejected before streaming headers')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    a=pool.submit(call,paris);b=pool.submit(call,prompt('Reply with only the number: 17 + 25 ='))
    assert content(a.result()).strip()=='Paris' and content(b.result()).strip()=='42'
record('concurrent HTTP requests preserve isolated model state')
response=request(prompt('Write a very long story about computers.',stream=True,max_tokens=1000))
for raw in response:
    if raw.startswith(b'data: '):break
response.close()
time.sleep(.2)
assert content(call(paris)).strip()=='Paris'
record('stream disconnect and recovery')
plain={'prompt':'<|im_start|>user\nReply with only the number: 17 + 25 =<|im_end|>\n<|im_start|>assistant\n<think></think>','temperature':0,'max_tokens':8}
with request(plain,path='/v1/completions') as response:r=json.loads(response.read())
assert r['choices'][0]['text'].strip()=='42',r
record('text completion endpoint')
with open(options.output,'w') as f:json.dump({'complete':True,'passed':True,'records':records},f,indent=2)
print(json.dumps({'passed':True,'checks':len(records)}),flush=True)
