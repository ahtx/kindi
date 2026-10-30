import json
import numpy as np
from kindi.lightning_server import split_text,hold_suffix,sample
from kindi.prefix_cache import PrefixCache,snapshot_bytes

checks=[]
tools=[{'type':'function','function':{'name':'weather','parameters':{'properties':{'city':{'type':'string'},'days':{'type':'integer'}}}}}]
examples=[('Привет, мир!',False,[]),('reason\n</think>answer',True,[]),('<think>理由</think>答案',False,[]),
    ('\n<tool_call>\n<function=weather>\n<parameter=city>Хайфа</parameter><parameter=days>2</parameter></function></tool_call>',False,tools),
    ('text <tool_call>unfinished',False,tools)]
for text,thinking,ts in examples:
    last_reason='';last_content=''
    for n in range(len(text)+1):
        reason,content,_=split_text(text[:n],thinking,ts)
        assert reason.startswith(last_reason),(text,n,last_reason,reason)
        assert content.startswith(last_content),(text,n,last_content,content)
        last_reason,last_content=reason,content
    reason,content,calls=split_text(text,thinking,ts,True)
    assert reason.startswith(last_reason) and content.startswith(last_content)
    if calls:assert json.loads(calls[0]['function']['arguments'])=={'city':'Хайфа','days':2}
checks.append('all incremental markup boundaries are monotonic')
assert hold_suffix('abcEN',['END','ND'])=='abc'
assert hold_suffix('abc',['END'])=='abc'
checks.append('overlapping stop prefixes')
def snapshot(tokens,fp='a'):
    return {'tokens':tokens,'fingerprint':fp,'states':{0:np.zeros(4,np.float32)},'histories':{},'keys':{},'values':{},'logits':np.zeros(8,np.float32)}
s=snapshot([1,2]);size=snapshot_bytes(s);cache=PrefixCache(size*2+100,2)
cache.put(s);cache.put(snapshot([3,4]));assert cache.best([1,2,5],'a') is s
cache.put(snapshot([5,6]));assert cache.best([3,4],'a') is None
assert cache.best([1,9],'a') is None and cache.best([1,2],'b') is None
assert cache.used<=cache.budget and len(cache.entries)<=2
cache.clear();assert cache.used==0 and not cache.entries
checks.append('prefix branching, fingerprint and LRU budget')
class Logits(object):
    def get(self):return np.array([1,2,3],np.float32)
    def argmax(self):return Scalar()
class Scalar(object):
    def get(self):return 2
assert sample(Logits(),np.random.RandomState(3),0,0,1)==2
assert sample(Logits(),np.random.RandomState(3),1e-308,0,1)==2
x=[sample(Logits(),np.random.RandomState(13),.8,0,.9) for _ in range(3)]
assert len(set(x))==1
checks.append('greedy, tiny temperature and seeded sampling')
print(json.dumps({'passed':True,'checks':checks}))
