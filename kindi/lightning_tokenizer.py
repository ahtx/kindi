"""Nota/Pixtral byte-level BPE, dependency-free Unicode pre-tokenizer.

The exact case-sensitive split expression is from the pinned Nota tokenizer.json.
No Qwen/Gemma pre-tokenization assumptions are inherited.
"""
import functools
import heapq
import re
import unicodedata
from .gguf import GGUF
from .tokenizer import _B2U

_U2B={v:k for k,v in _B2U.items()}
@functools.lru_cache(maxsize=16384)
def category(c):return unicodedata.category(c)
def upper(c):return category(c) in ('Lu','Lt','Lm','Lo') or category(c)[0]=='M'
def lower(c):return category(c) in ('Ll','Lm','Lo') or category(c)[0]=='M'
def prefix(c):return c not in '\r\n' and category(c)[0] not in ('L','N')

def pretoken(text):
    i=0;n=len(text)
    while i<n:
        match=None
        starts=[i+1,i] if prefix(text[i]) else [i]
        # Regex alternative one: optional prefix, upper*, lower+.
        for start in starts:
            u=start
            while u<n and upper(text[u]):u+=1
            end=u
            while end<n and lower(text[end]):end+=1
            if end>u:match=end;break
            if u>start and lower(text[u-1]):match=u;break
        if match is None:
            # Alternative two: optional prefix, upper+, lower*.
            for start in starts:
                end=start
                while end<n and upper(text[end]):end+=1
                if end==start:continue
                while end<n and lower(text[end]):end+=1
                match=end;break
        if match is None and category(text[i])[0]=='N':match=i+1
        if match is None:
            def other(c):return not c.isspace() and category(c)[0] not in ('L','N')
            start=i+1 if text[i]==' ' and i+1<n and other(text[i+1]) else i
            end=start
            while end<n and other(text[end]):end+=1
            if end>start:
                while end<n and text[end] in '\r\n/':end+=1
                match=end
        if match is None and text[i].isspace():
            end=i+1
            while end<n and text[end].isspace():end+=1
            newlines=[p for p in range(i,end) if text[p] in '\r\n']
            if newlines:match=newlines[-1]+1
            elif end==n:match=end
            elif end-i>1:match=end-1
            else:match=end
        if match is None:match=i+1
        yield text[i:match];i=match


class Tok(object):
    def __init__(self,path):
        kv=GGUF(path).kv
        if kv.get('tokenizer.ggml.model')!='gpt2' or kv.get('tokenizer.ggml.pre')!='pixtral':
            raise ValueError('Nota tokenizer requires gpt2/pixtral metadata')
        self.tokens=kv['tokenizer.ggml.tokens'];self.ids={t:i for i,t in enumerate(self.tokens)}
        self.ranks={tuple(m.split(' ')):i for i,m in enumerate(kv['tokenizer.ggml.merges'])}
        self.special={t:i for i,(t,ty) in enumerate(zip(self.tokens,kv['tokenizer.ggml.token_type'])) if ty in (3,4)}
        # AddedToken(special=False) still bypasses BPE in the source tokenizer.
        # This GGUF marked the six reasoning/tool AddedTokens as ordinary vocabulary.
        for i,text in enumerate(('<think>','</think>','<tool_call>','</tool_call>','<tool_response>','</tool_response>'),12):
            if self.tokens[i]!=text:raise ValueError('unsupported Nota added-token mapping')
            self.special[text]=i
        self.rx=re.compile('|'.join(re.escape(s) for s in sorted(self.special,key=len,reverse=True)))
        self.stop_ids={2,11};self.bos=1;self.eos=11
        self.template=kv['tokenizer.chat_template']

    @functools.lru_cache(maxsize=32768)
    def _bpe(self,word):
        pieces=[_B2U[b] for b in word.encode('utf-8')]
        if not pieces:return ()
        # Linked list + heap preserves the leftmost tie while avoiding quadratic long-token merges.
        prev=list(range(-1,len(pieces)-1));nxt=list(range(1,len(pieces)))+[-1]
        heap=[]
        def push(i):
            if i>=0 and nxt[i]>=0:
                j=nxt[i];rank=self.ranks.get((pieces[i],pieces[j]))
                if rank is not None:heapq.heappush(heap,(rank,i,j,pieces[i],pieces[j]))
        for i in range(len(pieces)-1):push(i)
        while heap:
            rank,i,j,a,b=heapq.heappop(heap)
            if nxt[i]!=j or pieces[i]!=a or pieces[j]!=b:continue
            pieces[i]+=pieces[j];pieces[j]='';nxt[i]=nxt[j]
            if nxt[j]>=0:prev[nxt[j]]=i
            nxt[j]=-1;push(prev[i]);push(i)
        return tuple(self.ids[p] for p in pieces if p)

    def encode(self,text):
        if not isinstance(text,str):raise ValueError('text must be a string')
        out=[];pos=0
        for m in self.rx.finditer(text):
            for chunk in pretoken(text[pos:m.start()]):out.extend(self._bpe(chunk))
            out.append(self.special[m.group()]);pos=m.end()
        for chunk in pretoken(text[pos:]):out.extend(self._bpe(chunk))
        return out

    def decode_bytes(self,index):
        token=self.tokens[int(index)]
        if token in self.special:return token.encode('utf-8')
        return bytes(_U2B[c] for c in token)

    def decode(self,ids):return b''.join(self.decode_bytes(i) for i in ids).decode('utf-8',errors='replace')

    def chat(self,messages,tools=None,enable_thinking=False):
        import json
        import copy
        from collections.abc import Mapping
        from jinja2 import Undefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        messages=copy.deepcopy(messages)
        for message in messages:
            for call in message.get('tool_calls') or []:
                function=call.get('function',call)
                if isinstance(function.get('arguments'),str):function['arguments']=json.loads(function['arguments'])
                if function.get('arguments') is not None and not isinstance(function['arguments'],dict):raise ValueError('tool arguments must be a JSON object')
        env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
        def items(value):
            if isinstance(value,Undefined):return ()
            if not isinstance(value,Mapping):raise ValueError('items requires a mapping')
            return value.items()
        env.filters['items']=items # Jinja 3.0/Python 3.6 compatibility with original Nota template
        # Match Transformers' chat-template JSON filter, including insertion order and Unicode.
        env.filters['tojson']=lambda value,ensure_ascii=False,indent=None,separators=None,sort_keys=False:json.dumps(
            value,ensure_ascii=ensure_ascii,indent=indent,separators=separators,sort_keys=sort_keys)
        def fail(message):raise ValueError(message)
        env.globals['raise_exception']=fail
        return env.from_string(self.template).render(messages=messages,tools=tools or [],add_generation_prompt=True,
            enable_thinking=enable_thinking,bos_token='<s>',eos_token='<|im_end|>')
