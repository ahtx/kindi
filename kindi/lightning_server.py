"""Serialized native Lightning API; all CUDA calls run on one dedicated thread."""
import asyncio
import codecs
import concurrent.futures
import json
import os
import threading
import time
import uuid
import numpy as np
from .lightning import Engine
from .lightning_tokenizer import Tok
from .chat import parse_tool_calls
from .prefix_cache import PrefixCache


def sample(logits,rng,temperature,top_k,top_p):
    if temperature==0:return int(logits.argmax().get())
    values=logits.get().astype(np.float64)
    if not np.isfinite(values).all():raise RuntimeError('nonfinite logits')
    with np.errstate(over='ignore'):values=(values-values.max())/temperature
    if 0<top_k<len(values):
        cutoff=np.partition(values,len(values)-top_k)[len(values)-top_k]
        better=np.flatnonzero(values>cutoff)
        tied=np.flatnonzero(values==cutoff)[:top_k-len(better)]
        ids=np.concatenate((better,tied))
        # Restore canonical lower-ID ties before the small stable sort.
        ids.sort();ids=ids[np.argsort(-values[ids],kind='stable')]
    else:ids=np.argsort(-values,kind='stable')
    probs=np.exp(values[ids]-values[ids].max());probs/=probs.sum()
    if top_p<1:
        keep=max(1,int(np.searchsorted(np.cumsum(probs),top_p))+1)
        ids=ids[:keep];probs=probs[:keep];probs/=probs.sum()
    return int(rng.choice(ids,p=probs))


def split_text(text,thinking,tools,final=False):
    reason='';content=text
    if thinking or text.startswith('<think>'):
        body=text[len('<think>'):] if text.startswith('<think>') else text
        if '</think>' in body:reason,content=body.split('</think>',1)
        else:reason=body;content=''
    calls=[]
    if tools:
        if final:
            _,calls=parse_tool_calls(content,tools)
            if calls:content=content[:content.find('<tool_call>')]
        elif '<tool_call>' in content:content=content.split('<tool_call>',1)[0]
    if not final:
        # Withhold partial markup until the following token resolves it.
        reason=hold_suffix(reason,('</think>','<tool_call>','<think>'))
        content=hold_suffix(content,('</think>','<tool_call>','<think>'))
    return reason,content,calls


def hold_suffix(text,markers):
    length=max([0]+[n for marker in markers for n in range(1,len(marker)) if text.endswith(marker[:n])])
    return text[:-length] if length else text


class Runtime(object):
    def __init__(self,cfg,log):
        manifest=json.load(open(os.path.join(cfg['pack'],'kindi_pack.json')))
        if cfg['head']!='exact':raise ValueError('Lightning currently requires exact full-vocabulary head')
        self.engine=Engine(manifest['model_path'],ctx=cfg['ctx'],log=log,kv_dtype=cfg.get('lightning_kv'))
        if manifest.get('source_sha256')!=self.engine.source_sha256:raise ValueError('model fingerprint differs from pack')
        self.tok=Tok(manifest['model_path']);self.cfg=cfg
        self.cache=PrefixCache((cfg.get('checkpoint_budget_mb',0) or 512)*1024**2,cfg.get('prefix_max',8))
        self.prefill_chunk=int(os.environ.get('KINDI_LIGHTNING_PREFILL_CHUNK','128'))
        if not 0<=self.prefill_chunk<=256:raise ValueError('invalid Lightning prefill chunk')
        self.busy=False;self.phase='idle';self.requests=0

    def prepare(self,req):
        for name in ('max_tokens','max_completion_tokens','top_k','n'):
            if name in req and (not isinstance(req[name],int) or isinstance(req[name],bool)):raise ValueError(name+' must be an integer')
        try:
            temp=float(req.get('temperature',self.cfg['temperature']))
            top_p=float(req.get('top_p',self.cfg['top_p']));top_k=int(req.get('top_k',self.cfg['top_k']))
            maximum=int(req.get('max_tokens',req.get('max_completion_tokens',self.cfg['max_tokens'])))
        except (ValueError,TypeError,OverflowError):raise ValueError('invalid sampling parameters')
        if not np.isfinite(temp) or temp<0 or not 0<top_p<=1 or top_k<0 or maximum<1:raise ValueError('invalid sampling parameters')
        if req.get('n',1)!=1:raise ValueError('only n=1 is supported')
        seed=req.get('seed')
        if seed is not None and (not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**32):raise ValueError('seed must be an integer in [0, 2^32)')
        for name in ('frequency_penalty','presence_penalty','logit_bias','logprobs','top_logprobs','min_p'):
            if req.get(name):raise ValueError(name+' is not supported by this backend')
        fmt=req.get('response_format') or {}
        if not isinstance(fmt,dict) or fmt.get('type','text')!='text':raise ValueError('only text response_format is supported')
        if not isinstance(req.get('stream',False),bool):raise ValueError('stream must be boolean')
        stream_options=req.get('stream_options') or {}
        if not isinstance(stream_options,dict) or not isinstance(stream_options.get('include_usage',False),bool):raise ValueError('invalid stream_options')
        if req.get('parallel_tool_calls',True) is not True:raise ValueError('disabling parallel_tool_calls is not supported')
        tools=req.get('tools') or []
        if not isinstance(tools,list) or any(not isinstance(t,dict) or not isinstance(t.get('function'),dict) for t in tools):raise ValueError('tools must be function definitions')
        for tool in tools:
            fn=tool['function']
            if tool.get('type','function')!='function' or not isinstance(fn.get('name'),str) or not fn['name'] or not isinstance(fn.get('parameters',{}),dict):raise ValueError('invalid function definition')
        choice=req.get('tool_choice','auto')
        if choice=='none':tools=[]
        elif choice!='auto':raise ValueError('tool_choice supports auto or none')
        kwargs=req.get('chat_template_kwargs') or {}
        if not isinstance(kwargs,dict):raise ValueError('chat_template_kwargs must be an object')
        thinking=kwargs.get('enable_thinking',req.get('enable_thinking',False))
        if not isinstance(thinking,bool):raise ValueError('enable_thinking must be boolean')
        if 'messages' in req:
            if not isinstance(req['messages'],list) or not req['messages']:raise ValueError('messages must be a nonempty list')
            for message in req['messages']:
                if not isinstance(message,dict) or message.get('role') not in ('system','user','assistant','tool'):raise ValueError('invalid message')
                if message.get('content') is not None and not isinstance(message['content'],str):raise ValueError('Lightning accepts text messages')
            try:prompt=self.tok.chat(req['messages'],tools,thinking)
            except (ValueError,TypeError,AttributeError) as exc:raise ValueError('invalid chat template inputs: '+str(exc))
        else:
            prompt=req.get('prompt')
            if not isinstance(prompt,str):raise ValueError('prompt must be text')
        tokens=self.tok.encode(prompt)
        if not tokens:raise ValueError('prompt must contain tokens')
        if len(tokens)+maximum>self.engine.ctx:raise ValueError('prompt + max_tokens exceeds context capacity')
        stop=req.get('stop') or [];stop=[stop] if isinstance(stop,str) else stop
        if not isinstance(stop,list) or any(not isinstance(x,str) or not x for x in stop):raise ValueError('invalid stop strings')
        return tokens,thinking,tools,temp,top_p,top_k,maximum,stop

    def generate(self,req,cancel,emit,prepared=None):
        self.busy=True;self.phase='prefill'
        eng=self.engine;start=time.monotonic();completed=False
        try:
            tokens,thinking,tools,temp,top_p,top_k,maximum,stop=prepared if prepared is not None else self.prepare(req)
            if cancel.is_set():raise InterruptedError('request cancelled')
            cached=0
            if eng.tokens and len(eng.tokens)<=len(tokens) and eng.tokens==tokens[:len(eng.tokens)]:
                cached=len(eng.tokens)
            else:
                snapshot=self.cache.best(tokens,eng.fingerprint)
                if snapshot is None:eng.reset()
                else:eng.restore(snapshot);cached=eng.position
            if cached<len(tokens):logits=eng.prefill(tokens[cached:],cancel=cancel.is_set,chunk_size=self.prefill_chunk)
            elif eng.last_logits is not None:logits=eng.last_logits
            else:
                eng.reset();cached=0;logits=eng.prefill(tokens,cancel=cancel.is_set,chunk_size=self.prefill_chunk)
            eng.cp.cuda.Stream.null.synchronize();prefill_done=time.monotonic()
            if len(tokens)>=self.cfg.get('prefix_min',512):self.cache.put(eng.snapshot())
            self.phase='decode';rng=np.random.RandomState(req.get('seed'))
            decoder=codecs.getincrementaldecoder('utf-8')('replace')
            text='';generated=[];finish='length';sent_reason='';sent_content='';text_stop=False
            for index in range(maximum):
                if cancel.is_set():raise InterruptedError('request cancelled')
                token=sample(logits,rng,temp,top_k,top_p)
                generated.append(token)
                if token in self.tok.stop_ids:finish='stop';break
                text+=decoder.decode(self.tok.decode_bytes(token))
                hits=[text.find(s) for s in stop if s in text]
                if hits:text=text[:min(hits)];finish='stop';text_stop=True
                visible=text
                if not hits:visible=hold_suffix(visible,stop)
                reason,content,_=split_text(visible,thinking,tools)
                delta={}
                if reason.startswith(sent_reason) and reason!=sent_reason:delta['reasoning_content']=reason[len(sent_reason):];sent_reason=reason
                if content.startswith(sent_content) and content!=sent_content:delta['content']=content[len(sent_content):];sent_content=content
                if delta:emit(delta)
                if hits:break
                # Keep cache at the exact consumed-token boundary, including the last returned token.
                logits=eng.forward(token)
            if not text_stop:text+=decoder.decode(b'',final=True)
            reason,content,calls=split_text(text,thinking,tools,True)
            delta={}
            if reason.startswith(sent_reason) and reason!=sent_reason:delta['reasoning_content']=reason[len(sent_reason):]
            if content.startswith(sent_content) and content!=sent_content:delta['content']=content[len(sent_content):]
            if calls:delta['tool_calls']=[dict(call,index=i) for i,call in enumerate(calls)];finish='tool_calls'
            if delta:emit(delta)
            elapsed=time.monotonic()-prefill_done
            result={'content':content,'reasoning_content':reason,'tool_calls':calls,'finish_reason':finish,
                'usage':{'prompt_tokens':len(tokens),'completion_tokens':len(generated),'total_tokens':len(tokens)+len(generated),
                    'prompt_tokens_details':{'cached_tokens':cached}},
                'timing':{'prompt_s':prefill_done-start,'gen_s':elapsed,'gen_tok_s':len(generated)/max(elapsed,1e-9)},
                'generated_ids':generated}
            if eng.position>=self.cfg.get('prefix_min',512):self.cache.put(eng.snapshot())
            self.requests+=1;completed=True
            return result
        finally:
            if not completed:eng.invalidate()
            self.busy=False;self.phase='failed' if getattr(eng,'failed',False) else 'idle'


def create_app(cfg,log=print):
    from fastapi import FastAPI,Request
    from fastapi.responses import JSONResponse,StreamingResponse
    worker=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='lightning-gpu')
    runtime=worker.submit(Runtime,cfg,log).result()
    from . import VERSION
    from .source_offer import SourceOfferMiddleware
    app=FastAPI(title='Kindi Lightning V3',version=VERSION,
        description='Modified Kindi by slavalevski-tech. Original Kindi copyright Amir Husain. '
                    'AGPL-3.0-only, without warranty. [Complete source and legal notices](/source).')
    app.add_middleware(SourceOfferMiddleware,root=os.path.dirname(os.path.dirname(__file__)))
    model=cfg['model_name'];active=set()
    @app.on_event('shutdown')
    def shutdown():
        for cancel in list(active):cancel.set()
        worker.shutdown(wait=True)
    def auth(request):return not cfg['api_key'] or request.headers.get('authorization')=='Bearer '+cfg['api_key']
    def error(message,status=400):return JSONResponse({'error':{'message':message,'type':'invalid_request_error' if status==400 else 'server_error'}},status_code=status)
    @app.get('/health')
    def health():
        failed=getattr(runtime.engine,'failed',False)
        body={'ok':not failed,'backend':'kindi_lightning','model':model,'ctx':runtime.engine.ctx,'head':'exact','kv':runtime.engine.kv_dtype,
            'busy':runtime.busy,'phase':runtime.phase,'requests':runtime.requests,'memory_plan':runtime.engine.memory,
            'prefix_cache_bytes':runtime.cache.used,'prefix_cache_entries':len(runtime.cache.entries),'prefill_chunk':runtime.prefill_chunk}
        return JSONResponse(body,status_code=503 if failed else 200)
    @app.get('/v1/models')
    def models(request:Request):
        if not auth(request):return error('unauthorized',401)
        return {'object':'list','data':[{'id':model,'object':'model','owned_by':'local'}]}
    async def respond(request,chat):
        if not auth(request):return error('unauthorized',401)
        try:req=await request.json()
        except Exception:return error('invalid JSON')
        if not isinstance(req,dict):return error('request must be an object')
        cancel=threading.Event();cid='chatcmpl-'+uuid.uuid4().hex;created=int(time.time())
        loop=asyncio.get_event_loop()
        if req.get('model',model)!=model:return error('model not found',404)
        try:prepared=await loop.run_in_executor(None,runtime.prepare,req)
        except (ValueError,TypeError,AttributeError) as exc:return error(str(exc))
        def envelope(delta=None,result=None):
            if req.get('stream'):
                choice={'index':0,'delta':delta or {},'finish_reason':None if result is None else result['finish_reason']}
                if not chat:choice={'index':0,'text':(delta or {}).get('content',''),'finish_reason':choice['finish_reason']}
                out={'id':cid,'created':created,'model':model,'object':'chat.completion.chunk' if chat else 'text_completion','choices':[choice]}
                if result and (req.get('stream_options') or {}).get('include_usage'):out['usage']=result['usage']
                return out
            message={'role':'assistant','content':result['content']}
            if result['reasoning_content']:message['reasoning_content']=result['reasoning_content']
            if result['tool_calls']:message['tool_calls']=result['tool_calls']
            choice={'index':0,'finish_reason':result['finish_reason']}
            choice['message' if chat else 'text']=message if chat else result['content']
            return {'id':cid,'created':created,'model':model,'object':'chat.completion' if chat else 'text_completion',
                'choices':[choice],'usage':result['usage'],'timing':result['timing']}
        if req.get('stream'):
            async def stream():
                active.add(cancel)
                q=asyncio.Queue()
                def emit(delta):loop.call_soon_threadsafe(q.put_nowait,('delta',delta))
                def run():
                    try:
                        if cancel.is_set():return
                        result=runtime.generate(req,cancel,emit,prepared)
                        loop.call_soon_threadsafe(q.put_nowait,('done',result))
                    except Exception as exc:loop.call_soon_threadsafe(q.put_nowait,('error',str(exc)))
                    finally:loop.call_soon_threadsafe(q.put_nowait,('end',None))
                future=worker.submit(run)
                try:
                    yield 'data: '+json.dumps(envelope({'role':'assistant'}))+'\n\n'
                    while True:
                        try:kind,value=await asyncio.wait_for(q.get(),timeout=15)
                        except asyncio.TimeoutError:
                            yield ': keep-alive\n\n';continue
                        if kind=='end':break
                        if kind=='error':out={'error':{'message':value,'type':'server_error'}}
                        elif kind=='done':out=envelope(result=value)
                        else:out=envelope(delta=value)
                        yield 'data: '+json.dumps(out,ensure_ascii=False)+'\n\n'
                    yield 'data: [DONE]\n\n'
                finally:cancel.set();active.discard(cancel)
            return StreamingResponse(stream(),media_type='text/event-stream')
        active.add(cancel)
        future=loop.run_in_executor(worker,runtime.generate,req,cancel,lambda delta:None,prepared)
        try:
            while not future.done():
                await asyncio.wait([future],timeout=1)
                if await request.is_disconnected():cancel.set()
            return envelope(result=future.result())
        except (ValueError,OverflowError) as exc:return error(str(exc))
        except InterruptedError:return error('cancelled',499)
        except Exception as exc:return error(str(exc),500)
        finally:cancel.set();active.discard(cancel)
    @app.post('/v1/chat/completions')
    async def chats(request:Request):return await respond(request,True)
    @app.post('/v1/completions')
    async def completions(request:Request):return await respond(request,False)
    return app
