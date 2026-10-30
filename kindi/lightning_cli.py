"""CLI integration for the native Lightning adapter."""
import json
import os
import sys
import threading
import time
import numpy as np
from .lightning import Engine
from .lightning_tokenizer import Tok


def manifest(cfg):
    path=os.path.join(cfg['pack'],'kindi_pack.json')
    if not os.path.isfile(path):return {}
    with open(path) as f:return json.load(f)


def selected(cfg):return manifest(cfg).get('backend')=='kindi_lightning'


def verify(cfg,log=print):
    model=manifest(cfg);eng=Engine(model['model_path'],ctx=cfg['ctx'],log=log,kv_dtype=cfg.get('lightning_kv'))
    if eng.source_sha256!=model.get('source_sha256'):raise ValueError('model SHA differs from manifest')
    expected_hash='38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3'
    if eng.source_sha256!=expected_hash:raise ValueError('no numerical reference for this checkpoint')
    suffix='' if eng.kv_dtype=='fp32' else '-'+eng.kv_dtype
    reference=np.load(os.path.join(os.path.dirname(__file__),'lightning_reference'+suffix+'.npz'))
    errors=[]
    for step,token in enumerate((1,22177,1046)):
        actual=eng.forward(token).get();expected=reference['step%d_logits'%step]
        np.testing.assert_allclose(actual,expected,atol=3e-3,rtol=2e-3)
        if actual.argmax()!=expected.argmax():raise ValueError('greedy mismatch')
        errors.append(float(np.max(np.abs(actual-expected))))
    state=eng.snapshot();token=int(eng.last_logits.argmax());first=eng.forward(token).get()
    eng.restore(state);second=eng.forward(token).get()
    if not np.array_equal(first,second):raise ValueError('snapshot restoration is not exact')
    log(json.dumps({'passed':True,'head_max_abs':errors,'snapshot_bit_exact':True,'source_sha256':eng.source_sha256,
        'numeric_profile':eng.numeric_profile}))
    return 0


def chat(cfg,args,log=print):
    from .lightning_server import Runtime
    runtime=Runtime(cfg,log);messages=[]
    if args.system:messages.append({'role':'system','content':args.system})
    print('Kindi Lightning V3; modified by slavalevski-tech, 2026-10-30. Original Kindi: Amir Husain.')
    print('AGPL-3.0-only; WITHOUT ANY WARRANTY. See LICENSE and MODIFICATIONS.md for copying terms/source.')
    print('Kindi Lightning: /reset starts over; /exit quits.')
    while True:
        try:line=input('\nyou> ')
        except (EOFError,KeyboardInterrupt):print();return 0
        if line.strip() in ('/exit','/quit'):return 0
        if line.strip()=='/reset':
            messages=messages[:1] if args.system else [];runtime.engine.reset();runtime.cache.clear();continue
        if not line.strip():continue
        messages.append({'role':'user','content':line});sys.stdout.write('kindi> ');sys.stdout.flush()
        def emit(delta):sys.stdout.write(delta.get('content',''));sys.stdout.flush()
        try:result=runtime.generate({'messages':messages,'temperature':args.temperature,'max_tokens':args.max_tokens},threading.Event(),emit)
        except KeyboardInterrupt:
            messages.pop();print('\n(cancelled)');continue
        except ValueError as exc:
            messages.pop();print('\n'+str(exc));continue
        messages.append({'role':'assistant','content':result['content']})
        print('\n  (%.1f tokens/s)'%result['timing']['gen_tok_s'])


def bench(cfg,log=print):
    model=manifest(cfg);start=time.perf_counter();eng=Engine(model['model_path'],ctx=cfg['ctx'],log=log,kv_dtype=cfg.get('lightning_kv'));load=time.perf_counter()-start
    tok=Tok(model['model_path']);tokens=tok.encode(tok.chat([{'role':'user','content':'Explain arithmetic and memory with examples. '*30}]))[:128]
    if eng.ctx<len(tokens)+64:raise ValueError('benchmark requires context >=192')
    eng.prefill(tokens[:8]);eng.cp.cuda.Stream.null.synchronize();eng.reset()
    start=time.perf_counter();logits=eng.prefill(tokens,chunk_size=int(os.environ.get('KINDI_LIGHTNING_PREFILL_CHUNK','128')))
    eng.cp.cuda.Stream.null.synchronize();prefill=time.perf_counter()-start;start=time.perf_counter()
    for _ in range(64):logits=eng.forward(int(logits.argmax()))
    eng.cp.cuda.Stream.null.synchronize();decode=time.perf_counter()-start
    log(json.dumps({'model_load_s':load,'prompt_tokens':len(tokens),'prefill_tps':len(tokens)/prefill,'generated_tokens':64,
        'decode_tps':64/decode,'eos_policy':'ignored for fixed-length benchmark','numeric_profile':eng.numeric_profile}))
    return 0
