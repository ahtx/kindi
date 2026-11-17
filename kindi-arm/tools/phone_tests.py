#!/usr/bin/env python3
import argparse, json, os, sys, time, traceback
import numpy as np

def log(msg):
    print(msg, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True)
    ap.add_argument('--ctx-max', type=int, default=8192)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    results = {'meta': {'pack': args.pack, 'ctx_max': args.ctx_max, 'threads': args.threads}, 'tests': {}}

    from kindi.cpu import Engine
    from kindi.tokenizer import Tok
    from kindi.chat import render, template_variant

    t0 = time.time()
    eng = Engine(args.pack, ctx=max(512, args.ctx_max), threads=args.threads, log=log)
    man = json.load(open(os.path.join(args.pack, 'kindi_pack.json')))
    tok = Tok(json.load(open(os.path.join(eng.m and '', '')))) if False else None
    # build tokenizer: need vocab and merges
    vocab = json.load(open(os.path.join(args.pack, 'vocab.json')))
    tok = Tok(vocab, os.path.join(args.pack, 'merges.txt'))
    variant = template_variant(args.pack)
    results['meta']['load_s'] = round(time.time()-t0, 2)
    results['meta']['ctx'] = eng.ctx
    log('LOADED in %.2fs ctx=%d' % (results['meta']['load_s'], eng.ctx))

    def enc_prompt(user, system=None, thinking=False):
        msgs = []
        if system: msgs.append({'role':'system','content':system})
        msgs.append({'role':'user','content':user})
        body, gp = render(msgs, tools=None, enable_thinking=thinking, variant=variant)
        return tok.encode(body + gp)

    def generate_text(user, max_new=128, system=None, ctx=None, greedy=True):
        ids = enc_prompt(user, system=system)
        prompt_n = len(ids)
        eng.ctx if ctx is None else ctx
        eng.reset()
        t0 = time.perf_counter()
        # feed prompt
        for i,t in enumerate(ids[:-1]):
            eng.feed(t,i)
        t_prefill = time.perf_counter()
        out = [eng.step(ids[-1], len(ids)-1)]
        pos = len(ids)
        eos = {151643,151645} # common qwen eos/im_end; stop on eos-like if any
        stopped = 'max'
        for _ in range(max_new-1):
            # stop on im_end/eos tokens if known
            if out[-1] in (151645, 151643, 151644):
                stopped = 'eos'
                break
            nt = eng.step(out[-1], pos)
            out.append(nt)
            pos += 1
            if nt in (151645, 151643, 151644):
                stopped = 'eos'
                break
        t_end = time.perf_counter()
        # trim trailing eos
        gen_ids = list(out)
        while gen_ids and gen_ids[-1] in (151645,151643,151644):
            gen_ids.pop()
        text = tok.decode(gen_ids)
        prefill_s = t_prefill - t0
        decode_s = t_end - t_prefill
        gen_n = len(gen_ids)
        return {
            'prompt_tokens': prompt_n,
            'gen_tokens': gen_n,
            'prefill_s': round(prefill_s,4),
            'decode_s': round(decode_s,4),
            'prefill_tok_s': round(prompt_n/prefill_s,3) if prefill_s>0 else None,
            'decode_tok_s': round(gen_n/decode_s,3) if decode_s>0 and gen_n>0 else None,
            'stopped': stopped,
            'text': text,
            'gen_ids': gen_ids[:40],
        }

    # 1) Math and common questions
    qa = []
    prompts = [
        ('math_add', 'What is 17 + 28? Answer with just the number.', None, 16),
        ('math_mul', 'What is 13 * 7? Answer with just the number.', None, 16),
        ('math_sub', 'What is 100 - 37? Answer with just the number.', None, 16),
        ('math_div', 'What is 144 divided by 12? Answer with just the number.', None, 16),
        ('math_word', 'If I have 3 apples and buy 5 more, then give away 2, how many apples do I have? Answer with just the number.', None, 24),
        ('common_capital', 'What is the capital of France? Answer in one word.', None, 16),
        ('common_planet', 'Which planet is known as the Red Planet? Answer in one word.', None, 16),
        ('common_water', 'What is the chemical formula for water? Answer briefly.', None, 16),
        ('common_author', 'Who wrote Romeo and Juliet? Answer with the name only.', None, 24),
        ('common_days', 'How many days are in a week? Answer with just the number.', None, 12),
        ('common_color', 'What color do you get by mixing red and white? Answer in one word.', None, 12),
        ('common_ocean', 'What is the largest ocean on Earth? Answer briefly.', None, 16),
    ]
    for name, user, system, mx in prompts:
        try:
            log('QA %s' % name)
            r = generate_text(user, max_new=mx, system=system)
            r['name'] = name; r['prompt'] = user
            qa.append(r)
            log('  -> gen=%d %.2f tok/s text=%r' % (r['gen_tokens'], r.get('decode_tok_s') or 0, r['text'][:80]))
        except Exception as e:
            qa.append({'name': name, 'error': str(e), 'trace': traceback.format_exc()})
            log('  ERR %s' % e)
    results['tests']['qa'] = qa

    # 2) Speed/read/write
    speed = {}
    # prompt reading (prefill) at different sizes - build synthetic prompts by repeating tokens
    # use a simple token sequence from encoding a repeated phrase
    base = tok.encode('hello world ' * 200)
    read_points = [32,64,128,256,512,1024]
    prefill = []
    for n in read_points:
        if n > len(base):
            ids = (base * ((n//len(base))+1))[:n]
        else:
            ids = base[:n]
        try:
            eng.reset()
            t0 = time.perf_counter()
            for i,t in enumerate(ids[:-1]):
                eng.feed(t,i)
            # one step to include final token processing comparable to bench
            _ = eng.step(ids[-1], len(ids)-1)
            t1 = time.perf_counter()
            prefill.append({'prompt_tokens': n, 's': round(t1-t0,4), 'tok_s': round(n/(t1-t0),3)})
            log('PREFILL n=%d -> %.2f tok/s' % (n, n/(t1-t0)))
        except Exception as e:
            prefill.append({'prompt_tokens': n, 'error': str(e)})
            log('PREFILL n=%d ERR %s' % (n,e))
    speed['prefill'] = prefill

    # decode speed at different gen lengths (short prompt)
    short_ids = tok.encode('The quick brown fox')
    decode = []
    for g in [16, 32, 64, 128, 256]:
        try:
            eng.reset()
            for i,t in enumerate(short_ids[:-1]): eng.feed(t,i)
            _ = eng.step(short_ids[-1], len(short_ids)-1)
            t0 = time.perf_counter()
            out=[]
            pos=len(short_ids)
            prev=_ if isinstance(_,int) else short_ids[-1]
            # we already did first step above returned token
            first = _
            out=[first]; pos=len(short_ids)
            for _k in range(g-1):
                nt=eng.step(out[-1], pos); out.append(nt); pos+=1
            t1=time.perf_counter()
            decode.append({'gen_tokens': g, 's': round(t1-t0,4), 'tok_s': round(g/(t1-t0),3)})
            log('DECODE g=%d -> %.2f tok/s' % (g, g/(t1-t0)))
        except Exception as e:
            decode.append({'gen_tokens': g, 'error': str(e)})
            log('DECODE g=%d ERR %s' % (g,e))
    speed['decode'] = decode

    # write test: generate a short paragraph
    write_tests = []
    for name, user, mx in [
        ('write_sentence', 'Write one sentence about the ocean.', 64),
        ('write_para', 'Write a short paragraph (3-4 sentences) about why reading is useful.', 160),
    ]:
        try:
            log('WRITE %s' % name)
            r = generate_text(user, max_new=mx)
            r['name']=name; r['prompt']=user
            write_tests.append(r)
            log('  -> gen=%d %.2f tok/s' % (r['gen_tokens'], r.get('decode_tok_s') or 0))
        except Exception as e:
            write_tests.append({'name': name, 'error': str(e), 'trace': traceback.format_exc()})
    speed['write'] = write_tests
    results['tests']['speed'] = speed

    # 3) Incremental max context test
    # step up: try prompt lengths approaching ctx; generate a few tokens after
    ctx_steps = [256,512,768,1024,1536,2048,3072,4096,5120,6144,7168,8192,9216,10240,12288,14336,16384]
    ctx_steps = [s for s in ctx_steps if s <= args.ctx_max]
    base_long = tok.encode(('This is a context length test sentence. ' * 500))
    ctx_results = []
    for target in ctx_steps:
        prompt_n = target - 8  # leave room for generation
        if prompt_n < 8: continue
        ids = (base_long * ((prompt_n//len(base_long))+2))[:prompt_n]
        try:
            eng.reset()
            t0 = time.perf_counter()
            ok = True
            err = None
            for i,t in enumerate(ids[:-1]):
                eng.feed(t,i)
            last = eng.step(ids[-1], len(ids)-1)
            out=[last]; pos=len(ids)
            for _k in range(8):
                nt=eng.step(out[-1], pos); out.append(nt); pos+=1
            t1 = time.perf_counter()
            txt = tok.decode(out)
            ctx_results.append({'target_ctx': target, 'prompt_tokens': len(ids), 'gen_tokens': len(out), 'ok': True, 's': round(t1-t0,4), 'tail_text': txt[:60]})
            log('CTX %d OK (%.2fs, tail=%r)' % (target, t1-t0, txt[:40]))
        except Exception as e:
            err = str(e)
            ctx_results.append({'target_ctx': target, 'prompt_tokens': len(ids), 'ok': False, 'error': err, 'trace': traceback.format_exc()})
            log('CTX %d FAIL: %s' % (target, err))
            # if hard failure (OOM/overflow), break early to avoid repeated bad state? reset and continue a bit
            try:
                eng.reset()
            except Exception:
                pass
            # stop at first failure to be safe/informative
            break
    results['tests']['max_context'] = ctx_results

    with open(args.out,'w') as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    log('WROTE %s' % args.out)

if __name__ == '__main__':
    main()
