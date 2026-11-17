#!/usr/bin/env python3
import argparse, json, os, time, traceback

def log(msg):
    print(msg, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True)
    ap.add_argument('--ctx-max', type=int, default=8192)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--start', type=int, default=3072)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))

    from kindi.cpu import Engine
    from kindi.tokenizer import Tok

    t0 = time.time()
    eng = Engine(args.pack, ctx=max(512, args.ctx_max), threads=args.threads, log=log)
    vocab = json.load(open(os.path.join(args.pack, 'vocab.json')))
    tok = Tok(vocab, os.path.join(args.pack, 'merges.txt'))
    log('LOADED in %.2fs ctx=%d' % (time.time()-t0, eng.ctx))

    ctx_steps = [256,512,768,1024,1536,2048,3072,4096,5120,6144,7168,8192,10240,12288,16384,20480,24576,32768]
    ctx_steps = [s for s in ctx_steps if s >= args.start and s <= args.ctx_max]
    base_long = tok.encode(('This is a context length test sentence. ' * 800))
    results = {'meta': {'pack': args.pack, 'ctx_max': args.ctx_max, 'threads': args.threads, 'start': args.start, 'engine_ctx': eng.ctx}, 'max_context': []}
    for target in ctx_steps:
        prompt_n = target - 8
        if prompt_n < 8: continue
        ids = (base_long * ((prompt_n//len(base_long))+2))[:prompt_n]
        try:
            eng.reset()
            t0 = time.perf_counter()
            for i,t in enumerate(ids[:-1]):
                eng.feed(t,i)
            last = eng.step(ids[-1], len(ids)-1)
            out=[last]; pos=len(ids)
            for _k in range(8):
                nt=eng.step(out[-1], pos); out.append(nt); pos+=1
            t1 = time.perf_counter()
            txt = tok.decode(out)
            results['max_context'].append({'target_ctx': target, 'prompt_tokens': len(ids), 'gen_tokens': len(out), 'ok': True, 's': round(t1-t0,4), 'tail_text': txt[:60]})
            log('CTX %d OK (%.2fs, tail=%r)' % (target, t1-t0, txt[:40]))
            with open(args.out,'w') as f: json.dump(results, f, indent=1)
        except Exception as e:
            err = str(e)
            results['max_context'].append({'target_ctx': target, 'prompt_tokens': len(ids), 'ok': False, 'error': err, 'trace': traceback.format_exc()})
            log('CTX %d FAIL: %s' % (target, err))
            try:
                eng.reset()
            except Exception:
                pass
            with open(args.out,'w') as f: json.dump(results, f, indent=1)
            break
    log('DONE %s' % args.out)

if __name__ == '__main__':
    main()
