"""Diagnostic CUDA-event instrumentation, not an end-to-end speed benchmark.

Use the intended numerical-profile environment. Imports and --help do not
create a CUDA context. Compatible with Python 3.6 and CuPy 9.6.
"""
import argparse
import collections
import hashlib
import json
import os
import time


def write_json(path, value):
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    deadline = time.perf_counter() + 2.0
    while True:
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise
            time.sleep(min(0.05, remaining))


def build_chat_tokens(tokenizer, length):
    """Exact-length ASCII user content, preserving the model's chat suffix."""
    marker = 'KINDI_BENCH_BODY_79031'
    template = tokenizer.chat([{'role': 'user', 'content': marker}])
    if template.count(marker) != 1:
        raise ValueError('cannot locate user content in the chat template')
    prefix, suffix = template.split(marker)
    suffix_ids = tokenizer.encode(suffix)
    minimum = len(tokenizer.encode(prefix)) + len(suffix_ids) + 16
    if length < minimum or not suffix_ids:
        raise ValueError('token count is too small for a complete chat prompt')
    records = max(32, length // 16)
    while True:
        body = ('Read the following engineering notebook. Explain the memory '
                'tradeoffs and verify the arithmetic in records 7 and 19.\n')
        body += ''.join('Record %d: %d batches contain %d items each; total %d. '
                        'Keep the source data and compare the two methods.\n' %
                        (i, i + 3, i % 13 + 2, (i + 3) * (i % 13 + 2))
                        for i in range(records))
        encoded = tokenizer.encode(prefix + body + suffix)
        if encoded[-len(suffix_ids):] != suffix_ids:
            raise ValueError('chat suffix does not have a stable token boundary')
        if len(encoded) >= length:
            ids = encoded[:length - len(suffix_ids)] + suffix_ids
            return ids, {'policy': 'tokenized ASCII notebook; truncated inside user content; real chat suffix retained',
                         'tokens': len(ids), 'suffix_tokens': len(suffix_ids),
                         'sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).hexdigest()}
        records *= 2


def kernel_label(name, args):
    if name == 'q2_tile8':
        return '%s/%dx%d/%s/per_assignment=%d/items=%d/topk=%d' % (
            name, int(args[3]), int(args[4]), 'expert' if hasattr(args[9], 'get') else 'dense',
            int(args[7]), int(args[5]), int(args[6]))
    if name == 'packed_mm':
        return '%s/%dx%d/%s/per_assignment=%d' % (
            name, int(args[3]), int(args[4]), 'expert' if hasattr(args[10], 'get') else 'dense', int(args[8]))
    if name in ('packed_mv', 'q2_mv', 'q8_mv'):
        index = 7 if name == 'packed_mv' else 6
        return '%s/%dx%d/%s/selected=%d/per_input=%d' % (
            name, int(args[4]), int(args[5]), 'expert' if hasattr(args[3], 'get') else 'dense',
            int(args[index]), int(args[index + 1]))
    if name == 'grouped_mv':
        return '%s/%dx%d/expert/per_assignment=%d' % (name, int(args[6]), int(args[7]), int(args[9]))
    if name == 'grouped_mv4':
        return '%s/%dx%d/expert/per_assignment=%d' % (name, int(args[8]), int(args[9]), int(args[11]))
    if name.startswith('attention') and name not in ('attention', 'attention_merge_tiles'):
        return '%s/tokens=%d/heads=%d/kvheads=%d/dim=%d' % (
            name, int(args[4]), int(args[6]), int(args[7]), int(args[8]))
    if name == 'ssm_chunk':
        return '%s/tokens=%d/heads=%d/headdim=%d/state=%d' % (
            name, int(args[7]), int(args[8]), int(args[9]), int(args[10]))
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokens', type=int, default=1024)
    parser.add_argument('--ctx', type=int, default=32768)
    parser.add_argument('--chunk', type=int, default=128, help='0: sequential; otherwise 1..256')
    args = parser.parse_args()
    warmup_tokens = max(256, args.tokens)
    if not 0 <= args.chunk <= 256 or args.tokens < 1 or max(args.tokens, warmup_tokens + 1) > args.ctx:
        parser.error('require chunk 0..256, positive tokens, and ctx fitting full-prompt warmup plus one decode token')
    if args.ctx > 32768:
        parser.error('this experiment is capped at 32768 context tokens')
    report = {'complete': False, 'diagnostic_instrumentation': True,
              'warning': 'CUDA events around registered Kindi kernels perturb launch timing; CuPy internals are not individually instrumented; use bench_prefill_chunks.py for speed comparisons',
              'phase': 'loading', 'tokens': args.tokens, 'ctx': args.ctx, 'chunk': args.chunk}
    write_json(args.output, report)
    try:
        from kindi.lightning import Engine
        from kindi.lightning_tokenizer import Tok
        engine = Engine(args.model, ctx=args.ctx)
        cp = engine.cp
        tokenizer = Tok(args.model)
        tokens, prompt = build_chat_tokens(tokenizer, args.tokens)
        warmup, _ = build_chat_tokens(tokenizer, warmup_tokens)
        report.update(prompt=prompt, numeric_profile=engine.numeric_profile,
                      source_sha256=engine.source_sha256, phase='warming', warmup_tokens=warmup_tokens)
        write_json(args.output, report)
        engine.reset()
        logits = engine.prefill(warmup, chunk_size=args.chunk)
        engine.forward(int(logits.argmax()))
        engine.synchronize()
        engine.reset()
        engine.synchronize()
        events = []
        original = dict(engine.ops.k)
        attributes = {name: {key: int(value) for key, value in kernel.attributes.items()}
                      for name, kernel in original.items()}

        def instrument(name, kernel):
            def run(grid, block, kernel_args, **kwargs):
                label = kernel_label(name, kernel_args)
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                kernel(grid, block, kernel_args, **kwargs)
                end.record()
                events.append((label, start, end))
            return run

        for name, kernel in original.items():
            engine.ops.k[name] = instrument(name, kernel)
        report['phase'] = 'profiling'
        write_json(args.output, report)
        start = time.perf_counter()
        try:
            engine.prefill(tokens, chunk_size=args.chunk)
            engine.synchronize()
            elapsed = time.perf_counter() - start
        finally:
            engine.ops.k.update(original)
        totals, counts = collections.defaultdict(float), collections.Counter()
        for label, first, last in events:
            totals[label] += float(cp.cuda.get_elapsed_time(first, last))
            counts[label] += 1
        report.update(complete=True, phase='complete', wall_s=elapsed,
                      gpu_event_sum_s=sum(totals.values()) / 1000.0,
                      event_pairs=len(events), kernel_attributes=attributes,
                      kernels=[{'name': name, 'total_ms': value, 'count': counts[name],
                                'per_call_ms': value / counts[name]}
                               for name, value in sorted(totals.items(), key=lambda item: -item[1])])
        write_json(args.output, report)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        report.update(phase='failed', error=type(exc).__name__ + ': ' + str(exc))
        write_json(args.output, report)
        raise


if __name__ == '__main__':
    main()
