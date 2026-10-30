"""ABBA tile8/tile16 prefill comparison at fixed chunk128.

Only one Engine is resident. Every new Engine gets a full prompt+32-token
warmup. Loading, warmup, unload, snapshots and comparisons are outside timings.
Validation-only mode uses two independent full prompts and short warmups; its
times are diagnostics, not a performance comparison. Imports and --help do not
initialize CUDA. Python 3.6 / CuPy 9.6 compatible.
"""
import argparse
import gc
import math
import os
import statistics
import time
import weakref

try:
    from .profile_lightning import build_chat_tokens, write_json
    from .bench_prefill_chunks import NvmlTelemetry, compare_array, compare_snapshot, decode
except ImportError:
    from profile_lightning import build_chat_tokens, write_json
    from bench_prefill_chunks import NvmlTelemetry, compare_array, compare_snapshot, decode


def profile_errors(profile, mode, kv, baseline=None):
    errors = []
    for key, expected in (('dense_prefill', mode), ('expert_prefill', mode),
                          ('kv', kv), ('state', 'fp32'), ('kernels', 'optimized')):
        if profile.get(key) != expected:
            errors.append('%s: expected %s, got %r' % (key, expected, profile.get(key)))
    if baseline is not None:
        if set(profile) != set(baseline):
            errors.append('numeric_profile field set changed')
        for key in set(profile) | set(baseline):
            if key not in ('dense_prefill', 'expert_prefill') and profile.get(key) != baseline.get(key):
                errors.append('unexpected numeric_profile change: ' + key)
    return errors


def compare_variant_snapshot(expected, actual, expected_fingerprint, actual_fingerprint, different_profile, kv):
    """Validate profile identity separately; never modify either snapshot."""
    result = compare_snapshot(expected, actual)
    fingerprint_valid = (
        isinstance(expected_fingerprint, str) and isinstance(actual_fingerprint, str)
        and len(expected_fingerprint) == 64 and len(actual_fingerprint) == 64
        and expected.get('fingerprint') == expected_fingerprint
        and actual.get('fingerprint') == actual_fingerprint
        and (expected_fingerprint != actual_fingerprint) == different_profile)
    result['fingerprint_check'] = {
        'reference': expected.get('fingerprint'), 'actual': actual.get('fingerprint'),
        'expected_difference': different_profile, 'passed': fingerprint_valid}
    # The common validator still checks every field and array. A fingerprint
    # difference is allowed only after the independently recorded profiles pass.
    result['metadata_differences'] = [item for item in result['metadata_differences']
                                      if item != 'fingerprint mismatch']
    scale_checks = []
    if kv == 'q8_0':
        import numpy as np
        for label, snapshot in (('reference', expected), ('actual', actual)):
            for group in ('keys', 'values'):
                for layer, values in snapshot.get(group, {}).items():
                    entry = {'path': '%s/%s/%s/q8_scales' % (label, group, layer), 'passed': False}
                    if values.dtype == np.uint8 and values.shape[-1] % 34 == 0:
                        blocks = np.ascontiguousarray(values).reshape(-1, 34)
                        scales = blocks[:, :2].copy().view(np.float16)
                        entry['passed'] = bool(np.isfinite(scales).all())
                    if not entry['passed']:
                        entry['error'] = 'invalid q8 block layout or non-finite fp16 scale'
                    scale_checks.append(entry)
    result['q8_scale_checks'] = scale_checks
    result['passed'] = (not result['metadata_differences'] and fingerprint_valid
                        and all(item['passed'] for item in result['arrays'])
                        and all(item['passed'] for item in scale_checks))
    return result


def release_memory(cp, reference):
    """Caller must drop its Engine reference before invoking this function."""
    cp.cuda.Stream.null.synchronize()
    collected = gc.collect()
    if reference() is not None:
        import types
        holders=[]
        for holder in gc.get_referrers(reference()):
            entry={'type':type(holder).__name__}
            if isinstance(holder,types.FrameType):
                entry.update(function=holder.f_code.co_name,file=holder.f_code.co_filename,line=holder.f_lineno)
            elif isinstance(holder,dict):
                entry['reference_keys']=[str(key) for key,value in holder.items() if value is reference()]
            holders.append(entry)
        raise RuntimeError('previous Engine is still referenced; refusing a second Engine; holders=%r'%holders)
    cp.cuda.Stream.null.synchronize()
    pool = cp.get_default_memory_pool()
    before = int(pool.total_bytes())
    pool.free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    free, total = cp.cuda.runtime.memGetInfo()
    return {'engine_collected': True, 'gc_objects': collected, 'pool_before_bytes': before,
            'pool_after_bytes': int(pool.total_bytes()), 'pool_used_bytes': int(pool.used_bytes()),
            'device_free_bytes': int(free), 'device_total_bytes': int(total)}


def timed_trial(engine, tokens, chunk, generated, result, phase, save, progress=False):
    import numpy as np
    engine.reset()
    engine.synchronize()
    phase('trial%d_prefill' % result['trial'])
    save()
    options={'chunk_size':chunk}
    if progress:
        last=[time.perf_counter()]
        def report_progress(done,total):
            now=time.perf_counter()
            if now-last[0]>=30 or done==total:
                last[0]=now
                print('validation progress mode=%s tokens=%d/%d seconds=%.2f'%
                      (result['mode'],done,total,now-begin),flush=True)
        options['progress']=report_progress
    begin = time.perf_counter()
    logits = engine.prefill(tokens, **options)
    engine.synchronize()
    end = time.perf_counter()
    elapsed = end - begin
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError('invalid prefill duration')
    result.update(prefill_s=elapsed, prefill_tps=len(tokens) / elapsed, prefill_interval=[begin, end])
    phase('trial%d_snapshot_prefill' % result['trial'])
    save()
    prefill_state = engine.snapshot()
    result['prefill_coverage'] = {
        'expected_position': len(tokens), 'actual_position': prefill_state.get('position'),
        'tokens_identical': prefill_state.get('tokens') == tokens}
    if prefill_state.get('position') != len(tokens) or prefill_state.get('tokens') != tokens:
        raise AssertionError('prefill did not consume the complete requested prompt')
    phase('trial%d_decode' % result['trial'])
    save()
    decode_begin = time.perf_counter()
    ids, outputs, latencies, decode_s = decode(engine, logits, generated)
    decode_end = time.perf_counter()
    result['decode_coverage'] = {'expected_tokens': 32, 'actual_tokens': len(ids),
                               'logit_vectors': len(outputs), 'latencies': len(latencies)}
    if generated != 32 or len(ids) != 32 or len(outputs) != 32 or len(latencies) != 32:
        raise AssertionError('decode must produce exactly 32 IDs, logit vectors and latency samples')
    if not all(math.isfinite(value) and value > 0 for value in [decode_s] + latencies):
        raise RuntimeError('invalid decode duration')
    result.update(generated_tokens=len(ids), decode_s=decode_s, decode_tps=generated / decode_s,
                  latency_p50_ms=statistics.median(latencies) * 1000.0,
                  latency_p95_ms=sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000.0,
                  ids=ids, decode_interval=[decode_begin, decode_end])
    phase('trial%d_comparison' % result['trial'])
    save()
    final_state = engine.snapshot()
    result['final_coverage'] = {
        'expected_position': len(tokens) + 32, 'actual_position': final_state.get('position'),
        'tokens_identical': final_state.get('tokens') == tokens + ids}
    if final_state.get('position') != len(tokens) + 32 or final_state.get('tokens') != tokens + ids:
        raise AssertionError('final state did not consume the complete prompt plus all 32 generated IDs')
    decode_logits = np.stack([value.get() for value in outputs])
    # Only CPU arrays escape this function; logits/outputs cannot retain VRAM
    # after the caller unloads Engine.
    return prefill_state, final_state, decode_logits, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokens', type=int, choices=(1024, 4096, 32736), default=1024)
    parser.add_argument('--ctx', type=int, default=32768)
    parser.add_argument('--kv', choices=('fp32', 'fp16', 'q8_0'), default='fp32')
    parser.add_argument('--chunk', type=int, choices=(128,), default=128)
    parser.add_argument('--telemetry-ms', type=int, default=200)
    parser.add_argument('--validation-only', action='store_true',
                        help='tile8/tile16 once each, 256-token warmups; no performance comparison')
    args = parser.parse_args()
    generated = 32
    if args.tokens + generated > args.ctx or args.ctx > 32768:
        parser.error('require tokens+32<=ctx<=32768')
    if args.telemetry_ms < 100:
        parser.error('telemetry-ms must be at least 100')
    if args.tokens == 32736 and not args.validation_only:
        parser.error('32736-token acceptance requires --validation-only; regular ABBA supports 1024/4096')
    order = ['tile8', 'tile16'] if args.validation_only else ['tile8', 'tile16', 'tile16', 'tile8']
    report = {'complete': False, 'passed': False, 'phase': 'starting', 'results': [],
              'instances': [], 'unloads': [], 'order': order,
              'validation_only': args.validation_only, 'performance_comparison': not args.validation_only,
              'ctx': args.ctx, 'chunk': args.chunk, 'kv': args.kv,
              'prompt_tokens': args.tokens, 'generated_tokens': generated,
              'timing_policy': 'wall clock with synchronization; load/warmup/unload/snapshots/copies/comparisons excluded',
              'comparison_policy': 'exact state bytes, finite floats, all decode logits and greedy IDs against first tile8; profile fingerprints separately validated',
              'load_policy': 'one Engine at a time; reuse adjacent equal modes; 256-token warmup in validation-only, otherwise full prompt; 32 warmup decode tokens per new instance',
              'eos_policy': 'ignored for fixed-length test; all 32 generated tokens consumed'}
    engine = None
    cp = None
    mode_loaded = None
    telemetry = None
    baseline_profile = None
    baseline_source = None
    reference = None
    fingerprints = {}
    env_names = ('KINDI_LIGHTNING_DENSE_PREFILL', 'KINDI_LIGHTNING_EXPERT_PREFILL')
    old_environment = {name: os.environ.get(name) for name in env_names}

    def phase(name):
        report['phase'] = name
        if telemetry is not None:
            telemetry.set_phase(name)

    def save():
        if telemetry is not None:
            report['telemetry'] = telemetry.export()
        write_json(args.output, report)

    save()
    try:
        # CuPy 9.6 retains optional pytest import exceptions and their import
        # stack on Python 3.6. Import before constructing any Engine so those
        # cached traceback frames cannot retain its 9 GiB of resident weights.
        import cupy as cp
        from kindi.lightning import Engine
        from kindi.lightning_tokenizer import Tok
        tokenizer = Tok(args.model)
        tokens, report['prompt'] = build_chat_tokens(tokenizer, args.tokens)
        warmup_tokens = build_chat_tokens(tokenizer, 256)[0] if args.validation_only else tokens
        if args.validation_only:
            report['timing_warning'] = 'short warmups and one trial per mode; durations are validation diagnostics, not evidence of performance gain'
        for trial, mode in enumerate(report['order']):
            if mode != mode_loaded:
                if engine is not None:
                    phase('unload_' + mode_loaded)
                    save()
                    engine.synchronize()
                    previous = weakref.ref(engine)
                    engine = None
                    report['unloads'].append(release_memory(cp, previous))
                    mode_loaded = None
                    save()
                for name in env_names:
                    os.environ[name] = mode
                phase('load_' + mode)
                save()
                start = time.perf_counter()
                engine = Engine(args.model, ctx=args.ctx, kv_dtype=args.kv)
                loaded_s = time.perf_counter() - start
                cp = engine.cp
                profile = dict(engine.numeric_profile)
                errors = profile_errors(profile, mode, args.kv, baseline_profile)
                instance = {'instance': len(report['instances']), 'mode': mode, 'load_s': loaded_s,
                            'numeric_profile': profile, 'fingerprint': engine.fingerprint,
                            'source_sha256': engine.source_sha256, 'memory': dict(engine.memory),
                            'profile_errors': errors, 'warmup_complete': False}
                report['instances'].append(instance)
                save()
                if errors:
                    raise AssertionError('incorrect numeric_profile: ' + '; '.join(errors))
                if baseline_profile is None:
                    baseline_profile, baseline_source = profile, engine.source_sha256
                if engine.source_sha256 != baseline_source:
                    raise AssertionError('model content changed between Engine instances')
                if mode in fingerprints and fingerprints[mode] != engine.fingerprint:
                    raise AssertionError('same mode produced a different fingerprint')
                fingerprints[mode] = engine.fingerprint
                if len(fingerprints) == 2 and fingerprints['tile8'] == fingerprints['tile16']:
                    raise AssertionError('tile8/tile16 fingerprints unexpectedly match')
                if telemetry is None:
                    try:
                        telemetry = NvmlTelemetry(cp.cuda.Device().pci_bus_id, args.telemetry_ms / 1000.0)
                        telemetry.start()
                    except Exception as exc:
                        telemetry = None
                        report['telemetry'] = {'available': False, 'rows': [],
                                               'error': type(exc).__name__ + ': ' + str(exc)}
                phase('warmup_' + mode)
                save()
                engine.reset()
                engine.synchronize()
                start = time.perf_counter()
                logits = engine.prefill(warmup_tokens, chunk_size=args.chunk)
                ids, outputs, _, _ = decode(engine, logits, generated)
                instance.update(warmup_complete=True, warmup_s=time.perf_counter() - start,
                                warmup_prompt_tokens=len(warmup_tokens), warmup_decode_tokens=len(ids))
                del outputs, logits
                engine.reset()
                engine.synchronize()
                mode_loaded = mode
                save()
            result = {'trial': trial, 'mode': mode, 'chunk': args.chunk, 'kv': args.kv,
                      'instance': instance['instance'], 'prompt_tokens': len(tokens), 'complete': False,
                      'numeric_profile': dict(engine.numeric_profile), 'fingerprint': engine.fingerprint}
            report['results'].append(result)
            save()
            observed = timed_trial(engine, tokens, args.chunk, generated, result, phase, save,
                                   progress=args.validation_only and len(tokens)>4096)
            if reference is None:
                reference = observed
            comparisons = {'reference_trial': 0, 'reference_mode': 'tile8',
                'prefill_snapshot': compare_variant_snapshot(reference[0], observed[0], fingerprints['tile8'], engine.fingerprint, mode != 'tile8', args.kv),
                'final_snapshot': compare_variant_snapshot(reference[1], observed[1], fingerprints['tile8'], engine.fingerprint, mode != 'tile8', args.kv),
                'decode_logits': compare_array(reference[2], observed[2], 'decode_logits'),
                'greedy_ids_identical': reference[3] == observed[3]}
            exact = (all(comparisons[key]['passed'] for key in ('prefill_snapshot', 'final_snapshot', 'decode_logits'))
                     and comparisons['greedy_ids_identical'])
            rows = telemetry.export()['rows'] if telemetry is not None else []
            windows = (result['prefill_interval'], result['decode_interval'])
            samples = [row for row in rows if any(begin <= row['monotonic_s'] <= end for begin, end in windows)]
            metrics = {key: sorted(set(row[key] for row in samples if row[key] is not None)) or None
                       for key in ('sm_clock_mhz', 'graphics_clock_mhz', 'memory_clock_mhz', 'temperature_c')}
            result.update(exact=exact, comparison_to_tile8=comparisons, observed_nvml=metrics,
                          timed_telemetry_samples=len(samples), complete=True)
            save()
            print('%s trial=%d mode=%s kv=%s prefill=%.6fs decode=%.3f tok/s exact=%s' %
                  ('validation' if args.validation_only else 'benchmark', trial, mode, args.kv,
                   result['prefill_s'], result['decode_tps'], exact), flush=True)
            if not exact:
                raise AssertionError('strict tile8 comparison failed in trial %d' % trial)
            del observed
        if args.validation_only:
            report['summary'] = {'validation_trials': len(report['results']),
                                 'exact': all(row['exact'] for row in report['results'])}
        else:
            report['summary'] = {mode: {'trials': 2,
                'prefill_median_s': statistics.median(row['prefill_s'] for row in report['results'] if row['mode'] == mode),
                'decode_median_tps': statistics.median(row['decode_tps'] for row in report['results'] if row['mode'] == mode)}
                for mode in ('tile8', 'tile16')}
        report['telemetry_complete'] = all(all(row['observed_nvml'][key] is not None
            for key in ('sm_clock_mhz', 'memory_clock_mhz', 'temperature_c')) for row in report['results'])
        report.update(complete=True, passed=True, phase='complete')
    except BaseException as exc:
        report.update(phase='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        if engine is not None:
            try:
                engine.synchronize()
                previous = weakref.ref(engine)
                engine = None
                report['unloads'].append(release_memory(cp, previous))
            except Exception as exc:
                report.update(complete=False, passed=False, unload_error=str(exc))
        if telemetry is not None:
            try:
                telemetry.stop()
            except Exception as exc:
                report.update(telemetry_shutdown_error=str(exc), telemetry_complete=False)
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        save()
    if not report['passed']:
        raise RuntimeError('benchmark did not pass; inspect partial JSON')


if __name__ == '__main__':
    main()
