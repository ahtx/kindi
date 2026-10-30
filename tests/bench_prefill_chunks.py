"""Single-engine chunk128/chunk256 ABBA benchmark with exact state comparison.

Full prompt+32-token warmups precede ABBA. Snapshots, host logits copies and JSON
writes are outside timings. Python 3.6/CuPy 9.6 and Python 3.12/CuPy 13 compatible.
Imports and --help do not initialize CUDA.
"""
import argparse
import ctypes
import datetime
import os
import statistics
import threading
import time

try:
    from .profile_lightning import build_chat_tokens, write_json
except ImportError:
    from profile_lightning import build_chat_tokens, write_json


class NvmlTelemetry(object):
    """Read-only NVML sampling of the CUDA device, selected by PCI bus ID."""
    def __init__(self, pci_bus_id, interval=0.2):
        self.rows, self.errors = [], []
        self.phase = 'starting'
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.interval = interval
        self.thread = None
        self.initialized = False
        self.pci_bus_id = pci_bus_id.decode('ascii') if isinstance(pci_bus_id, bytes) else str(pci_bus_id)
        paths = ['libnvidia-ml.so.1']
        loader = ctypes.CDLL
        if os.name == 'nt':
            loader = ctypes.WinDLL
            paths = [os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32', 'nvml.dll'),
                     os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'NVIDIA Corporation', 'NVSMI', 'nvml.dll'),
                     'nvml.dll']
        self.library = None
        failures = []
        for path in paths:
            try:
                self.library = loader(path)
                self.library_path = path
                break
            except OSError as exc:
                failures.append(str(exc))
        if self.library is None:
            raise RuntimeError('NVML unavailable: ' + '; '.join(failures))
        self._call('nvmlInit_v2')
        self.initialized = True
        try:
            self.handle = ctypes.c_void_p()
            self._call('nvmlDeviceGetHandleByPciBusId_v2', ctypes.c_char_p(self.pci_bus_id.encode('ascii')), ctypes.byref(self.handle))
            uuid = ctypes.create_string_buffer(96)
            self._call('nvmlDeviceGetUUID', self.handle, uuid, ctypes.c_uint(len(uuid)))
            self.uuid = uuid.value.decode('ascii')
            self.rows.append(self.sample())
        except BaseException:
            self.stop()
            raise

    def _call(self, name, *args):
        function = getattr(self.library, name)
        function.restype = ctypes.c_int
        status = function(*args)
        if status != 0:
            raise RuntimeError('%s returned NVML status %d' % (name, status))

    def scalar(self, name, *extra):
        value = ctypes.c_uint()
        try:
            self._call(name, self.handle, *extra, ctypes.byref(value))
            return int(value.value)
        except (RuntimeError, AttributeError) as exc:
            with self.lock:
                if str(exc) not in self.errors:
                    self.errors.append(str(exc))
            return None

    def sample(self):
        with self.lock:
            phase = self.phase
        return {'monotonic_s': time.perf_counter(),
                'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'phase': phase,
                'sm_clock_mhz': self.scalar('nvmlDeviceGetClockInfo', ctypes.c_uint(1)),
                'graphics_clock_mhz': self.scalar('nvmlDeviceGetClockInfo', ctypes.c_uint(0)),
                'memory_clock_mhz': self.scalar('nvmlDeviceGetClockInfo', ctypes.c_uint(2)),
                'temperature_c': self.scalar('nvmlDeviceGetTemperature', ctypes.c_uint(0)),
                'power_mw': self.scalar('nvmlDeviceGetPowerUsage')}

    def start(self):
        def worker():
            while not self.stop_event.wait(self.interval):
                try:
                    row = self.sample()
                    with self.lock:
                        self.rows.append(row)
                except Exception as exc:
                    with self.lock:
                        self.errors.append(type(exc).__name__ + ': ' + str(exc))
                    return
        self.thread = threading.Thread(target=worker)
        self.thread.daemon = True
        self.thread.start()

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase

    def export(self):
        with self.lock:
            return {'available': True, 'backend': 'read-only NVML ctypes', 'library': self.library_path,
                    'pci_bus_id': self.pci_bus_id, 'uuid': self.uuid,
                    'sample_interval_s': self.interval, 'rows': list(self.rows), 'errors': list(self.errors)}

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError('NVML sampling thread did not stop')
        if self.initialized:
            self._call('nvmlShutdown')
            self.initialized = False


def compare_array(expected, actual, path):
    import numpy as np
    result = {'path': path, 'passed': False, 'bytes_compared': 0}
    if not isinstance(expected, np.ndarray) or not isinstance(actual, np.ndarray):
        result['error'] = 'both values must be numpy arrays'
        return result
    result.update(dtype=actual.dtype.str, shape=list(actual.shape))
    if expected.dtype != actual.dtype or expected.shape != actual.shape:
        result.update(error='dtype or shape mismatch', expected_dtype=expected.dtype.str, expected_shape=list(expected.shape))
        return result
    if expected.dtype.kind in 'fc' and (not np.isfinite(expected).all() or not np.isfinite(actual).all()):
        result['error'] = 'non-finite floating-point value'
        return result
    if expected.dtype.kind not in 'biufc':
        result['error'] = 'unsupported array dtype for exact comparison'
        return result
    left = np.ascontiguousarray(expected).view(np.uint8).reshape(-1)
    right = np.ascontiguousarray(actual).view(np.uint8).reshape(-1)
    result['bytes_compared'] = int(left.size)
    result['passed'] = bool(np.array_equal(left, right))
    if not result['passed']:
        indices = np.flatnonzero(left != right)
        result.update(error='byte mismatch', different_bytes=int(indices.size), first_byte_offsets=indices[:8].tolist())
    return result


def compare_snapshot(expected, actual):
    differences, arrays = [], []
    required = {'version', 'fingerprint', 'position', 'tokens', 'states', 'histories', 'keys', 'values', 'logits'}
    if set(expected) != required or set(actual) != required:
        differences.append('snapshot field set mismatch or unknown fields')
    for key in ('version', 'fingerprint', 'position', 'tokens'):
        if key not in expected or key not in actual or type(expected[key]) is not type(actual[key]) or expected[key] != actual[key]:
            differences.append(key + ' mismatch')
    for label, snapshot in (('reference', expected), ('actual', actual)):
        if snapshot.get('position') != len(snapshot.get('tokens', [])):
            differences.append(label + ' position/token-count mismatch')
    for group in ('states', 'histories', 'keys', 'values'):
        left, right = expected.get(group), actual.get(group)
        if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
            differences.append(group + ' layer set mismatch')
            continue
        for layer in sorted(left):
            if group in ('keys', 'values'):
                if left[layer].shape[0] != expected['position'] or right[layer].shape[0] != actual['position']:
                    differences.append('%s/%s contains unconsumed or missing KV' % (group, layer))
            arrays.append(compare_array(left[layer], right[layer], '%s/%s' % (group, layer)))
    arrays.append(compare_array(expected.get('logits'), actual.get('logits'), 'logits'))
    return {'passed': not differences and all(item['passed'] for item in arrays),
            'metadata_differences': differences, 'arrays': arrays,
            'bytes_compared': sum(item['bytes_compared'] for item in arrays)}


def decode(engine, logits, count):
    ids, outputs, latencies = [], [], []
    start = time.perf_counter()
    for _ in range(count):
        tick = time.perf_counter()
        token = int(logits.argmax())
        ids.append(token)
        logits = engine.forward(token)
        outputs.append(logits)  # Retain GPU arrays; host copies happen after timing.
        engine.synchronize()
        latencies.append(time.perf_counter() - tick)
    return ids, outputs, latencies, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokens', type=int, default=1024)
    parser.add_argument('--ctx', type=int, default=32768)
    parser.add_argument('--telemetry-ms', type=int, default=200)
    args = parser.parse_args()
    generated = 32
    if args.tokens < 256 or args.tokens + generated > args.ctx or args.ctx > 32768:
        parser.error('require tokens>=256 and tokens+32<=ctx<=32768')
    if args.telemetry_ms < 100:
        parser.error('telemetry-ms must be at least 100')
    report = {'complete': False, 'passed': False, 'phase': 'loading', 'results': [], 'warmups': [],
              'order': [128, 256, 256, 128], 'ctx': args.ctx, 'prompt_tokens': args.tokens,
              'generated_tokens': generated,
              'timing_policy': 'wall clock including synchronization; model load, warmup, snapshots, host logits copies and comparisons excluded',
              'eos_policy': 'ignored for fixed-length benchmark; all 32 generated tokens are consumed',
              'load_policy': 'one Engine; existing model reused; no file-cache eviction',
              'comparison_policy': 'exact metadata/dtype/shape/finite floats/raw uint8 bytes against first chunk128 trial; no tolerance'}
    telemetry = None

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
        import numpy as np
        from kindi.lightning import Engine
        from kindi.lightning_tokenizer import Tok
        start = time.perf_counter()
        engine = Engine(args.model, ctx=args.ctx)
        report['model_load_s'] = time.perf_counter() - start
        tokens, prompt = build_chat_tokens(Tok(args.model), args.tokens)
        report.update(prompt=prompt, numeric_profile=engine.numeric_profile,
                      source_sha256=engine.source_sha256, memory=engine.memory)
        try:
            telemetry = NvmlTelemetry(engine.cp.cuda.Device().pci_bus_id, args.telemetry_ms / 1000.0)
            telemetry.start()
        except Exception as exc:
            report['telemetry'] = {'available': False, 'rows': [],
                                   'error': type(exc).__name__ + ': ' + str(exc)}
        for chunk in (128, 256):
            phase('warmup_chunk%d' % chunk)
            save()
            engine.reset()
            engine.synchronize()
            start = time.perf_counter()
            logits = engine.prefill(tokens, chunk_size=chunk)
            ids, outputs, _, _ = decode(engine, logits, generated)
            report['warmups'].append({'chunk': chunk, 'prompt_tokens': len(tokens),
                                      'generated_tokens': len(ids), 'seconds': time.perf_counter() - start})
            del outputs, logits
            engine.reset()
            engine.synchronize()
            save()
        reference = None
        for trial, chunk in enumerate(report['order']):
            engine.reset()
            engine.synchronize()
            phase('trial%d_prefill' % trial)
            result = {'trial': trial, 'chunk': chunk, 'prompt_tokens': len(tokens), 'complete': False}
            report['results'].append(result)
            save()
            begin = time.perf_counter()
            logits = engine.prefill(tokens, chunk_size=chunk)
            engine.synchronize()
            end = time.perf_counter()
            result.update(prefill_s=end - begin, prefill_tps=len(tokens) / (end - begin), prefill_interval=[begin, end])
            phase('trial%d_snapshot_prefill' % trial)
            save()
            prefill_state = engine.snapshot()
            phase('trial%d_decode' % trial)
            save()
            decode_begin = time.perf_counter()
            ids, outputs, latencies, decode_s = decode(engine, logits, generated)
            decode_end = time.perf_counter()
            result.update(generated_tokens=len(ids), decode_s=decode_s, decode_tps=generated / decode_s,
                          latency_p50_ms=statistics.median(latencies) * 1000.0,
                          latency_p95_ms=sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000.0,
                          ids=ids, decode_interval=[decode_begin, decode_end])
            phase('trial%d_comparison' % trial)
            save()
            final_state = engine.snapshot()
            decode_logits = np.stack([item.get() for item in outputs])
            del outputs, logits
            if reference is None:
                reference = (prefill_state, final_state, decode_logits, list(ids))
            comparisons = {'reference_trial': 0, 'reference_chunk': 128,
                           'prefill_snapshot': compare_snapshot(reference[0], prefill_state),
                           'final_snapshot': compare_snapshot(reference[1], final_state),
                           'decode_logits': compare_array(reference[2], decode_logits, 'decode_logits'),
                           'greedy_ids_identical': reference[3] == ids}
            exact = all(comparisons[key]['passed'] for key in ('prefill_snapshot', 'final_snapshot', 'decode_logits')) and comparisons['greedy_ids_identical']
            rows = telemetry.export()['rows'] if telemetry is not None else []
            samples = [row for row in rows if begin <= row['monotonic_s'] <= end or decode_begin <= row['monotonic_s'] <= decode_end]
            observed = {key: sorted(set(row[key] for row in samples if row[key] is not None)) or None
                        for key in ('sm_clock_mhz', 'graphics_clock_mhz', 'memory_clock_mhz', 'temperature_c')}
            result.update(observed_nvml=observed, timed_telemetry_samples=len(samples),
                          exact=exact, comparison_to_chunk128=comparisons, complete=True)
            save()
            print('trial=%d chunk=%d prefill=%.6fs decode=%.3f tok/s exact=%s' %
                  (trial, chunk, result['prefill_s'], result['decode_tps'], exact), flush=True)
            if not exact:
                raise AssertionError('strict chunk128 comparison failed in trial %d' % trial)
            del prefill_state, final_state, decode_logits
        report['summary'] = {str(chunk): {'trials': 2,
            'prefill_median_s': statistics.median(row['prefill_s'] for row in report['results'] if row['chunk'] == chunk),
            'decode_median_tps': statistics.median(row['decode_tps'] for row in report['results'] if row['chunk'] == chunk)}
            for chunk in (128, 256)}
        report['telemetry_complete'] = all(all(row['observed_nvml'][key] is not None
            for key in ('sm_clock_mhz', 'memory_clock_mhz', 'temperature_c')) for row in report['results'])
        report.update(complete=True, passed=True, phase='complete')
    except BaseException as exc:
        report.update(phase='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        if telemetry is not None:
            try:
                telemetry.stop()
            except Exception as exc:
                report.update(telemetry_shutdown_error=str(exc), telemetry_complete=False)
        save()


if __name__ == '__main__':
    main()
