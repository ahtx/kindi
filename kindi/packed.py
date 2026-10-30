"""Validated resident GGUF tensors and independent CPU decoders.

Q2_0: upstream GGML type 42, fp16 scale followed by 16 bytes, 64 values.
The loader does not reinterpret Q2_K/TQ2_0 as Q2_0 or requantize weights.
"""
import os
import hashlib
import numpy as np
from .gguf import GGUF

FORMATS = {0: (1, 4), 8: (32, 34), 42: (64, 18)}


class Tensor(object):
    def __init__(self, name, dims, kind, offset, data_start):
        if kind not in FORMATS:
            raise ValueError('unsupported tensor type %s: %s' % (kind, name))
        if not dims or any(int(d) <= 0 for d in dims):
            raise ValueError('invalid tensor dimensions: ' + name)
        self.name, self.dims, self.kind = name, tuple(dims), int(kind)
        self.shape = tuple(reversed(self.dims))
        self.cols = int(dims[0])
        block, size = FORMATS[kind]
        if self.cols % block:
            raise ValueError('row is not quant-block aligned: ' + name)
        self.elements = 1
        for d in dims:self.elements *= int(d)
        self.nbytes = self.elements // block * size
        self.offset = int(data_start + offset)
        self.row_bytes = self.cols // block * size


class PackedFile(object):
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.gguf = GGUF(self.path)
        self.tensors = {}
        file_size = os.path.getsize(self.path)
        ranges = []
        for name, dims, kind, offset in self.gguf.tensors:
            if name in self.tensors:
                raise ValueError('duplicate tensor: ' + name)
            t = Tensor(name, dims, kind, offset, self.gguf.data_start)
            if offset < 0 or t.offset + t.nbytes > file_size:
                raise ValueError('tensor outside file: ' + name)
            ranges.append((t.offset, t.offset + t.nbytes, name))
            self.tensors[name] = t
        ranges.sort()
        for a, b in zip(ranges, ranges[1:]):
            if a[1] > b[0]:
                raise ValueError('overlapping tensors: %s / %s' % (a[2], b[2]))
        self.resident = {}

    def raw(self, name):
        t = self.tensors[name]
        return np.memmap(self.path, mode='r', dtype=np.uint8, offset=t.offset, shape=(t.nbytes,))

    def cpu(self, name, expert=None):
        t = self.tensors[name]
        raw = self.raw(name)
        shape = t.shape
        if expert is not None:
            if len(shape) != 3 or not 0 <= expert < shape[0]:
                raise ValueError('invalid expert index')
            size = t.nbytes // shape[0]
            raw = raw[expert * size:(expert + 1) * size]
            shape = shape[1:]
        return dequantize(raw, t.kind).reshape(shape)

    def load_gpu(self, cp, reserve_bytes=512 * 1024**2):
        free, total = cp.cuda.runtime.memGetInfo()
        needed = sum(t.nbytes for t in self.tensors.values())
        if needed + reserve_bytes > free:
            raise MemoryError('packed weights need %d bytes plus %d reserve; free %d' % (needed, reserve_bytes, free))
        try:
            for name, t in self.tensors.items():
                a = cp.asarray(self.raw(name))
                self.resident[name] = a.view(cp.float32).reshape(t.shape) if t.kind == 0 else a
        except Exception:
            self.resident.clear()
            raise
        return needed

    def source_hash(self):
        h=hashlib.sha256()
        with open(self.path,'rb') as f:
            for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
        return h.hexdigest()


def dequantize(raw, kind):
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    if kind == 0:
        return raw.view('<f4').copy()
    block, size = FORMATS[kind]
    b = raw.reshape(-1, size)
    scale = b[:, :2].copy().view('<f2').astype(np.float32)
    if kind == 8:
        codes = b[:, 2:].view(np.int8).astype(np.float32)
    elif kind == 42:
        q = b[:, 2:]
        codes = np.stack([(q >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(-1, 64).astype(np.float32) - 1
    else:
        raise ValueError('unsupported tensor type')
    return (codes * scale).reshape(-1)
