"""Reusable packed linear, routing and selected-expert CUDA operations."""
import os
import hashlib
import numpy as np


class CudaOps(object):
    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.profile=os.environ.get('KINDI_LIGHTNING_KERNELS','optimized')
        if self.profile not in ('baseline','optimized'):raise ValueError('unknown Lightning numerical profile')
        code = open(os.path.join(os.path.dirname(__file__), 'lightning.cu'), encoding='utf-8').read()
        code += '\n'+open(os.path.join(os.path.dirname(__file__),'lightning_batch.cu'),encoding='utf-8').read()
        code += '\n'+open(os.path.join(os.path.dirname(__file__),'lightning_kv.cu'),encoding='utf-8').read()
        self.code_hash=hashlib.sha256(code.encode('utf-8')).hexdigest()
        self.module = cp.RawModule(code=code, options=('--std=c++11',))
        self.k = {name:self.module.get_function(name) for name in (
            'packed_mv','q2_mv','q8_mv','embedding','rms','relu2','residual','route','combine',
            'causal_conv','ssm_step','attention','embedding_batch','rms_batch','route_batch','expert_histogram',
            'expert_offsets','expert_scatter','packed_mm','grouped_mv','combine_batch','conv_chunk','ssm_chunk','attention_tiled',
            'unpack_dense','unpack_jobs','gather_jobs','scatter_jobs','grouped_mv4','q2_tile8','q2_tile16',
            'kv_store_fp16','kv_store_q8','attention_fp16','attention_q8','kv_decode_fp16','kv_decode_q8',
            'attention_warp','attention_fp16_warp','attention_q8_warp',
            'attention_tiles_fp32','attention_tiles_fp16','attention_tiles_q8','attention_merge_tiles')}
        self.null = np.uint64(0)

    def linear(self, packed, x, rows, cols, kind, ids=None, selected=1, per_input=False):
        y = self.cp.empty((selected, rows), self.cp.float32)
        if self.profile=='optimized' and kind in (8,42):
            self.k['q2_mv' if kind==42 else 'q8_mv'](((rows+3)//4,selected),(128,),
                (packed,x,y,ids if ids is not None else self.null,np.int32(rows),np.int32(cols),np.int32(selected),np.int32(per_input)))
            return y[0] if selected==1 else y
        self.k['packed_mv'](((rows+3)//4, selected),(128,),
            (packed,x,y,ids if ids is not None else self.null,np.int32(rows),np.int32(cols),
             np.int32(kind),np.int32(selected),np.int32(per_input)))
        return y[0] if selected == 1 else y

    def norm(self, x, w, eps=1e-5, width=None, gate=None):
        width = int(width or x.size)
        if x.size % width or w.size != x.size:
            raise ValueError('normalization shape mismatch')
        y = self.cp.empty_like(x)
        self.k['rms']((x.size//width,),(256,),
            (x,w,y,np.int32(width),np.int32(x.size//width),np.float32(eps),gate if gate is not None else self.null))
        return y

    def activate(self, x):
        self.k['relu2'](((x.size+255)//256,),(256,),(x,np.int32(x.size)))
        return x

    def add(self, x, y):
        self.k['residual'](((x.size+255)//256,),(256,),(x,y,np.int32(x.size)))

    def router(self, logits, bias, topk, scale):
        if not 1 <= topk <= logits.size or logits.size != bias.size:
            raise ValueError('invalid routing dimensions')
        ids=self.cp.empty(topk,self.cp.int32)
        weights=self.cp.empty(topk,self.cp.float32)
        if self.profile=='optimized' and logits.size<=256:
            threads=1
            while threads<logits.size:threads*=2
            self.k['route_batch']((1,),(threads,),(logits,bias,ids,weights,np.int32(1),np.int32(logits.size),np.int32(topk),np.float32(scale)))
            return ids,weights
        self.k['route']((1,),(1,),(logits,bias,ids,weights,np.int32(logits.size),np.int32(topk),np.float32(scale)))
        return ids, weights

    def combine(self, expert, weights, shared):
        topk,width=expert.shape
        y=self.cp.empty(width,self.cp.float32)
        self.k['combine'](((width+255)//256,),(256,),
            (expert,weights,shared,y,np.int32(width),np.int32(topk)))
        return y
