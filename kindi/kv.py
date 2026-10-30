"""Explicit KV storage contracts; arithmetic remains FP32 on every GPU."""
import os
import numpy as np

MODES=('fp32','fp16','q8_0')

class KVCacheOps(object):
    def __init__(self,ops,mode,read_mode=None):
        if mode not in MODES:raise ValueError('KV format must be fp32, fp16 or q8_0')
        self.ops=ops;self.cp=ops.cp;self.mode=mode;self.error=self.cp.zeros(1,self.cp.int32)
        self.read_mode=read_mode or os.environ.get('KINDI_LIGHTNING_KV_READ','auto')
        if self.read_mode not in ('direct','expanded','auto'):raise ValueError('KV read mode must be direct, expanded or auto')
        self.attention_mode=os.environ.get('KINDI_LIGHTNING_ATTENTION','tiled')
        if self.attention_mode not in ('tiled','warp','split'):raise ValueError('attention mode must be tiled, warp or split')
        self.split_workspace_bytes=int(os.environ.get('KINDI_LIGHTNING_ATTENTION_WORKSPACE_MB','128'))*1024**2
        if not 1024**2<=self.split_workspace_bytes<=512*1024**2:raise ValueError('attention workspace must be 1..512 MiB')
        self.capacity=0;self.scratch=None;self.split_scratch=None

    def row_bytes(self,heads,dim):
        if self.mode=='q8_0':
            if dim%32:raise ValueError('Q8_0 KV requires head dimension divisible by 32')
            return heads*(dim//32)*34
        return heads*dim*(4 if self.mode=='fp32' else 2)

    def allocate(self,ctx,heads,dim):
        if min(ctx,heads,dim)<1:raise ValueError('invalid KV dimensions')
        self.capacity=max(ctx,self.capacity)
        self.row_bytes(heads,dim)
        shape=(ctx,heads,(dim//32)*34 if self.mode=='q8_0' else dim)
        dtype={'fp32':self.cp.float32,'fp16':self.cp.float16,'q8_0':self.cp.uint8}[self.mode]
        return self.cp.empty(shape,dtype)

    def store(self,dst,src,position,heads,dim):
        self.validate_storage(dst,heads,dim)
        if not isinstance(src,self.cp.ndarray):raise ValueError('KV source must be a GPU array')
        count=src.size;tokens=count//(heads*dim)
        if tokens<1 or count!=tokens*heads*dim or position<0 or position+tokens>dst.shape[0]:raise ValueError('invalid KV write range')
        if src.dtype!=self.cp.float32 or not src.flags.c_contiguous:raise ValueError('KV source must be contiguous FP32')
        target=dst[position:position+tokens]
        if self.mode=='fp32':target[:]=src.reshape(tokens,heads,dim)
        elif self.mode=='fp16':
            self.ops.k['kv_store_fp16'](((count+255)//256,),(256,),(src,target,np.int32(count),self.error))
        else:
            self.ops.k['kv_store_q8']((((count//32)+3)//4,),(128,),(src,target,np.int32(count//32),self.error))

    def attention(self,q,k,v,past,tokens,qheads,kvheads,dim):
        self.validate_storage(k,kvheads,dim);self.validate_storage(v,kvheads,dim)
        if min(tokens,qheads,kvheads,dim)<1 or dim>128 or qheads%kvheads or past<0 or past+tokens>min(k.shape[0],v.shape[0]):raise ValueError('invalid attention dimensions or range')
        if not isinstance(q,self.cp.ndarray) or q.dtype!=self.cp.float32 or not q.flags.c_contiguous or q.size!=tokens*qheads*dim:raise ValueError('invalid attention query')
        out=self.cp.empty((tokens,qheads*dim),self.cp.float32)
        name={'fp32':'attention_tiled','fp16':'attention_fp16','q8_0':'attention_q8'}[self.mode]
        length=past+tokens
        expand_auto=(self.mode=='q8_0' and (tokens>=8 or length>=4096)) or (self.mode=='fp16' and self.attention_mode=='split' and tokens>=8 and length>=512)
        if self.mode!='fp32' and (self.read_mode=='expanded' or (self.read_mode=='auto' and expand_auto)):
            # One reusable layer-sized scratch, not an expanded copy of every layer's KV.
            self.capacity=max(self.capacity,k.shape[0])
            if self.scratch is None or self.scratch[0].shape[0]<self.capacity or self.scratch[0].shape[1:]!=(kvheads,dim):
                self.scratch=(self.cp.empty((self.capacity,kvheads,dim),self.cp.float32),self.cp.empty((self.capacity,kvheads,dim),self.cp.float32))
            count=length*kvheads*dim;decode=self.ops.k['kv_decode_fp16' if self.mode=='fp16' else 'kv_decode_q8']
            decode(((count+255)//256,),(256,),(k,self.scratch[0],np.int32(count)))
            decode(((count+255)//256,),(256,),(v,self.scratch[1],np.int32(count)))
            k,v=self.scratch;name='attention_tiled'
        if self.attention_mode=='split' and length>=512:
            if self.split_scratch is None:self.split_scratch=self.cp.empty(self.split_workspace_bytes//4,self.cp.float32)
            tiles=(length+127)//128;per_query=qheads*tiles*(dim+2)
            batch=min(tokens,self.split_scratch.size//per_query)
            if batch<1:raise MemoryError('attention tile workspace too small for this geometry')
            dtype='fp32' if name=='attention_tiled' else ('fp16' if self.mode=='fp16' else 'q8')
            query=q.reshape(tokens,qheads*dim)
            for start in range(0,tokens,batch):
                count=min(batch,tokens-start);position=past+start
                self.ops.k['attention_tiles_'+dtype]((qheads,count,tiles),(128,),
                    (query[start:start+count],k,v,self.split_scratch,np.int32(count),np.int32(position),np.int32(qheads),np.int32(kvheads),np.int32(dim),np.int32(tiles)))
                self.ops.k['attention_merge_tiles']((qheads,count),(128,),
                    (self.split_scratch,out[start:start+count],np.int32(count),np.int32(position),np.int32(qheads),np.int32(dim),np.int32(tiles)))
            return out
        if self.attention_mode in ('warp','split'):name='attention_warp' if name=='attention_tiled' else name+'_warp'
        self.ops.k[name]((qheads,tokens),(128,),(q,k,v,out,np.int32(tokens),np.int32(past),np.int32(qheads),np.int32(kvheads),np.int32(dim)))
        return out

    def validate_storage(self,array,heads,dim):
        if min(heads,dim)<1:raise ValueError('invalid KV geometry')
        self.row_bytes(heads,dim)
        tail=(heads,(dim//32)*34 if self.mode=='q8_0' else dim)
        dtype={'fp32':self.cp.float32,'fp16':self.cp.float16,'q8_0':self.cp.uint8}[self.mode]
        if not isinstance(array,self.cp.ndarray) or array.ndim!=3 or array.shape[1:]!=tail or array.dtype!=dtype or not array.flags.c_contiguous:raise ValueError('invalid KV storage layout')

    def check(self):
        if self.mode!='fp32' and int(self.error.get()[0]):raise FloatingPointError('KV storage rejected non-finite or out-of-range values')

    def reset_error(self):self.error.fill(0)
