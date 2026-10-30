"""Native Kindi CUDA adapter for Nota Nemotron Lightning, sequential correctness path.

NoPE attention, sigmoid corrected top-k, ReLU2 experts, FP32 Mamba2 states.
Weights stay packed on the selected GPU. No model computation falls back to CPU.
"""
import hashlib
import json
import os
import numpy as np
from .packed import PackedFile
from .moe import CudaOps


class ModelSpec(object):
    def __init__(self, gguf):
        kv=gguf.kv
        if kv.get('general.architecture')!='nemotron_h_moe':
            raise ValueError('Lightning adapter requires nemotron_h_moe')
        p='nemotron_h_moe.'
        def get(key):
            if p+key not in kv: raise ValueError('missing model metadata: '+p+key)
            return kv[p+key]
        self.width=int(get('embedding_length'))
        self.layers=int(get('block_count'))
        ff=get('feed_forward_length'); kvh=get('attention.head_count_kv')
        if not isinstance(ff,list) or not isinstance(kvh,list) or len(ff)!=self.layers or len(kvh)!=self.layers:
            raise ValueError('explicit layer schedule required')
        self.blocks=['moe' if ff[i] else ('attention' if kvh[i] else 'mamba') for i in range(self.layers)]
        self.qheads=int(get('attention.head_count'))
        self.kvheads=max(kvh)
        self.hd=int(get('attention.key_length'))
        self.eps=float(get('attention.layer_norm_rms_epsilon'))
        self.experts=int(get('expert_count'));self.topk=int(get('expert_used_count'))
        self.scale=float(get('expert_weights_scale'))
        self.routed_width=int(get('expert_feed_forward_length'));self.shared_width=int(get('expert_shared_feed_forward_length'))
        self.inner=int(get('ssm.inner_size'));self.heads=int(get('ssm.time_step_rank'))
        self.state_size=int(get('ssm.state_size'));self.groups=int(get('ssm.group_count'))
        self.conv=int(get('ssm.conv_kernel'))
        if min(self.width,self.layers,self.qheads,self.kvheads,self.hd,self.inner,self.heads,self.state_size,self.groups,self.routed_width,self.shared_width)<=0:
            raise ValueError('model dimensions must be positive')
        if not 1<=self.topk<=self.experts or not np.isfinite(self.scale) or not np.isfinite(self.eps) or self.eps<=0:
            raise ValueError('invalid expert or normalization metadata')
        if any(v not in (0,self.kvheads) for v in kvh) or any(v not in (0,self.routed_width) for v in ff):
            raise ValueError('heterogeneous attention/expert widths require another adapter')
        self.head_dim=self.inner//self.heads
        self.conv_dim=self.inner+2*self.groups*self.state_size
        self.vocab=len(kv['tokenizer.ggml.tokens'])
        # Conservative checkpoint contract: do not silently run a different Nemotron variant.
        if self.inner%self.heads or self.heads%self.groups or self.inner%self.groups or self.conv<2:
            raise ValueError('invalid Mamba dimensions')
        if self.hd != int(get('attention.value_length')) or self.hd>256 or self.qheads%self.kvheads:
            raise ValueError('unsupported attention dimensions')
        if not get('expert_weights_norm') or int(get('expert_group_count'))!=1 or int(get('expert_group_used_count'))!=1 or int(get('expert_shared_count'))!=1:
            raise ValueError('unsupported router policy')
        # Community GGUF advertises extrapolated 1M; original Nota checkpoint specifies 256K.
        self.max_context=min(int(get('context_length')),262144)


class Engine(object):
    backend='kindi_lightning'
    state_version=4
    def __init__(self, path, ctx=8192, log=print,kv_dtype=None,kv_read=None):
        self.file=PackedFile(path);self.spec=s=ModelSpec(self.file.gguf)
        self.ctx=int(ctx)
        if not 1<=self.ctx<=s.max_context:raise ValueError('invalid context capacity')
        self._validate_tensors()
        self.ops=CudaOps();self.cp=cp=self.ops.cp
        self.kv_dtype=kv_dtype or os.environ.get('KINDI_LIGHTNING_KV','fp32')
        from .kv import KVCacheOps
        self.kv=KVCacheOps(self.ops,self.kv_dtype,kv_read)
        self.kv_read=self.kv.read_mode
        if self.kv_dtype!='fp32' and (self.ops.profile!='optimized' or s.hd>128):raise ValueError('compressed KV requires optimized attention with dim<=128')
        self.memory=self.memory_plan(self.ctx)
        free,total=cp.cuda.runtime.memGetInfo()
        if self.memory['total_with_reserve']>free:
            raise MemoryError('memory plan requires %d bytes, free %d' % (self.memory['total_with_reserve'],free))
        self.file.load_gpu(cp,reserve_bytes=self.memory['total_with_reserve']-self.memory['weights'])
        self.w=self.file.resident
        self.states={};self.histories={};self.keys={};self.values={}
        for i,kind in enumerate(s.blocks):
            if kind=='mamba':
                self.states[i]=cp.zeros((s.heads,s.head_dim,s.state_size),cp.float32)
                self.histories[i]=cp.zeros((s.conv_dim,s.conv-1),cp.float32)
            elif kind=='attention':
                self.keys[i]=self.kv.allocate(self.ctx,s.kvheads,s.hd)
                self.values[i]=cp.empty_like(self.keys[i])
        self.position=0;self.last_logits=None;self.tokens=[];self.failed=False
        self.source_sha256=self.file.source_hash()
        properties=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        self.numeric_profile={'kernels':self.ops.profile,'dense_prefill':os.environ.get('KINDI_LIGHTNING_DENSE_PREFILL','batched_mv'),
            'expert_prefill':os.environ.get('KINDI_LIGHTNING_EXPERT_PREFILL','grouped_mv'),'state':'fp32','kv':self.kv_dtype,'kv_read':self.kv_read,'attention':self.kv.attention_mode,'position':'nope',
            'dt':'softplus_no_extra_clamp','state_version':self.state_version,'kernel_hash':self.ops.code_hash,
            'compute_capability':[properties['major'],properties['minor']],'cuda_runtime':cp.cuda.runtime.runtimeGetVersion()}
        source=hashlib.sha256()
        for name in ('lightning.py','lightning_batch.py','lightning_tokenizer.py','packed.py','moe.py','kv.py'):
            with open(os.path.join(os.path.dirname(__file__),name),'rb') as f:source.update(f.read())
        self.numeric_profile['adapter_hash']=source.hexdigest()
        self.numeric_profile['attention_workspace_bytes']=self.memory['attention_scratch']
        self.fingerprint=hashlib.sha256((self.source_sha256+json.dumps(self.numeric_profile,sort_keys=True)).encode()).hexdigest()
        log('Lightning CUDA loaded: %.3f GiB weights; context %d; FP32 state, %s KV' % (self.memory['weights']/2**30,self.ctx,self.kv_dtype))

    def _validate_tensors(self):
        s=self.spec;t=self.file.tensors
        def require(name,shape,kind=None):
            if name not in t or t[name].shape!=tuple(shape):raise ValueError('missing/incorrect tensor: '+name)
            if kind is not None and t[name].kind!=kind:raise ValueError('wrong tensor format: '+name)
        require('token_embd.weight',(s.vocab,s.width),8)
        require('output.weight',(s.vocab,s.width),8)
        require('output_norm.weight',(s.width,),0)
        for i,kind in enumerate(s.blocks):
            p='blk.%d.'%i;require(p+'attn_norm.weight',(s.width,),0)
            if kind=='mamba':
                for name,shape in [('ssm_a',(s.heads,1)),('ssm_d',(s.heads,1)),
                    ('ssm_dt.bias',(s.heads,)),('ssm_conv1d.bias',(s.conv_dim,)),
                    ('ssm_conv1d.weight',(s.conv_dim,s.conv)),('ssm_norm.weight',(s.groups,s.inner//s.groups))]:require(p+name,shape,0)
                require(p+'ssm_in.weight',(s.inner+s.conv_dim+s.heads,s.width),42)
                require(p+'ssm_out.weight',(s.width,s.inner),42)
            elif kind=='moe':
                require(p+'ffn_gate_inp.weight',(s.experts,s.width),0)
                require(p+'exp_probs_b.bias',(s.experts,),0)
                for suffix in ('exps','shexp'):
                    up=t.get(p+'ffn_up_'+suffix+'.weight');down=t.get(p+'ffn_down_'+suffix+'.weight')
                    if up is None or down is None or up.kind!=42 or down.kind!=42:raise ValueError('missing expert matrices')
                    hidden=s.routed_width if suffix=='exps' else s.shared_width
                    expected=(s.experts,hidden,s.width) if suffix=='exps' else (hidden,s.width)
                    require(up.name,expected,42)
                    require(down.name,((s.experts,s.width,up.shape[-2]) if suffix=='exps' else (s.width,up.shape[-2])),42)
            else:
                for name,n in [('attn_q',s.qheads),('attn_k',s.kvheads),('attn_v',s.kvheads)]:require(p+name+'.weight',(n*s.hd,s.width),42)
                require(p+'attn_output.weight',(s.width,s.qheads*s.hd),42)

    def memory_plan(self,ctx):
        s=self.spec
        weights=sum(t.nbytes for t in self.file.tensors.values())
        state=s.blocks.count('mamba')*(s.heads*s.head_dim*s.state_size+s.conv_dim*(s.conv-1))*4 if 'mamba' in s.blocks else 0
        dtype=getattr(self,'kv_dtype','fp32')
        if dtype=='q8_0' and s.hd%32:raise ValueError('Q8_0 KV requires head dimension divisible by 32')
        row_bytes=s.kvheads*((s.hd//32)*34 if dtype=='q8_0' else s.hd*(2 if dtype=='fp16' else 4))
        kv=s.blocks.count('attention')*2*ctx*row_bytes
        scratch=2*ctx*s.kvheads*s.hd*4 if dtype!='fp32' and getattr(self,'kv_read','auto')!='direct' else 0
        attention_scratch=self.kv.split_workspace_bytes if hasattr(self,'kv') and self.kv.attention_mode=='split' else 0
        return {'weights':weights,'state':state,'kv':kv,'kv_scratch':scratch,'attention_scratch':attention_scratch,'reserve':512*1024**2,'total_with_reserve':weights+state+kv+scratch+attention_scratch+512*1024**2}

    def linear(self,name,x,ids=None,per_input=False):
        t=self.file.tensors[name]
        n=t.shape[-2];k=t.shape[-1]
        selected=int(ids.size) if ids is not None else 1
        return self.ops.linear(self.w[name],x,n,k,t.kind,ids,selected,per_input)

    def kv_ops(self):
        if not hasattr(self,'kv'):
            from .kv import KVCacheOps
            self.kv=KVCacheOps(self.ops,getattr(self,'kv_dtype','fp32'))
        return self.kv

    def store_kv(self,layer,position,key,value):
        s=self.spec;kv=self.kv_ops()
        kv.store(self.keys[layer],key,position,s.kvheads,s.hd)
        kv.store(self.values[layer],value,position,s.kvheads,s.hd)

    def block(self,i,x):
        s=self.spec;cp=self.cp;p='blk.%d.'%i;kind=s.blocks[i]
        z=self.ops.norm(x,self.w[p+'attn_norm.weight'],s.eps)
        if kind=='moe':
            logits=self.linear(p+'ffn_gate_inp.weight',z)
            ids,weights=self.ops.router(logits,self.w[p+'exp_probs_b.bias'],s.topk,s.scale)
            up=self.ops.activate(self.linear(p+'ffn_up_exps.weight',z,ids))
            down=self.linear(p+'ffn_down_exps.weight',up,ids,True)
            shared=self.linear(p+'ffn_down_shexp.weight',self.ops.activate(self.linear(p+'ffn_up_shexp.weight',z)))
            y=self.ops.combine(down,weights,shared)
        elif kind=='mamba':
            projected=self.linear(p+'ssm_in.weight',z)
            conv=cp.empty(s.conv_dim,cp.float32)
            self.ops.k['causal_conv'](((s.conv_dim+255)//256,),(256,),
                (projected[s.inner:s.inner+s.conv_dim],self.w[p+'ssm_conv1d.weight'],self.w[p+'ssm_conv1d.bias'],
                 self.histories[i],conv,np.int32(s.conv_dim),np.int32(s.conv)))
            y=cp.empty(s.inner,cp.float32)
            self.ops.k['ssm_step']((s.inner,),(128,),
                (conv,projected[s.inner+s.conv_dim:],self.w[p+'ssm_dt.bias'],self.w[p+'ssm_a'],self.w[p+'ssm_d'],
                 self.states[i],y,np.int32(s.heads),np.int32(s.head_dim),np.int32(s.state_size),np.int32(s.groups)))
            y=self.ops.norm(y,self.w[p+'ssm_norm.weight'].reshape(-1),s.eps,s.inner//s.groups,projected[:s.inner])
            y=self.linear(p+'ssm_out.weight',y)
        else:
            q=self.linear(p+'attn_q.weight',z)
            k=self.linear(p+'attn_k.weight',z);v=self.linear(p+'attn_v.weight',z)
            self.store_kv(i,self.position,k,v)
            y=cp.empty(s.qheads*s.hd,cp.float32)
            if self.ops.profile=='optimized' and s.hd<=128:
                y=self.kv_ops().attention(q,self.keys[i],self.values[i],self.position,1,s.qheads,s.kvheads,s.hd)
            else:
                if self.kv_ops().mode!='fp32':raise ValueError('compressed KV requires optimized attention with dim<=128')
                self.ops.k['attention']((s.qheads,),(128 if s.hd<=128 else 256,),
                    (q,self.keys[i],self.values[i],y,np.int32(self.position+1),np.int32(s.qheads),np.int32(s.kvheads),np.int32(s.hd)))
            y=self.linear(p+'attn_output.weight',y)
        self.ops.add(x,y)
        return x

    def forward(self,token,logits=True,trace=None):
        if getattr(self,'failed',False):raise RuntimeError('CUDA context failed; restart this backend process')
        token=int(token);s=self.spec;cp=self.cp
        if not 0<=token<s.vocab:raise ValueError('token out of range')
        if self.position>=self.ctx:raise OverflowError('Lightning context full')
        x=cp.empty(s.width,cp.float32);t=self.file.tensors['token_embd.weight']
        self.ops.k['embedding'](((s.width+255)//256,),(256,),(self.w[t.name],x,np.int32(token),np.int32(s.width),np.int32(t.kind)))
        try:
            for i in range(s.layers):
                x=self.block(i,x)
                if trace is not None:trace(i,x,self)
            result=self.linear('output.weight',self.ops.norm(x,self.w['output_norm.weight'],s.eps)) if logits else None
            if logits:self.kv_ops().check()
            self.position+=1;self.tokens.append(token);self.last_logits=result
            return result
        except Exception:
            self.invalidate() # keep the original error if the CUDA context is no longer usable
            raise

    def prefill(self,tokens,progress=None,cancel=None,chunk_size=0):
        if getattr(self,'failed',False):raise RuntimeError('CUDA context failed; restart this backend process')
        tokens=list(tokens)
        if not tokens:raise ValueError('empty token sequence')
        if self.position+len(tokens)>self.ctx:raise OverflowError('Lightning context full')
        if chunk_size:
            if not 1<=chunk_size<=256:raise ValueError('chunk size must be in [1,256]')
            from .lightning_batch import BatchOps
            batch=BatchOps(self)
            for start in range(0,len(tokens),chunk_size):
                if cancel is not None and cancel():raise InterruptedError('prefill cancelled at chunk boundary')
                end=min(len(tokens),start+chunk_size)
                result=batch.forward(tokens[start:end],last_logits=(end==len(tokens)))
                # Bound the driver launch queue even in offline benchmarks and long-context tools.
                self.synchronize()
                if progress is not None:progress(end,len(tokens))
            return result
        result=None
        for i,token in enumerate(tokens):
            if cancel is not None and cancel():raise InterruptedError('prefill cancelled at token boundary')
            result=self.forward(token,logits=(i==len(tokens)-1))
            if cancel is not None or (i+1)%32==0 or i+1==len(tokens):self.synchronize()
            if progress is not None:progress(i+1,len(tokens))
        return result

    def reset(self):
        if getattr(self,'failed',False):raise RuntimeError('CUDA context failed; restart this backend process')
        for a in self.states.values():a.fill(0)
        for a in self.histories.values():a.fill(0)
        self.position=0;self.tokens=[];self.last_logits=None
        if hasattr(self,'kv'):self.kv.reset_error()

    def invalidate(self):
        try:
            self.reset()
            self.cp.cuda.Stream.null.synchronize()
        except Exception:
            self.failed=True;self.position=0;self.tokens=[];self.last_logits=None

    def synchronize(self):
        try:
            self.cp.cuda.Stream.null.synchronize();self.kv_ops().check()
        except Exception:
            self.invalidate();raise

    def snapshot(self):
        if getattr(self,'failed',False):raise RuntimeError('CUDA context failed; restart this backend process')
        self.kv_ops().check()
        n=self.position
        return {'version':self.state_version,'fingerprint':self.fingerprint,'tokens':list(self.tokens),'position':n,
            'states':{i:x.get() for i,x in self.states.items()},'histories':{i:x.get() for i,x in self.histories.items()},
            'keys':{i:x[:n].get() for i,x in self.keys.items()},'values':{i:x[:n].get() for i,x in self.values.items()},
            'logits':self.last_logits.get() if self.last_logits is not None else None}

    def restore(self,snapshot):
        if getattr(self,'failed',False):raise RuntimeError('CUDA context failed; restart this backend process')
        required=('version','fingerprint','position','tokens','logits','states','histories','keys','values')
        if not isinstance(snapshot,dict) or any(k not in snapshot for k in required):raise ValueError('incomplete snapshot')
        if snapshot['version']!=self.state_version or snapshot['fingerprint']!=self.fingerprint:
            raise ValueError('incompatible snapshot')
        n=snapshot['position']
        if not isinstance(n,(int,np.integer)) or isinstance(n,bool) or not isinstance(snapshot['tokens'],list):raise ValueError('invalid snapshot position/tokens')
        if n<0 or n>self.ctx or n!=len(snapshot['tokens']):raise ValueError('invalid snapshot position')
        if any(not isinstance(t,(int,np.integer)) or not 0<=t<self.spec.vocab for t in snapshot['tokens']):raise ValueError('invalid snapshot token')
        if snapshot['logits'] is not None and (not isinstance(snapshot['logits'],np.ndarray) or snapshot['logits'].shape!=(self.spec.vocab,) or snapshot['logits'].dtype!=np.float32):
            raise ValueError('invalid snapshot logits')
        for kind in ('states','histories','keys','values'):
            dst=getattr(self,kind)
            if not isinstance(snapshot[kind],dict) or set(dst)!=set(snapshot[kind]):raise ValueError('snapshot layer mismatch')
            for i,array in snapshot[kind].items():
                target=dst[i][:n] if kind in ('keys','values') else dst[i]
                if not isinstance(array,np.ndarray) or target.shape!=array.shape or array.dtype!=target.dtype:raise ValueError('invalid snapshot array')
        for kind in ('states','histories','keys','values'):
            for i,array in snapshot[kind].items():
                target=getattr(self,kind)[i][:n] if kind in ('keys','values') else getattr(self,kind)[i]
                target.set(array)
        self.position=n;self.tokens=list(snapshot['tokens'])
        self.last_logits=self.cp.asarray(snapshot['logits']) if snapshot['logits'] is not None else None
