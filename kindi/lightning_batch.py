"""Bounded-memory prefill: GPU expert grouping, SIMT GEMM, recurrent chunk scan."""
import os
import numpy as np


class BatchOps(object):
    def __init__(self,engine):
        self.eng=engine;self.ops=engine.ops;self.cp=engine.cp;self.k=self.ops.k;self.null=self.ops.null
        self.expert_mode=os.environ.get('KINDI_LIGHTNING_EXPERT_PREFILL','grouped_mv')
        self.dense_mode=os.environ.get('KINDI_LIGHTNING_DENSE_PREFILL','batched_mv')
        if self.expert_mode not in ('gemm','grouped_mv','grouped_mv4','cublas','tile8','tile16'):raise ValueError('unknown expert prefill mode')
        if self.dense_mode not in ('tiled','cublas','batched_mv','tile8','tile16'):raise ValueError('unknown dense prefill mode')
        self.alpha=np.array(1,dtype=np.float32);self.beta=np.array(0,dtype=np.float32)
        self.handle=None

    def blas(self):
        if self.handle is None:
            self.handle=self.cp.cuda.device.get_cublas_handle()
            self.cp.cuda.cublas.setPointerMode(self.handle,0)
            self.cp.cuda.cublas.setMathMode(self.handle,0) # FP32 SGEMM; no TF32/FP16 conversion
        self.cp.cuda.cublas.setStream(self.handle,self.cp.cuda.get_current_stream().ptr)
        return self.handle

    def norm(self,x,w,groups=1,gate=None):
        m,n=x.shape
        if m<1 or groups<1 or n%groups or w.size!=n:raise ValueError('invalid batched normalization shape')
        if gate is not None and (gate.shape[0]!=m or gate.shape[1]<n):raise ValueError('invalid batched gate shape')
        out=self.cp.empty_like(x)
        self.k['rms_batch']((m*groups,),(256,),(x,w,out,np.int32(n),np.int32(groups),np.int32(m),np.float32(self.eng.spec.eps),
            gate if gate is not None else self.null,np.int32(gate.shape[1] if gate is not None else 0)))
        return out

    def route(self,logits,bias,topk,scale):
        m,experts=logits.shape
        if not 1<=experts<=256 or m<1 or not 1<=topk<=experts or bias.size!=experts:raise ValueError('invalid batched router dimensions (maximum 256 experts)')
        ids=self.cp.empty((m,topk),self.cp.int32);weights=self.cp.empty((m,topk),self.cp.float32)
        threads=1
        while threads<experts:threads*=2
        self.k['route_batch']((m,),(threads,),(logits,bias,ids,weights,np.int32(m),np.int32(experts),np.int32(topk),np.float32(scale)))
        return ids,weights

    def dispatch(self,ids,experts):
        cp=self.cp;a=ids.size;maxjobs=(a+7)//8+experts
        counts=cp.zeros(experts,cp.int32);offsets=cp.empty(experts+1,cp.int32);cursors=cp.empty(experts,cp.int32)
        order=cp.empty(a,cp.int32);je=cp.empty(maxjobs,cp.int32);js=cp.empty_like(je);jl=cp.empty_like(je);jc=cp.empty(1,cp.int32)
        self.k['expert_histogram'](((a+255)//256,),(256,),(ids,counts,np.int32(a)))
        self.k['expert_offsets']((1,),(1,),(counts,offsets,cursors,je,js,jl,jc,np.int32(experts)))
        self.k['expert_scatter'](((a+255)//256,),(256,),(ids,cursors,order,np.int32(a)))
        return {'order':order,'ids':ids,'expert':je,'start':js,'length':jl,'count':jc,'maxjobs':maxjobs,'assignments':a,'topk':ids.shape[1]}

    def linear(self,name,x,dispatch=None,per_assignment=False):
        t=self.eng.file.tensors[name];rows,cols=t.shape[-2:];cp=self.cp
        m=x.shape[0] if dispatch is None else dispatch['assignments'];out=cp.empty((m,rows),cp.float32)
        mode=self.dense_mode if dispatch is None else self.expert_mode
        if mode in ('tile8','tile16'):
            if t.kind!=42:
                if dispatch is not None:raise ValueError(mode+' requires Q2_0 experts')
                return self.ops.linear(self.eng.w[name],x,rows,cols,t.kind,selected=m,per_input=True).reshape(m,rows)
            jobs=(m+7)//8 if dispatch is None else dispatch['maxjobs']
            arrays=[self.null]*5 if dispatch is None else [dispatch[k] for k in ('order','expert','start','length','count')]
            row_tile=16 if mode=='tile16' else 8
            self.k['q2_'+mode](((rows+row_tile-1)//row_tile,jobs),(256,),tuple([self.eng.w[name],x,out,np.int32(rows),np.int32(cols),np.int32(m),
                np.int32(dispatch['topk'] if dispatch else 1),np.int32(per_assignment if dispatch else True)]+arrays))
            return out
        if dispatch is None and self.dense_mode=='batched_mv':
            return self.ops.linear(self.eng.w[name],x,rows,cols,t.kind,selected=m,per_input=True).reshape(m,rows)
        if dispatch is not None and self.expert_mode=='grouped_mv4':
            if t.kind!=42:raise ValueError('grouped GEMV currently requires Q2_0')
            self.k['grouped_mv4']((256,),(128,),
                (self.eng.w[name],x,out,dispatch['order'],dispatch['expert'],dispatch['start'],dispatch['length'],dispatch['count'],
                 np.int32(rows),np.int32(cols),np.int32(dispatch['topk']),np.int32(per_assignment)))
            return out
        if dispatch is None and self.dense_mode=='cublas':
            if t.kind==0:dense=self.eng.w[name]
            else:
                dense=cp.empty((rows,cols),cp.float32)
                self.k['unpack_dense'](((dense.size+255)//256,),(256,),
                    (self.eng.w[name],dense,np.int32(rows),np.int32(cols),np.int32(t.kind)))
            cp.cuda.cublas.sgemm(self.blas(),1,0,rows,m,cols,self.alpha.ctypes.data,dense.data.ptr,cols,
                x.data.ptr,cols,self.beta.ctypes.data,out.data.ptr,rows)
            return out
        if dispatch is not None and self.expert_mode=='cublas':
            # Only a bounded group of four expert matrices is unpacked at once.
            # The outer loop is over grouped GEMM jobs, never token x expert evaluation.
            jobs=int(dispatch['count'].get()[0])
            capacity=min(4,jobs)
            dense=cp.empty((capacity,rows,cols),cp.float32);inputs=cp.empty((capacity,8,cols),cp.float32);outputs=cp.empty((capacity,8,rows),cp.float32)
            for first in range(0,jobs,4):
                count=min(4,jobs-first)
                self.k['unpack_jobs'](((dense.size+255)//256,),(256,),
                    (self.eng.w[name],dense,dispatch['expert'],np.int32(first),np.int32(count),np.int32(rows),np.int32(cols),np.int32(t.kind)))
                self.k['gather_jobs'](((inputs.size+255)//256,),(256,),
                    (x,inputs,dispatch['order'],dispatch['start'],dispatch['length'],np.int32(first),np.int32(count),np.int32(cols),np.int32(dispatch['topk']),np.int32(per_assignment)))
                cp.cuda.cublas.sgemmStridedBatched(self.blas(),1,0,rows,8,cols,self.alpha.ctypes.data,
                    dense.data.ptr,cols,rows*cols,inputs.data.ptr,cols,8*cols,self.beta.ctypes.data,outputs.data.ptr,rows,8*rows,count)
                self.k['scatter_jobs'](((outputs.size+255)//256,),(256,),
                    (outputs,out,dispatch['order'],dispatch['start'],dispatch['length'],np.int32(first),np.int32(count),np.int32(rows)))
            return out
        if dispatch is not None and self.expert_mode=='grouped_mv':
            if t.kind!=42:raise ValueError('grouped GEMV currently requires Q2_0')
            self.k['grouped_mv']((256,),(128,),(self.eng.w[name],x,out,dispatch['ids'],dispatch['order'],np.int32(m),
                np.int32(rows),np.int32(cols),np.int32(dispatch['topk']),np.int32(per_assignment)))
            return out
        jobs=(m+7)//8 if dispatch is None else dispatch['maxjobs']
        arrays=[self.null]*5 if dispatch is None else [dispatch[k] for k in ('order','expert','start','length','count')]
        self.k['packed_mm'](((rows+15)//16,jobs),(128,),
            tuple([self.eng.w[name],x,out,np.int32(rows),np.int32(cols),np.int32(t.kind),np.int32(m),
                np.int32(1 if dispatch is None else dispatch['topk']),np.int32(per_assignment)]+arrays))
        return out

    def block(self,i,x,past):
        eng=self.eng;s=eng.spec;cp=self.cp;m=x.shape[0];p='blk.%d.'%i
        z=self.norm(x,eng.w[p+'attn_norm.weight'])
        if s.blocks[i]=='moe':
            logits=self.linear(p+'ffn_gate_inp.weight',z)
            ids,weights=self.route(logits,eng.w[p+'exp_probs_b.bias'],s.topk,s.scale)
            dispatch=self.dispatch(ids,s.experts)
            up=self.ops.activate(self.linear(p+'ffn_up_exps.weight',z,dispatch))
            down=self.linear(p+'ffn_down_exps.weight',up,dispatch,True)
            shared=self.linear(p+'ffn_down_shexp.weight',self.ops.activate(self.linear(p+'ffn_up_shexp.weight',z)))
            self.k['combine_batch'](((x.size+255)//256,),(256,),(down,weights,shared,x,np.int32(m),np.int32(s.width),np.int32(s.topk)))
            return x
        elif s.blocks[i]=='mamba':
            projected=self.linear(p+'ssm_in.weight',z);stride=projected.shape[1]
            conv=cp.empty((m,s.conv_dim),cp.float32)
            self.k['conv_chunk'](((s.conv_dim+255)//256,),(256,),
                (projected,eng.w[p+'ssm_conv1d.weight'],eng.w[p+'ssm_conv1d.bias'],eng.histories[i],conv,
                 np.int32(m),np.int32(s.conv_dim),np.int32(s.conv),np.int32(stride),np.int32(s.inner)))
            y=cp.empty((m,s.inner),cp.float32)
            self.k['ssm_chunk']((s.inner,),(128,),
                (conv,projected,eng.w[p+'ssm_dt.bias'],eng.w[p+'ssm_a'],eng.w[p+'ssm_d'],eng.states[i],y,
                 np.int32(m),np.int32(s.heads),np.int32(s.head_dim),np.int32(s.state_size),np.int32(s.groups),np.int32(stride)))
            y=self.norm(y,eng.w[p+'ssm_norm.weight'],s.groups,projected)
            y=self.linear(p+'ssm_out.weight',y)
        else:
            q=self.linear(p+'attn_q.weight',z);k=self.linear(p+'attn_k.weight',z);v=self.linear(p+'attn_v.weight',z)
            eng.store_kv(i,past,k,v)
            y=eng.kv_ops().attention(q,eng.keys[i],eng.values[i],past,m,s.qheads,s.kvheads,s.hd)
            y=self.linear(p+'attn_output.weight',y)
        self.ops.add(x,y);return x

    def forward(self,tokens,last_logits=True,trace=None):
        eng=self.eng;s=eng.spec;cp=self.cp;tokens=list(tokens);m=len(tokens);past=eng.position
        if getattr(eng,'failed',False):raise RuntimeError('CUDA context failed; restart this backend process')
        if not m or past+m>eng.ctx:raise ValueError('invalid prefill chunk length')
        if any(not 0<=int(t)<s.vocab for t in tokens):raise ValueError('token out of range')
        if ('mamba' in s.blocks and s.state_size!=128) or s.hd>128:raise ValueError('this profile requires Mamba state_size=128 and attention dim<=128')
        x=cp.empty((m,s.width),cp.float32);t=eng.file.tensors['token_embd.weight']
        try:
            self.k['embedding_batch'](((x.size+255)//256,),(256,),
                (eng.w[t.name],cp.asarray(tokens,dtype=cp.int32),x,np.int32(m),np.int32(s.width),np.int32(t.kind)))
            for i in range(s.layers):
                x=self.block(i,x,past)
                if trace is not None:trace(i,x,eng)
            result=eng.linear('output.weight',self.ops.norm(x[-1],eng.w['output_norm.weight'],s.eps)) if last_logits else None
            if last_logits:eng.kv_ops().check()
            eng.tokens.extend(int(t) for t in tokens);eng.position+=m;eng.last_logits=result
            return result
        except Exception:
            eng.invalidate();raise
