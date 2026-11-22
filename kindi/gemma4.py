"""Kindi-owned Gemma 4 GGUF reader and CUDA primitives."""
import mmap
from collections import OrderedDict
import os
import numpy as np
from .gguf import GGUF

def ctypes_ptr(v):
    import ctypes
    return ctypes.c_void_p(v)

# GGUF K-quant type -> (dp4a matvec kernel, rows per warp)
_KQ = {10:("q2k_mv",4), 11:("q3k_mv",4), 13:("q5k_mv",4), 14:("q6k_mv",2), 111:("q3p_mv",8), 114:("q6p_mv",4)}
try:                                    # the head kernel takes 2 rows per warp on ROCm (8 on CUDA)
    import cupy as _cp
    if getattr(_cp.cuda.runtime,"is_hip",False): _KQ[111]=("q3p_mv",2)
except ImportError:
    pass

def _ptx_module(src,arch):
    """NVRTC 10.2 has no sm_80+ target: compile for compute_75, raise the PTX version and target, and let the driver
    build it for this card (the way tc.cu is loaded). The CUDA headers' directory is passed explicitly."""
    import re
    import cupy as cp
    from cupy.cuda import compiler,function
    inc=os.path.join(cp.cuda.get_cuda_path() or "/usr/local/cuda","include")
    ptx,_=compiler._NVRTCProgram(src,"gemma4_tc.cu").compile(("-arch=compute_75","-I"+inc))
    if isinstance(ptx,bytes): ptx=ptx.decode()
    ptx=re.sub(r"\.version \d+\.\d+",".version 7.1",ptx,count=1)
    ptx=re.sub(r"\.target sm_\d+",".target "+(arch if arch in ("sm_80","sm_86") else "sm_80"),ptx,count=1)   # PTX 7.1: sm_80 / sm_86 only (Ada sm_89 and later: sm_80 code)
    m=function.Module(); m.load(ptx.encode())
    return m

class Kernels:
    NAMES=("k_g4_attn_ffn_norm_residual","k_g4_rmsnorm","k_g4_residual","k_g4_gelu_mul","k_g4_ple_gate","k_g4_softcap","k_g4_rope","k_g4_q8_mv","k_g4_f32_mv","k_g4_bf16_mv","k_g4_q32","k_g4_quant_mm4","k_g4_qkv_mm4","k_g4_embed_batch","k_g4_rope_batch","k_g4_iq4xs_dp4a_mv","k_g4_iq4xs_dual_dp4a_mv","k_g4_quant_mv","k_g4_embed","k_g4_scale_vector","k_g4_dequant_q3_rows","k_g4_q3_mv_selected","k_g4_qkv_mv","k_g4_dual_quant_mv","k_g4_attn_scores","k_g4_attn_softmax","k_g4_attn_values","k_g4_attn_split_part","k_g4_attn_split_merge",
         "k_g4_embed_dev","k_g4_rope_dev","k_g4_kv_store","k_g4_attn_dev_part_g1","k_g4_attn_dev_part_g2","k_g4_attn_dev_part_g4","k_g4_attn_dev_part_g8",
         "k_g4_attn_dev_merge","k_g4_attn_dev_merge_live","k_g4_kv_store_batch",
         )+tuple("k_g4_%s_w%d"%(k,w) for k in ("iq4w_mv","iq4w_dual_mv","q3k_mv","q6k_mv","q2k_mv","q5k_mv","q3p_mv","q6p_mv","q8k_mv","k4_mv","k4_dual_mv") for w in (1,2,4,8))\
         +tuple("k_g4_mmq_t%d"%t for t in (8,10,11,13,14,23,123,111,114,108))+tuple("k_g4_dequant_t%d"%t for t in (8,10,11,13,14,23,123,111,114,108))+("k_g4_iq4_repack","k_g4_pad_blocks")+('k_g4_norm_q32', 'k_g4_qnorm_rope', 'k_g4_kv_norm_rope_store', 'k_g4_attn_merge_q32', 'k_g4_norm_residual_q32', 'k_g4_gelu_mul_q32', 'k_g4_norm_add_q32', 'k_g4_norm_add_scale', 'k_g4_ple_gate_q32', 'k_g4_q8_mv256', 'k_g4_q8_repack', 'k_g4_q8r_mv', 'k_g4_softmax_mask', 'k_g4_attn_merge64_q32', 'k_g4_tp_xsum', 'k_g4_gelu_mul2_q32', 'k_g4_softmax_mask_h')
    def __init__(self):
        import cupy as cp
        source=open(os.path.join(os.path.dirname(__file__),"gemma4.cu"),encoding="utf-8").read()
        self.mod=mod=cp.RawModule(code=source)
        for name in self.NAMES: setattr(self,name,mod.get_function(name))

class Ops:
    """Launch Kindi's fused Gemma 4 elementwise and normalization kernels."""
    def __init__(self,kernels=None):
        import cupy as cp
        self.cp=cp
        self.k=kernels or Kernels()
        self.iq4_dp4a=os.environ.get("KINDI_G4_IQ4_DP4A","1")!="0"
        self.q5_dp4a=os.environ.get("KINDI_G4_Q5_DP4A","1")!="0"
        self.kq_dp4a=os.environ.get("KINDI_G4_KQ_DP4A","1")!="0"
        self.mmq_on=os.environ.get("KINDI_G4_MMQ","1")!="0"
        cc=cp.cuda.Device().compute_capability if not getattr(cp.cuda.runtime,"is_hip",False) else "hip"
        self.dp4a=os.environ.get("KINDI_G4_DP4A",("1" if cc=="hip" or int(cc)>=61 else "0"))!="0"   # sm_61+ and ROCm
        self.tcmm=None                     # the prompt GEMM on int8 tensor cores (sm_80+, CUDA)
        self.tcmm_tokens=64                # tokens per block of tcmm (the 780M WMMA kernel takes 128, except Q2_K / Q5_K)
        self.hgemm=None
        if cc!="hip" and int(cc)>=70 and os.environ.get("KINDI_G4_TC","1")!="0":     # cuBLAS on tensor cores
            from cupy_backends.cuda.libs import cublas
            ctypes_mod=__import__("ctypes")
            one=np.array([1.0],dtype=np.float32); zero=np.array([0.0],dtype=np.float32)
            own=cublas.create()      # a handle of its own: CuPy's handle and its state are never touched
            cublas.setPointerMode(own,0)
            self._cublas=own
            def hgemm(a,b,c,ldb=None):   # c[M,N] = a[M,K] b[N,K]^T, float16 in, float32 sums and out (column-major: c^T = b a^T)
                M,Kd=a.shape; N=b.shape[0]
                cublas.setStream(own,cp.cuda.get_current_stream().ptr)
                cublas.gemmEx(own,1,0,N,M,Kd,one.ctypes.data,b.data.ptr,2,ldb or Kd,
                              a.data.ptr,2,Kd,zero.ctypes.data,c.data.ptr,0,N,0,99)   # 2 = R_16F, 0 = R_32F, 99 = tensor op
            self.hgemm=hgemm
        if cc=="hip" and os.environ.get("KINDI_G4_TC","1")!="0":
            try:
                fn=self.k.mod.get_function("k_g4_wmma_hgemm")
                def hgemm(a,b,c,ldb=None):   # c[M,N] = a[M,K] b[N,K]^T, float16 in, float32 out; ldb: b's row stride
                    M,Kd=a.shape; N=b.shape[0]
                    fn(((M+63)//64,(N+63)//64),(256,),(a,np.int32(Kd),b,np.int32(ldb or Kd),c,np.int32(N),np.int32(M),np.int32(N),np.int32(Kd)))
                self.hgemm=hgemm
            except Exception: self.hgemm=None
        if cc=="hip" and os.environ.get("KINDI_G4_TC","1")!="0":       # RDNA3 WMMA (gfx11)
            try: self.tcmm={t:self.k.mod.get_function("k_g4_wmma_t%d"%t) for t in (8,10,11,13,14,23,123,111,114,108)}; self.tcmm_tokens=128
            except Exception: self.tcmm=None
        if cc!="hip" and int(cc)>=80 and os.environ.get("KINDI_G4_TC","1")!="0":
            here=os.path.dirname(__file__)
            src=open(os.path.join(here,"gemma4.cu"),encoding="utf-8").read()+open(os.path.join(here,"gemma4_tc.cu"),encoding="utf-8").read()
            mod=_ptx_module(src,"sm_"+cc)
            self.tcmm={t:mod.get_function("k_g4_tcmm_t%d"%t) for t in (8,10,11,13,14,23,123,111,114,108)}
        # start-up checks of the matrix-core paths (each is turned off, with a log line, if it fails)
        if self.hgemm is not None and not self._hgemm_ok(): self.hgemm=None
        if self.tcmm and self.dp4a and not self._tcmm_ok(): self.tcmm=None
        self.warps=int(os.environ.get("KINDI_G4_WARPS","2560"))     # split-K target: warps per matvec launch
    def _vec(self,fn,*arrays,extra=()):
        n=int(arrays[0].size)
        out=self.cp.empty_like(arrays[0])
        fn(((n+255)//256,),(256,),(*arrays,out,np.int32(n),*extra))
        return out
    def attn_ffn_norm_residual(self,x,att,post_attn_w,ffn_w,eps=1e-6):
        cp=self.cp; x=cp.ascontiguousarray(x,dtype=cp.float32).reshape(-1)
        att=cp.ascontiguousarray(att,dtype=cp.float32).reshape(-1)
        wa=cp.ascontiguousarray(post_attn_w,dtype=cp.float32).reshape(-1)
        wf=cp.ascontiguousarray(ffn_w,dtype=cp.float32).reshape(-1)
        width=int(x.size); h=cp.empty_like(x); fn=cp.empty_like(x)
        self.k.k_g4_attn_ffn_norm_residual((1,),(256,),(x,att,wa,wf,h,fn,np.int32(width),np.float32(eps)))
        return h,fn
    def attn_ffn_norm_residual_batch(self,x,att,post_attn_w,ffn_w,eps=1e-6):
        cp=self.cp;x=cp.ascontiguousarray(x,dtype=cp.float32);att=cp.ascontiguousarray(att,dtype=cp.float32)
        wa=cp.ascontiguousarray(post_attn_w,dtype=cp.float32);wf=cp.ascontiguousarray(ffn_w,dtype=cp.float32)
        nt,width=map(int,x.shape);h=cp.empty_like(x);fn=cp.empty_like(x)
        self.k.k_g4_attn_ffn_norm_residual((nt,),(256,),(x,att,wa,wf,h,fn,np.int32(width),np.float32(eps)))
        return h,fn
    def rmsnorm(self,x,w,eps=1e-6):
        x=self.cp.ascontiguousarray(x); w=self.cp.ascontiguousarray(w)
        width=int(x.shape[-1]); rows=int(x.size//width); out=self.cp.empty_like(x)
        self.k.k_g4_rmsnorm((rows,),(256,),(x,w,out,np.int32(width),np.int32(rows),np.float32(eps)))
        return out.reshape(x.shape)
    def residual(self,x,branch,scale=1.0):
        return self._vec(self.k.k_g4_residual,self.cp.ascontiguousarray(x),
                         self.cp.ascontiguousarray(branch),extra=(np.float32(scale),))
    def gelu_mul(self,gate,up):
        return self._vec(self.k.k_g4_gelu_mul,self.cp.ascontiguousarray(gate),self.cp.ascontiguousarray(up))
    def ple_gate(self,mapped,ple):
        return self._vec(self.k.k_g4_ple_gate,self.cp.ascontiguousarray(mapped),self.cp.ascontiguousarray(ple))
    def softcap(self,x,cap=30.0):
        return self._vec(self.k.k_g4_softcap,self.cp.ascontiguousarray(x),extra=(np.float32(cap),))
    def embedding_batch(self,packed,tokens,width,gguf_type):
        cp=self.cp; tokens=cp.ascontiguousarray(tokens,dtype=cp.int32).reshape(-1); n=int(tokens.size)
        out=cp.empty((n,int(width)),dtype=cp.float32)
        self.k.k_g4_embed_batch((n,),(256,),(packed,tokens,out,np.int32(n),np.int32(width),np.int32(gguf_type)))
        return out
    def qkv_batch(self,qw,kw,vw,x,qrows,kvrows,cols,qtype,ktype,vtype,with_kv=True):
        cp=self.cp;x=cp.ascontiguousarray(x,dtype=cp.float32);nt=int(x.shape[0])
        qrows,kvrows,cols=int(qrows),int(kvrows),int(cols);total=qrows+(2*kvrows if with_kv else 0)
        out=cp.empty((nt,total),dtype=cp.float32)
        if self.mmq_on and cols%256==0 and all(t in (8,10,11,13,14,23,123,111,114,108) for t in (qtype,ktype,vtype)):
            xq=self.quant_input(x.reshape(-1),nt*cols)
            self.mmq(qw,x,qrows,cols,qtype,out,total,xq)
            if with_kv:
                self.mmq(kw,x,kvrows,cols,ktype,out[:,qrows:],total,xq)
                self.mmq(vw,x,kvrows,cols,vtype,out[:,qrows+kvrows:],total,xq)
            return out
        self.k.k_g4_qkv_mm4((total,(nt+3)//4),(128,),(qw,kw,vw,x,out,np.int32(qrows),np.int32(kvrows),
             np.int32(cols),np.int32(qtype),np.int32(ktype),np.int32(vtype),np.int32(with_kv),np.int32(nt)))
        return out
    def mmq(self,packed,x,rows,cols,gguf_type,out,ldy,xq=None):
        """prompt GEMM: out[t*ldy+row] for every token t of x (nt x cols, float32); xq: x already quantized"""
        cp=self.cp; nt=int(x.shape[0])
        if not self.dp4a:                 # no integer dot product (sm_50): dequantize to float32 and use cuBLAS
            wf=cp.empty((rows,cols),dtype=cp.float32); n=rows*(cols//256)*16
            getattr(self.k,"k_g4_dequant_t%d"%gguf_type)(((n+255)//256,),(256,),(packed,wf,np.int32(rows),np.int32(cols)))
            if ldy==rows and out.flags.c_contiguous: cp.matmul(x,wf.T,out=out)
            else: out[:,:rows]=x@wf.T     # q/k/v write column slices of one (tokens, q+k+v) array
            return out
        qx,xs=xq if xq is not None else self.quant_input(x.reshape(-1),nt*cols)
        tn=128 if (self.tcmm and self.tcmm_tokens==128 and gguf_type not in (10,13)) else 64   # tokens per block of the kernel
        grid=((nt+tn-1)//tn,(rows+63)//64) if (self.tcmm and self.tcmm_tokens==128) else ((rows+63)//64,(nt+tn-1)//tn)   # 780M: token tiles first
        (self.tcmm[gguf_type] if self.tcmm else getattr(self.k,"k_g4_mmq_t%d"%gguf_type))(grid,(256,),
            (packed,qx,xs,out,np.int32(rows),np.int32(cols),np.int32(nt),np.int32(ldy)))
        return out
    def linear_batch(self,packed,x,rows,cols,gguf_type,xq=None):
        if gguf_type in (8,10,11,13,14,23,123,111,114,108) and int(cols)%256==0 and self.mmq_on:
            x=self.cp.ascontiguousarray(x,dtype=self.cp.float32); rows,cols=int(rows),int(cols)
            return self.mmq(packed,x,rows,cols,gguf_type,self.cp.empty((int(x.shape[0]),rows),dtype=self.cp.float32),rows,xq)
        if gguf_type==30 and self.mmq_on:     # BF16: widen to float32 for this batch and use cuBLAS
            cp=self.cp; x=cp.ascontiguousarray(x,dtype=cp.float32)
            w=(packed.view(cp.uint16).astype(cp.uint32)<<16).view(cp.float32).reshape(int(rows),int(cols))
            return x@w.T
        cp=self.cp; x=cp.ascontiguousarray(x,dtype=cp.float32); nt=int(x.shape[0]); rows,cols=int(rows),int(cols)
        out=cp.empty((nt,rows),dtype=cp.float32)
        self.k.k_g4_quant_mm4((rows,(nt+3)//4),(128,),(packed,x,out,np.int32(rows),np.int32(cols),np.int32(gguf_type),np.int32(nt)))
        return out
    def embedding(self,packed,token,width,gguf_type):
        cp=self.cp
        if gguf_type not in (11,20,111):
            raise ValueError("Kindi embedding lookup does not yet support GGUF type %d" % int(gguf_type))
        out=cp.empty(int(width),dtype=cp.float32)
        if isinstance(token,cp.ndarray):      # graph decode: st[0] on the device
            self.k.k_g4_embed_dev((1,),(256,),(packed,out,token,np.int32(width),np.int32(gguf_type)))
            return out
        self.k.k_g4_embed((1,),(256,),(packed,out,np.int32(token),np.int32(width),np.int32(gguf_type)))
        return out
    def scale(self,x,factor):
        x=self.cp.ascontiguousarray(x,dtype=self.cp.float32)
        out=self.cp.empty_like(x); n=int(x.size)
        self.k.k_g4_scale_vector(((n+255)//256,),(256,),(x,out,np.int32(n),np.float32(factor)))
        return out.reshape(x.shape)
    def _tcmm_ok(self):
        """the matrix-core prompt GEMM against the dp4a GEMM on a Q8_0 matrix (64 x 256) and 300 tokens: equal
        results, every row written (a tile or grid mismatch leaves rows unwritten)"""
        cp=self.cp; rs=np.random.RandomState(3); rows,cols,nt=64,256,300
        blk=np.zeros((rows,8,34),dtype=np.uint8)
        blk[:,:,0:2]=np.frombuffer(np.full(rows*8,0.01,dtype=np.float16).tobytes(),dtype=np.uint8).reshape(rows,8,2)
        blk[:,:,2:]=rs.randint(-127,128,size=(rows,8,32)).astype(np.int8).view(np.uint8)
        w=cp.asarray(blk.reshape(-1)); x=cp.asarray(rs.standard_normal((nt,cols)).astype(np.float32))
        keep=self.tcmm; a=cp.full((nt,rows),cp.nan,dtype=cp.float32); b=cp.full((nt,rows),cp.nan,dtype=cp.float32)
        try:
            self.tcmm=None; self.mmq(w,x,rows,cols,8,a,rows)
            self.tcmm=keep; self.mmq(w,x,rows,cols,8,b,rows)
        finally:
            self.tcmm=keep
        ok=bool(cp.isfinite(b).all()) and float(cp.abs(a-b).max())<=1e-6*float(cp.abs(a).max())
        if not ok: print("kindi: the matrix-core prompt GEMM fails its start-up check; the dp4a GEMM is used")
        return ok
    def _hgemm_ok(self):
        """the float16 product against float32 on an odd shape and on the prompt attention's two shapes"""
        cp=self.cp; rs=np.random.RandomState(7)
        for M,N,K in ((37,53,520),(2048,1500,512),(2048,512,1500),(2048,1499,512),(2048,512,1499)):
            A=cp.asarray(rs.standard_normal((M,K)).astype(np.float16)); B=cp.asarray(rs.standard_normal((N,K)).astype(np.float16))
            c=cp.empty((M,N),dtype=cp.float32); self.hgemm(A,B,c)
            ref=A.astype(cp.float32)@B.astype(cp.float32).T
            if float(cp.abs(c-ref).max())>1e-3*float(cp.abs(ref).max()):
                print("kindi: the float16 tensor-core product fails its check at %dx%dx%d; prompt attention stays float32" % (M,N,K))
                return False
        return True
    def quant_input(self,x,cols):
        qx=self.cp.empty(cols,dtype=self.cp.int8); xs=self.cp.empty(cols//32,dtype=self.cp.float32)
        self.k.k_g4_q32((cols//32,),(32,),(x,qx,xs,np.int32(cols)))
        return qx,xs
    def kq(self,packed,qx,xs,out,rows,cols,gguf_type):
        """dp4a K-quant matvec into out (a contiguous array or slice)"""
        name,rpw=_KQ[gguf_type]
        self.rows_launch(name,rows,rpw,(packed,qx,xs,out,np.int32(rows),np.int32(cols)))
    def rows_launch(self,name,rows,rpw,args):
        """launch a split-K matvec: WPR warps per row group, chosen so the launch has about 2560 warps"""
        groups=(rows+rpw-1)//rpw; wpr=1
        while wpr<8 and groups*wpr<self.warps: wpr*=2
        getattr(self.k,"k_g4_%s_w%d"%(name,wpr))(((groups*wpr+7)//8,),(256,),args)
    def linear(self,packed,x,rows,cols,gguf_type):
        """Kindi CUDA row-vector GEMV for GGUF Q8_0, F32 and BF16 tensors."""
        cp=self.cp
        x=cp.ascontiguousarray(x,dtype=cp.float32).reshape(-1)
        rows,cols=int(rows),int(cols)
        out=cp.empty(rows,dtype=cp.float32)
        if gguf_type==8:
            if cols%32: raise ValueError("Q8_0 GEMV input width must be divisible by 32")
            fn=self.k.k_g4_q8_mv
        elif gguf_type==0: fn=self.k.k_g4_f32_mv
        elif gguf_type==30: fn=self.k.k_g4_bf16_mv
        elif gguf_type==108:                  # Kindi Q8 layout
            qx,xs=self.quant_input(x,cols)
            self.k.k_g4_q8r_mv(((rows+15)//16,),(256,),(packed,qx,xs,out,np.int32(rows),np.int32(cols)))
            return out
        elif gguf_type==123:                  # Kindi IQ4 layout
            qx,xs=self.quant_input(x,cols)
            self.rows_launch("k4_mv",rows,4,(packed,qx,xs,out,np.int32(rows),np.int32(cols)))
            return out
        elif gguf_type==23 and self.iq4_dp4a:
            qx=cp.empty(cols,dtype=cp.int8); xs=cp.empty(cols//32,dtype=cp.float32)
            self.k.k_g4_q32((cols//32,),(32,),(x,qx,xs,np.int32(cols)))
            self.rows_launch("iq4w_mv",rows,4,(packed,qx,xs,out,np.int32(rows),np.int32(cols)))
            return out
        elif gguf_type in _KQ and cols%256==0 and self.kq_dp4a:
            qx,xs=self.quant_input(x,cols)
            self.kq(packed,qx,xs,out,rows,cols,gguf_type)
            return out
        elif gguf_type in (10,11,13,14,23):
            fn=self.k.k_g4_quant_mv
            fn((rows,),(128,),(packed,x,out,np.int32(rows),np.int32(cols),np.int32(gguf_type)))
            return out
        else: raise ValueError("Kindi CUDA GEMV does not yet support GGUF type %d" % int(gguf_type))
        fn((rows,),(256,),(packed,x,out,np.int32(rows),np.int32(cols)))
        return out
    def qkv(self,qw,kw,vw,x,qrows,kvrows,cols,qtype,ktype,vtype,with_kv=True):
        cp=self.cp
        x=cp.ascontiguousarray(x,dtype=cp.float32).reshape(-1)
        qrows,kvrows,cols=int(qrows),int(kvrows),int(cols)
        total=qrows+(2*kvrows if with_kv else 0)
        out=cp.empty(total,dtype=cp.float32)
        if self.kq_dp4a and cols%256==0 and all(t in _KQ for t in ((qtype,ktype,vtype) if with_kv else (qtype,))):
            qx,xs=self.quant_input(x,cols)
            self.kq(qw,qx,xs,out[:qrows],qrows,cols,qtype)
            if with_kv:
                self.kq(kw,qx,xs,out[qrows:qrows+kvrows],kvrows,cols,ktype)
                self.kq(vw,qx,xs,out[qrows+kvrows:],kvrows,cols,vtype)
            return out
        use_q5=self.q5_dp4a and qtype==13
        xq=cp.empty(cols,dtype=cp.int8);xs=cp.empty(cols//32,dtype=cp.float32)
        if use_q5:
            self.k.k_g4_q32((cols//32,),(32,),(x,xq,xs,np.int32(cols)))
        self.k.k_g4_qkv_mv((total,),(128,),(qw,kw,vw,x,xq,xs,np.int32(use_q5),out,np.int32(qrows),np.int32(kvrows),
          np.int32(cols),np.int32(qtype),np.int32(ktype),np.int32(vtype),np.int32(with_kv)))
        return out
    def linear_dual_quant(self,aw,bw,x,rows,cols,gguf_type):
        cp=self.cp
        x=cp.ascontiguousarray(x,dtype=cp.float32).reshape(-1)
        rows,cols=int(rows),int(cols)
        out=cp.empty(rows*2,dtype=cp.float32)
        if gguf_type==123:
            qx,xs=self.quant_input(x,cols)
            self.rows_launch("k4_dual_mv",rows,2,(aw,bw,qx,xs,out,np.int32(rows),np.int32(cols)))
        elif gguf_type==23 and self.iq4_dp4a:
            qx=cp.empty(cols,dtype=cp.int8); xs=cp.empty(cols//32,dtype=cp.float32)
            self.k.k_g4_q32((cols//32,),(32,),(x,qx,xs,np.int32(cols)))
            self.rows_launch("iq4w_dual_mv",rows,2,(aw,bw,qx,xs,out,np.int32(rows),np.int32(cols)))
        else:
            self.k.k_g4_dual_quant_mv((rows,),(128,),(aw,bw,x,out,np.int32(rows),np.int32(cols),np.int32(gguf_type)))
        return out
    def linear_selected_q3(self,packed,x,selected,cols,bb=110):
        cp=self.cp
        x=cp.ascontiguousarray(x,dtype=cp.float32).reshape(-1)
        selected=cp.ascontiguousarray(selected,dtype=cp.int32).reshape(-1)
        out=cp.empty(int(selected.size),dtype=cp.float32)
        n=int(selected.size)
        self.k.k_g4_q3_mv_selected((n,),(128,),(packed,x,out,selected,np.int32(n),np.int32(cols),np.int32(bb)))
        return out
    def rope_batch(self,x,positions,rope_dim,base,scale=1.0):
        cp=self.cp; x=cp.ascontiguousarray(x).copy(); positions=cp.ascontiguousarray(positions,dtype=cp.int32)
        tokens,heads,dim=map(int,x.shape)
        self.k.k_g4_rope_batch((heads,tokens),(dim//2,),(x,positions,np.int32(tokens),np.int32(heads),np.int32(dim),
                          np.int32(rope_dim),np.float32(base),np.float32(scale)))
        return x
    def rope(self,x,pos,rope_dim,base,scale=1.0):
        x=self.cp.ascontiguousarray(x).copy()
        if x.ndim != 2:
            raise ValueError("Gemma 4 token RoPE expects [heads, head_dim]")
        heads,dim=map(int,x.shape)
        if isinstance(pos,self.cp.ndarray):   # graph decode: st[1] on the device
            self.k.k_g4_rope_dev((heads,),(dim//2,),(x,np.int32(heads),np.int32(dim),np.int32(rope_dim),
                                pos,np.float32(base),np.float32(scale)))
            return x
        self.k.k_g4_rope((heads,),(dim//2,),(x,np.int32(heads),np.int32(dim),np.int32(rope_dim),
                        np.int32(pos),np.float32(base),np.float32(scale)))
        return x

# GGML quant type -> (elements per block, bytes per block)
_QUANTS = {0:(1,4), 8:(32,34), 10:(256,84), 11:(256,110), 12:(256,144),
           13:(256,176), 14:(256,210), 20:(32,18), 23:(256,136), 30:(1,2)}
_NAMES = {0:"F32",8:"Q8_0",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",
          14:"Q6_K",20:"IQ4_NL",23:"IQ4_XS",30:"BF16"}

def inspect_model(path):
    g=GGUF(path)
    arch=g.kv.get("general.architecture")
    if arch != "gemma4": raise ValueError("kindi: expected Gemma 4 GGUF, got %r" % arch)
    prefix="gemma4."
    dims={k[len(prefix):]:v for k,v in g.kv.items() if k.startswith(prefix)}
    counts,seen={},set()
    for name,ne,ty,off in g.tensors:
        if ty not in _QUANTS: raise ValueError("kindi: unsupported GGUF tensor type %d in %s" % (ty,name))
        block,size=_QUANTS[ty]; n=1
        for d in ne: n*=d
        if n % block: raise ValueError("kindi: tensor %s has %d elements, not aligned to quant block %d" % (name,n,block))
        counts[_NAMES[ty]]=counts.get(_NAMES[ty],0)+1
        if name in seen: raise ValueError("kindi: duplicate tensor %s" % name)
        seen.add(name)
    required={"token_embd.weight","output_norm.weight","per_layer_token_embd.weight","per_layer_model_proj.weight","rope_freqs.weight"}
    for i in range(int(dims.get("block_count",0))):
        root="blk.%d." % i
        required.update(root+n for n in ("attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight",
          "attn_q_norm.weight","attn_k_norm.weight","attn_norm.weight","post_attention_norm.weight","ffn_gate.weight",
          "ffn_up.weight","ffn_down.weight","ffn_norm.weight","post_ffw_norm.weight","post_norm.weight",
          "layer_output_scale.weight","inp_gate.weight","proj.weight"))
    missing=sorted(required-seen)
    if missing: raise ValueError("kindi: Gemma 4 GGUF missing required tensors: %s" % ", ".join(missing[:12]))
    if int(dims.get("block_count",0))!=42 or int(dims.get("embedding_length",0))!=2560:
        raise ValueError("kindi: Gemma 4 E4B requires 42 layers and width 2560")
    pattern=dims.get("attention.sliding_window_pattern",[])
    if len(pattern)!=42:
        raise ValueError("kindi: Gemma 4 E4B needs a 42-entry sliding/global attention pattern")
    local=[i for i,sliding in enumerate(pattern) if sliding]
    global_layers=[i for i,sliding in enumerate(pattern) if not sliding]
    return {"name":g.kv.get("general.name","Gemma 4"),"context_length":int(dims.get("context_length",8192)),
      "dimensions":{"layers":int(dims["block_count"]),"width":int(dims["embedding_length"]),
                    "vocab":len(g.kv.get("tokenizer.ggml.tokens",[])),
                    "heads":int(dims["attention.head_count"]),"kv_heads":int(dims["attention.head_count_kv"]),
                    "ffn_length":int(dims["feed_forward_length"]),
                    "local_head_dim":int(dims["attention.key_length_swa"]),
                    "global_head_dim":int(dims["attention.key_length"]),
                    "sliding_window":int(dims["attention.sliding_window"]),
                    "shared_kv_layers":int(dims["attention.shared_kv_layers"]),
                    "per_layer_dim":int(dims["embedding_length_per_layer_input"]),
                    "local_layers":local,"global_layers":global_layers,
                    "rope_base_global":float(dims["rope.freq_base"]),
                    "rope_base_local":float(dims["rope.freq_base_swa"]),
                    "rope_dim_global":int(dims["rope.dimension_count"]),
                    "rope_dim_local":int(dims["rope.dimension_count_swa"]),
                    "rms_epsilon":float(dims["attention.layer_norm_rms_epsilon"])},
      "tensor_count":len(g.tensors),"tensor_types":counts}

class TensorFile:
    """Memory mapped GGUF tensors and their GPU-resident copies."""
    def __init__(self,path):
        self.gguf=GGUF(path); self.file=open(path,"rb")
        self.map=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
        self.tensors={n:{"shape":tuple(reversed(ne)),"type":ty,"offset":self.gguf.data_start+off}
                      for n,ne,ty,off in self.gguf.tensors}
        self._gpu_cache=OrderedDict(); self._gpu_cache_bytes=0; self._resident={}
    def raw(self,name):
        t=self.tensors[name]; block,size=_QUANTS[t["type"]]; n=int(np.prod(t["shape"]))
        return memoryview(self.map)[t["offset"]:t["offset"]+(n//block)*size]
    @staticmethod
    def _to_card(cupy,host):
        """host bytes -> card with a plain synchronous copy. cupy.asarray stages each copy in a pinned host buffer of the
        same size, and the buffers stay in CuPy's pinned pool: the server kept GBs of host RAM after the load (#44).
        engine._upload copies the 27B's weights the same way."""
        g=cupy.empty(host.shape,dtype=host.dtype); g.set(host,stream=cupy.cuda.Stream.null); return g
    def gpu(self,name,cupy,max_cache_bytes=512*1024*1024):
        if name in self._gpu_cache:
            value,size=self._gpu_cache.pop(name); self._gpu_cache[name]=(value,size); return value
        value=self._to_card(cupy,np.frombuffer(self.raw(name),dtype=np.uint8)); size=int(value.nbytes)
        if size<=max_cache_bytes:
            while self._gpu_cache and self._gpu_cache_bytes+size>max_cache_bytes:
                _,(_,old)=self._gpu_cache.popitem(last=False); self._gpu_cache_bytes-=old
            self._gpu_cache[name]=(value,size); self._gpu_cache_bytes+=size
        return value
    def load_resident(self,cupy,reserve_bytes=1200*1024*1024):
        props=cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
        name=props["name"].decode() if isinstance(props["name"],bytes) else str(props["name"])
        free,total=cupy.cuda.runtime.memGetInfo()
        # the per-layer token embedding is read one row per token: it stays in host memory when the card is short
        # (KINDI_G4_PLE_HOST=1 forces this, 0 forbids it)
        mode=os.environ.get("KINDI_G4_PLE_HOST","auto"); big="per_layer_token_embd.weight"
        allsize=sum(len(self.raw(n)) for n in self.tensors)
        self.host=set()
        if big in self.tensors and (mode=="1" or (mode=="auto" and allsize+reserve_bytes>free)): self.host={big}
        required=sum(len(self.raw(n)) for n in self.tensors if n not in self.host)
        if required+reserve_bytes>free:
            raise MemoryError("packed weights need %d MiB + %d MiB reserve; %d MiB is free on %s" %
              (required//1048576,reserve_bytes//1048576,free//1048576,name))
        self._resident={n:self._to_card(cupy,np.frombuffer(self.raw(n),dtype=np.uint8)) for n in self.tensors if n not in self.host}
        cupy.cuda.Stream.null.synchronize()
        return {"device":name,"weight_bytes":required,"free_bytes":free,"total_bytes":total,"host_tensors":sorted(self.host)}
    def layer(self,index):
        prefix="blk.%d." % int(index)
        return {n[len(prefix):]:t for n,t in self.tensors.items() if n.startswith(prefix)}
    def resident_layer(self,index):
        prefix="blk.%d." % int(index)
        return {n[len(prefix):]:v for n,v in self._resident.items() if n.startswith(prefix)}
    def clear_gpu(self):
        self._resident.clear(); self._gpu_cache.clear(); self._gpu_cache_bytes=0
    def close(self):
        self.clear_gpu(); self.map.close(); self.file.close()

class EmbeddingPipeline:
    """Kindi CUDA input-embedding and per-layer-embedding path for Gemma 4 E4B."""
    def __init__(self,gguf_path,ctx=8192,reserve_bytes=1200*1024*1024):
        import cupy as cp
        info=inspect_model(gguf_path)
        self.cp=cp; self.info=info; self.ctx=int(ctx)
        self.file=TensorFile(gguf_path)
        self.residency=self.file.load_resident(cp,reserve_bytes=reserve_bytes)
        self.ops=Ops()
        self.D=info["dimensions"]["width"]; self.NL=info["dimensions"]["layers"]; self.P=info["dimensions"]["per_layer_dim"]
    def _f32(self,name,shape):
        raw=self.file._resident[name]
        return raw.view(self.cp.float32).reshape(shape)
    def inputs(self,token):
        cp=self.cp; token=int(token)
        if token<0 or token>=self.info["dimensions"]["vocab"]:
            raise ValueError("token id is outside the Gemma 4 vocabulary")
        w=self.file._resident
        x=self.ops.embedding(w["token_embd.weight"],token,self.D,11)
        x=self.ops.scale(x,np.sqrt(float(self.D)))
        ple_tok=self.ops.embedding(w["per_layer_token_embd.weight"],token,self.NL*self.P,20)
        ple_tok=self.ops.scale(ple_tok,np.sqrt(float(self.P))).reshape(self.NL,self.P)
        proj=self.ops.linear(w["per_layer_model_proj.weight"],x,self.NL*self.P,self.D,30)
        proj=proj.reshape(self.NL,self.P)
        norm=self._f32("per_layer_proj_norm.weight",(self.P,))
        proj=self.ops.rmsnorm(proj,norm)
        ple=self.ops.residual(proj,ple_tok,0.7071067811865475)
        return x,ple
    def close(self):
        self.file.close()

class Engine(object):
    """Single-stream Kindi Gemma 4 E4B CUDA decoder; packed GGUF weights stay resident."""
    def __init__(self,manifest,ctx=8192,log=print,file=None,info=None):
        import cupy as cp
        self.cp=cp; self.log=log
        path=manifest["model_path"]; self.info=info or inspect_model(path)
        self.file=file or TensorFile(path)                # gemma4_tp gives each card a file of its slices
        reserve=int(os.environ.get("KINDI_G4_RESERVE_MB","1024"))*1024*1024
        self.residency=self.file.load_resident(cp,reserve_bytes=reserve)
        self.ops=Ops()
        d=self.info["dimensions"]
        self.D=d["width"]; self.NL=d["layers"]; self.V=d["vocab"]
        self.NH=d["heads"]; self.NHKV=d["kv_heads"]; self.P=d["per_layer_dim"]
        self.local=set(d["local_layers"]); self.global_layers=set(d["global_layers"])
        self.HDL=d["local_head_dim"]; self.HDG=d["global_head_dim"]
        self.unshared=self.NL-d["shared_kv_layers"]
        fixed=sum(2*self.NHKV*self.HDL*1024*2 for i in self.local if i<self.unshared)
        perpos=sum(2*self.NHKV*self.HDG*2 for i in self.global_layers if i<self.unshared)
        free,_=cp.cuda.runtime.memGetInfo()
        fit=max(0,free-reserve-fixed)//max(1,perpos)
        self.ctx=min(max(1,int(ctx)),int(self.info["context_length"]),int(fit))
        if self.ctx<1024:
            self.file.close()
            raise MemoryError("kindi: this GPU cannot fit the Gemma 4 weights and a 1024-token KV context")
        self.cache={}
        for i in range(self.unshared):
            if i in self.local:
                size=1024; hd=self.HDL         # the 512 window plus one prompt batch (forward_batch writes up to 512 at once)
            else:
                size=self.ctx; hd=self.HDG
            self.cache[i]={"k":cp.zeros((size,self.NHKV,hd),dtype=cp.float16),
                           "v":cp.zeros((size,self.NHKV,hd),dtype=cp.float16)}
        self.onesL=cp.ones(self.HDL,dtype=cp.float32)
        self.onesG=cp.ones(self.HDG,dtype=cp.float32)
        self.qrL=d["rope_dim_local"]; self.qrG=int(self.HDG*0.25)
        self.baseL=d["rope_base_local"]; self.baseG=d["rope_base_global"]
        self.eps=d["rms_epsilon"]
        self.layer_scale=[float(cp.asnumpy(self._f32("blk.%d.layer_output_scale.weight"%i,(1,)))[0]) for i in range(self.NL)]
        self.shortlist=0
        self.fast_assign=None; self.fast_centroids=None; self.fast_special=None
        cluster_path=os.path.join(manifest.get("pack_path",""),"head_clusters.npz")
        if cluster_path and os.path.isfile(cluster_path):
            z=np.load(cluster_path)
            assign=np.asarray(z["assign"],dtype=np.int32)
            cent=np.asarray(z["centroids"],dtype=np.float32)
            if assign.shape==(self.V,) and cent.ndim==2 and cent.shape[1]==self.D:
                self.fast_assign=cp.asarray(assign)
                self.fast_centroids=cp.asarray(cent)
                types=np.asarray(self.file.gguf.kv.get("tokenizer.ggml.token_type",[1]*self.V))
                self.fast_special=cp.asarray(np.flatnonzero(types!=1).astype(np.int32))
        self.last_token_usage={}
        self._graph=None
        self.ple_host="per_layer_token_embd.weight" in self.file.host
        if self.ple_host:
            rb=(self.NL*self.P//32)*18
            self._ple_table=np.frombuffer(self.file.raw("per_layer_token_embd.weight"),dtype=np.uint8).reshape(-1,rb)
            self._ple_row=cp.empty(rb,dtype=cp.uint8)
            self.log("per-layer token embedding in host memory (%.1f GiB); one row is copied per token" %
                     (self._ple_table.nbytes/2**30))
        if os.environ.get("KINDI_G4_K4","1")!="0": self.repack_iq4()
        if os.environ.get("KINDI_G4_PAD","1")!="0" and self.ops.kq_dp4a and self.ops.mmq_on: self.pad_kquants()
        self.log("Kindi Gemma 4 CUDA ready on %s: %.1f GiB packed weights, ctx %d" %
                 (self.residency["device"],self.residency["weight_bytes"]/2**30,self.ctx))
    def repack_iq4(self):
        """IQ4_XS -> Kindi IQ4 layout (type 123): 16-byte aligned codes, scales apart; one tensor at a time"""
        cp=self.cp; n=0
        for name,t in self.file.tensors.items():
            if t["type"]!=23 or name not in self.file._resident: continue
            rows,cols=map(int,t["shape"]); src=self.file._resident[name]
            dst=cp.empty(rows*(cols//2)+rows*(cols//256)*2+rows*(cols//32),dtype=cp.uint8)
            m=rows*(cols//256)
            self.ops.k.k_g4_iq4_repack(((m+255)//256,),(256,),(src,dst,np.int32(rows),np.int32(cols)))
            self.file._resident[name]=dst; t["type"]=123; n+=1; del src
        cp.cuda.Stream.null.synchronize(); cp.get_default_memory_pool().free_all_blocks()
        self.log("Kindi IQ4 layout: %d tensors repacked" % n)
    def pad_kquants(self):
        """Q3_K -> 112-byte blocks (type 111), Q6_K -> 224-byte blocks (type 114): one aligned load per 16-byte run"""
        cp=self.cp; n=0
        for name,t in self.file.tensors.items():
            if t["type"]==8 and name in self.file._resident and int(t["shape"][1])%256==0:     # Q8_0 -> Kindi Q8 layout
                rows,cols=map(int,t["shape"]); src=self.file._resident[name]; dst=cp.empty(rows*cols+rows*(cols//32)*2,dtype=cp.uint8)
                self.ops.k.k_g4_q8_repack(((rows*(cols//32)+255)//256,),(256,),(src,dst,np.int32(rows),np.int32(cols)))
                self.file._resident[name]=dst; t["type"]=108; n+=1; del src; continue
            if t["type"] not in (11,14) or name not in self.file._resident: continue
            sb,db,nt=(110,112,111) if t["type"]==11 else (210,224,114)
            src=self.file._resident[name]; nblocks=int(src.size)//sb; dst=cp.empty(nblocks*db,dtype=cp.uint8)
            self.ops.k.k_g4_pad_blocks(((nblocks*db+255)//256,),(256,),(src,dst,np.int64(nblocks),np.int32(sb),np.int32(db)))
            self.file._resident[name]=dst; t["type"]=nt; n+=1; del src
        cp.cuda.Stream.null.synchronize(); cp.get_default_memory_pool().free_all_blocks()
        self.log("Kindi padded K-quant blocks: %d tensors" % n)
    def _f32(self,name,shape):
        return self.file._resident[name].view(self.cp.float32).reshape(shape)
    def _linear(self,name,x):
        t=self.file.tensors[name]; rows,cols=t["shape"]
        return self.ops.linear(self.file._resident[name],x,rows,cols,t["type"])
    def reset(self):
        # Each request overwrites every position it reads; stale cache slots are outside the new position range.
        return None
    def _attend(self,q,k,v,pos,layer,owner,local,hd,write=True):
        cp=self.cp; cache=self.cache[owner]; cap=int(cache["k"].shape[0])
        if isinstance(pos,cp.ndarray):
            return self._attend_dev(q,k,v,pos,layer,owner,local,hd,cache,cap,write)
        if owner==layer and write:
            slot=pos%cap
            cache["k"][slot]=k.astype(cp.float16)
            cache["v"][slot]=v.astype(cp.float16)
        start=max(0,pos-(511 if local else pos))
        length=pos-start+1
        heads=int(q.shape[0]); groups=heads//self.NHKV
        scores=cp.empty((heads,length),dtype=cp.float32)
        probs=cp.empty_like(scores) if length<512 else None
        out=cp.empty((heads,hd),dtype=cp.float32)
        self.ops.k.k_g4_attn_scores((heads*length,),(256,),
            (q,cache["k"],scores,np.int32(start),np.int32(pos),np.int32(cap),
             np.int32(self.NHKV),np.int32(groups),np.int32(hd),np.int32(length)))
        if length>=512:
            nchunks=(length+255)//256
            pm=cp.empty((heads,nchunks),dtype=cp.float32); pl=cp.empty_like(pm)
            pa=cp.empty((heads,nchunks,hd),dtype=cp.float32)
            self.ops.k.k_g4_attn_split_part((heads,nchunks),(256,),
                (scores,cache["v"],pm,pl,pa,np.int32(start),np.int32(cap),np.int32(self.NHKV),
                 np.int32(groups),np.int32(hd),np.int32(length),np.int32(nchunks)))
            self.ops.k.k_g4_attn_split_merge((heads,),(256,),
                (pm,pl,pa,out,np.int32(nchunks),np.int32(hd)))
        else:
            self.ops.k.k_g4_attn_softmax((heads,),(256,),(scores,probs,np.int32(heads),np.int32(length)))
            self.ops.k.k_g4_attn_values((heads,),(256,),
                (probs,cache["v"],out,np.int32(start),np.int32(cap),np.int32(self.NHKV),
                 np.int32(groups),np.int32(hd),np.int32(length)))
        return out.reshape(heads*hd)
    def _attend_batch(self,q,pos,nt,owner,local,hd):
        """causal attention for nt prompt queries at positions pos..pos+nt-1 (keys already in the cache)"""
        cp=self.cp; cache=self.cache[owner]; cap=int(cache["k"].shape[0]); G=self.NHKV; gs=self.NH//G
        lo=max(0,pos-511) if local else 0; kp=cp.arange(lo,pos+nt,dtype=cp.int32); L=int(kp.size)
        slots=kp%cap
        if self.ops.hgemm is not None:                     # 780M: float16 matrix cores, float32 sums
            return self._attend_batch_h(q,pos,nt,cache,cap,local,hd,lo,kp,slots,L)
        K=cache["k"][slots].astype(cp.float32); V=cache["v"][slots].astype(cp.float32)     # [L, G, hd]
        q=q.reshape(nt,G,gs,hd); out=cp.empty((nt,G,gs,hd),dtype=cp.float32)
        sub=max(1,min(nt,(96<<20)//(gs*L*4)))            # queries per product: about 96 MB of scores
        for b0 in range(0,nt,sub):
            m=min(sub,nt-b0)
            for g in range(G):
                S=cp.ascontiguousarray(cp.matmul(q[b0:b0+m,g].reshape(m*gs,hd),K[:,g].T))
                self.ops.k.k_g4_softmax_mask((m*gs,),(256,),(S,np.int32(L),np.int32(gs),np.int32(pos+b0),np.int32(lo),
                    np.int32(512 if local else 0)))
                out[b0:b0+m,g]=cp.matmul(S,V[:,g]).reshape(m,gs,hd)
        return out.reshape(nt,G*gs*hd)
    def _attend_batch_h(self,q,pos,nt,cache,cap,local,hd,lo,kp,slots,L):
        """_attend_batch with the two products on float16 matrix cores (scores and the output in float32)"""
        cp=self.cp; G=self.NHKV; gs=self.NH//G; hg=self.ops.hgemm
        s0=lo%cap
        if s0+L<=cap: K=cache["k"][s0:s0+L]; V=cache["v"][s0:s0+L]               # contiguous positions: views, no copy
        else: K=cache["k"][slots]; V=cache["v"][slots]                            # [L, G, hd] float16
        q=q.reshape(nt,G,gs,hd); out=cp.empty((nt,G,gs,hd),dtype=cp.float32)
        sub=max(1,min(nt,(96<<20)//(gs*L*4)))
        L8=L
        for g in range(G):
            Kg=K[:,g]; Vt=cp.ascontiguousarray(V[:,g].T)       # [L, hd] read in place (row stride G*hd), [hd, L]
            for b0 in range(0,nt,sub):
                m=min(sub,nt-b0); Q16=cp.ascontiguousarray(q[b0:b0+m,g].reshape(m*gs,hd)).astype(cp.float16)
                S=cp.empty((m*gs,L8),dtype=cp.float32); hg(Q16,Kg,S,ldb=int(K.strides[0])//2)
                P16=cp.empty((m*gs,L8),dtype=cp.float16)
                self.ops.k.k_g4_softmax_mask_h((m*gs,),(256,),(S,P16,np.int32(L8),np.int32(gs),np.int32(pos+b0),np.int32(lo),
                    np.int32(512 if local else 0)))
                O=cp.empty((m*gs,hd),dtype=cp.float32); hg(P16,Vt,O)
                out[b0:b0+m,g]=O.reshape(m,gs,hd)
        return out.reshape(nt,G*gs*hd)
    ring_slots=1024                         # positions a sliding-window ring holds (the 512 window plus one batch)
    def _attend_dev(self,q,k,v,st,layer,owner,local,hd,cache,cap,write):
        cp=self.cp; heads=int(q.shape[0]); groups=heads//self.NHKV
        if owner==layer and write:
            n=int(k.size)
            self.ops.k.k_g4_kv_store(((n+255)//256,),(256,),(k,v,cache["k"],cache["v"],st,np.int32(cap),np.int32(n)))
        nch=(cap+31)//32                                 # 32-position chunks (G4_ACH)
        pm=cp.empty((heads,nch),dtype=cp.float32); pl=cp.empty_like(pm)
        pa=cp.empty((heads,nch,hd),dtype=cp.float32); out=cp.empty((heads,hd),dtype=cp.float32)
        getattr(self.ops.k,"k_g4_attn_dev_part_g%d"%groups)((self.NHKV,nch),(256,),(q,cache["k"],cache["v"],pm,pl,pa,st,
            np.int32(512 if local else 0),np.int32(cap),np.int32(self.NHKV),np.int32(hd),np.int32(nch)))
        self.ops.k.k_g4_attn_dev_merge_live((heads,),(256,),(pm,pl,pa,out,st,np.int32(512 if local else 0),np.int32(32),
            np.int32(nch),np.int32(hd)))
        return out.reshape(heads*hd)
    def _xch(self,partial):
        """tensor parallel: the sum of this card's partial output and the other card's (one card: unchanged)"""
        return partial
    def _xch_b(self,partial):
        return partial
    def _linq(self,name,qx,xs,out=None):
        """matvec of a resident tensor on an input that is already quantized (graph path)"""
        cp=self.cp; t=self.file.tensors[name]; rows,cols=map(int,t["shape"]); ty=t["type"]
        out=cp.empty(rows,dtype=cp.float32) if out is None else out
        if ty==123: self.ops.rows_launch("k4_mv",rows,4,(self.file._resident[name],qx,xs,out,np.int32(rows),np.int32(cols)))
        else: self.ops.kq(self.file._resident[name],qx,xs,out,rows,cols,ty)
        return out
    def _forward_graph(self,st):
        """one decode step with fused kernels, token and position in st (captured once as a CUDA graph)"""
        cp=self.cp; k=self.ops.k; D=self.D; f32=self._f32; R=self.file._resident
        eps=np.float32(self.eps); e6=np.float32(1e-6)           # e6: the default of Ops.rmsnorm, as in _forward
        x=(self.ops.embedding(self._emb_row,0,D,11) if getattr(self,"emb_host",False) else
           self.ops.embedding(R["token_embd.weight"],st,D,self.file.tensors["token_embd.weight"]["type"]))
        x=self.ops.scale(x,np.sqrt(float(D)))
        ple_tok=(self.ops.embedding(self._ple_row,0,self.NL*self.P,20) if self.ple_host else
                 self.ops.embedding(R["per_layer_token_embd.weight"],st,self.NL*self.P,20))
        ple_tok=self.ops.scale(ple_tok,np.sqrt(float(self.P))).reshape(self.NL,self.P)
        ple_proj=self._linear("per_layer_model_proj.weight",x).reshape(self.NL,self.P)
        ple_proj=self.ops.rmsnorm(ple_proj,f32("per_layer_proj_norm.weight",(self.P,)))
        ple=(ple_proj+ple_tok)*0.7071067811865475
        nff=self.info["dimensions"]["ffn_length"]
        xq=cp.empty(max(D,nff),dtype=cp.int8); xs=cp.empty(max(D,nff)//32,dtype=cp.float32)
        h=cp.empty(D,dtype=cp.float32)
        for i in range(self.NL):
            local=i in self.local; hd=self.HDL if local else self.HDG; kvh=self.NHKV; qrows=self.NH*hd; kvrows=kvh*hd
            owner=i if i<self.unshared else (22 if local else 23); pre="blk.%d."%i
            rd=self.qrL if local else self.qrG; base=np.float32(self.baseL if local else self.baseG)
            k.k_g4_norm_q32((1,),(1024,),(x,f32(pre+"attn_norm.weight",(D,)),xq,xs,np.int32(D),e6))
            raw=cp.empty(qrows+(2*kvrows if owner==i else 0),dtype=cp.float32)
            self._linq(pre+"attn_q.weight",xq,xs,raw[:qrows])
            q=cp.empty(qrows,dtype=cp.float32)
            k.k_g4_qnorm_rope((self.NH,),(256,),(raw,f32(pre+"attn_q_norm.weight",(hd,)),q,np.int32(self.NH),np.int32(hd),
                np.int32(rd),st,base,np.float32(1.0),e6))
            cache=self.cache[owner]; cap=int(cache["k"].shape[0])
            if owner==i:
                self._linq(pre+"attn_k.weight",xq,xs,raw[qrows:qrows+kvrows]); self._linq(pre+"attn_v.weight",xq,xs,raw[qrows+kvrows:])
                k.k_g4_kv_norm_rope_store((kvh,),(256,),(raw[qrows:qrows+kvrows],raw[qrows+kvrows:],f32(pre+"attn_k_norm.weight",(hd,)),
                    cache["k"],cache["v"],np.int32(kvh),np.int32(hd),np.int32(rd),st,base,np.float32(1.0),e6,np.int32(cap)))
            heads=self.NH; groups=heads//kvh; nch=(cap+31)//32      # 32-position chunks (G4_ACH)
            pm=cp.empty((heads,nch),dtype=cp.float32); pl=cp.empty_like(pm); pa=cp.empty((heads,nch,hd),dtype=cp.float32)
            getattr(k,"k_g4_attn_dev_part_g%d"%groups)((kvh,nch),(256,),(q,cache["k"],cache["v"],pm,pl,pa,st,
                np.int32(512 if local else 0),np.int32(cap),np.int32(kvh),np.int32(hd),np.int32(nch)))
            k.k_g4_attn_merge64_q32((heads,hd//64),(256,),(pm,pl,pa,xq,xs,st,np.int32(512 if local else 0),np.int32(32),np.int32(nch),np.int32(hd)))
            att=self._xch(self._linq(pre+"attn_output.weight",xq,xs))
            k.k_g4_norm_residual_q32((1,),(1024,),(x,att,f32(pre+"post_attention_norm.weight",(D,)),f32(pre+"ffn_norm.weight",(D,)),
                h,xq,xs,np.int32(D),eps))
            gu=cp.empty(2*nff,dtype=cp.float32); gn=pre+"ffn_gate.weight"; un=pre+"ffn_up.weight"
            if self.file.tensors[gn]["type"]==123 and self.file.tensors[un]["type"]==123:
                self.ops.rows_launch("k4_dual_mv",nff,2,(R[gn],R[un],xq,xs,gu,np.int32(nff),np.int32(D)))
            else:
                self._linq(gn,xq,xs,gu[:nff]); self._linq(un,xq,xs,gu[nff:])
            k.k_g4_gelu_mul_q32(((nff//32+7)//8,),(256,),(gu,xq,xs,np.int32(nff)))
            ff=self._xch(self._linq(pre+"ffn_down.weight",xq,xs))
            k.k_g4_norm_add_q32((1,),(1024,),(ff,f32(pre+"post_ffw_norm.weight",(D,)),h,xq,xs,np.int32(D),e6))
            mapped=self._linq(pre+"inp_gate.weight",xq,xs)
            P=self.P; pq=cp.empty(P,dtype=cp.int8); ps=cp.empty(P//32,dtype=cp.float32)
            k.k_g4_ple_gate_q32(((P//32+7)//8,),(256,),(mapped,cp.ascontiguousarray(ple[i]),pq,ps,np.int32(P)))
            pt=self.file.tensors[pre+"proj.weight"]; prow,pcol=map(int,pt["shape"])
            if pt["type"]==108:
                mapped=cp.empty(prow,dtype=cp.float32)
                k.k_g4_q8r_mv(((prow+15)//16,),(256,),(R[pre+"proj.weight"],pq,ps,mapped,np.int32(prow),np.int32(pcol)))
            elif pt["type"]==8 and pcol==256:
                mapped=cp.empty(prow,dtype=cp.float32)
                k.k_g4_q8_mv256(((prow+15)//16,),(256,),(R[pre+"proj.weight"],pq,ps,mapped,np.int32(prow)))
            else:
                mapped=self._linear(pre+"proj.weight",self.ops.ple_gate(mapped,ple[i]))
            xn=cp.empty(D,dtype=cp.float32)
            k.k_g4_norm_add_scale((1,),(1024,),(mapped,f32(pre+"post_norm.weight",(D,)),h,xn,np.int32(D),e6,np.float32(self.layer_scale[i])))
            x=xn
        x=self.ops.rmsnorm(x,f32("output_norm.weight",(D,)))
        return self.ops.softcap(self._linear("token_embd.weight",x),30.0)
    def _graph_forward(self,token,pos):
        """Replay the captured decode graph; the logits buffer is reused by the next call."""
        cp=self.cp; g=self._graph
        if self.ple_host: self._ple_row.set(self._ple_table[int(token)])    # ordered before the graph (blocking stream)
        if g is None:
            import ctypes
            from . import gpudrv
            drv=gpudrv.driver(); P=ctypes.c_void_p                     # libcuda, or its HIP stand-in on ROCm
            st=cp.zeros(2,dtype=cp.int32); stream=cp.cuda.Stream()      # blocking: ordered with the null stream
            pool=cp.cuda.MemoryPool()                                    # only the graph uses these blocks
            cp.cuda.Stream.null.synchronize()
            with cp.cuda.using_allocator(pool.malloc), stream:
                st.set(np.array([token,pos],dtype=np.int32),stream=stream)
                self._forward_graph(st); stream.synchronize()            # compiles every kernel before capture
                assert drv.cuStreamBeginCapture_v2(P(stream.ptr),2)==0   # relaxed mode
                out=self._forward_graph(st)
                graph=P(); assert drv.cuStreamEndCapture(P(stream.ptr),ctypes.byref(graph))==0
            ex=P(); assert drv.cuGraphInstantiateWithFlags(ctypes.byref(ex),graph,ctypes.c_ulonglong(0))==0
            g=self._graph=dict(drv=drv,st=st,stream=stream,pool=pool,graph=graph,exec=ex,out=out)
        g["st"].set(np.array([token,pos],dtype=np.int32),stream=g["stream"])
        assert g["drv"].cuGraphLaunch(g["exec"],ctypes_ptr(g["stream"].ptr))==0
        return g["out"]
    def forward(self,token,pos,logits=True):
        cp=self.cp; token=int(token); pos=int(pos)
        if pos<0 or pos>=self.ctx: raise OverflowError("Gemma 4 context limit reached")
        if logits and self.shortlist<=0 and os.environ.get("KINDI_G4_GRAPH","1")!="0":
            return self._graph_forward(token,pos)
        return self._forward(token,pos,logits)
    def _forward(self,token,pos,logits=True):
        cp=self.cp
        x=self.ops.embedding(self.file._resident["token_embd.weight"],token,self.D,self.file.tensors["token_embd.weight"]["type"])
        x=self.ops.scale(x,np.sqrt(float(self.D)))
        if self.ple_host: self._ple_row.set(self._ple_table[int(token)])
        ple_tok=(self.ops.embedding(self._ple_row,0,self.NL*self.P,20) if self.ple_host else
                 self.ops.embedding(self.file._resident["per_layer_token_embd.weight"],token,self.NL*self.P,20))
        ple_tok=self.ops.scale(ple_tok,np.sqrt(float(self.P))).reshape(self.NL,self.P)
        ple_proj=self._linear("per_layer_model_proj.weight",x).reshape(self.NL,self.P)
        ple_proj=self.ops.rmsnorm(ple_proj,self._f32("per_layer_proj_norm.weight",(self.P,)))
        ple=(ple_proj+ple_tok)*0.7071067811865475
        for i in range(self.NL):
            local=i in self.local; hd=self.HDL if local else self.HDG
            kvheads=self.NHKV; qrows=self.NH*hd; kvrows=kvheads*hd
            owner=i if i<self.unshared else (22 if local else 23)
            inp=self.ops.rmsnorm(x,self._f32("blk.%d.attn_norm.weight"%i,(self.D,)))
            qwname="blk.%d.attn_q.weight"%i
            kwname="blk.%d.attn_k.weight"%i
            vwname="blk.%d.attn_v.weight"%i
            qw=self.file._resident[qwname]; kw=self.file._resident[kwname]; vw=self.file._resident[vwname]
            qt=self.file.tensors[qwname]["type"]; kt=self.file.tensors[kwname]["type"]; vt=self.file.tensors[vwname]["type"]
            qkv=self.ops.qkv(qw,kw,vw,inp,qrows,kvrows,self.D,qt,kt,vt,with_kv=(owner==i))
            q=qkv[:qrows].reshape(self.NH,hd)
            q=self.ops.rmsnorm(q,self._f32("blk.%d.attn_q_norm.weight"%i,(hd,)))
            q=self.ops.rope(q,pos,self.qrL if local else self.qrG,self.baseL if local else self.baseG)
            if owner==i:
                k=qkv[qrows:qrows+kvrows].reshape(kvheads,hd)
                k=self.ops.rmsnorm(k,self._f32("blk.%d.attn_k_norm.weight"%i,(hd,)))
                k=self.ops.rope(k,pos,self.qrL if local else self.qrG,self.baseL if local else self.baseG)
                v=qkv[qrows+kvrows:].reshape(kvheads,hd)
                v=self.ops.rmsnorm(v,self.onesL if local else self.onesG)
            else:
                k=v=None
            att=self._attend(q,k,v,pos,i,owner,local,hd)
            att=self._linear("blk.%d.attn_output.weight"%i,att)
            h,fn=self.ops.attn_ffn_norm_residual(x,att,
                self._f32("blk.%d.post_attention_norm.weight"%i,(self.D,)),
                self._f32("blk.%d.ffn_norm.weight"%i,(self.D,)),self.eps)
            gname="blk.%d.ffn_gate.weight"%i; uname="blk.%d.ffn_up.weight"%i
            gw=self.file._resident[gname]; uw=self.file._resident[uname]
            gt=self.file.tensors[gname]["type"]; ut=self.file.tensors[uname]["type"]
            if gt==ut and gt in (10,11,13,14,23,123):
                both=self.ops.linear_dual_quant(gw,uw,fn,self.info["dimensions"]["ffn_length"],self.D,gt)
                nff=self.info["dimensions"]["ffn_length"]
                gate,up=both[:nff],both[nff:]
            else:
                gate=self._linear(gname,fn); up=self._linear(uname,fn)
            hidden=self.ops.gelu_mul(gate,up)
            ff=self._linear("blk.%d.ffn_down.weight"%i,hidden)
            ff=self.ops.rmsnorm(ff,self._f32("blk.%d.post_ffw_norm.weight"%i,(self.D,)))
            h=h+ff
            mapped=self._linear("blk.%d.inp_gate.weight"%i,h)
            mapped=self.ops.ple_gate(mapped,ple[i])
            mapped=self._linear("blk.%d.proj.weight"%i,mapped)
            mapped=self.ops.rmsnorm(mapped,self._f32("blk.%d.post_norm.weight"%i,(self.D,)))
            x=(h+mapped)*self.layer_scale[i]
        x=self.ops.rmsnorm(x,self._f32("output_norm.weight",(self.D,)))
        if not logits: return x
        if self.shortlist>0 and self.fast_assign is not None and self.fast_centroids is not None:
            count=min(int(self.shortlist),int(self.fast_centroids.shape[0]))
            cluster_score=self.fast_centroids@x
            chosen=cp.argpartition(cluster_score,-count)[-count:]
            candidate=cp.nonzero(cp.isin(self.fast_assign,chosen))[0].astype(cp.int32)
            if self.fast_special is not None and self.fast_special.size:
                candidate=cp.unique(cp.concatenate((candidate,self.fast_special)))
            packed=self.file._resident["token_embd.weight"]
            selected=self.ops.linear_selected_q3(packed,x,candidate,self.D,112 if self.file.tensors["token_embd.weight"]["type"]==111 else 110)
            scores=cp.full(self.V,-cp.inf,dtype=cp.float32)
            scores[candidate]=selected
        else:
            scores=self._linear("token_embd.weight",x)
        return self.ops.softcap(scores,30.0)
    def forward_batch(self,tokens,pos):
        """Kindi-native causal prefill: share each packed weight read across groups of four tokens."""
        cp=self.cp; tokens=list(map(int,tokens)); nt=len(tokens); pos=int(pos)
        if not nt:return
        if nt>512:                         # the local KV ring holds the window plus one batch of at most 512
            for b in range(0,nt,512): self.forward_batch(tokens[b:b+512],pos+b)
            return None
        if pos<0 or pos+nt>self.ctx:raise OverflowError("Gemma 4 prefill exceeds context")
        ids=cp.asarray(tokens,dtype=cp.int32); positions=cp.arange(pos,pos+nt,dtype=cp.int32)
        if getattr(self,"emb_host",False):
            x=self.ops.embedding_batch(cp.asarray(self._emb_table[np.asarray(tokens,dtype=np.int64)]),cp.arange(nt,dtype=cp.int32),self.D,11)
        else:
            x=self.ops.embedding_batch(self.file._resident["token_embd.weight"],ids,self.D,self.file.tensors["token_embd.weight"]["type"])
        x=self.ops.scale(x,np.sqrt(float(self.D)))
        if self.ple_host:                  # gather this batch's rows on the host, then decode them on the card
            rows=cp.asarray(self._ple_table[np.asarray(tokens,dtype=np.int64)])
            ple_tok=self.ops.embedding_batch(rows,cp.arange(nt,dtype=cp.int32),self.NL*self.P,20)
        else:
            ple_tok=self.ops.embedding_batch(self.file._resident["per_layer_token_embd.weight"],ids,self.NL*self.P,20)
        ple_tok=self.ops.scale(ple_tok,np.sqrt(float(self.P))).reshape(nt,self.NL,self.P)
        proj=self.ops.linear_batch(self.file._resident["per_layer_model_proj.weight"],x,self.NL*self.P,self.D,30)
        proj=self.ops.rmsnorm(proj.reshape(nt,self.NL,self.P),self._f32("per_layer_proj_norm.weight",(self.P,)))
        ple=(proj+ple_tok)*0.7071067811865475
        for i in range(self.NL):
            local=i in self.local;hd=self.HDL if local else self.HDG;kvheads=self.NHKV;qrows=self.NH*hd;kvrows=kvheads*hd
            owner=i if i<self.unshared else (22 if local else 23);prefix="blk.%d."%i
            inp=self.ops.rmsnorm(x,self._f32(prefix+"attn_norm.weight",(self.D,)))
            qwname=prefix+"attn_q.weight";kwname=prefix+"attn_k.weight";vwname=prefix+"attn_v.weight"
            qtype=self.file.tensors[qwname]["type"]
            ktype=self.file.tensors[kwname]["type"];vtype=self.file.tensors[vwname]["type"]
            qkv=self.ops.qkv_batch(self.file._resident[qwname],self.file._resident[kwname],self.file._resident[vwname],
                                   inp,qrows,kvrows,self.D,qtype,ktype,vtype,with_kv=(owner==i))
            qraw=qkv[:,:qrows]
            q=self.ops.rmsnorm(qraw.reshape(nt,self.NH,hd),self._f32(prefix+"attn_q_norm.weight",(hd,)))
            q=self.ops.rope_batch(q,positions,self.qrL if local else self.qrG,self.baseL if local else self.baseG)
            if owner==i:
                kraw=qkv[:,qrows:qrows+kvrows]
                vraw=qkv[:,qrows+kvrows:]
                k=self.ops.rmsnorm(kraw.reshape(nt,kvheads,hd),self._f32(prefix+"attn_k_norm.weight",(hd,)))
                k=self.ops.rope_batch(k,positions,self.qrL if local else self.qrG,self.baseL if local else self.baseG)
                v=self.ops.rmsnorm(vraw.reshape(nt,kvheads,hd),self.onesL if local else self.onesG)
                cache=self.cache[owner];cap=int(cache["k"].shape[0]);n=kvheads*hd
                k=cp.ascontiguousarray(k,dtype=cp.float32);v=cp.ascontiguousarray(v,dtype=cp.float32)
                self.ops.k.k_g4_kv_store_batch(((n*nt+255)//256,),(256,),(k,v,cache["k"],cache["v"],np.int32(pos),
                    np.int32(cap),np.int32(n),np.int32(nt)))
            else:k=v=None
            att=self._attend_batch(cp.ascontiguousarray(q,dtype=cp.float32),pos,nt,owner,local,hd)
            att=self.ops.linear_batch(self.file._resident[prefix+"attn_output.weight"],att,
                                      self.D,int(att.shape[1]),self.file.tensors[prefix+"attn_output.weight"]["type"])
            att=self._xch_b(att)
            h,fn=self.ops.attn_ffn_norm_residual_batch(x,att,self._f32(prefix+"post_attention_norm.weight",(self.D,)),
                                                       self._f32(prefix+"ffn_norm.weight",(self.D,)),self.eps)
            gwname=prefix+"ffn_gate.weight";uwname=prefix+"ffn_up.weight";nff=self.info["dimensions"]["ffn_length"]
            fq=self.ops.quant_input(cp.ascontiguousarray(fn,dtype=cp.float32).reshape(-1),nt*self.D) if self.ops.mmq_on else None
            gate=self.ops.linear_batch(self.file._resident[gwname],fn,nff,self.D,self.file.tensors[gwname]["type"],fq)
            up=self.ops.linear_batch(self.file._resident[uwname],fn,nff,self.D,self.file.tensors[uwname]["type"],fq)
            dn=prefix+"ffn_down.weight"
            if fq is not None and self.ops.dp4a and nff%32==0:   # gelu * up straight into the down projection's int8 input
                n=nt*nff; hq=cp.empty(n,dtype=cp.int8); hs=cp.empty(n//32,dtype=cp.float32)
                self.ops.k.k_g4_gelu_mul2_q32(((n//32+7)//8,),(256,),(cp.ascontiguousarray(gate),cp.ascontiguousarray(up),hq,hs,np.uint64(n)))
                ff=self.ops.linear_batch(self.file._resident[dn],gate,self.D,nff,self.file.tensors[dn]["type"],(hq,hs))
            else:
                hidden=self.ops.gelu_mul(gate,up)
                ff=self.ops.linear_batch(self.file._resident[dn],hidden,self.D,nff,self.file.tensors[dn]["type"])
            ff=self._xch_b(ff)
            ff=self.ops.rmsnorm(ff,self._f32(prefix+"post_ffw_norm.weight",(self.D,)))
            h=h+ff
            mapped=self.ops.linear_batch(self.file._resident[prefix+"inp_gate.weight"],h,self.P,self.D,
                                         self.file.tensors[prefix+"inp_gate.weight"]["type"])
            mapped=self.ops.ple_gate(mapped,ple[:,i,:])
            mapped=self.ops.linear_batch(self.file._resident[prefix+"proj.weight"],mapped,self.D,self.P,
                                         self.file.tensors[prefix+"proj.weight"]["type"])
            mapped=self.ops.rmsnorm(mapped,self._f32(prefix+"post_norm.weight",(self.D,)))
            x=(h+mapped)*self.layer_scale[i]
        return None

    def step(self,token,pos):
        return int(self.cp.argmax(self.forward(token,pos)).item())
    def close(self):
        self.file.close()


def build_fast_head(model_path,out_dir,K=1024,iters=10,log=print):
    """Build Kindi's optional centroid shortlist for a GGUF Q3_K output embedding."""
    import cupy as cp
    import cupyx
    from pathlib import Path
    tf=TensorFile(model_path)
    try:
        t=tf.tensors["token_embd.weight"]
        rows,width=t["shape"]
        if t["type"]!=11 or width%256:
            raise ValueError("Kindi fast head currently requires a Q3_K token embedding")
        kernels=Kernels()
        packed=cp.asarray(np.frombuffer(tf.raw("token_embd.weight"),dtype=np.uint8))
        dequant=kernels.k_g4_dequant_q3_rows
        K=min(int(K),int(rows))
        if K<2: raise ValueError("Kindi fast head needs at least two clusters")
        chunk=8192
        rng=np.random.RandomState(0)
        seeds=np.sort(rng.choice(rows,K,replace=False)).astype(np.int32)
        def decode(ids):
            ids=cp.asarray(np.asarray(ids,dtype=np.int32))
            out=cp.empty((int(ids.size),width),dtype=cp.float32)
            n=int(ids.size)
            dequant((n,),(256,),(packed,ids,out,np.int32(n),np.int32(width),np.int32(110)))   # the GGUF file layout
            out/=cp.maximum(cp.linalg.norm(out,axis=1,keepdims=True),1e-12)
            return out
        cent=decode(seeds)
        assign=cp.zeros(rows,dtype=cp.int32)
        for it in range(int(iters)):
            sums=cp.zeros((K,width),dtype=cp.float32)
            counts=cp.zeros(K,dtype=cp.float32)
            for start in range(0,rows,chunk):
                n=min(chunk,rows-start)
                X=decode(np.arange(start,start+n,dtype=np.int32))
                a=cp.argmax(X@cent.T,axis=1).astype(cp.int32)
                assign[start:start+n]=a
                cupyx.scatter_add(sums,a,X)
                cupyx.scatter_add(counts,a,cp.ones(n,dtype=cp.float32))
            cent=cp.where(counts[:,None]>0,sums/cp.maximum(counts[:,None],1),cent)
            cent/=cp.maximum(cp.linalg.norm(cent,axis=1,keepdims=True),1e-12)
            log("Kindi Gemma 4 fast head: cluster pass %d/%d complete"%(it+1,int(iters)))
        os.makedirs(out_dir,exist_ok=True)
        out=os.path.join(out_dir,"head_clusters.npz")
        np.savez(out,assign=cp.asnumpy(assign),centroids=cp.asnumpy(cent))
        log("Kindi Gemma 4 fast head wrote %s (%d clusters)"%(out,K))
        return out
    finally:
        tf.close()
