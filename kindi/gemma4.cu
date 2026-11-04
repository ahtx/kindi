/* HIP (ROCm) or CUDA: KHIP is set only when hipRTC compiles this file. CUDA builds keep their exact
   instructions; the HIP alternatives sit beside them. */
#ifndef KHIP   /* KINDI-PORT-BEGIN */
#if defined(__HIPCC_RTC__) || defined(__HIP_DEVICE_COMPILE__) || defined(__HIP_PLATFORM_AMD__)
#define KHIP 1
#define KSHFLX(v, o) __shfl_xor((v), (o))
#define KSHFL(v, l) __shfl((v), (l))
#else
#define KHIP 0
#define KSHFLX(v, o) __shfl_xor_sync(0xffffffffu, (v), (o))
#define KSHFL(v, l) __shfl_sync(0xffffffffu, (v), (l))
#endif
#endif         /* KINDI-PORT-END */
#if KHIP
#define KSHFLD(v, d) __shfl_down((v), (d))
#else
#define KSHFLD(v, d) __shfl_down_sync(0xffffffffu, (v), (d))
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

// Kindi-owned Gemma 4 primitives (CUDA and HIP).

extern "C" __global__ void k_g4_rmsnorm(const float* x, const float* w, float* y,
                                         int width, int rows, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sum[256];
    float v = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float a = x[row * width + i];
        v = fmaf(a, a, v);
    }
    sum[threadIdx.x] = v;
    __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) {
        if (threadIdx.x < d) sum[threadIdx.x] += sum[threadIdx.x + d];
        __syncthreads();
    }
    float r = rsqrtf(sum[0] / width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        y[row * width + i] = x[row * width + i] * r * w[i];
}

extern "C" __global__ void k_g4_attn_ffn_norm_residual(const float* x,const float* att,const float* wa,const float* wf,float* h,float* fn,int width,float eps) {
    __shared__ float partial[256];
    __shared__ float temp[4096];
    int t=threadIdx.x,row=blockIdx.x; const float* xr=x+(size_t)row*width; const float* ar=att+(size_t)row*width;
    float* hr=h+(size_t)row*width; float* fnr=fn+(size_t)row*width; float sum=0.0f;
    for(int i=t;i<width;i+=blockDim.x)sum=fmaf(ar[i],ar[i],sum);
    partial[t]=sum; __syncthreads();
    for(int d=blockDim.x>>1;d;d>>=1){if(t<d)partial[t]+=partial[t+d];__syncthreads();}
    float r=rsqrtf(partial[0]/width+eps); sum=0.0f;
    for(int i=t;i<width;i+=blockDim.x){float v=xr[i]+ar[i]*r*wa[i];temp[i]=v;sum=fmaf(v,v,sum);}
    partial[t]=sum; __syncthreads();
    for(int d=blockDim.x>>1;d;d>>=1){if(t<d)partial[t]+=partial[t+d];__syncthreads();}
    r=rsqrtf(partial[0]/width+eps);
    for(int i=t;i<width;i+=blockDim.x){float v=temp[i];hr[i]=v;fnr[i]=v*r*wf[i];}
}

extern "C" __global__ void k_g4_residual(const float* x, const float* branch,
                                           float* y, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (x[i] + branch[i]) * scale;
}

extern "C" __global__ void k_g4_gelu_mul(const float* gate, const float* up,
                                           float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = gate[i];
        float gelu = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
        y[i] = gelu * up[i];
    }
}

extern "C" __global__ void k_g4_ple_gate(const float* mapped, const float* ple,
                                           float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = mapped[i];
        y[i] = (0.5f * x * (1.0f + erff(x * 0.7071067811865475f))) * ple[i];
    }
}

extern "C" __global__ void k_g4_softcap(const float* x, float* y, int n, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cap * tanhf(x[i] / cap);
}

__device__ __forceinline__ void g4_rope_one(float* x, int heads, int head_dim, int rope_dim,
                                      int pos, float base, float scale) {
    int h = blockIdx.x;
    int i = threadIdx.x;
    int half = head_dim >> 1;
    int angles = rope_dim >> 1;
    if (h >= heads || i >= angles) return;
    float exponent = (2.0f / rope_dim) * i;
    float angle = pos / (powf(base, exponent) * scale);
    float sn, cs;
    sincosf(angle, &sn, &cs);
    int lo = h * head_dim + i;
    int hi = lo + half;
    float a = x[lo], b = x[hi];
    x[lo] = a * cs - b * sn;
    x[hi] = b * cs + a * sn;
}
extern "C" __global__ void k_g4_rope(float* x, int heads, int head_dim, int rope_dim,
                                      int pos, float base, float scale) {
    g4_rope_one(x, heads, head_dim, rope_dim, pos, base, scale);
}

// Row-major GGUF linear kernels used by Kindi's E4B projection paths.
__device__ __forceinline__ float g4_half(const unsigned char* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}
__device__ __forceinline__ float g4_warp_sum(float v) {
    for(int delta=16;delta;delta>>=1)v+=KSHFLD(v,delta);
    return v;
}
extern "C" __global__ void k_g4_q8_mv(const unsigned char* w,const float* x,float* y,int rows,int cols) {
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    if(row>=rows)return;
    int blocks=cols/32;
    const unsigned char* wr=w+(size_t)row*blocks*34;
    float sum=0.0f;
    for(int j=tid;j<cols;j+=blockDim.x) {
        const unsigned char* b=wr+(j>>5)*34;
        sum=fmaf(x[j],g4_half(b)*(float)*(const signed char*)(b+2+(j&31)),sum);
    }
    sum=g4_warp_sum(sum);
    __shared__ float partial[8];
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(warp==0) {
        sum=lane<(blockDim.x>>5)?partial[lane]:0.0f;
        sum=g4_warp_sum(sum);
        if(lane==0)y[row]=sum;
    }
}
extern "C" __global__ void k_g4_f32_mv(const float* w,const float* x,float* y,int rows,int cols) {
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    if(row>=rows)return;
    float sum=0.0f;
    for(int j=tid;j<cols;j+=blockDim.x)sum=fmaf(w[(size_t)row*cols+j],x[j],sum);
    sum=g4_warp_sum(sum);
    __shared__ float partial[8];
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(warp==0) {
        sum=lane<(blockDim.x>>5)?partial[lane]:0.0f;
        sum=g4_warp_sum(sum);
        if(lane==0)y[row]=sum;
    }
}
extern "C" __global__ void k_g4_bf16_mv(const unsigned short* w,const float* x,float* y,int rows,int cols) {
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    if(row>=rows)return;
    float sum=0.0f;
    for(int j=tid;j<cols;j+=blockDim.x)sum=fmaf(__uint_as_float((unsigned)w[(size_t)row*cols+j]<<16),x[j],sum);
    sum=g4_warp_sum(sum);
    __shared__ float partial[8];
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(warp==0) {
        sum=lane<(blockDim.x>>5)?partial[lane]:0.0f;
        sum=g4_warp_sum(sum);
        if(lane==0)y[row]=sum;
    }
}

// 6-bit scale and min j (0..7) of a Q4_K/Q5_K block from its 12 scale bytes s. Let k = j mod 4 and u = j / 4; with
// x = s[k], y = s[k+4], z = s[k+8]: the low 4 bits come from x/y (u = 0) or from the two nibbles of z (u = 1), and
// bits 4-5 come from bits 4-5 (u = 0) or 6-7 (u = 1) of x for the scale and of y for the min
__device__ __forceinline__ void g4_scale_min(int j,const unsigned char* s,int* sc,int* mn) {
    const int k=j&3,u=j>>2; const unsigned int x=s[k],y=s[k+4],z=s[k+8],sh=2*u;
    *sc=(int)(((u?z:x)&15u)|((x>>sh)&48u)); *mn=(int)(((u?z>>4:y)&15u)|((y>>sh)&48u));
}
template<int TYPE> __device__ __forceinline__ float g4_qweight(const unsigned char* row,int col) {
    const int block=col>>8, i=col&255;
    const unsigned char* b;
    if(TYPE==23) {
        b=row+block*136;
        int sb=i>>5, lane=i&31, qidx=sb*16+(lane&15);
        unsigned short sh=*reinterpret_cast<const unsigned short*>(b+2);
        int sl=(b[4+(sb>>1)]>>(4*(sb&1)))&15;
        int scale=sl|(((sh>>(2*sb))&3)<<4);
        int q=(b[8+qidx]>>(lane>=16?4:0))&15;
        const signed char lut[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        return g4_half(b)*(scale-32)*(float)lut[q];
    }
    if(TYPE==14||TYPE==114) {
        b=row+block*(TYPE==114?224:210);
        int half=i>>7, local=i&127, lane=local&31, qgroup=local>>5;
        int pair=qgroup&1, quarter=qgroup>>1, shift=qgroup*2;
        int qlidx=half*64+lane+(pair?32:0);
        unsigned char lo=b[qlidx], hi=b[128+half*32+lane];
        int q=quarter?(lo>>4):(lo&15);
        q|=((hi>>shift)&3)<<4;
        int scale=*(const signed char*)(b+192+half*8+lane/16+qgroup*2);
        return g4_half(b+208)*(float)scale*(float)(q-32);
    }
    if(TYPE==13) {
        b=row+block*176;
        int group=i>>6, lane=i&63, half=lane>>5, l=lane&31;
        int sc,mn; g4_scale_min(group*2+half,b+4,&sc,&mn);
        int mask=1<<(2*group+half);
        int qn=((b[48+group*32+l]>>(half?4:0))&15)+((b[16+l]&mask)?16:0);
        return g4_half(b)*(float)(sc*qn)-g4_half(b+2)*(float)mn;
    }
    if(TYPE==11||TYPE==111) {
        b=row+block*(TYPE==111?112:110);
        int group=i>>4, lane=i&15, half=group>>3, g=group&7;
        int pair=g&1, quarter=g>>1, qi=pair*16+lane;
        int qidx=half*32+pair*16+lane, shift=quarter*2;
        unsigned char q=(b[32+qidx]>>shift)&3;
        int h=(b[qi]&(1<<(half*4+quarter)))?0:4;
        const unsigned char* sc=b+96;
        int lo=group<8?(sc[group]&15):(sc[group-8]>>4);
        int sh=(sc[8+(group&3)]>>(2*(group>>2)))&3;
        int scale=(lo|(sh<<4))-32;
        return g4_half(b+108)*(float)scale*(float)(q-h);
    }
    if(TYPE==10) {
        b=row+block*84;
        int group=i>>4, local=i&127, lane=local&31, shift=((local>>5)&3)*2;
        int half=i>>7, qidx=half*32+lane;
        unsigned char packed=b[16+qidx];
        int q=(packed>>shift)&3, sm=b[group];
        return g4_half(b+80)*(float)(sm&15)*(float)q-g4_half(b+82)*(float)(sm>>4);
    }
    return 0.0f;
}
template<int TYPE> __device__ __forceinline__ float g4_weight_value(const unsigned char* row,const float* x,int col) {
    if(TYPE==0)return reinterpret_cast<const float*>(row)[col]*x[col];
    if(TYPE==30)return __uint_as_float((unsigned)reinterpret_cast<const unsigned short*>(row)[col]<<16)*x[col];
    if(TYPE==8) {
        const unsigned char* b=row+(col>>5)*34;
        int q=*(const signed char*)(b+2+(col&31));
        return g4_half(b)*(float)q*x[col];
    }
    return g4_qweight<TYPE>(row,col)*x[col];
}
template<int TYPE> __device__ __forceinline__ float g4_dot(const unsigned char* row,const float* x,int cols,int lane) {
    float sum=0.0f;
    for(int j=lane;j<cols;j+=blockDim.x)sum=fmaf(g4_weight_value<TYPE>(row,x,j),1.0f,sum);
    return sum;
}
__device__ __forceinline__ float g4_dot_dispatch(const unsigned char* row,const float* x,int cols,int type,int lane) {
    switch(type) {
        case 0:return g4_dot<0>(row,x,cols,lane);
        case 8:return g4_dot<8>(row,x,cols,lane);
        case 10:return g4_dot<10>(row,x,cols,lane);
        case 11:return g4_dot<11>(row,x,cols,lane);
        case 13:return g4_dot<13>(row,x,cols,lane);
        case 14:return g4_dot<14>(row,x,cols,lane);
        case 23:return g4_dot<23>(row,x,cols,lane);
        case 30:return g4_dot<30>(row,x,cols,lane);
        default:return 0.0f;
    }
}
// bytewise a-b (mod 256 per byte)
__device__ __forceinline__ unsigned int g4_vsub4(unsigned int a,unsigned int b) {
#if KHIP
    return ((a|0x80808080u)-(b&0x7f7f7f7fu))^((a^~b)&0x80808080u);
#else
    return __vsub4(a,b);
#endif
}
__device__ __forceinline__ int g4_dp4a(int a,int b,int c) {
#if KHIP
    return __builtin_amdgcn_sudot4(true,a,true,b,c,false);       /* v_dot4_i32_iu8, both operands signed */
#elif __CUDA_ARCH__ >= 610
    int d; asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
#else
    const signed char* aa=(const signed char*)&a;const signed char* bb=(const signed char*)&b;
    return c+(int)aa[0]*bb[0]+(int)aa[1]*bb[1]+(int)aa[2]*bb[2]+(int)aa[3]*bb[3];
#endif
}
__device__ __forceinline__ int g4_q5_value(const unsigned char* row,int col) {
    const unsigned char* b=row+(col>>8)*176;int i=col&255,group=i>>6,lane=i&63,half=lane>>5,l=lane&31;
    int mask=1<<(2*group+half);
    return ((b[48+group*32+l]>>(half?4:0))&15)+((b[16+l]&mask)?16:0);
}
__device__ __forceinline__ float g4_q5_partial(const unsigned char* row,const signed char* xq,const float* xs,int cols,int lane) {
    int warp=threadIdx.x>>5;float total=0.0f;
    for(int cb=warp*128;cb<cols;cb+=blockDim.x*4) {
        int base=cb+(lane>>3)*32+(lane&7)*4;unsigned int qp=0;
#pragma unroll
        for(int u=0;u<4;u++)qp|=((unsigned int)g4_q5_value(row,base+u)&255u)<<(8*u);
        int xp=*reinterpret_cast<const int*>(xq+base);
        int dot=g4_dp4a((int)qp,xp,0),sx=g4_dp4a(0x01010101,xp,0);
        dot+=KSHFLX(dot,1);dot+=KSHFLX(dot,2);dot+=KSHFLX(dot,4);
        sx+=KSHFLX(sx,1);sx+=KSHFLX(sx,2);sx+=KSHFLX(sx,4);
        if((lane&7)==0) {
            const unsigned char* b=row+(base>>8)*176;int gi=(base&255)>>5,sc,mn;g4_scale_min(gi,b+4,&sc,&mn);
            float scale=xs[base>>5];total+=g4_half(b)*(float)sc*scale*(float)dot-g4_half(b+2)*(float)mn*scale*(float)sx;
        }
    }
    return total;
}

extern "C" __global__ void k_g4_qkv_mv(const unsigned char* qw,const unsigned char* kw,
                                         const unsigned char* vw,const float* x,const signed char* xq,const float* xs,int use_q5_dp4a,float* y,
                                         int qrows,int kvrows,int cols,int qt,int kt,int vt,int with_kv) {
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    int total=qrows+(with_kv?2*kvrows:0);
    if(row>=total)return;
    const unsigned char* w;int wr,ty;
    if(row<qrows){w=qw;wr=row;ty=qt;}
    else if(row<qrows+kvrows){w=kw;wr=row-qrows;ty=kt;}
    else {w=vw;wr=row-qrows-kvrows;ty=vt;}
    size_t row_bytes=ty==8?(size_t)(cols/32)*34:ty==0?(size_t)cols*4:
                     ty==30?(size_t)cols*2:(size_t)(cols/256)*(ty==23?136:ty==14?210:ty==13?176:ty==11?110:84);
    const unsigned char* wrow=w+(size_t)wr*row_bytes;
    float sum=(ty==13&&use_q5_dp4a)?g4_q5_partial(wrow,xq,xs,cols,lane):g4_dot_dispatch(wrow,x,cols,ty,tid);
    sum=g4_warp_sum(sum);
    __shared__ float partial[8];
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(warp==0) {
        sum=lane<(blockDim.x>>5)?partial[lane]:0.0f;
        sum=g4_warp_sum(sum);
        if(lane==0)y[row]=sum;
    }
}

__device__ __forceinline__ float g4_value_only(const unsigned char* row,int col,int type) {
    switch(type) {
        case 0:return reinterpret_cast<const float*>(row)[col];
        case 8:{const unsigned char* b=row+(col>>5)*34;return g4_half(b)*(float)*(const signed char*)(b+2+(col&31));}
        case 10:return g4_qweight<10>(row,col);
        case 11:return g4_qweight<11>(row,col);
        case 13:return g4_qweight<13>(row,col);
        case 14:return g4_qweight<14>(row,col);
        case 23:return g4_qweight<23>(row,col);
        case 30:return __uint_as_float((unsigned)reinterpret_cast<const unsigned short*>(row)[col]<<16);
        default:return 0.0f;
    }
}

extern "C" __global__ void k_g4_qkv_mm4(const unsigned char* qw,const unsigned char* kw,const unsigned char* vw,
                                          const float* x,float* y,int qrows,int kvrows,int cols,int qt,int kt,int vt,
                                          int with_kv,int tokens) {
    int row=blockIdx.x,tile=blockIdx.y,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    int total=qrows+(with_kv?2*kvrows:0);if(row>=total)return;
    const unsigned char* w;int wr,ty;
    if(row<qrows){w=qw;wr=row;ty=qt;}
    else if(row<qrows+kvrows){w=kw;wr=row-qrows;ty=kt;}
    else {w=vw;wr=row-qrows-kvrows;ty=vt;}
    size_t rb=ty==8?(size_t)(cols/32)*34:ty==0?(size_t)cols*4:ty==30?(size_t)cols*2:
       (size_t)(cols/256)*(ty==23?136:ty==14?210:ty==13?176:ty==11?110:ty==10?84:84);
    const unsigned char* wrp=w+(size_t)wr*rb;int t0=tile*4,active=min(4,tokens-t0);
    float a0=0,a1=0,a2=0,a3=0;
    for(int j=tid;j<cols;j+=blockDim.x){float v=g4_value_only(wrp,j,ty);
        if(active>0)a0=fmaf(v,x[(size_t)(t0+0)*cols+j],a0);
        if(active>1)a1=fmaf(v,x[(size_t)(t0+1)*cols+j],a1);
        if(active>2)a2=fmaf(v,x[(size_t)(t0+2)*cols+j],a2);
        if(active>3)a3=fmaf(v,x[(size_t)(t0+3)*cols+j],a3);}
    a0=g4_warp_sum(a0);a1=g4_warp_sum(a1);a2=g4_warp_sum(a2);a3=g4_warp_sum(a3);
    __shared__ float part[4][8];if(lane==0){part[0][warp]=a0;part[1][warp]=a1;part[2][warp]=a2;part[3][warp]=a3;}__syncthreads();
    if(warp==0){a0=lane<(blockDim.x>>5)?part[0][lane]:0;a1=lane<(blockDim.x>>5)?part[1][lane]:0;
        a2=lane<(blockDim.x>>5)?part[2][lane]:0;a3=lane<(blockDim.x>>5)?part[3][lane]:0;
        a0=g4_warp_sum(a0);a1=g4_warp_sum(a1);a2=g4_warp_sum(a2);a3=g4_warp_sum(a3);
        if(lane==0){if(active>0)y[(size_t)(t0+0)*total+row]=a0;if(active>1)y[(size_t)(t0+1)*total+row]=a1;
            if(active>2)y[(size_t)(t0+2)*total+row]=a2;if(active>3)y[(size_t)(t0+3)*total+row]=a3;}}
}

extern "C" __global__ void k_g4_dual_quant_mv(const unsigned char* aw,const unsigned char* bw,
                                                const float* x,float* y,int rows,int cols,int type) {
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    if(row>=rows)return;
    int block_bytes=type==23?136:type==14?210:type==13?176:type==11?110:84;
    const unsigned char* ar=aw+(size_t)row*(cols/256)*block_bytes;
    const unsigned char* br=bw+(size_t)row*(cols/256)*block_bytes;
    float sa=g4_dot_dispatch(ar,x,cols,type,tid),sb=g4_dot_dispatch(br,x,cols,type,tid);
    sa=g4_warp_sum(sa);sb=g4_warp_sum(sb);
    __shared__ float pa[8],pb[8];
    if(lane==0){pa[warp]=sa;pb[warp]=sb;}
    __syncthreads();
    if(warp==0) {
        sa=lane<(blockDim.x>>5)?pa[lane]:0.0f;sb=lane<(blockDim.x>>5)?pb[lane]:0.0f;
        sa=g4_warp_sum(sa);sb=g4_warp_sum(sb);
        if(lane==0){y[row]=sa;y[rows+row]=sb;}
    }
}
extern "C" __global__ void k_g4_dequant_q3_rows(const unsigned char* weights,const int* row_ids,
                                                  float* out,int rows,int width,int bb) {
    int r=blockIdx.x;
    if(r>=rows)return;
    int source_row=row_ids[r];
    const unsigned char* row=weights+(size_t)source_row*(width/256)*bb;
    float* y=out+(size_t)r*width;
    for(int i=threadIdx.x;i<width;i+=blockDim.x)y[i]=bb==112?g4_qweight<111>(row,i):g4_qweight<11>(row,i);
}
extern "C" __global__ void k_g4_q3_mv_selected(const unsigned char* w,const float* x,float* y,
                                                const int* selected,int nsel,int cols,int bb) {
    int outrow=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    if(outrow>=nsel)return;
    int row=selected[outrow];
    const unsigned char* wr=w+(size_t)row*(cols/256)*bb;
    float sum=0.0f;
    for(int j=tid;j<cols;j+=blockDim.x)sum=fmaf(bb==112?g4_qweight<111>(wr,j):g4_qweight<11>(wr,j),x[j],sum);
    sum=g4_warp_sum(sum);
    __shared__ float partial[8];
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(warp==0) {
        sum=lane<(blockDim.x>>5)?partial[lane]:0.0f;
        sum=g4_warp_sum(sum);
        if(lane==0)y[outrow]=sum;
    }
}
__device__ __constant__ signed char g4_iq4xs_lut[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
__device__ __forceinline__ int g4_iq4xs_value(const unsigned char* row,int col) {
    const unsigned char* b=row+(col>>8)*136;int i=col&255,sb=i>>5,lane=i&31;
    int qi=sb*16+(lane&15);unsigned char v=b[8+qi];int code=(v>>(lane>=16?4:0))&15;
    return (int)g4_iq4xs_lut[code];
}
__device__ __forceinline__ int g4_iq4xs_scale(const unsigned char* row,int col) {
    const unsigned char* b=row+(col>>8)*136;int sb=(col&255)>>5;
    unsigned short sh=*reinterpret_cast<const unsigned short*>(b+2);
    int sl=(b[4+(sb>>1)]>>(4*(sb&1)))&15;
    return (sl|(((sh>>(2*sb))&3)<<4))-32;
}
__device__ __forceinline__ float g4_iq4xs_d(const unsigned char* row,int col) {
    const unsigned char* b=row+(col>>8)*136;return g4_half(b);
}
extern "C" __global__ void k_g4_q32(const float* x,signed char* q,float* scale,int n) {
    int g=blockIdx.x,l=threadIdx.x,i=g*32+l;__shared__ float sh[32];
    float v=i<n?x[i]:0.0f;sh[l]=fabsf(v);__syncthreads();
    for(int d=16;d;d>>=1){if(l<d)sh[l]=fmaxf(sh[l],sh[l+d]);__syncthreads();}
    float sc=sh[0]>0.0f?sh[0]/127.0f:1.0f;
    if(l==0)scale[g]=sc;
    if(i<n)q[i]=(signed char)__float2int_rn(v/sc);
}
extern "C" __global__ void k_g4_iq4xs_dp4a_mv(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows,int cols) {
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;if(row>=rows)return;
    const unsigned char* wr=w+(size_t)row*(cols/256)*136;float total=0.0f;
    for(int cb=warp*128;cb<cols;cb+=blockDim.x*4) {
        int base=cb+(lane>>3)*32+(lane&7)*4;unsigned int qw=0;
#pragma unroll
        for(int u=0;u<4;u++)qw|=((unsigned int)g4_iq4xs_value(wr,base+u)&255u)<<(8*u);
        int xp=*reinterpret_cast<const int*>(xq+base);int di=g4_dp4a((int)qw,xp,0);
        di+=KSHFLX(di,1);di+=KSHFLX(di,2);di+=KSHFLX(di,4);
        if((lane&7)==0)total+=(float)di*(g4_iq4xs_d(wr,base)*(float)g4_iq4xs_scale(wr,base)*xs[base>>5]);
    }
    total=g4_warp_sum(total);__shared__ float partial[8];if(lane==0)partial[warp]=total;__syncthreads();
    if(warp==0){float z=lane<(blockDim.x>>5)?partial[lane]:0.0f;z=g4_warp_sum(z);if(lane==0)y[row]=z;}
}
extern "C" __global__ void k_g4_iq4xs_dual_dp4a_mv(const unsigned char* aw,const unsigned char* bw,const signed char* xq,const float* xs,float* y,int rows,int cols) {
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;if(row>=rows)return;
    const unsigned char* ar=aw+(size_t)row*(cols/256)*136;const unsigned char* br=bw+(size_t)row*(cols/256)*136;
    float totalA=0.0f,totalB=0.0f;
    for(int cb=warp*128;cb<cols;cb+=blockDim.x*4) {
        int base=cb+(lane>>3)*32+(lane&7)*4;unsigned int qa=0,qb=0;
#pragma unroll
        for(int u=0;u<4;u++){qa|=((unsigned int)g4_iq4xs_value(ar,base+u)&255u)<<(8*u);qb|=((unsigned int)g4_iq4xs_value(br,base+u)&255u)<<(8*u);}
        int xp=*reinterpret_cast<const int*>(xq+base);int da=g4_dp4a((int)qa,xp,0),db=g4_dp4a((int)qb,xp,0);
        da+=KSHFLX(da,1);da+=KSHFLX(da,2);da+=KSHFLX(da,4);
        db+=KSHFLX(db,1);db+=KSHFLX(db,2);db+=KSHFLX(db,4);
        if((lane&7)==0){
            totalA+=(float)da*(g4_iq4xs_d(ar,base)*(float)g4_iq4xs_scale(ar,base)*xs[base>>5]);
            totalB+=(float)db*(g4_iq4xs_d(br,base)*(float)g4_iq4xs_scale(br,base)*xs[base>>5]);
        }
    }
    totalA=g4_warp_sum(totalA);totalB=g4_warp_sum(totalB);__shared__ float pa[8],pb[8];
    if(lane==0){pa[warp]=totalA;pb[warp]=totalB;}__syncthreads();
    if(warp==0){float a=lane<(blockDim.x>>5)?pa[lane]:0.0f;float b=lane<(blockDim.x>>5)?pb[lane]:0.0f;
        a=g4_warp_sum(a);b=g4_warp_sum(b);if(lane==0){y[row]=a;y[rows+row]=b;}}
}

extern "C" __global__ void k_g4_quant_mv(const unsigned char* w,const float* x,float* y,
                                          int rows,int cols,int type) {
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    if(row>=rows)return;
    int block_bytes=type==23?136:type==14?210:type==13?176:type==11?110:84;
    const unsigned char* wr=w+(size_t)row*(cols/256)*block_bytes;
    float sum=g4_dot_dispatch(wr,x,cols,type,tid);
    sum=g4_warp_sum(sum);
    __shared__ float partial[8];
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(warp==0) {
        sum=lane<(blockDim.x>>5)?partial[lane]:0.0f;
        sum=g4_warp_sum(sum);
        if(lane==0)y[row]=sum;
    }
}


extern "C" __global__ void k_g4_quant_mm4(const unsigned char* w,const float* x,float* y,
                                            int rows,int cols,int type,int tokens) {
    int row=blockIdx.x,tile=blockIdx.y,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    if(row>=rows)return;
    int t0=tile*4,active=min(4,tokens-t0);
    int block_bytes=type==23?136:type==14?210:type==13?176:type==11?110:type==10?84:type==8?34:type==30?2:4;
    size_t row_bytes=type==8?(size_t)(cols/32)*34:type==0?(size_t)cols*4:type==30?(size_t)cols*2:(size_t)(cols/256)*block_bytes;
    const unsigned char* wr=w+(size_t)row*row_bytes;
    float a0=0,a1=0,a2=0,a3=0;
    for(int j=tid;j<cols;j+=blockDim.x) {
        float v=g4_value_only(wr,j,type);
        if(active>0)a0=fmaf(v,x[(size_t)(t0+0)*cols+j],a0);
        if(active>1)a1=fmaf(v,x[(size_t)(t0+1)*cols+j],a1);
        if(active>2)a2=fmaf(v,x[(size_t)(t0+2)*cols+j],a2);
        if(active>3)a3=fmaf(v,x[(size_t)(t0+3)*cols+j],a3);
    }
    a0=g4_warp_sum(a0);a1=g4_warp_sum(a1);a2=g4_warp_sum(a2);a3=g4_warp_sum(a3);
    __shared__ float part[4][8];
    if(lane==0){part[0][warp]=a0;part[1][warp]=a1;part[2][warp]=a2;part[3][warp]=a3;}
    __syncthreads();
    if(warp==0) {
        a0=lane<(blockDim.x>>5)?part[0][lane]:0;a1=lane<(blockDim.x>>5)?part[1][lane]:0;
        a2=lane<(blockDim.x>>5)?part[2][lane]:0;a3=lane<(blockDim.x>>5)?part[3][lane]:0;
        a0=g4_warp_sum(a0);a1=g4_warp_sum(a1);a2=g4_warp_sum(a2);a3=g4_warp_sum(a3);
        if(lane==0){if(active>0)y[(size_t)(t0+0)*rows+row]=a0;if(active>1)y[(size_t)(t0+1)*rows+row]=a1;
                    if(active>2)y[(size_t)(t0+2)*rows+row]=a2;if(active>3)y[(size_t)(t0+3)*rows+row]=a3;}
    }
}

__device__ __forceinline__ void g4_embed_row(const unsigned char* weights,float* out,int token,int width,int type) {
    int lane=threadIdx.x;
    int bytes_per_row=type==11?(width/256)*110:type==111?(width/256)*112:(width/32)*18;
    const unsigned char* row=weights+(size_t)token*bytes_per_row;
    const signed char lut[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    for(int i=lane;i<width;i+=blockDim.x) {
        float v;
        if(type==11) v=g4_qweight<11>(row,i);
        else if(type==111) v=g4_qweight<111>(row,i);
        else {
            const unsigned char* b=row+(i/32)*18;
            int local=i%32;
            int q=(b[2+(local&15)]>>(local>=16?4:0))&15;
            v=g4_half(b)*(float)lut[q];
        }
        out[i]=v;
    }
}
extern "C" __global__ void k_g4_embed(const unsigned char* weights,float* out,int token,int width,int type) {
    g4_embed_row(weights,out,token,width,type);
}

extern "C" __global__ void k_g4_embed_batch(const unsigned char* weights,const int* tokens,float* out,
                                               int ntokens,int width,int type) {
    int t=blockIdx.x,lane=threadIdx.x;
    int bytes_per_row=type==11?(width/256)*110:type==111?(width/256)*112:(width/32)*18;
    const unsigned char* row=weights+(size_t)tokens[t]*bytes_per_row;
    const signed char lut[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    for(int i=lane;i<width;i+=blockDim.x) {
        float v;
        if(type==11)v=g4_qweight<11>(row,i);
        else if(type==111)v=g4_qweight<111>(row,i);
        else {const unsigned char* b=row+(i/32)*18;int local=i%32;int q=(b[2+(local&15)]>>(local>=16?4:0))&15;v=g4_half(b)*(float)lut[q];}
        out[(size_t)t*width+i]=v;
    }
}

extern "C" __global__ void k_g4_scale_vector(const float* x,float* y,int n,float scale) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)y[i]=x[i]*scale;
}


extern "C" __global__ void k_g4_rope_batch(float* x,const int* positions,int tokens,int heads,int head_dim,int rope_dim,float base,float scale) {
    int h=blockIdx.x,t=blockIdx.y,i=threadIdx.x,half=head_dim>>1,angles=rope_dim>>1;
    if(h>=heads||t>=tokens||i>=angles)return;
    float angle=positions[t]/(powf(base,(2.0f/rope_dim)*i)*scale);float sn,cs;sincosf(angle,&sn,&cs);
    size_t off=(size_t)t*heads*head_dim+(size_t)h*head_dim+i;
    float a=x[off],b=x[off+half];x[off]=a*cs-b*sn;x[off+half]=b*cs+a*sn;
}

extern "C" __global__ void k_g4_attn_scores(const float* q,const __half* keys,float* scores,
                                              int start,int pos,int cap,int kvheads,int groups,int hd,int len) {
    int task=blockIdx.x,head=task/len,ti=task-head*len,lane=threadIdx.x;
    if(head>=groups*kvheads)return;
    int t=start+ti,slot=t%cap,kv=head/groups;
    float sum=0.0f;
    for(int d=lane;d<hd;d+=blockDim.x) {
        float kval=__half2float(keys[((size_t)slot*kvheads+kv)*hd+d]);
        sum=fmaf(q[(size_t)head*hd+d],kval,sum);
    }
    __shared__ float sh[256];sh[lane]=sum;__syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]+=sh[lane+d];__syncthreads();}
    if(lane==0)scores[(size_t)head*len+ti]=sh[0];
}
extern "C" __global__ void k_g4_attn_softmax(const float* scores,float* probs,int heads,int len) {
    int head=blockIdx.x,lane=threadIdx.x;
    __shared__ float sh[256];
    float vmax=-3.402823466e+38F;
    for(int t=lane;t<len;t+=blockDim.x)vmax=fmaxf(vmax,scores[(size_t)head*len+t]);
    sh[lane]=vmax;__syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]=fmaxf(sh[lane],sh[lane+d]);__syncthreads();}
    float mx=sh[0],sum=0.0f;
    for(int t=lane;t<len;t+=blockDim.x)sum+=expf(scores[(size_t)head*len+t]-mx);
    sh[lane]=sum;__syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]+=sh[lane+d];__syncthreads();}
    float den=sh[0];
    for(int t=lane;t<len;t+=blockDim.x)probs[(size_t)head*len+t]=expf(scores[(size_t)head*len+t]-mx)/den;
}
extern "C" __global__ void k_g4_attn_split_part(const float* scores,const __half* values,float* pm,float* pl,float* pa,
                                                     int start,int cap,int kvheads,int groups,int hd,int len,int nchunks) {
    int head=blockIdx.x,chunk=blockIdx.y,lane=threadIdx.x,kv=head/groups;
    int begin=chunk*256,n=min(256,len-begin); if(n<=0)return;
    __shared__ float sh[256]; __shared__ float ex[256];
    float mx=-3.402823466e+38F;
    for(int t=lane;t<n;t+=blockDim.x)mx=fmaxf(mx,scores[(size_t)head*len+begin+t]);
    sh[lane]=mx; __syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]=fmaxf(sh[lane],sh[lane+d]);__syncthreads();}
    mx=sh[0]; float z=0.0f;
    for(int t=lane;t<n;t+=blockDim.x){float e=expf(scores[(size_t)head*len+begin+t]-mx);ex[t]=e;z+=e;}
    sh[lane]=z; __syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]+=sh[lane+d];__syncthreads();}
    size_t ci=(size_t)head*nchunks+chunk; pm[ci]=mx; pl[ci]=sh[0];
    for(int d=lane;d<hd;d+=blockDim.x){float a=0.0f;
        for(int t=0;t<n;t++){int slot=(start+begin+t)%cap;float v=__half2float(values[((size_t)slot*kvheads+kv)*hd+d]);a=fmaf(ex[t],v,a);}
        pa[ci*hd+d]=a;
    }
}
extern "C" __global__ void k_g4_attn_split_merge(const float* pm,const float* pl,const float* pa,float* out,
                                                      int nchunks,int hd) {
    int head=blockIdx.x,lane=threadIdx.x; __shared__ float sh[256];
    float mx=-3.402823466e+38F;
    for(int c=lane;c<nchunks;c+=blockDim.x)mx=fmaxf(mx,pm[(size_t)head*nchunks+c]);
    sh[lane]=mx; __syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]=fmaxf(sh[lane],sh[lane+d]);__syncthreads();}
    mx=sh[0]; float z=0.0f;
    for(int c=lane;c<nchunks;c+=blockDim.x)z+=pl[(size_t)head*nchunks+c]*expf(pm[(size_t)head*nchunks+c]-mx);
    sh[lane]=z; __syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]+=sh[lane+d];__syncthreads();}
    float den=sh[0];
    for(int d=lane;d<hd;d+=blockDim.x){float a=0.0f;
        for(int c=0;c<nchunks;c++){size_t ci=(size_t)head*nchunks+c;a=fmaf(pa[ci*hd+d],expf(pm[ci]-mx),a);}
        out[(size_t)head*hd+d]=a/den;
    }
}

extern "C" __global__ void k_g4_attn_values(const float* probs,const __half* values,float* out,
                                              int start,int cap,int kvheads,int groups,int hd,int len) {
    int head=blockIdx.x,lane=threadIdx.x,kv=head/groups;
    for(int d=lane;d<hd;d+=blockDim.x) {
        float sum=0.0f;
        for(int t=0;t<len;t++) {
            int slot=(start+t)%cap;
            float v=__half2float(values[((size_t)slot*kvheads+kv)*hd+d]);
            sum=fmaf(probs[(size_t)head*len+t],v,sum);
        }
        out[(size_t)head*hd+d]=sum;
    }
}

// ---- graph decode: the token and the position come from device memory (st[0], st[1]), so one captured graph
// ---- serves every position. Grids are sized for the longest window; blocks past the live length write an
// ---- empty partial and leave.
extern "C" __global__ void k_g4_embed_dev(const unsigned char* weights,float* out,const int* st,int width,int type) {
    g4_embed_row(weights,out,st[0],width,type);
}
extern "C" __global__ void k_g4_rope_dev(float* x,int heads,int head_dim,int rope_dim,const int* st,float base,float scale) {
    g4_rope_one(x,heads,head_dim,rope_dim,st[1],base,scale);
}
extern "C" __global__ void k_g4_kv_store(const float* k,const float* v,__half* ck,__half* cv,const int* st,int cap,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    size_t o=(size_t)(st[1]%cap)*n+i; ck[o]=__float2half(k[i]); cv[o]=__float2half(v[i]);
}
// one block per (head, 256-position chunk): scores (one warp per position), chunk softmax, chunk-weighted values
// one block per (KV group, G4_ACH-position chunk): the G query heads of the group share every key and value read.
// Partials are [head][chunk] with head = kv*G + h, as k_g4_attn_dev_merge expects.
#define G4_ACH 32      /* decode attention: positions per chunk */
template<int G> __device__ __forceinline__ void g4_attn_group(const float* q,const __half* keys,const __half* values,
        float* pm,float* pl,float* pa,int pos,int kv,int chunk,int nchunks,int window,int cap,int kvheads,int hd) {
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int start=window>0?max(0,pos-(window-1)):0,len=pos-start+1,begin=chunk*G4_ACH,n=min(G4_ACH,len-begin);
    if(n<=0)return;                                // past the live length: the merge stops before this chunk
    __shared__ float qs[G][512]; __shared__ float ex[G][64];
    for(int i=threadIdx.x;i<G*hd;i+=blockDim.x)qs[i/hd][i%hd]=q[(size_t)kv*G*hd+i];
    __syncthreads();
    for(int t=warp;t<n;t+=8) {
        const __half* kp=keys+((size_t)((start+begin+t)%cap)*kvheads+kv)*hd; float sc[G];
#pragma unroll
        for(int h=0;h<G;h++)sc[h]=0.0f;
        const __half2* kp2=reinterpret_cast<const __half2*>(kp);
        for(int d=lane;d<(hd>>1);d+=32){float2 k=__half22float2(kp2[d]);
#pragma unroll
            for(int h=0;h<G;h++)sc[h]=fmaf(qs[h][2*d],k.x,fmaf(qs[h][2*d+1],k.y,sc[h]));}
#pragma unroll
        for(int h=0;h<G;h++){float v=g4_warp_sum(sc[h]);if(lane==0)ex[h][t]=v;}
    }
    __syncthreads();
    if(warp<G) {                                   // warp h: softmax statistics of head h over this chunk
        float mx=-3.402823466e+38F,z=0.0f;
        for(int t=lane;t<n;t+=32)mx=fmaxf(mx,ex[warp][t]);
        for(int o=16;o;o>>=1)mx=fmaxf(mx,KSHFLX(mx,o));
        for(int t=lane;t<n;t+=32){float e=expf(ex[warp][t]-mx);ex[warp][t]=e;z+=e;}
        z=g4_warp_sum(z);
        if(lane==0){size_t ci=(size_t)(kv*G+warp)*nchunks+chunk;pm[ci]=mx;pl[ci]=z;}
    }
    __syncthreads();
    for(int d=threadIdx.x;d<(hd>>1);d+=blockDim.x) {        // two dimensions per thread
        float ax[G],ay[G];
#pragma unroll
        for(int h=0;h<G;h++)ax[h]=ay[h]=0.0f;
#pragma unroll 4
        for(int t=0;t<n;t++){float2 v=__half22float2(reinterpret_cast<const __half2*>(values+((size_t)((start+begin+t)%cap)*kvheads+kv)*hd)[d]);
#pragma unroll
            for(int h=0;h<G;h++){ax[h]=fmaf(ex[h][t],v.x,ax[h]);ay[h]=fmaf(ex[h][t],v.y,ay[h]);}}
#pragma unroll
        for(int h=0;h<G;h++){size_t o=((size_t)(kv*G+h)*nchunks+chunk)*hd+2*d;pa[o]=ax[h];pa[o+1]=ay[h];}
    }
}
#define G4_ATTN(G) extern "C" __global__ void k_g4_attn_dev_part_g##G(const float* q,const __half* keys,const __half* values, \
        float* pm,float* pl,float* pa,const int* st,int window,int cap,int kvheads,int hd,int nchunks) \
    { g4_attn_group<G>(q,keys,values,pm,pl,pa,st[1],blockIdx.x,blockIdx.y,nchunks,window,cap,kvheads,hd); }
G4_ATTN(1) G4_ATTN(2) G4_ATTN(4) G4_ATTN(8)
extern "C" __global__ void k_g4_kv_store_batch(const float* k,const float* v,__half* ck,__half* cv,int pos0,int cap,int n,int nt) {
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)n*nt)return;
    int t=(int)(i/n),e=(int)(i%n); size_t o=(size_t)((pos0+t)%cap)*n+e;
    ck[o]=__float2half(k[i]); cv[o]=__float2half(v[i]);
}
// decode merge: only the chunks that hold live positions (the grid covers the whole cache)
extern "C" __global__ void k_g4_attn_dev_merge_live(const float* pm,const float* pl,const float* pa,float* out,
        const int* st,int window,int chunk,int nchunks,int hd) {
    int pos=st[1],len=window>0?min(pos+1,window):pos+1,live=min(nchunks,(len+chunk-1)/chunk);
    int head=blockIdx.x,lane=threadIdx.x; __shared__ float sh[256];
    float mx=-3.402823466e+38F;
    for(int c=lane;c<live;c+=blockDim.x)if(pl[(size_t)head*nchunks+c]>0.0f)mx=fmaxf(mx,pm[(size_t)head*nchunks+c]);
    sh[lane]=mx; __syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]=fmaxf(sh[lane],sh[lane+d]);__syncthreads();}
    mx=sh[0]; __syncthreads(); float z=0.0f;
    for(int c=lane;c<live;c+=blockDim.x){size_t ci=(size_t)head*nchunks+c;if(pl[ci]>0.0f)z+=pl[ci]*expf(pm[ci]-mx);}
    sh[lane]=z; __syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]+=sh[lane+d];__syncthreads();}
    float den=sh[0];
    for(int d=lane;d<hd;d+=blockDim.x){float acc=0.0f;
        for(int c=0;c<live;c++){size_t ci=(size_t)head*nchunks+c;if(pl[ci]>0.0f)acc=fmaf(pa[ci*hd+d],expf(pm[ci]-mx),acc);}
        out[(size_t)head*hd+d]=acc/den;
    }
}
extern "C" __global__ void k_g4_attn_dev_merge(const float* pm,const float* pl,const float* pa,float* out,int nchunks,int hd) {
    int head=blockIdx.x,lane=threadIdx.x; __shared__ float sh[256];
    float mx=-3.402823466e+38F;
    for(int c=lane;c<nchunks;c+=blockDim.x)if(pl[(size_t)head*nchunks+c]>0.0f)mx=fmaxf(mx,pm[(size_t)head*nchunks+c]);
    sh[lane]=mx; __syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]=fmaxf(sh[lane],sh[lane+d]);__syncthreads();}
    mx=sh[0]; __syncthreads(); float z=0.0f;
    for(int c=lane;c<nchunks;c+=blockDim.x){size_t ci=(size_t)head*nchunks+c;if(pl[ci]>0.0f)z+=pl[ci]*expf(pm[ci]-mx);}
    sh[lane]=z; __syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]+=sh[lane+d];__syncthreads();}
    float den=sh[0];
    for(int d=lane;d<hd;d+=blockDim.x){float a=0.0f;
        for(int c=0;c<nchunks;c++){size_t ci=(size_t)head*nchunks+c;if(pl[ci]>0.0f)a=fmaf(pa[ci*hd+d],expf(pm[ci]-mx),a);}
        out[(size_t)head*hd+d]=a/den;
    }
}

// ---- IQ4_XS matvec, one warp per row. Each lane takes half a 32-weight sub-block per step: 8 bytes = 16 codes
// ---- (8 low nibbles are columns j, 8 high nibbles are columns j+16), decoded through a shared table and
// ---- multiplied with the int8 input by dp4a. NM matrices share the input (NM=2: gate and up in one pass).
__device__ __forceinline__ unsigned int g4_prmt(unsigned int a,unsigned int b,unsigned int sel) {
#if KHIP
    unsigned long long x=((unsigned long long)b<<32)|a; unsigned int r=0;     /* prmt default mode, byte by byte */
    for(int i=0;i<4;i++){unsigned int n=(sel>>(4*i))&15u,v=(unsigned int)(x>>(8*(n&7)))&255u;
        if(n&8)v=(v&128u)?255u:0u; r|=v<<(8*i);}
    return r;
#else
    unsigned int r; asm("prmt.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(a),"r"(b),"r"(sel)); return r;
#endif
}
// four codes in the low 16 bits of w -> four int8 values: the table permutes pick by bits 0-2 of each code; the
// third permute replicates bit 3 of each code (moved to a byte's sign bit) into a byte mask choosing the table half
__device__ __forceinline__ int g4_iq4_4(const unsigned int* t,unsigned int w) {
#if KHIP
    /* v_perm_b32 takes one selector byte per output byte: spread the 4 codes to bytes, pick from each table half */
    unsigned int b=(w&0xFu)|((w&0xF0u)<<4)|((w&0xF00u)<<8)|((w&0xF000u)<<12),sel=b&0x07070707u;
    unsigned int lo=__builtin_amdgcn_perm(t[1],t[0],sel),hi=__builtin_amdgcn_perm(t[3],t[2],sel),m=((b>>3)&0x01010101u)*0xffu;
#else
    unsigned int sel=w&0x7777u,lo=g4_prmt(t[0],t[1],sel),hi=g4_prmt(t[2],t[3],sel),m=g4_prmt(w<<4,w,0xD9C8u);
#endif
    return (int)((lo&~m)|(hi&m));
}
// four IQ4 codes (the low nibbles of the bytes of w>>shift) -> four int8 values. byte_perm picks from the 16-byte
// table held in four registers; bit 3 of each code chooses the low or the high half of the table
__device__ __forceinline__ int g4_iq4_pack(const unsigned int* t,unsigned int w,int shift) {
    unsigned int n=(w>>shift)&0x0f0f0f0fu,c=n|(n>>4),sel=((c&0xffu)|((c>>8)&0xff00u))&0x7777u;
    unsigned int lo=__byte_perm(t[0],t[1],sel),hi=__byte_perm(t[2],t[3],sel),m=((n&0x08080808u)>>3)*0xffu;
    return (int)((lo&~m)|(hi&m));
}
// partial sums of the WPR warps of a row group meet in shared memory; warp part 0 writes the row
template<int N> __device__ __forceinline__ void g4_rows_out(float (&acc)[N],float* y,int r0,int rows,int stride,
        int rpw,int wpr,int part,int slot) {
    __shared__ float red[8][N];
    int lane=threadIdx.x&31;
#pragma unroll
    for(int i=0;i<N;i++){float v=g4_warp_sum(acc[i]);if(lane==0)red[threadIdx.x>>5][i]=v;}
    __syncthreads();
    if(part==0&&lane<N) {
        float v=0.0f; for(int w=0;w<wpr;w++)v+=red[slot*wpr+w][lane];
        int m=lane/rpw,r=lane%rpw; if(r0+r<rows)y[m*stride+r0+r]=v;
    }
}
template<int NM,int RPW,int WPR> __device__ __forceinline__ void g4_iq4w_rows(const unsigned char* a,const unsigned char* b,
        const signed char* xq,const float* xs,float* y,int rows,int cols) {
    unsigned int lut[4];
#pragma unroll
    for(int i=0;i<4;i++)lut[i]=reinterpret_cast<const unsigned int*>(g4_iq4xs_lut)[i];
    int lane=threadIdx.x&31,w=threadIdx.x>>5,slot=w/WPR,part=w%WPR;
    int r0=(blockIdx.x*(8/WPR)+slot)*RPW;
    int nb=cols>>8,halves=nb*16; float acc[NM*RPW];
#pragma unroll
    for(int i=0;i<NM*RPW;i++)acc[i]=0.0f;
    if(r0<rows)
    for(int h0=part*32;h0<halves;h0+=160*WPR) {
        uint2 q[NM][RPW][5];
#pragma unroll
        for(int m=0;m<NM;m++)
#pragma unroll
        for(int r=0;r<RPW;r++)
#pragma unroll
        for(int j=0;j<5;j++) {
            int h=h0+j*32*WPR+lane,rr=min(r0+r,rows-1);
            q[m][r][j]=h<halves?*reinterpret_cast<const uint2*>((m?b:a)+((size_t)rr*nb+(h>>4))*136+8+((h>>1)&7)*16+(h&1)*8):make_uint2(0u,0u);
        }
#pragma unroll
        for(int j=0;j<5;j++) {
            int h=h0+j*32*WPR+lane; if(h>=halves)break;
            int bi=h>>4,sb=(h>>1)&7,col=bi*256+sb*32;
            const int* xi=reinterpret_cast<const int*>(xq+col+(h&1)*8);
            int x0=xi[0],x1=xi[1],x2=xi[4],x3=xi[5]; float xsc=xs[col>>5];
#pragma unroll
            for(int m=0;m<NM;m++)
#pragma unroll
            for(int r=0;r<RPW;r++) {
                uint2 wq=q[m][r][j];
                int s=g4_dp4a(g4_iq4_pack(lut,wq.x,0),x0,0); s=g4_dp4a(g4_iq4_pack(lut,wq.y,0),x1,s);
                s=g4_dp4a(g4_iq4_pack(lut,wq.x,4),x2,s); s=g4_dp4a(g4_iq4_pack(lut,wq.y,4),x3,s);
                const unsigned char* blk=(m?b:a)+((size_t)min(r0+r,rows-1)*nb+bi)*136;
                uint2 hd=*reinterpret_cast<const uint2*>(blk);     /* d, scales_h, scales_l in one load */
                int sc=(((hd.y>>(4*sb))&15u)|(((hd.x>>(16+2*sb))&3u)<<4))-32;
                acc[m*RPW+r]+=(float)s*(__half2float(__ushort_as_half((unsigned short)(hd.x&0xffffu)))*(float)sc*xsc);
            }
        }
    }
    g4_rows_out<NM*RPW>(acc,y,r0,rows,rows,RPW,WPR,part,slot);
}
#define G4_IQ4W(W) \
extern "C" __global__ void k_g4_iq4w_mv_w##W(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows,int cols) \
    { g4_iq4w_rows<1,4,W>(w,w,xq,xs,y,rows,cols); } \
extern "C" __global__ void k_g4_iq4w_dual_mv_w##W(const unsigned char* aw,const unsigned char* bw,const signed char* xq,const float* xs,float* y,int rows,int cols) \
    { g4_iq4w_rows<2,2,W>(aw,bw,xq,xs,y,rows,cols); }
G4_IQ4W(1) G4_IQ4W(2) G4_IQ4W(4) G4_IQ4W(8)

// ---- K-quant matvec (Q3_K type 11, Q6_K type 14) with int8 input. Run t of a block (0..15): half n=t>>3,
// ---- group g=(t>>1)&3, sub-run r=t&1; it covers columns n*128+g*32+r*16 .. +15 (one int8 input group of 32).
__device__ __forceinline__ unsigned int g4_u32_at(const unsigned char* p) {     // 2-byte aligned 4-byte read
    const unsigned short* h=reinterpret_cast<const unsigned short*>(p);
    return (unsigned int)h[0]|((unsigned int)h[1]<<16);
}
// first column of run t (16 weights) of a 256-weight block
template<int TYPE> __device__ __forceinline__ int g4_kcol(int t) {
    return TYPE==8?t*16:TYPE==13?(t>>2)*64+((t>>1)&1)*32+(t&1)*16:(t>>3)*128+((t>>1)&3)*32+(t&1)*16;
}
// 16 bytes at p: one load for the 16-byte aligned layouts (padded Q3_K/Q6_K, Q5_K), else 2-byte pieces
template<int TYPE> __device__ __forceinline__ uint4 g4_w16(const unsigned char* p) {
    if(TYPE==111||TYPE==114||TYPE==13)return *reinterpret_cast<const uint4*>(p);
    return make_uint4(g4_u32_at(p),g4_u32_at(p+4),g4_u32_at(p+8),g4_u32_at(p+12));
}
__device__ __forceinline__ unsigned int g4_wk(const uint4& v,int k){return k==0?v.x:k==1?v.y:k==2?v.z:v.w;}
// run t of a block: its two 16-byte pieces (g4_kraw), then the 16 int8 weights, scale and minimum (g4_kfrom)
template<int TYPE> __device__ __forceinline__ void g4_kraw(const unsigned char* blk,int t,uint4& a,uint4& b) {
    const int B=TYPE==111?11:TYPE==114?14:TYPE;
    int n=t>>3,g=(t>>1)&3,r=t&1;
    if(B==8){a=g4_w16<TYPE>(blk+(t>>1)*34+2+(t&1)*16);b=a;}
    else if(B==11){a=g4_w16<TYPE>(blk+32+n*32+r*16);b=g4_w16<TYPE>(blk+r*16);}
    else if(B==14){a=g4_w16<TYPE>(blk+n*64+(g&1)*32+r*16);b=g4_w16<TYPE>(blk+128+n*32+r*16);}
    else if(B==10){a=g4_w16<TYPE>(blk+16+n*32+r*16);b=a;}
    else {a=g4_w16<TYPE>(blk+48+(t>>2)*32+r*16);b=g4_w16<TYPE>(blk+16+r*16);}
}
template<int TYPE> __device__ __forceinline__ void g4_kfrom(const unsigned char* blk,int t,const uint4& q,const uint4& h,
        unsigned int (&v)[4],float& sc,float& mn) {
    const int B=TYPE==111?11:TYPE==114?14:TYPE;
    int n=t>>3,g=(t>>1)&3,r=t&1; mn=0.0f;
    if(B==8){v[0]=q.x;v[1]=q.y;v[2]=q.z;v[3]=q.w;sc=g4_half(blk+(t>>1)*34);}
    else if(B==11) {
#pragma unroll
        for(int k=0;k<4;k++)v[k]=g4_vsub4((g4_wk(q,k)>>(2*g))&0x03030303u,(((g4_wk(h,k)>>(n*4+g))&0x01010101u)^0x01010101u)<<2);
        const unsigned char* c=blk+96; int is=n*8+g*2+r;
        sc=g4_half(blk+108)*(float)(((is<8?c[is]&15:c[is-8]>>4)|(((c[8+(is&3)]>>(2*(is>>2)))&3)<<4))-32);
    } else if(B==14) {
#pragma unroll
        for(int k=0;k<4;k++)v[k]=g4_vsub4(((g4_wk(q,k)>>(4*(g>>1)))&0x0f0f0f0fu)|(((g4_wk(h,k)>>(2*g))&0x03030303u)<<4),0x20202020u);
        sc=g4_half(blk+208)*(float)(signed char)blk[192+n*8+r+2*g];
    } else if(B==10) {
#pragma unroll
        for(int k=0;k<4;k++)v[k]=(g4_wk(q,k)>>(2*g))&0x03030303u;
        unsigned char c=blk[n*8+g*2+r]; sc=g4_half(blk+80)*(float)(c&15); mn=g4_half(blk+82)*(float)(c>>4);
    } else {   // 13: Q5_K
        int j=t>>2,hs=(t>>1)&1,is=2*j+hs;
#pragma unroll
        for(int k=0;k<4;k++)v[k]=((g4_wk(q,k)>>(4*hs))&0x0f0f0f0fu)|(((g4_wk(h,k)>>is)&0x01010101u)<<4);
        const unsigned char* c=blk+4; int a,m;
        g4_scale_min(is,c,&a,&m);
        sc=g4_half(blk)*(float)a; mn=g4_half(blk+2)*(float)m;
    }
}
template<int TYPE> __device__ __forceinline__ void g4_kdecode(const unsigned char* blk,int t,unsigned int (&v)[4],float& sc,float& mn) {
    uint4 a,b; g4_kraw<TYPE>(blk,t,a,b); g4_kfrom<TYPE>(blk,t,a,b,v,sc,mn);
}
template<int TYPE> __device__ __forceinline__ float g4_krun(const unsigned char* blk,int t,const int* xi,float xsc) {
    unsigned int v[4]; float sc,mn; g4_kdecode<TYPE>(blk,t,v,sc,mn);
    int s=0;
#pragma unroll
    for(int k=0;k<4;k++)s=g4_dp4a((int)v[k],xi[k],s);
    if(TYPE==10||TYPE==13) {
        int xsum=0;
#pragma unroll
        for(int k=0;k<4;k++)xsum=g4_dp4a(xi[k],0x01010101,xsum);
        return ((float)s*sc-(float)xsum*mn)*xsc;
    }
    return (float)s*sc*xsc;
}
template<int TYPE,int RPW,int WPR,int J> __device__ __forceinline__ void g4_kq_rows(const unsigned char* w,const signed char* xq,
        const float* xs,float* y,int rows,int cols) {
    const int BB=TYPE==8?272:TYPE==11?110:TYPE==111?112:TYPE==14?210:TYPE==114?224:TYPE==10?84:176;
    int lane=threadIdx.x&31,wi=threadIdx.x>>5,slot=wi/WPR,part=wi%WPR;
    int r0=(blockIdx.x*(8/WPR)+slot)*RPW;
    int nb=cols>>8,runs=nb*16; float acc[RPW];
#pragma unroll
    for(int r=0;r<RPW;r++)acc[r]=0.0f;
    if(r0<rows)
    for(int t0=part*32+lane;t0<runs;t0+=32*WPR*J) {
        uint4 qa[RPW][J],qb[RPW][J];
#pragma unroll
        for(int r=0;r<RPW;r++)
#pragma unroll
        for(int j=0;j<J;j++){int t=t0+j*32*WPR;
            if(t<runs)g4_kraw<TYPE>(w+((size_t)min(r0+r,rows-1)*nb+(t>>4))*BB,t&15,qa[r][j],qb[r][j]);}
#pragma unroll
        for(int j=0;j<J;j++) {
            int t=t0+j*32*WPR; if(t>=runs)break;
            int bi=t>>4,u=t&15,col=bi*256+g4_kcol<TYPE>(u);
            const int* xi=reinterpret_cast<const int*>(xq+col); float xsc=xs[col>>5];
            int xsum=0;
            if(TYPE==10||TYPE==13)
#pragma unroll
                for(int k=0;k<4;k++)xsum=g4_dp4a(xi[k],0x01010101,xsum);
#pragma unroll
            for(int r=0;r<RPW;r++) {
                unsigned int v[4]; float sc,mn;
                g4_kfrom<TYPE>(w+((size_t)min(r0+r,rows-1)*nb+bi)*BB,u,qa[r][j],qb[r][j],v,sc,mn);
                int s=0;
#pragma unroll
                for(int k=0;k<4;k++)s=g4_dp4a((int)v[k],xi[k],s);
                acc[r]+=((float)s*sc-(float)xsum*mn)*xsc;
            }
        }
    }
    g4_rows_out<RPW>(acc,y,r0,rows,rows,RPW,WPR,part,slot);
}
#define G4_KQ(W) \
extern "C" __global__ void k_g4_q3k_mv_w##W(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows,int cols) \
    { g4_kq_rows<11,4,W,1>(w,xq,xs,y,rows,cols); } \
extern "C" __global__ void k_g4_q6k_mv_w##W(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows,int cols) \
    { g4_kq_rows<14,2,W,(W>=8?1:W>=4?2:4)>(w,xq,xs,y,rows,cols); } \
extern "C" __global__ void k_g4_q3p_mv_w##W(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows,int cols) \
    { g4_kq_rows<111,(KHIP?2:8),W,1>(w,xq,xs,y,rows,cols); } \
extern "C" __global__ void k_g4_q6p_mv_w##W(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows,int cols) \
    { g4_kq_rows<114,4,W,(W>=4?1:2)>(w,xq,xs,y,rows,cols); } \
extern "C" __global__ void k_g4_q8k_mv_w##W(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows,int cols) \
    { g4_kq_rows<8,2,W,(W>=8?1:W>=4?2:4)>(w,xq,xs,y,rows,cols); } \
extern "C" __global__ void k_g4_q2k_mv_w##W(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows,int cols) \
    { g4_kq_rows<10,4,W,(W>=4?1:2)>(w,xq,xs,y,rows,cols); } \
extern "C" __global__ void k_g4_q5k_mv_w##W(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows,int cols) \
    { g4_kq_rows<13,4,W,(W>=4?1:2)>(w,xq,xs,y,rows,cols); }
G4_KQ(1) G4_KQ(2) G4_KQ(4) G4_KQ(8)

// ---- prompt GEMM. g4_run<TYPE>: run u (0..15) of a 256-weight block -> 16 int8 weights, scale, minimum, first column
template<int TYPE> __device__ __forceinline__ int g4_run_col(int u) {
    return (TYPE==23||TYPE==8||TYPE==123||TYPE==108)?u*16:g4_kcol<TYPE>(u);
}
template<int TYPE> __device__ __forceinline__ int g4_blk_bytes() {
    return TYPE==23?136:TYPE==11?110:TYPE==111?112:TYPE==14?210:TYPE==114?224:TYPE==10?84:TYPE==13?176:272;
}
template<int TYPE> __device__ __forceinline__ void g4_run(const unsigned char* blk,int u,uint4& q,float& sc,float& mn) {
    unsigned int v[4]; mn=0.0f;
    if(TYPE==23) {
        unsigned int lut[4];
#pragma unroll
        for(int i=0;i<4;i++)lut[i]=reinterpret_cast<const unsigned int*>(g4_iq4xs_lut)[i];
        int sb=u>>1,part=u&1;
        const unsigned int* w=reinterpret_cast<const unsigned int*>(blk+8+sb*16);
#pragma unroll
        for(int k=0;k<4;k++)v[k]=(unsigned int)g4_iq4_pack(lut,w[k],part*4);
        uint2 hd=*reinterpret_cast<const uint2*>(blk);
        int s6=(((hd.y>>(4*sb))&15u)|(((hd.x>>(16+2*sb))&3u)<<4))-32;
        sc=__half2float(__ushort_as_half((unsigned short)(hd.x&0xffffu)))*(float)s6;
    } else if(TYPE==8) {
        const unsigned char* b=blk+(u>>1)*34;
#pragma unroll
        for(int k=0;k<4;k++)v[k]=g4_u32_at(b+2+(u&1)*16+4*k);
        sc=g4_half(b);
    } else g4_kdecode<TYPE>(blk,u,v,sc,mn);
    q=make_uint4(v[0],v[1],v[2],v[3]);
}
// run u of 256-weight block b of a row, in any layout the prompt GEMM reads -> 16 int8 weights, scale, minimum
template<int TYPE> __device__ __forceinline__ void g4_load_run(const unsigned char* w,int rows,int cols,int nb,int row,int b,int u,
        uint4& q,float& sc,float& mn) {
    if(TYPE==108) {                              // Kindi Q8 layout
        q=*reinterpret_cast<const uint4*>(w+(size_t)row*cols+b*256+u*16);
        sc=__half2float(__ushort_as_half(reinterpret_cast<const unsigned short*>(w+(size_t)rows*cols)[(size_t)row*(cols>>5)+b*8+(u>>1)]));
        mn=0.0f;
    } else if(TYPE==123) {                       // Kindi IQ4 layout
        unsigned int lut[4];
#pragma unroll
        for(int i=0;i<4;i++)lut[i]=reinterpret_cast<const unsigned int*>(g4_iq4xs_lut)[i];
        int sb=b*8+(u>>1),S=cols>>5;
        uint2 raw=*reinterpret_cast<const uint2*>(w+(size_t)row*(cols/2)+(size_t)sb*16+(u&1)*8);
        q=make_uint4(g4_iq4_4(lut,raw.x),g4_iq4_4(lut,raw.x>>16),g4_iq4_4(lut,raw.y),g4_iq4_4(lut,raw.y>>16));
        size_t qsz=(size_t)rows*(cols/2);
        sc=__half2float(__ushort_as_half(reinterpret_cast<const unsigned short*>(w+qsz)[(size_t)row*nb+b]))
          *(float)reinterpret_cast<const signed char*>(w+qsz+(size_t)rows*nb*2)[(size_t)row*S+sb];
        mn=0.0f;
    } else
    g4_run<TYPE>(w+((size_t)row*nb+b)*g4_blk_bytes<TYPE>(),u,q,sc,mn);
}
// y[t*ldy+row] = sum_k W[row,k]*x[t,k] for 64 rows x 64 tokens per block; xq/xs: int8 input and its per-32 scales
template<int TYPE> __device__ __forceinline__ void g4_mmq(const unsigned char* w,const signed char* xq,const float* xs,
        float* y,int rows,int cols,int nt,int ldy) {
    const int MINS=(TYPE==10||TYPE==13);
    __shared__ uint4 W[64][16];            // [row][run]
    __shared__ uint4 X[64][17];            // [token][16-column group]
    __shared__ float WS[64][16],WM[MINS?64:1][16],XS[64][8];
    __shared__ int XSUM[MINS?64:1][16];
    int tid=threadIdx.x,tx=tid&15,ty=tid>>4,row0=blockIdx.x*64,tok0=blockIdx.y*64,nb=cols>>8;
    float acc[4][4];
#pragma unroll
    for(int i=0;i<4;i++)
#pragma unroll
        for(int j=0;j<4;j++)acc[i][j]=0.0f;
    for(int b=0;b<nb;b++) {
#pragma unroll
        for(int k=0;k<4;k++) {                           // weights: 1024 runs, four per thread
            int idx=tid+k*256,r=idx>>4,u=idx&15,row=min(row0+r,rows-1);
            uint4 q; float sc,mn;
            g4_load_run<TYPE>(w,rows,cols,nb,row,b,u,q,sc,mn);
            W[r][u]=q; WS[r][u]=sc; if(MINS)WM[r][u]=mn;
        }
#pragma unroll
        for(int k=0;k<4;k++) {                           // input: 64 tokens x 16 groups of 16 bytes
            int idx=tid+k*256,t=idx>>4,c=idx&15,tok=tok0+t;
            X[t][c]=tok<nt?*reinterpret_cast<const uint4*>(xq+(size_t)tok*cols+b*256+c*16):make_uint4(0u,0u,0u,0u);
        }
        for(int idx=tid;idx<512;idx+=256){int t=idx>>3,g=idx&7,tok=tok0+t;XS[t][g]=tok<nt?xs[((size_t)tok*cols+b*256)/32+g]:0.0f;}
        __syncthreads();
        if(MINS)
#pragma unroll
        for(int k=0;k<4;k++){int idx=tid+k*256,t=idx>>4,u=idx&15;uint4 v=X[t][g4_run_col<TYPE>(u)>>4];
            XSUM[t][u]=g4_dp4a((int)v.w,0x01010101,g4_dp4a((int)v.z,0x01010101,g4_dp4a((int)v.y,0x01010101,g4_dp4a((int)v.x,0x01010101,0))));}
        if(MINS)__syncthreads();
        // runs u and u+1 share one scale for IQ4, Q8_0 and Q5_K: their integer dots add before the float scaling
        const int PAIR=(TYPE==23||TYPE==123||TYPE==8||TYPE==108||TYPE==13);
        if(PAIR)
#pragma unroll 2
        for(int u=0;u<16;u+=2) {
            int c=g4_run_col<TYPE>(u)>>4,g=c>>1;
            uint4 wa[4],wb[4],xa[4],xb[4]; float ws[4],wm[4],xsc[4]; int xsum[4];
#pragma unroll
            for(int i=0;i<4;i++){wa[i]=W[ty+16*i][u];wb[i]=W[ty+16*i][u+1];ws[i]=WS[ty+16*i][u];if(MINS)wm[i]=WM[ty+16*i][u];}
#pragma unroll
            for(int j=0;j<4;j++){xa[j]=X[tx+16*j][c];xb[j]=X[tx+16*j][c+1];xsc[j]=XS[tx+16*j][g];if(MINS)xsum[j]=XSUM[tx+16*j][u]+XSUM[tx+16*j][u+1];}
#pragma unroll
            for(int i=0;i<4;i++)
#pragma unroll
            for(int j=0;j<4;j++) {
                int d=g4_dp4a((int)wa[i].x,(int)xa[j].x,0); d=g4_dp4a((int)wa[i].y,(int)xa[j].y,d);
                d=g4_dp4a((int)wa[i].z,(int)xa[j].z,d); d=g4_dp4a((int)wa[i].w,(int)xa[j].w,d);
                d=g4_dp4a((int)wb[i].x,(int)xb[j].x,d); d=g4_dp4a((int)wb[i].y,(int)xb[j].y,d);
                d=g4_dp4a((int)wb[i].z,(int)xb[j].z,d); d=g4_dp4a((int)wb[i].w,(int)xb[j].w,d);
                float f=(float)d*ws[i]; if(MINS)f-=(float)xsum[j]*wm[i];
                acc[i][j]=fmaf(f,xsc[j],acc[i][j]);
            }
        }
        else
#pragma unroll 4
        for(int u=0;u<16;u++) {
            int c=g4_run_col<TYPE>(u)>>4,g=c>>1;
            uint4 wv[4],xv[4]; float ws[4],wm[4],xsc[4]; int xsum[4];
#pragma unroll
            for(int i=0;i<4;i++){wv[i]=W[ty+16*i][u];ws[i]=WS[ty+16*i][u];if(MINS)wm[i]=WM[ty+16*i][u];}
#pragma unroll
            for(int j=0;j<4;j++){xv[j]=X[tx+16*j][c];xsc[j]=XS[tx+16*j][g];if(MINS)xsum[j]=XSUM[tx+16*j][u];}
#pragma unroll
            for(int i=0;i<4;i++)
#pragma unroll
            for(int j=0;j<4;j++) {
                int d=g4_dp4a((int)wv[i].x,(int)xv[j].x,0); d=g4_dp4a((int)wv[i].y,(int)xv[j].y,d);
                d=g4_dp4a((int)wv[i].z,(int)xv[j].z,d); d=g4_dp4a((int)wv[i].w,(int)xv[j].w,d);
                float f=(float)d*ws[i]; if(MINS)f-=(float)xsum[j]*wm[i];
                acc[i][j]=fmaf(f,xsc[j],acc[i][j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int i=0;i<4;i++)
#pragma unroll
    for(int j=0;j<4;j++){int row=row0+ty+16*i,tok=tok0+tx+16*j;if(row<rows&&tok<nt)y[(size_t)tok*ldy+row]=acc[i][j];}
}
#define G4_MMQ(T) extern "C" __global__ void __launch_bounds__(256) k_g4_mmq_t##T(const unsigned char* w,const signed char* xq,const float* xs, \
    float* y,int rows,int cols,int nt,int ldy){ g4_mmq<T>(w,xq,xs,y,rows,cols,nt,ldy); }
G4_MMQ(8) G4_MMQ(10) G4_MMQ(11) G4_MMQ(13) G4_MMQ(14) G4_MMQ(23) G4_MMQ(123) G4_MMQ(111) G4_MMQ(114) G4_MMQ(108)

// ---- Kindi IQ4 layout (type 123): QS | D | SC, see gemma4.py repack_iq4
extern "C" __global__ void k_g4_iq4_repack(const unsigned char* src,unsigned char* dst,int rows,int cols) {
    int nb=cols>>8; size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)rows*nb)return;
    int row=(int)(i/nb),b=(int)(i%nb); const unsigned char* blk=src+i*136;
    unsigned char* qs=dst+(size_t)row*(cols/2)+b*128;
    unsigned short* d=reinterpret_cast<unsigned short*>(dst+(size_t)rows*(cols/2))+i;
    signed char* sc=reinterpret_cast<signed char*>(dst+(size_t)rows*(cols/2)+(size_t)rows*nb*2)+(size_t)row*(cols/32)+b*8;
    for(int sb=0;sb<8;sb++)                    /* byte b of a sub-block: columns 2b (low nibble) and 2b+1 */
        for(int b2=0;b2<16;b2++){int c0=2*b2,c1=2*b2+1;
            unsigned char n0=(blk[8+sb*16+(c0&15)]>>(c0>=16?4:0))&15,n1=(blk[8+sb*16+(c1&15)]>>(c1>=16?4:0))&15;
            qs[sb*16+b2]=(unsigned char)(n0|(n1<<4));}
    *d=*reinterpret_cast<const unsigned short*>(blk);
    unsigned short sh=*reinterpret_cast<const unsigned short*>(blk+2);
    for(int sb=0;sb<8;sb++)sc[sb]=(signed char)((((blk[4+(sb>>1)]>>(4*(sb&1)))&15)|(((sh>>(2*sb))&3)<<4))-32);
}
// matvec on the Kindi IQ4 layout: a half-warp per row (RPH rows per half-warp), a lane takes a whole 32-weight
// sub-block per step (one 16-byte load), WPR warps split the columns
template<int NM,int RPH,int WPR> __device__ __forceinline__ void g4_k4_rows(const unsigned char* a,const unsigned char* b,
        const signed char* xq,const float* xs,float* y,int rows,int cols) {
    unsigned int lut[4];
#pragma unroll
    for(int i=0;i<4;i++)lut[i]=reinterpret_cast<const unsigned int*>(g4_iq4xs_lut)[i];
    const int RPW=2*RPH;
    int lane=threadIdx.x&31,hw=lane>>4,l16=lane&15,w=threadIdx.x>>5,slot=w/WPR,part=w%WPR;
    int r0=(blockIdx.x*(8/WPR)+slot)*RPW,nb=cols>>8,S=cols>>5;
    size_t qsz=(size_t)rows*(cols/2),dsz=(size_t)rows*nb*2;
    float acc[NM*RPW],mine[NM][RPH];
#pragma unroll
    for(int m=0;m<NM;m++)
#pragma unroll
        for(int r=0;r<RPH;r++)mine[m][r]=0.0f;
    if(r0<rows)
    for(int s0=part*16;s0<S;s0+=80*WPR) {
        uint4 q[NM][RPH][5];
#pragma unroll
        for(int m=0;m<NM;m++)
#pragma unroll
        for(int r=0;r<RPH;r++)
#pragma unroll
        for(int j=0;j<5;j++) {
            int sb=s0+j*16*WPR+l16,row=min(r0+hw*RPH+r,rows-1);
            q[m][r][j]=sb<S?*reinterpret_cast<const uint4*>((m?b:a)+(size_t)row*(cols/2)+(size_t)sb*16):make_uint4(0u,0u,0u,0u);
        }
#pragma unroll
        for(int j=0;j<5;j++) {
            int sb=s0+j*16*WPR+l16; if(sb>=S)break;
            const int4* xi=reinterpret_cast<const int4*>(xq+sb*32); int4 xa=xi[0],xb=xi[1]; float xsc=xs[sb];
#pragma unroll
            for(int m=0;m<NM;m++)
#pragma unroll
            for(int r=0;r<RPH;r++) {
                uint4 v=q[m][r][j]; int row=min(r0+hw*RPH+r,rows-1);
                int s=g4_dp4a(g4_iq4_4(lut,v.x),xa.x,0); s=g4_dp4a(g4_iq4_4(lut,v.x>>16),xa.y,s);
                s=g4_dp4a(g4_iq4_4(lut,v.y),xa.z,s); s=g4_dp4a(g4_iq4_4(lut,v.y>>16),xa.w,s);
                s=g4_dp4a(g4_iq4_4(lut,v.z),xb.x,s); s=g4_dp4a(g4_iq4_4(lut,v.z>>16),xb.y,s);
                s=g4_dp4a(g4_iq4_4(lut,v.w),xb.z,s); s=g4_dp4a(g4_iq4_4(lut,v.w>>16),xb.w,s);
                const unsigned char* base=m?b:a;
                float d=__half2float(__ushort_as_half(reinterpret_cast<const unsigned short*>(base+qsz)[(size_t)row*nb+(sb>>3)]));
                float sc=(float)reinterpret_cast<const signed char*>(base+qsz+dsz)[(size_t)row*S+sb];
                mine[m][r]+=(float)s*(d*sc*xsc);
            }
        }
    }
    // each half-warp holds its own rows; g4_rows_out sums all 32 lanes, so the other half adds zeros
#pragma unroll
    for(int m=0;m<NM;m++)
#pragma unroll
    for(int r=0;r<RPH;r++){acc[m*RPW+r]=hw==0?mine[m][r]:0.0f;acc[m*RPW+RPH+r]=hw==1?mine[m][r]:0.0f;}
    g4_rows_out<NM*RPW>(acc,y,r0,rows,rows,RPW,WPR,part,slot);
}
#define G4_K4(W) \
extern "C" __global__ void k_g4_k4_mv_w##W(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows,int cols) \
    { g4_k4_rows<1,2,W>(w,w,xq,xs,y,rows,cols); } \
extern "C" __global__ void k_g4_k4_dual_mv_w##W(const unsigned char* aw,const unsigned char* bw,const signed char* xq,const float* xs,float* y,int rows,int cols) \
    { g4_k4_rows<2,1,W>(aw,bw,xq,xs,y,rows,cols); }
G4_K4(1) G4_K4(2) G4_K4(4) G4_K4(8)

// copy blocks into a wider stride (Q3_K 110 -> 112, Q6_K 210 -> 224 bytes); the tail of each block is zero
extern "C" __global__ void k_g4_pad_blocks(const unsigned char* src,unsigned char* dst,long long nblocks,int sb,int db) {
    long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x; if(i>=nblocks*db)return;
    long long blk=i/db; int o=(int)(i%db); dst[i]=o<sb?src[blk*sb+o]:0;
}

// ---- fused decode kernels (graph path) ----
__device__ __forceinline__ float g4_block_sum256(float v,float* sh) {     // the k_g4_rmsnorm reduction
    sh[threadIdx.x]=v; __syncthreads();
    for(int d=blockDim.x>>1;d;d>>=1){if(threadIdx.x<d)sh[threadIdx.x]+=sh[threadIdx.x+d];__syncthreads();}
    float r=sh[0]; __syncthreads(); return r;
}
// quantize buf[0..n) (shared memory) in groups of 32, as k_g4_q32 does
__device__ __forceinline__ void g4_q32_block(const float* buf,int n,signed char* q,float* scale) {
    int lane=threadIdx.x&31;
    for(int g=threadIdx.x>>5;g<n/32;g+=blockDim.x>>5) {          // one warp per group
        float v=buf[g*32+lane],m=fabsf(v);
        for(int o=16;o;o>>=1)m=fmaxf(m,KSHFLX(m,o));
        float sc=m>0.0f?m/127.0f:1.0f; if(lane==0)scale[g]=sc;
        q[g*32+lane]=(signed char)__float2int_rn(v/sc);
    }
}
// y = rmsnorm(x)*w, quantized (the attention input)
extern "C" __global__ void k_g4_norm_q32(const float* x,const float* w,signed char* q,float* scale,int width,float eps) {
    __shared__ float sh[1024]; __shared__ float buf[4096];
    float v=0.0f; for(int i=threadIdx.x;i<width;i+=blockDim.x)v=fmaf(x[i],x[i],v);
    float r=rsqrtf(g4_block_sum256(v,sh)/width+eps);
    for(int i=threadIdx.x;i<width;i+=blockDim.x)buf[i]=x[i]*r*w[i];
    __syncthreads(); g4_q32_block(buf,width,q,scale);
}
// q: per-head rmsnorm, then rope at st[1] (one block per head)
extern "C" __global__ void k_g4_qnorm_rope(const float* q,const float* w,float* out,int heads,int hd,int rope_dim,
        const int* st,float base,float scale,float eps) {
    __shared__ float sh[1024]; const float* x=q+(size_t)blockIdx.x*hd; float* y=out+(size_t)blockIdx.x*hd;
    float v=0.0f; for(int i=threadIdx.x;i<hd;i+=blockDim.x)v=fmaf(x[i],x[i],v);
    float r=rsqrtf(g4_block_sum256(v,sh)/hd+eps);
    for(int i=threadIdx.x;i<hd;i+=blockDim.x)y[i]=x[i]*r*w[i];
    __syncthreads(); g4_rope_one(out,heads,hd,rope_dim,st[1],base,scale);
}
// k: per-head rmsnorm and rope; v: per-head rmsnorm without weight; both stored to the cache slot of st[1]
extern "C" __global__ void k_g4_kv_norm_rope_store(const float* k,const float* v,const float* kw,__half* ck,__half* cv,
        int kvheads,int hd,int rope_dim,const int* st,float base,float scale,float eps,int cap) {
    __shared__ float sh[1024]; __shared__ float kb[512];
    int h=blockIdx.x; const float* kx=k+(size_t)h*hd; const float* vx=v+(size_t)h*hd;
    size_t o=((size_t)(st[1]%cap)*kvheads+h)*hd;
    float a=0.0f; for(int i=threadIdx.x;i<hd;i+=blockDim.x)a=fmaf(kx[i],kx[i],a);
    float r=rsqrtf(g4_block_sum256(a,sh)/hd+eps);
    for(int i=threadIdx.x;i<hd;i+=blockDim.x)kb[i]=kx[i]*r*kw[i];
    a=0.0f; for(int i=threadIdx.x;i<hd;i+=blockDim.x)a=fmaf(vx[i],vx[i],a);
    float rv=rsqrtf(g4_block_sum256(a,sh)/hd+eps);
    for(int i=threadIdx.x;i<hd;i+=blockDim.x)cv[o+i]=__float2half(vx[i]*rv*1.0f);
    __syncthreads();
    int half=hd>>1,i=threadIdx.x;
    if(i<(rope_dim>>1)) {
        float angle=st[1]/(powf(base,(2.0f/rope_dim)*i)*scale),sn,cs; sincosf(angle,&sn,&cs);
        float x0=kb[i],x1=kb[i+half]; kb[i]=x0*cs-x1*sn; kb[i+half]=x1*cs+x0*sn;
    }
    __syncthreads();
    for(int j=threadIdx.x;j<hd;j+=blockDim.x)ck[o+j]=__float2half(kb[j]);
}
// the attention merge, quantized for the output projection (one block per head; hd is a multiple of 32)
extern "C" __global__ void k_g4_attn_merge_q32(const float* pm,const float* pl,const float* pa,signed char* q,float* scale,
        const int* st,int window,int chunk,int nchunks,int hd) {
    int pos=st[1],len=window>0?min(pos+1,window):pos+1,live=min(nchunks,(len+chunk-1)/chunk);
    int head=blockIdx.x,lane=threadIdx.x; __shared__ float sh[1024]; __shared__ float buf[512];
    float mx=-3.402823466e+38F;
    for(int c=lane;c<live;c+=blockDim.x)if(pl[(size_t)head*nchunks+c]>0.0f)mx=fmaxf(mx,pm[(size_t)head*nchunks+c]);
    sh[lane]=mx; __syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]=fmaxf(sh[lane],sh[lane+d]);__syncthreads();}
    mx=sh[0]; __syncthreads(); float z=0.0f;
    for(int c=lane;c<live;c+=blockDim.x){size_t ci=(size_t)head*nchunks+c;if(pl[ci]>0.0f)z+=pl[ci]*expf(pm[ci]-mx);}
    sh[lane]=z; __syncthreads();
    for(int d=128;d;d>>=1){if(lane<d)sh[lane]+=sh[lane+d];__syncthreads();}
    float den=sh[0];
    for(int d=lane;d<hd;d+=blockDim.x){float acc=0.0f;
        for(int c=0;c<live;c++){size_t ci=(size_t)head*nchunks+c;if(pl[ci]>0.0f)acc=fmaf(pa[ci*hd+d],expf(pm[ci]-mx),acc);}
        buf[d]=acc/den;
    }
    __syncthreads(); g4_q32_block(buf,hd,q+(size_t)head*hd,scale+(size_t)head*hd/32);
}
// h = x + rmsnorm(att)*wa; fn = rmsnorm(h)*wf, quantized (k_g4_attn_ffn_norm_residual + k_g4_q32)
extern "C" __global__ void k_g4_norm_residual_q32(const float* x,const float* att,const float* wa,const float* wf,float* h,
        signed char* q,float* scale,int width,float eps) {
    __shared__ float sh[1024]; __shared__ float buf[4096];
    int t=threadIdx.x; float v=0.0f;
    for(int i=t;i<width;i+=blockDim.x)v=fmaf(att[i],att[i],v);
    float r=rsqrtf(g4_block_sum256(v,sh)/width+eps); v=0.0f;
    for(int i=t;i<width;i+=blockDim.x){float a=x[i]+att[i]*r*wa[i];buf[i]=a;h[i]=a;v=fmaf(a,a,v);}
    r=rsqrtf(g4_block_sum256(v,sh)/width+eps);
    for(int i=t;i<width;i+=blockDim.x)buf[i]=buf[i]*r*wf[i];
    __syncthreads(); g4_q32_block(buf,width,q,scale);
}
// gelu(gate)*up, quantized: one warp per group of 32 (gu = gate rows then up rows)
extern "C" __global__ void k_g4_gelu_mul_q32(const float* gu,signed char* q,float* scale,int n) {
    int g=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31,i=g*32+lane; if(g*32>=n)return;
    float x=gu[i],y=0.5f*x*(1.0f+erff(x*0.7071067811865475f))*gu[n+i],m=fabsf(y);
    for(int o=16;o;o>>=1)m=fmaxf(m,KSHFLX(m,o));
    float sc=m>0.0f?m/127.0f:1.0f; if(lane==0)scale[g]=sc; q[i]=(signed char)__float2int_rn(y/sc);
}
// h += rmsnorm(ff)*w; quantized h (the per-layer input gate reads it)
extern "C" __global__ void k_g4_norm_add_q32(const float* ff,const float* w,float* h,signed char* q,float* scale,int width,float eps) {
    __shared__ float sh[1024]; __shared__ float buf[4096];
    float v=0.0f; for(int i=threadIdx.x;i<width;i+=blockDim.x)v=fmaf(ff[i],ff[i],v);
    float r=rsqrtf(g4_block_sum256(v,sh)/width+eps);
    for(int i=threadIdx.x;i<width;i+=blockDim.x){float a=h[i]+ff[i]*r*w[i];h[i]=a;buf[i]=a;}
    __syncthreads(); g4_q32_block(buf,width,q,scale);
}
// x = (h + rmsnorm(mapped)*w) * layer_scale
extern "C" __global__ void k_g4_norm_add_scale(const float* mapped,const float* w,const float* h,float* x,int width,float eps,float ls) {
    __shared__ float sh[1024];
    float v=0.0f; for(int i=threadIdx.x;i<width;i+=blockDim.x)v=fmaf(mapped[i],mapped[i],v);
    float r=rsqrtf(g4_block_sum256(v,sh)/width+eps);
    for(int i=threadIdx.x;i<width;i+=blockDim.x)x[i]=(h[i]+mapped[i]*r*w[i])*ls;
}

// gelu(mapped)*ple, quantized for the per-layer projection (one warp per group of 32)
extern "C" __global__ void k_g4_ple_gate_q32(const float* mapped,const float* ple,signed char* q,float* scale,int n) {
    int g=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31,i=g*32+lane; if(g*32>=n)return;
    float x=mapped[i],y=(0.5f*x*(1.0f+erff(x*0.7071067811865475f)))*ple[i],m=fabsf(y);
    for(int o=16;o;o>>=1)m=fmaxf(m,KSHFLX(m,o));
    float sc=m>0.0f?m/127.0f:1.0f; if(lane==0)scale[g]=sc; q[i]=(signed char)__float2int_rn(y/sc);
}

// Q8_0 matvec for 256 columns (one row = 8 blocks = 16 runs): 16 lanes per row, one run each
extern "C" __global__ void k_g4_q8_mv256(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows) {
    int row=blockIdx.x*(blockDim.x>>4)+(threadIdx.x>>4),u=threadIdx.x&15; if(row>=rows)return;
    const unsigned char* b=w+(size_t)row*272+(u>>1)*34;
    const int* xi=reinterpret_cast<const int*>(xq+u*16); int s=0;
#pragma unroll
    for(int k=0;k<4;k++)s=g4_dp4a((int)g4_u32_at(b+2+(u&1)*16+4*k),xi[k],s);
    float v=(float)s*g4_half(b)*xs[u>>1];
    v+=KSHFLX(v,8); v+=KSHFLX(v,4); v+=KSHFLX(v,2); v+=KSHFLX(v,1);
    if(u==0)y[row]=v;
}

// ---- Kindi Q8 layout (type 108): QS | D
extern "C" __global__ void k_g4_q8_repack(const unsigned char* src,unsigned char* dst,int rows,int cols) {
    int nb=cols>>5; size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)rows*nb)return;
    const unsigned char* b=src+i*34; unsigned char* q=dst+i*32;
    for(int k=0;k<32;k++)q[k]=b[2+k];
    reinterpret_cast<unsigned short*>(dst+(size_t)rows*cols)[i]=*reinterpret_cast<const unsigned short*>(b);
}
// matvec on the Kindi Q8 layout: 16 lanes per row, one 16-weight run per lane per step
extern "C" __global__ void k_g4_q8r_mv(const unsigned char* w,const signed char* xq,const float* xs,float* y,int rows,int cols) {
    int row=blockIdx.x*(blockDim.x>>4)+(threadIdx.x>>4),u=threadIdx.x&15; if(row>=rows)return;
    const unsigned short* d=reinterpret_cast<const unsigned short*>(w+(size_t)rows*cols)+(size_t)row*(cols>>5);
    float v=0.0f;
    for(int c=u*16;c<cols;c+=256) {
        uint4 q=*reinterpret_cast<const uint4*>(w+(size_t)row*cols+c); int4 x=*reinterpret_cast<const int4*>(xq+c);
        int s=g4_dp4a((int)q.x,x.x,0); s=g4_dp4a((int)q.y,x.y,s); s=g4_dp4a((int)q.z,x.z,s); s=g4_dp4a((int)q.w,x.w,s);
        v+=(float)s*__half2float(__ushort_as_half(d[c>>5]))*xs[c>>5];
    }
    v+=KSHFLX(v,8); v+=KSHFLX(v,4); v+=KSHFLX(v,2); v+=KSHFLX(v,1);
    if(u==0)y[row]=v;
}

// prompt attention: in-place causal (and sliding-window) softmax of score rows. Row r is query r/gs at position
// q0+r/gs; column j is the key at position k0+j. One block per row.
extern "C" __global__ void k_g4_softmax_mask(float* S,int L,int gs,int q0,int k0,int window) {
    __shared__ float sh[256]; float* row=S+(size_t)blockIdx.x*L; int qp=q0+blockIdx.x/gs;
    float mx=-3.402823466e+38F;
    for(int j=threadIdx.x;j<L;j+=blockDim.x){int kp=k0+j;bool ok=kp<=qp&&(window<=0||kp>qp-window);if(ok)mx=fmaxf(mx,row[j]);}
    sh[threadIdx.x]=mx; __syncthreads();
    for(int d=blockDim.x>>1;d;d>>=1){if(threadIdx.x<d)sh[threadIdx.x]=fmaxf(sh[threadIdx.x],sh[threadIdx.x+d]);__syncthreads();}
    mx=sh[0]; __syncthreads(); float z=0.0f;
    for(int j=threadIdx.x;j<L;j+=blockDim.x){int kp=k0+j;bool ok=kp<=qp&&(window<=0||kp>qp-window);float e=ok?expf(row[j]-mx):0.0f;row[j]=e;z+=e;}
    sh[threadIdx.x]=z; __syncthreads();
    for(int d=blockDim.x>>1;d;d>>=1){if(threadIdx.x<d)sh[threadIdx.x]+=sh[threadIdx.x+d];__syncthreads();}
    float inv=1.0f/sh[0];
    for(int j=threadIdx.x;j<L;j+=blockDim.x)row[j]*=inv;
}
// decode merge, quantized: block (head, slice of 64 dimensions); each block redoes the small max/sum over the chunks
extern "C" __global__ void k_g4_attn_merge64_q32(const float* pm,const float* pl,const float* pa,signed char* q,float* scale,
        const int* st,int window,int chunk,int nchunks,int hd) {
    int pos=st[1],len=window>0?min(pos+1,window):pos+1,live=min(nchunks,(len+chunk-1)/chunk);
    int head=blockIdx.x,d0=blockIdx.y*64,t=threadIdx.x; __shared__ float sh[64]; __shared__ float part[4][64];
    const float* m=pm+(size_t)head*nchunks; const float* l=pl+(size_t)head*nchunks;
    float mx=-3.402823466e+38F; if(t<64)for(int c=t;c<live;c+=64)if(l[c]>0.0f)mx=fmaxf(mx,m[c]);   // warps 0-1 only
    for(int o=16;o;o>>=1)mx=fmaxf(mx,KSHFLX(mx,o));
    if((t&31)==0)sh[t>>5]=mx; __syncthreads(); mx=fmaxf(sh[0],sh[1]); __syncthreads();
    float z=0.0f; if(t<64)for(int c=t;c<live;c+=64)if(l[c]>0.0f)z+=l[c]*expf(m[c]-mx);
    for(int o=16;o;o>>=1)z+=KSHFLX(z,o);
    if((t&31)==0)sh[t>>5]=z; __syncthreads(); float den=sh[0]+sh[1]; __syncthreads();
    // 256 threads: 64 dimensions x 4 chunk lanes
    int d=t&63,cl=t>>6; float acc=0.0f;
    for(int c=cl;c<live;c+=4){size_t ci=(size_t)head*nchunks+c;if(pl[ci]>0.0f)acc=fmaf(pa[ci*hd+d0+d],expf(pm[ci]-mx),acc);}
    part[cl][d]=acc; __syncthreads();
    if(t<64){float v=(part[0][t]+part[1][t]+part[2][t]+part[3][t])/den; sh[t]=v;}
    __syncthreads();
    if(t<64){                                          // two groups of 32 in this slice
        int g=t>>5,lane=t&31; float v=sh[t],a=fabsf(v);
        for(int o=16;o;o>>=1)a=fmaxf(a,KSHFLX(a,o));
        float sc=a>0.0f?a/127.0f:1.0f; size_t gi=((size_t)head*hd+d0)/32+g;
        if(lane==0)scale[gi]=sc; q[(size_t)head*hd+d0+t]=(signed char)__float2int_rn(v/sc);
    }
}

// dequantize a whole matrix to float32 (rows x cols, row-major): the prompt path on cards without dp4a uses cuBLAS
template<int TYPE> __device__ __forceinline__ void g4_dequant(const unsigned char* w,float* out,int rows,int cols) {
    int nb=cols>>8; size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i>=(size_t)rows*nb*16)return;
    int row=(int)(i/((size_t)nb*16)),rem=(int)(i%((size_t)nb*16)),b=rem>>4,u=rem&15;
    uint4 q; float sc,mn; g4_load_run<TYPE>(w,rows,cols,nb,row,b,u,q,sc,mn);
    float* o=out+(size_t)row*cols+b*256+g4_run_col<TYPE>(u);
    unsigned int v[4]={q.x,q.y,q.z,q.w};
#pragma unroll
    for(int k=0;k<16;k++)o[k]=sc*(float)(signed char)((v[k>>2]>>(8*(k&3)))&255u)-mn;
}
#define G4_DQ(T) extern "C" __global__ void k_g4_dequant_t##T(const unsigned char* w,float* out,int rows,int cols){ g4_dequant<T>(w,out,rows,cols); }
G4_DQ(8) G4_DQ(10) G4_DQ(11) G4_DQ(13) G4_DQ(14) G4_DQ(23) G4_DQ(123) G4_DQ(111) G4_DQ(114) G4_DQ(108)

// ---- tensor parallel (gemma4_tp): out = partial(card 0) + partial(card 1), added in that order on both cards, so
// ---- the residual stream is the same on the two cards. The partials meet in mapped host memory: [2 parities][2
// ---- cards][D] floats; each card counts its exchanges (ctr) and waits until the other card's flag reaches the same
// ---- epoch. One block of 1024 threads.
extern "C" __global__ void __launch_bounds__(1024) k_g4_tp_xsum(const float* part,float* hb,volatile int* flags,int me,
        int* ctr,int* fail,float* out,int D) {
    __shared__ int ep,ok;
    int t=threadIdx.x;
    if(t==0)ep=++ctr[0];
    __syncthreads();
    int epoch=ep,par=epoch&1;
    float* mine=hb+(size_t)(par*2+me)*D; const float* other=hb+(size_t)(par*2+1-me)*D;
    for(int e=t;e<D;e+=blockDim.x)mine[e]=part[e];
    __threadfence_system(); __syncthreads();
    if(t==0) {
        flags[me]=epoch; __threadfence_system();
        long long spins=0; ok=1;
        while(flags[1-me]<epoch){ if(++spins>4000000000LL){ *fail=epoch; ok=0; break; } }
    }
    __syncthreads();
    if(!ok)return;
    for(int e=t;e<D;e+=blockDim.x) {
        float o;
#if KHIP
        o=((volatile const float*)other)[e];
#else
        asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(o) : "l"(other+e));
#endif
        float a=part[e]; out[e]=me==0?a+o:o+a;
    }
}

#if KHIP && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__))
// ---- the prompt GEMM on RDNA3 WMMA (v_wmma_i32_16x16x16_iu8, wave32). The tile and the decode are g4_mmq's: 64
// ---- weight rows x 64 tokens per block of 256 threads (8 waves); wave w takes rows 16 (w & 3) .. +15 and tokens
// ---- 32 (w >> 2) .. +31 (two 16 x 16 tiles). Lane l holds row (or token) l & 15 of a fragment; in the result, lane l
// ---- holds token l & 15 of rows 2 i + (l >> 4). Each run's int32 is scaled and added in fp32 in g4_mmq's order.
typedef int g4_v4i __attribute__((ext_vector_type(4)));
typedef int g4_v8i __attribute__((ext_vector_type(8)));
template<int TYPE,int NT> __device__ __forceinline__ void g4_wmma(const unsigned char* w,const signed char* xq,const float* xs,
        float* y,int rows,int cols,int nt,int ldy) {
    // NT tokens per block (64 or 128); wave w: rows 16 (w & 3) .. +15, tokens (NT / 2) (w >> 2) .. + NT / 2 - 1
    const int MINS=(TYPE==10||TYPE==13);
    const int PAIR=(TYPE==23||TYPE==123||TYPE==8||TYPE==108||TYPE==13);
    const int NJ=NT/32;                            // 16-token tiles per wave
    __shared__ uint4 W[64][17];
    __shared__ uint4 X[NT][17];
    __shared__ float WS[64][16],WM[MINS?64:1][16],XS[NT][8];
    __shared__ short XSUM[MINS?NT:1][16];
    int tid=threadIdx.x,lane=tid&31,wave=tid>>5,l16=lane&15,hi=lane>>4;
    int row0=blockIdx.y*64,tok0=blockIdx.x*NT,nb=cols>>8,mr=16*(wave&3),nc=(NT/2)*(wave>>2);   // token tiles first
    float acc[NJ][8];
#pragma unroll
    for(int j=0;j<NJ;j++)
#pragma unroll
        for(int i=0;i<8;i++)acc[j][i]=0.0f;
    // the next block's weights (decoded in registers) and input are fetched before the current block's math, so
    // their memory latency overlaps it
    uint4 pq[4]; float psc[4],pmn[4]; uint4 px[NT/16]; float pxs[NT/32];
    auto fetch=[&](int b) {
#pragma unroll
        for(int k=0;k<4;k++){int idx=tid+k*256,r=idx>>4,u=idx&15,row=min(row0+r,rows-1);
            g4_load_run<TYPE>(w,rows,cols,nb,row,b,u,pq[k],psc[k],pmn[k]);}
#pragma unroll
        for(int k=0;k<NT/16;k++){int idx=tid+k*256,t=idx>>4,c=idx&15,tok=tok0+t;
            px[k]=tok<nt?*reinterpret_cast<const uint4*>(xq+(size_t)tok*cols+b*256+c*16):make_uint4(0u,0u,0u,0u);}
#pragma unroll
        for(int k=0;k<NT/32;k++){int idx=tid+k*256,t=idx>>3,g=idx&7,tok=tok0+t;pxs[k]=tok<nt?xs[((size_t)tok*cols+b*256)/32+g]:0.0f;}
    };
    fetch(0);
    for(int b=0;b<nb;b++) {
#pragma unroll
        for(int k=0;k<4;k++){int idx=tid+k*256,r=idx>>4,u=idx&15;W[r][u]=pq[k]; WS[r][u]=psc[k]; if(MINS)WM[r][u]=pmn[k];}
#pragma unroll
        for(int k=0;k<NT/16;k++){int idx=tid+k*256,t=idx>>4,c=idx&15;X[t][c]=px[k];}
#pragma unroll
        for(int k=0;k<NT/32;k++){int idx=tid+k*256,t=idx>>3,g=idx&7;XS[t][g]=pxs[k];}
        __syncthreads();
        if(b+1<nb)fetch(b+1);
        if(MINS) {
            for(int idx=tid;idx<NT*16;idx+=256){int t=idx>>4,u=idx&15;uint4 v=X[t][g4_run_col<TYPE>(u)>>4];
                XSUM[t][u]=(short)g4_dp4a((int)v.w,0x01010101,g4_dp4a((int)v.z,0x01010101,g4_dp4a((int)v.y,0x01010101,g4_dp4a((int)v.x,0x01010101,0))));}
            __syncthreads();
        }
#pragma unroll
        for(int u=0;u<16;u+=(PAIR?2:1)) {
            int c=g4_run_col<TYPE>(u)>>4,g=c>>1;
            uint4 wa=W[mr+l16][u],wb=PAIR?W[mr+l16][u+1]:make_uint4(0u,0u,0u,0u);
            g4_v4i A0={(int)wa.x,(int)wa.y,(int)wa.z,(int)wa.w},A1={(int)wb.x,(int)wb.y,(int)wb.z,(int)wb.w};
            float ws[8],wm[8];
#pragma unroll
            for(int i=0;i<8;i++){ws[i]=WS[mr+2*i+hi][u];wm[i]=MINS?WM[mr+2*i+hi][u]:0.0f;}
#pragma unroll
            for(int j=0;j<NJ;j++) {
                int tk=nc+16*j+l16;
                uint4 xa=X[tk][c],xb=PAIR?X[tk][c+1]:make_uint4(0u,0u,0u,0u);
                g4_v4i B0={(int)xa.x,(int)xa.y,(int)xa.z,(int)xa.w},B1={(int)xb.x,(int)xb.y,(int)xb.z,(int)xb.w};
                g4_v8i C={0,0,0,0,0,0,0,0};
                C=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,A0,true,B0,C,false);
                if(PAIR)C=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,A1,true,B1,C,false);
                float xsc=XS[tk][g];
                float xsum=MINS?((float)XSUM[tk][u]+(PAIR?(float)XSUM[tk][u+1]:0.0f)):0.0f;
#pragma unroll
                for(int i=0;i<8;i++) {
                    float f=(float)C[i]*ws[i]; if(MINS)f-=xsum*wm[i];
                    acc[j][i]=fmaf(f,xsc,acc[j][i]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int j=0;j<NJ;j++)
#pragma unroll
        for(int i=0;i<8;i++){int row=row0+mr+2*i+hi,tok=tok0+nc+16*j+l16;if(row<rows&&tok<nt)y[(size_t)tok*ldy+row]=acc[j][i];}
}
// C[m][n] = sum_k A[m][k] B[n][k] in float32 from float16 inputs (v_wmma_f32_16x16x16_f16): the prompt
// attention's two products on the 780M (scores = Q K^T, out = P V). Block 256 threads = 8 waves, 64 x 64 of C; wave w
// takes rows 16 (w & 3) .. +15 and columns 32 (w >> 2) .. +31; k beyond K reads as zero.
typedef _Float16 g4_v16h __attribute__((ext_vector_type(16)));
typedef float g4_v8f __attribute__((ext_vector_type(8)));
__device__ __forceinline__ g4_v16h g4_frag16(const __half* p,int ld,int r,int R,int k0,int K) {
    g4_v16h v;
    const _Float16* q=reinterpret_cast<const _Float16*>(p)+(size_t)r*ld+k0;
    if(r<R&&k0+16<=K){
#pragma unroll
        for(int i=0;i<16;i++)v[i]=q[i];
    } else {
#pragma unroll
        for(int i=0;i<16;i++)v[i]=(r<R&&k0+i<K)?q[i]:(_Float16)0.0f;
    }
    return v;
}
extern "C" __global__ void __launch_bounds__(256) k_g4_wmma_hgemm(const __half* A,int lda,const __half* B,int ldb,float* C,int ldc,
        int M,int N,int K) {
    int lane=threadIdx.x&31,wave=threadIdx.x>>5,l16=lane&15,hi=lane>>4;
    int m0=blockIdx.x*64+16*(wave&3),n0=blockIdx.y*64+32*(wave>>2);
    g4_v8f c0={0,0,0,0,0,0,0,0},c1={0,0,0,0,0,0,0,0};
    for(int k0=0;k0<K;k0+=16) {
        g4_v16h a=g4_frag16(A,lda,m0+l16,M,k0,K);
        g4_v16h b0=g4_frag16(B,ldb,n0+l16,N,k0,K),b1=g4_frag16(B,ldb,n0+16+l16,N,k0,K);
        c0=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b0,c0);
        c1=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b1,c1);
    }
#pragma unroll
    for(int i=0;i<8;i++){int m=m0+2*i+hi;
        if(m<M&&n0+l16<N)C[(size_t)m*ldc+n0+l16]=c0[i];
        if(m<M&&n0+16+l16<N)C[(size_t)m*ldc+n0+16+l16]=c1[i];}
}
#define G4_WMMA(T) extern "C" __global__ void __launch_bounds__(256) k_g4_wmma_t##T(const unsigned char* w,const signed char* xq, \
    const float* xs,float* y,int rows,int cols,int nt,int ldy){ g4_wmma<T,(T==10||T==13)?64:128>(w,xq,xs,y,rows,cols,nt,ldy); }
G4_WMMA(8) G4_WMMA(10) G4_WMMA(11) G4_WMMA(13) G4_WMMA(14) G4_WMMA(23) G4_WMMA(123) G4_WMMA(111) G4_WMMA(114) G4_WMMA(108)
#endif

// prompt path: gelu(gate)*up quantized for the down projection, gate and up apart ([tokens][n] each); one warp per
// group of 32 (rows are multiples of 32)
extern "C" __global__ void k_g4_gelu_mul2_q32(const float* g,const float* u,signed char* q,float* scale,size_t n) {
    size_t gi=(size_t)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5); int lane=threadIdx.x&31; size_t i=gi*32+lane; if(gi*32>=n)return;
    float x=g[i],y=0.5f*x*(1.0f+erff(x*0.7071067811865475f))*u[i],m=fabsf(y);
    for(int o=16;o;o>>=1)m=fmaxf(m,KSHFLX(m,o));
    float sc=m>0.0f?m/127.0f:1.0f; if(lane==0)scale[gi]=sc; q[i]=(signed char)__float2int_rn(y/sc);
}
// prompt attention: softmax rows as float16 for the P.V product (the masked softmax, written as float16)
extern "C" __global__ void k_g4_softmax_mask_h(const float* S,__half* P,int L,int gs,int q0,int k0,int window) {
    __shared__ float sh[256]; const float* row=S+(size_t)blockIdx.x*L; __half* prow=P+(size_t)blockIdx.x*L; int qp=q0+blockIdx.x/gs;
    float mx=-3.402823466e+38F;
    for(int j=threadIdx.x;j<L;j+=blockDim.x){int kp=k0+j;bool ok=kp<=qp&&(window<=0||kp>qp-window);if(ok)mx=fmaxf(mx,row[j]);}
    sh[threadIdx.x]=mx; __syncthreads();
    for(int d=blockDim.x>>1;d;d>>=1){if(threadIdx.x<d)sh[threadIdx.x]=fmaxf(sh[threadIdx.x],sh[threadIdx.x+d]);__syncthreads();}
    mx=sh[0]; __syncthreads(); float z=0.0f;
    for(int j=threadIdx.x;j<L;j+=blockDim.x){int kp=k0+j;bool ok=kp<=qp&&(window<=0||kp>qp-window);z+=ok?expf(row[j]-mx):0.0f;}
    sh[threadIdx.x]=z; __syncthreads();
    for(int d=blockDim.x>>1;d;d>>=1){if(threadIdx.x<d)sh[threadIdx.x]+=sh[threadIdx.x+d];__syncthreads();}
    float inv=1.0f/sh[0];
    for(int j=threadIdx.x;j<L;j+=blockDim.x){int kp=k0+j;bool ok=kp<=qp&&(window<=0||kp>qp-window);prow[j]=__float2half(ok?expf(row[j]-mx)*inv:0.0f);}
}
