// Kindi Lightning SIMT baseline. CUDA 10.2+; no DP4A, Tensor Core or half arithmetic.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define INF __int_as_float(0x7f800000)

__device__ __forceinline__ float warp_sum(float x) {
    for (int d=16; d; d>>=1) x += __shfl_down_sync(0xffffffffu,x,d);
    return x;
}
__device__ __forceinline__ float sigmoid(float x) {
    return x >= 0 ? 1.f/(1.f+expf(-x)) : expf(x)/(1.f+expf(x));
}
__device__ __forceinline__ float softplus(float x) {
    return fmaxf(x,0.f)+log1pf(expf(-fabsf(x)));
}
__device__ __forceinline__ float weight(const unsigned char* row,int c,int type) {
    if (type==0) return reinterpret_cast<const float*>(row)[c];
    int bs = type==42 ? 64:32, bytes = type==42 ? 18:34;
    const unsigned char* b=row+(size_t)(c/bs)*bytes;
    float d=__half2float(*reinterpret_cast<const __half*>(b));
    if(type==8) return d*(float)reinterpret_cast<const signed char*>(b+2)[c%32];
    int q=(b[2+(c%64)/4]>>(2*(c%4)))&3;
    return d*(float)(q-1);
}

// Four warps/block, one output row/warp. Optional GPU expert IDs and per-assignment input.
extern "C" __global__ void packed_mv(const unsigned char* w,const float* x,float* y,
    const int* ids,int rows,int cols,int type,int selected,int per_input) {
    int lane=threadIdx.x&31, row=blockIdx.x*4+(threadIdx.x>>5), s=blockIdx.y;
    if(row>=rows || s>=selected) return; // all lanes in this warp take the same branch
    int e=ids ? ids[s]:0;
    size_t rb=type==0?(size_t)cols*4:(size_t)(cols/(type==42?64:32))*(type==42?18:34);
    const unsigned char* wr=w+((size_t)e*rows+row)*rb;
    const float* xr=x+(per_input?(size_t)s*cols:0);
    float sum=0;
    for(int c=lane;c<cols;c+=32) sum=fmaf(weight(wr,c,type),xr[c],sum);
    sum=warp_sum(sum);
    if(lane==0) y[(size_t)s*rows+row]=sum;
}

template<int TYPE> __device__ void packed_mv_specialized(const unsigned char* w,const float* x,float* y,
    const int* ids,int rows,int cols,int selected,int per_input) {
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5),s=blockIdx.y;
    if(row>=rows || s>=selected)return;
    int e=ids?ids[s]:0;
    const size_t rb=(size_t)(cols/(TYPE==42?64:32))*(TYPE==42?18:34);
    const unsigned char* wr=w+((size_t)e*rows+row)*rb;
    const float* xr=x+(per_input?(size_t)s*cols:0);
    float sum=0;
    if(TYPE==42) {
        // Four adjacent ternary values share one packed byte and one scale load.
        for(int c=lane*4;c<cols;c+=128) {
            const unsigned char* b=wr+(c/64)*18;
            float d=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(b))));
            unsigned q=__ldg(b+2+(c%64)/4);
            float partial=0;
            #pragma unroll
            for(int j=0;j<4;j++)partial=fmaf((float)((int)((q>>(2*j))&3)-1),__ldg(xr+c+j),partial);
            sum=fmaf(d,partial,sum);
        }
    } else {
        for(int c=lane;c<cols;c+=32) {
            const unsigned char* b=wr+(c/32)*34;
            float d=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(b))));
            sum=fmaf(d*(float)__ldg(reinterpret_cast<const signed char*>(b+2)+lane),__ldg(xr+c),sum);
        }
    }
    sum=warp_sum(sum);if(lane==0)y[(size_t)s*rows+row]=sum;
}
extern "C" __global__ void q2_mv(const unsigned char* w,const float* x,float* y,const int* ids,
    int rows,int cols,int selected,int per_input) {packed_mv_specialized<42>(w,x,y,ids,rows,cols,selected,per_input);}
extern "C" __global__ void q8_mv(const unsigned char* w,const float* x,float* y,const int* ids,
    int rows,int cols,int selected,int per_input) {packed_mv_specialized<8>(w,x,y,ids,rows,cols,selected,per_input);}

extern "C" __global__ void embedding(const unsigned char* w,float* x,int token,int width,int type) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    size_t rb=type==0?(size_t)width*4:(size_t)(width/(type==42?64:32))*(type==42?18:34);
    if(i<width) x[i]=weight(w+(size_t)token*rb,i,type);
}

extern "C" __global__ void rms(const float* x,const float* w,float* y,int width,int rows,float eps,const float* gate) {
    int row=blockIdx.x, tid=threadIdx.x;
    if(row>=rows) return;
    __shared__ float sums[256];
    float s=0;
    for(int j=tid;j<width;j+=blockDim.x) {
        int i=row*width+j;
        float v=x[i]; if(gate) v*=gate[i]*sigmoid(gate[i]);
        s=fmaf(v,v,s);
    }
    sums[tid]=s; __syncthreads();
    for(int d=blockDim.x/2;d;d/=2) {if(tid<d)sums[tid]+=sums[tid+d];__syncthreads();}
    float inv=rsqrtf(sums[0]/width+eps);
    for(int j=tid;j<width;j+=blockDim.x) {
        int i=row*width+j;
        float v=x[i]; if(gate) v*=gate[i]*sigmoid(gate[i]);
        y[i]=v*inv*w[i];
    }
}
extern "C" __global__ void relu2(float* x,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {float a=fmaxf(x[i],0.f);x[i]=a*a;}
}
extern "C" __global__ void residual(float* x,const float* y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=y[i];
}

// Scores computed in FP32; correction bias affects selection only. Canonical tie: smaller ID.
extern "C" __global__ void route(const float* logits,const float* bias,int* ids,float* weights,
    int experts,int topk,float scale) {
    if(threadIdx.x || blockIdx.x) return;
    float denom=0;
    for(int k=0;k<topk;k++) {
        int best=-1; float bs=-INF;
        for(int e=0;e<experts;e++) {
            bool used=false;for(int j=0;j<k;j++)if(ids[j]==e)used=true;
            float score=sigmoid(logits[e])+bias[e];
            if(!used && (best<0 || score>bs)) {best=e;bs=score;}
        }
        ids[k]=best;weights[k]=sigmoid(logits[best]);denom+=weights[k];
    }
    for(int k=0;k<topk;k++)weights[k]=denom>0 ? weights[k]/denom*scale:0.f;
}
extern "C" __global__ void combine(const float* expert,const float* weights,const float* shared,
    float* y,int width,int topk) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<width) {float v=0;for(int k=0;k<topk;k++)v=fmaf(expert[(size_t)k*width+i],weights[k],v);y[i]=v+shared[i];}
}

extern "C" __global__ void causal_conv(const float* input,const float* w,const float* bias,
    float* history,float* output,int channels,int kernel) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=channels)return;
    float s=bias[c];
    for(int j=0;j<kernel-1;j++)s=fmaf(history[(size_t)c*(kernel-1)+j],w[(size_t)c*kernel+j],s);
    s=fmaf(input[c],w[(size_t)c*kernel+kernel-1],s);
    for(int j=0;j<kernel-2;j++)history[(size_t)c*(kernel-1)+j]=history[(size_t)c*(kernel-1)+j+1];
    history[(size_t)c*(kernel-1)+kernel-2]=input[c];
    output[c]=s*sigmoid(s);
}

extern "C" __global__ void ssm_step(const float* conv,const float* dt,const float* dtbias,
    const float* a,const float* d,float* state,float* y,int heads,int headdim,int statesize,int groups) {
    int channel=blockIdx.x,h=channel/headdim,p=channel%headdim,t=threadIdx.x;
    if(h>=heads)return;
    int g=h/(heads/groups),inner=heads*headdim;
    float delta=softplus(dt[h]+dtbias[h]);
    float decay=expf(delta*a[h]),x=conv[channel],s=0;
    for(int n=t;n<statesize;n+=blockDim.x) {
        size_t index=((size_t)h*headdim+p)*statesize+n;
        float v=fmaf(delta*x,conv[inner+g*statesize+n],decay*state[index]);
        state[index]=v;
        s=fmaf(v,conv[inner+groups*statesize+g*statesize+n],s);
    }
    s=warp_sum(s);
    __shared__ float sums[8];
    if((t&31)==0)sums[t>>5]=s;__syncthreads();
    if(t<32) {s=t<(blockDim.x/32)?sums[t]:0;s=warp_sum(s);if(t==0)y[channel]=fmaf(d[h],x,s);}
}

// Stable online attention; one CTA/query head, streaming keys with bounded storage.
extern "C" __global__ void attention(const float* q,const float* k,const float* v,float* out,
    int length,int qheads,int kvheads,int dim) {
    int h=blockIdx.x,t=threadIdx.x,kv=h/(qheads/kvheads);
    __shared__ float dot[256];
    __shared__ float alpha,beta,m,l;
    if(t==0){m=-INF;l=0;}__syncthreads();
    float acc=0;
    for(int pos=0;pos<length;pos++) {
        float s=t<dim?q[h*dim+t]*k[((size_t)pos*kvheads+kv)*dim+t]:0;
        dot[t]=s;__syncthreads();
        for(int r=blockDim.x/2;r;r/=2){if(t<r)dot[t]+=dot[t+r];__syncthreads();}
        if(t==0){float score=dot[0]*rsqrtf((float)dim);float nm=fmaxf(m,score);alpha=expf(m-nm);beta=expf(score-nm);l=l*alpha+beta;m=nm;}
        __syncthreads();
        if(t<dim)acc=acc*alpha+beta*v[((size_t)pos*kvheads+kv)*dim+t];
        __syncthreads();
    }
    if(t<dim)out[h*dim+t]=acc/l;
}
