// Storage conversion only. Kepler performs attention arithmetic in FP32.
extern "C" __global__ void kv_store_fp16(const float* x,unsigned short* out,int n,int* error) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float value=x[i];
    if(!isfinite(value) || fabsf(value)>65504.f){atomicOr(error,1);value=0;}
    out[i]=__half_as_ushort(__float2half_rn(value));
}
extern "C" __global__ void kv_store_q8(const float* x,unsigned char* out,int blocks,int* error) {
    int lane=threadIdx.x&31,block=blockIdx.x*4+(threadIdx.x>>5);if(block>=blocks)return;
    float value=x[(size_t)block*32+lane];
    if(!isfinite(value)){atomicOr(error,1);value=0;}
    float maximum=fabsf(value);
    for(int shift=16;shift;shift/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,shift));
    maximum=__shfl_sync(0xffffffffu,maximum,0);
    float scale=maximum/127.f;
    if(scale>65504.f){if(lane==0)atomicOr(error,1);scale=0;}
    unsigned char* dst=out+(size_t)block*34;
    if(lane==0)*reinterpret_cast<unsigned short*>(dst)=__half_as_ushort(__float2half_rn(scale));
    // RNE, bounded to [-127,127]; scales use FP16, codes use the unrounded FP32 scale.
    int code=scale>0?__float2int_rn(value/scale):0;
    reinterpret_cast<signed char*>(dst+2)[lane]=(signed char)max(-127,min(127,code));
}
template<int TYPE> __device__ __forceinline__ float kv_read(const unsigned char* data,size_t index) {
    if(TYPE==0)return __ldg(reinterpret_cast<const float*>(data)+index);
    if(TYPE==1)return __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(data)+index)));
    const unsigned char* p=data+(index/32)*34;
    float scale=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(p))));
    return scale*(float)__ldg(reinterpret_cast<const signed char*>(p+2)+(index%32));
}
extern "C" __global__ void kv_decode_fp16(const unsigned char* src,float* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=kv_read<1>(src,i);
}
extern "C" __global__ void kv_decode_q8(const unsigned char* src,float* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=kv_read<8>(src,i);
}
template<int TYPE> __device__ void attention_storage(const float* q,const unsigned char* k,const unsigned char* v,float* out,
    int tokens,int past,int qheads,int kvheads,int dim) {
    int h=blockIdx.x,token=blockIdx.y,t=threadIdx.x,lane=t&31,warp=t>>5;
    if(token>=tokens)return;
    int kv=h/(qheads/kvheads),length=past+token+1;
    const float* query=q+((size_t)token*qheads+h)*dim;
    __shared__ float score[128],reduction[128],prob[128],alpha,beta,total,maxval;
    if(t==0){total=0;maxval=-INF;}__syncthreads();float acc=0;
    for(int base=0;base<length;base+=128) {
        int count=min(128,length-base);
        for(int j=warp;j<count;j+=4) {
            size_t key=((size_t)(base+j)*kvheads+kv)*dim;
            float dot=0;for(int c=lane;c<dim;c+=32)dot=fmaf(query[c],kv_read<TYPE>(k,key+c),dot);
            dot=warp_sum(dot);if(lane==0)score[j]=dot*rsqrtf((float)dim);
        }
        __syncthreads();reduction[t]=t<count?score[t]:-INF;__syncthreads();
        for(int d=64;d;d/=2){if(t<d)reduction[t]=fmaxf(reduction[t],reduction[t+d]);__syncthreads();}
        float tilemax=reduction[0];prob[t]=t<count?expf(score[t]-tilemax):0;__syncthreads();
        reduction[t]=prob[t];__syncthreads();
        for(int d=64;d;d/=2){if(t<d)reduction[t]+=reduction[t+d];__syncthreads();}
        if(t==0){float next=fmaxf(maxval,tilemax);alpha=expf(maxval-next);beta=expf(tilemax-next);total=total*alpha+reduction[0]*beta;maxval=next;}
        __syncthreads();
        if(t<dim){float sum=0;for(int j=0;j<count;j++)sum=fmaf(prob[j],kv_read<TYPE>(v,((size_t)(base+j)*kvheads+kv)*dim+t),sum);acc=acc*alpha+sum*beta;}
        __syncthreads();
    }
    if(t<dim)out[((size_t)token*qheads+h)*dim+t]=acc/total;
}
extern "C" __global__ void attention_fp16(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim) {attention_storage<1>(q,k,v,out,tokens,past,qheads,kvheads,dim);}
extern "C" __global__ void attention_q8(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim) {attention_storage<8>(q,k,v,out,tokens,past,qheads,kvheads,dim);}

// Same 128-element reduction tree and tile recurrence as attention_tiled.
// Shared loads plus warp shuffles replace fourteen block-wide reduction barriers.
template<int TYPE> __device__ void attention_storage_warp(const float* q,const unsigned char* k,const unsigned char* v,float* out,
    int tokens,int past,int qheads,int kvheads,int dim) {
    int h=blockIdx.x,token=blockIdx.y,t=threadIdx.x,lane=t&31,warp=t>>5;
    if(token>=tokens)return;
    int kv=h/(qheads/kvheads),length=past+token+1;
    const float* query=q+((size_t)token*qheads+h)*dim;
    __shared__ float score[128],prob[128],alpha,beta,total,maxval;
    if(t==0){total=0;maxval=-INF;}__syncthreads();float acc=0;
    float q0=lane<dim?query[lane]:0, q1=lane+32<dim?query[lane+32]:0;
    float q2=lane+64<dim?query[lane+64]:0, q3=lane+96<dim?query[lane+96]:0;
    for(int base=0;base<length;base+=128) {
        int count=min(128,length-base);
        for(int j=warp;j<128;j+=4) {
            float dot=0;
            if(j<count) {
                size_t key=((size_t)(base+j)*kvheads+kv)*dim;
                if(lane<dim)dot=fmaf(q0,kv_read<TYPE>(k,key+lane),dot);
                if(lane+32<dim)dot=fmaf(q1,kv_read<TYPE>(k,key+lane+32),dot);
                if(lane+64<dim)dot=fmaf(q2,kv_read<TYPE>(k,key+lane+64),dot);
                if(lane+96<dim)dot=fmaf(q3,kv_read<TYPE>(k,key+lane+96),dot);
                dot=warp_sum(dot);
            }
            if(lane==0)score[j]=j<count?dot*rsqrtf((float)dim):-INF;
        }
        __syncthreads();
        float maximum=fmaxf(fmaxf(score[lane],score[lane+64]),fmaxf(score[lane+32],score[lane+96]));
        for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
        float tilemax=__shfl_sync(0xffffffffu,maximum,0);
        prob[t]=t<count?expf(score[t]-tilemax):0;__syncthreads();
        if(warp==0) {
            float sum=prob[lane]+prob[lane+64];sum+=prob[lane+32]+prob[lane+96];sum=warp_sum(sum);
            if(lane==0){float next=fmaxf(maxval,tilemax);alpha=expf(maxval-next);beta=expf(tilemax-next);total=__fmaf_rn(total,alpha,__fmul_rn(sum,beta));maxval=next;}
        }
        __syncthreads();
        if(t<dim){float sum=0;for(int j=0;j<count;j++)sum=fmaf(prob[j],kv_read<TYPE>(v,((size_t)(base+j)*kvheads+kv)*dim+t),sum);acc=acc*alpha+sum*beta;}
        __syncthreads();
    }
    if(t<dim)out[((size_t)token*qheads+h)*dim+t]=acc/total;
}
extern "C" __global__ void attention_warp(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim) {attention_storage_warp<0>(q,k,v,out,tokens,past,qheads,kvheads,dim);}
extern "C" __global__ void attention_fp16_warp(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim) {attention_storage_warp<1>(q,k,v,out,tokens,past,qheads,kvheads,dim);}
extern "C" __global__ void attention_q8_warp(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim) {attention_storage_warp<8>(q,k,v,out,tokens,past,qheads,kvheads,dim);}

// Compute independent 128-key tiles in parallel; merge in the ORIGINAL tile order.
// Unlike a tree merge across tiles, this retains the online-softmax rounding path.
template<int TYPE> __device__ void attention_tiles(const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads,int kvheads,int dim,int tiles) {
    int h=blockIdx.x,token=blockIdx.y,tile=blockIdx.z,t=threadIdx.x,lane=t&31,warp=t>>5;
    int base=tile*128,length=past+token+1;if(token>=tokens || base>=length)return;
    int kv=h/(qheads/kvheads),count=min(128,length-base);
    const float* query=q+((size_t)token*qheads+h)*dim;
    __shared__ float score[128],prob[128];
    float q0=lane<dim?query[lane]:0,q1=lane+32<dim?query[lane+32]:0;
    float q2=lane+64<dim?query[lane+64]:0,q3=lane+96<dim?query[lane+96]:0;
    for(int j=warp;j<128;j+=4) {
        float dot=0;
        if(j<count) {
            size_t key=((size_t)(base+j)*kvheads+kv)*dim;
            if(lane<dim)dot=fmaf(q0,kv_read<TYPE>(k,key+lane),dot);
            if(lane+32<dim)dot=fmaf(q1,kv_read<TYPE>(k,key+lane+32),dot);
            if(lane+64<dim)dot=fmaf(q2,kv_read<TYPE>(k,key+lane+64),dot);
            if(lane+96<dim)dot=fmaf(q3,kv_read<TYPE>(k,key+lane+96),dot);
            dot=warp_sum(dot);
        }
        if(lane==0)score[j]=j<count?dot*rsqrtf((float)dim):-INF;
    }
    __syncthreads();
    float maximum=fmaxf(fmaxf(score[lane],score[lane+64]),fmaxf(score[lane+32],score[lane+96]));
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
    float tilemax=__shfl_sync(0xffffffffu,maximum,0);
    prob[t]=t<count?expf(score[t]-tilemax):0;__syncthreads();
    size_t offset=(((size_t)token*qheads+h)*tiles+tile)*(dim+2);
    if(warp==0) {
        float sum=prob[lane]+prob[lane+64];sum+=prob[lane+32]+prob[lane+96];sum=warp_sum(sum);
        if(lane==0){parts[offset+dim]=tilemax;parts[offset+dim+1]=sum;}
    }
    if(t<dim){float sum=0;for(int j=0;j<count;j++)sum=fmaf(prob[j],kv_read<TYPE>(v,((size_t)(base+j)*kvheads+kv)*dim+t),sum);parts[offset+t]=sum;}
}
extern "C" __global__ void attention_tiles_fp32(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim,int tiles) {attention_tiles<0>(q,k,v,out,tokens,past,qheads,kvheads,dim,tiles);}
extern "C" __global__ void attention_tiles_fp16(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim,int tiles) {attention_tiles<1>(q,k,v,out,tokens,past,qheads,kvheads,dim,tiles);}
extern "C" __global__ void attention_tiles_q8(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim,int tiles) {attention_tiles<8>(q,k,v,out,tokens,past,qheads,kvheads,dim,tiles);}
extern "C" __global__ void attention_merge_tiles(const float* parts,float* out,int tokens,int past,int qheads,int dim,int tiles) {
    int h=blockIdx.x,token=blockIdx.y,t=threadIdx.x,length=past+token+1;
    __shared__ float alpha,beta,total,maxval;
    if(t==0){total=0;maxval=-INF;}__syncthreads();float acc=0;
    for(int tile=0;tile*128<length;tile++) {
        size_t offset=(((size_t)token*qheads+h)*tiles+tile)*(dim+2);
        if(t==0){float tilemax=parts[offset+dim],next=fmaxf(maxval,tilemax);alpha=expf(maxval-next);beta=expf(tilemax-next);
            total=__fmaf_rn(total,alpha,__fmul_rn(parts[offset+dim+1],beta));maxval=next;}
        __syncthreads();
        if(t<dim)acc=__fmaf_rn(acc,alpha,__fmul_rn(parts[offset+t],beta));
        __syncthreads();
    }
    if(t<dim)out[((size_t)token*qheads+h)*dim+t]=acc/total;
}
