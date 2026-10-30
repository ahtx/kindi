// Appended to lightning.cu. FP32 SIMT tiles and bounded-memory chunk execution.
// Eight rows x eight tokens. Keep q2_mv's per-lane four-FMA/scale/tree order.
extern "C" __global__ void q2_tile8(const unsigned char* w,const float* x,float* y,int rows,int cols,
    int items,int topk,int per_assignment,const int* order,const int* job_expert,const int* job_start,
    const int* job_length,const int* job_count) {
    int job=blockIdx.y,t=threadIdx.x,lane=t&31,warp=t>>5;
    if(job>=(job_count?*job_count:(items+7)/8))return;
    int count=job_length?job_length[job]:min(8,items-job*8);
    int first=job_start?job_start[job]:job*8,expert=job_expert?job_expert[job]:0,row=blockIdx.x*8+warp;
    size_t rb=(size_t)(cols/64)*18;
    __shared__ float xs[8][4][32];__shared__ int assignment[8];
    if(t<8)assignment[t]=t<count?(order?order[first+t]:first+t):-1;
    __syncthreads();
    float sums[8]={0};
    for(int base=0;base<cols;base+=128) {
        int column=base+lane*4,aa=assignment[warp];
        if(aa>=0 && column<cols) {
            int input=per_assignment?aa:aa/topk;
            float4 value=__ldg(reinterpret_cast<const float4*>(x+(size_t)input*cols+column));
            xs[warp][0][lane]=value.x;xs[warp][1][lane]=value.y;xs[warp][2][lane]=value.z;xs[warp][3][lane]=value.w;
        }
        __syncthreads();
        if(row<rows && column<cols) {
            const unsigned char* b=w+((size_t)expert*rows+row)*rb+(column/64)*18;
            float d=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(b))));
            unsigned q=__ldg(b+2+(column%64)/4);
            float q0=(float)((int)(q&3)-1),q1=(float)((int)((q>>2)&3)-1),q2=(float)((int)((q>>4)&3)-1),q3=(float)((int)((q>>6)&3)-1);
            #pragma unroll
            for(int s=0;s<8;s++)if(s<count) {
                float partial=fmaf(q0,xs[s][0][lane],0.f);
                partial=fmaf(q1,xs[s][1][lane],partial);partial=fmaf(q2,xs[s][2][lane],partial);partial=fmaf(q3,xs[s][3][lane],partial);
                sums[s]=fmaf(d,partial,sums[s]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int s=0;s<8;s++)if(s<count) {
        float sum=warp_sum(sums[s]);if(lane==0 && row<rows)y[(size_t)assignment[s]*rows+row]=sum;
    }
}

// Experimental sixteen rows x eight assignments, two independent rows per warp.
// Keep the tile8/q2_mv four-FMA, scale-FMA and warp reduction order for each row.
extern "C" __global__ void q2_tile16(const unsigned char* w,const float* x,float* y,int rows,int cols,
    int items,int topk,int per_assignment,const int* order,const int* job_expert,const int* job_start,
    const int* job_length,const int* job_count) {
    int job=blockIdx.y,t=threadIdx.x,lane=t&31,warp=t>>5;
    if(job>=(job_count?*job_count:(items+7)/8))return;
    int count=job_length?job_length[job]:min(8,items-job*8);
    int first=job_start?job_start[job]:job*8,expert=job_expert?job_expert[job]:0;
    int row0=blockIdx.x*16+warp,row1=row0+8;
    size_t rb=(size_t)(cols/64)*18;
    __shared__ float xs[8][4][32];__shared__ int assignment[8];
    if(t<8)assignment[t]=t<count?(order?order[first+t]:first+t):-1;
    __syncthreads();
    float sums0[8]={0},sums1[8]={0};
    for(int base=0;base<cols;base+=128) {
        int column=base+lane*4,aa=assignment[warp];
        if(aa>=0 && column<cols) {
            int input=per_assignment?aa:aa/topk;
            float4 value=__ldg(reinterpret_cast<const float4*>(x+(size_t)input*cols+column));
            xs[warp][0][lane]=value.x;xs[warp][1][lane]=value.y;xs[warp][2][lane]=value.z;xs[warp][3][lane]=value.w;
        }
        __syncthreads();
        if(column<cols) {
            float d0=0.f,d1=0.f;unsigned q0=0,q1=0;
            if(row0<rows) {
                const unsigned char* b=w+((size_t)expert*rows+row0)*rb+(column/64)*18;
                d0=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(b))));
                q0=__ldg(b+2+(column%64)/4);
            }
            if(row1<rows) {
                const unsigned char* b=w+((size_t)expert*rows+row1)*rb+(column/64)*18;
                d1=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(b))));
                q1=__ldg(b+2+(column%64)/4);
            }
            float q00=(float)((int)(q0&3)-1),q01=(float)((int)((q0>>2)&3)-1),q02=(float)((int)((q0>>4)&3)-1),q03=(float)((int)((q0>>6)&3)-1);
            float q10=(float)((int)(q1&3)-1),q11=(float)((int)((q1>>2)&3)-1),q12=(float)((int)((q1>>4)&3)-1),q13=(float)((int)((q1>>6)&3)-1);
            #pragma unroll
            for(int s=0;s<8;s++)if(s<count) {
                float x0=xs[s][0][lane],x1=xs[s][1][lane],x2=xs[s][2][lane],x3=xs[s][3][lane];
                if(row0<rows) {
                    float partial=fmaf(q00,x0,0.f);
                    partial=fmaf(q01,x1,partial);partial=fmaf(q02,x2,partial);partial=fmaf(q03,x3,partial);
                    sums0[s]=fmaf(d0,partial,sums0[s]);
                }
                if(row1<rows) {
                    float partial=fmaf(q10,x0,0.f);
                    partial=fmaf(q11,x1,partial);partial=fmaf(q12,x2,partial);partial=fmaf(q13,x3,partial);
                    sums1[s]=fmaf(d1,partial,sums1[s]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int s=0;s<8;s++)if(s<count) {
        float sum0=warp_sum(sums0[s]);if(lane==0 && row0<rows)y[(size_t)assignment[s]*rows+row0]=sum0;
        float sum1=warp_sum(sums1[s]);if(lane==0 && row1<rows)y[(size_t)assignment[s]*rows+row1]=sum1;
    }
}

__device__ __forceinline__ float tile_weight(const unsigned char* row,int c,int type) {
    if(type==0)return __ldg(reinterpret_cast<const float*>(row)+c);
    if(type==42) {
        const unsigned char* b=row+(c/64)*18;
        float d=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(b))));
        unsigned q=__ldg(b+2+(c%64)/4);
        return d*(float)((int)((q>>(2*(c%4)))&3)-1);
    }
    const unsigned char* b=row+(c/32)*34;
    float d=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(b))));
    return d*(float)__ldg(reinterpret_cast<const signed char*>(b+2)+c%32);
}
extern "C" __global__ void embedding_batch(const unsigned char* w,const int* ids,float* x,int tokens,int width,int type) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=tokens*width)return;
    size_t rb=(size_t)(width/(type==42?64:32))*(type==42?18:34);
    x[index]=weight(w+(size_t)ids[index/width]*rb,index%width,type);
}

extern "C" __global__ void rms_batch(const float* x,const float* w,float* y,int columns,int groups,
    int tokens,float eps,const float* gate,int gate_stride) {
    int item=blockIdx.x,token=item/groups,g=item%groups,t=threadIdx.x,width=columns/groups;
    if(token>=tokens)return;
    __shared__ float sums[256];float sum=0;
    for(int j=t;j<width;j+=blockDim.x) {
        int c=g*width+j;float v=x[token*columns+c];if(gate){float z=gate[token*gate_stride+c];v*=z*sigmoid(z);}sum=fmaf(v,v,sum);
    }
    sums[t]=sum;__syncthreads();
    for(int d=blockDim.x/2;d;d/=2){if(t<d)sums[t]+=sums[t+d];__syncthreads();}
    float inv=rsqrtf(sums[0]/width+eps);
    for(int j=t;j<width;j+=blockDim.x) {
        int c=g*width+j;float v=x[token*columns+c];if(gate){float z=gate[token*gate_stride+c];v*=z*sigmoid(z);}y[token*columns+c]=v*inv*w[c];
    }
}

// One CTA/token; FP32 sigmoid, correction only for selection, canonical lower-ID ties.
extern "C" __global__ void route_batch(const float* logits,const float* bias,int* ids,float* weights,
    int tokens,int experts,int topk,float scale) {
    int token=blockIdx.x,t=threadIdx.x;if(token>=tokens)return;
    __shared__ float value[256],chosen[256];__shared__ int index[256],winner;
    float score=t<experts?sigmoid(logits[token*experts+t]):0;
    float candidate=t<experts?score+bias[t]:-INF;
    for(int slot=0;slot<topk;slot++) {
        value[t]=candidate;index[t]=t;__syncthreads();
        for(int d=blockDim.x/2;d;d/=2) {
            if(t<d && (value[t+d]>value[t] || (value[t+d]==value[t] && index[t+d]<index[t]))) {value[t]=value[t+d];index[t]=index[t+d];}
            __syncthreads();
        }
        if(t==0){winner=index[0];ids[token*topk+slot]=winner;}__syncthreads();
        if(t==winner){chosen[slot]=score;candidate=-INF;}__syncthreads();
    }
    if(t==0){float denom=0;for(int i=0;i<topk;i++)denom+=chosen[i];for(int i=0;i<topk;i++)weights[token*topk+i]=denom>0?chosen[i]/denom*scale:0;}
}
extern "C" __global__ void expert_histogram(const int* ids,int* counts,int assignments) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<assignments)atomicAdd(counts+ids[i],1);
}
extern "C" __global__ void expert_offsets(const int* counts,int* offsets,int* cursors,int* job_expert,
    int* job_start,int* job_length,int* job_count,int experts) {
    if(threadIdx.x || blockIdx.x)return;
    int offset=0,jobs=0;
    for(int e=0;e<experts;e++) {
        offsets[e]=offset;cursors[e]=offset;
        for(int n=0;n<counts[e];n+=8){job_expert[jobs]=e;job_start[jobs]=offset+n;job_length[jobs]=min(8,counts[e]-n);jobs++;}
        offset+=counts[e];
    }
    offsets[experts]=offset;*job_count=jobs;
}
extern "C" __global__ void expert_scatter(const int* ids,int* cursors,int* order,int assignments) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<assignments){int index=atomicAdd(cursors+ids[i],1);order[index]=i;}
}

// 16 output rows x 8 assigned tokens. Decode each weight once per tile; padded shared strides avoid bank conflicts.
extern "C" __global__ void packed_mm(const unsigned char* w,const float* x,float* y,int rows,int cols,
    int kind,int items,int topk,int per_assignment,const int* order,const int* job_expert,
    const int* job_start,const int* job_length,const int* job_count) {
    int job=blockIdx.y,t=threadIdx.x;
    if(job>=(job_count?*job_count:(items+7)/8))return;
    int count=job_length?job_length[job]:min(8,items-job*8);
    int first=job_start?job_start[job]:job*8,expert=job_expert?job_expert[job]:0;
    int rowbase=blockIdx.x*16;
    size_t rb=kind==0?(size_t)cols*4:(size_t)(cols/(kind==42?64:32))*(kind==42?18:34);
    const unsigned char* we=w+(size_t)expert*rows*rb;
    __shared__ float ws[16][65],xs[8][65];__shared__ int assignment[8];
    if(t<8)assignment[t]=t<count?(order?order[first+t]:first+t):-1;
    __syncthreads();
    int row=t/8,slot=t%8;float sum=0;
    for(int base=0;base<cols;base+=64) {
        for(int n=t;n<16*64;n+=128) {
            int r=n/64,c=n%64;ws[r][c]=(rowbase+r<rows && base+c<cols)?tile_weight(we+(size_t)(rowbase+r)*rb,base+c,kind):0;
        }
        for(int n=t;n<8*64;n+=128) {
            int s=n/64,c=n%64,a=assignment[s];
            int input=per_assignment?a:a/topk;
            xs[s][c]=(a>=0 && base+c<cols)?__ldg(x+(size_t)input*cols+base+c):0;
        }
        __syncthreads();
        float partial=0;
        #pragma unroll
        for(int c=0;c<64;c++)partial=fmaf(ws[row][c],xs[slot][c],partial);
        sum+=partial;
        __syncthreads();
    }
    if(rowbase+row<rows && assignment[slot]>=0)y[(size_t)assignment[slot]*rows+rowbase+row]=sum;
}

// Persistent grouped GEMV: no oversized empty grid and no padding for small expert groups.
extern "C" __global__ void grouped_mv(const unsigned char* w,const float* x,float* y,const int* ids,
    const int* order,int assignments,int rows,int cols,int topk,int per_assignment) {
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,rowtiles=(rows+3)/4;
    size_t rb=(size_t)(cols/64)*18;
    for(int job=blockIdx.x;job<assignments*rowtiles;job+=gridDim.x) {
        int assignment=order[job/rowtiles],row=(job%rowtiles)*4+warp;
        if(row>=rows)continue;
        int expert=ids[assignment],input=per_assignment?assignment:assignment/topk;
        const unsigned char* wr=w+((size_t)expert*rows+row)*rb;
        const float* xr=x+(size_t)input*cols;float sum=0;
        for(int c=lane*4;c<cols;c+=128) {
            const unsigned char* b=wr+(c/64)*18;
            float d=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(b))));
            unsigned q=__ldg(b+2+(c%64)/4);float partial=0;
            #pragma unroll
            for(int j=0;j<4;j++)partial=fmaf((float)((int)((q>>(2*j))&3)-1),__ldg(xr+c+j),partial);
            sum=fmaf(d,partial,sum);
        }
        sum=warp_sum(sum);if(lane==0)y[(size_t)assignment*rows+row]=sum;
    }
}

extern "C" __global__ void unpack_dense(const unsigned char* w,float* out,int rows,int cols,int kind) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=rows*cols)return;
    size_t rb=kind==0?(size_t)cols*4:(size_t)(cols/(kind==42?64:32))*(kind==42?18:34);
    out[index]=tile_weight(w+(size_t)(index/cols)*rb,index%cols,kind);
}

// Four token vectors share each packed-byte decode. No expanded expert matrices.
extern "C" __global__ void grouped_mv4(const unsigned char* w,const float* x,float* y,const int* order,
    const int* experts,const int* starts,const int* lengths,const int* job_count,
    int rows,int cols,int topk,int per_assignment) {
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,rowtiles=(rows+3)/4;
    size_t rb=(size_t)(cols/64)*18;int workcount=(*job_count)*2*rowtiles;
    for(int work=blockIdx.x;work<workcount;work+=gridDim.x) {
        int pair=work/rowtiles,job=pair/2,slot=(pair%2)*4,count=min(4,lengths[job]-slot);
        int row=(work%rowtiles)*4+warp;if(count<=0 || row>=rows)continue;
        int assignments[4];const float* inputs[4];float sums[4]={0,0,0,0};
        #pragma unroll
        for(int s=0;s<4;s++)if(s<count){int a=order[starts[job]+slot+s];assignments[s]=a;inputs[s]=x+(size_t)(per_assignment?a:a/topk)*cols;}
        const unsigned char* wr=w+((size_t)experts[job]*rows+row)*rb;
        for(int c=lane*4;c<cols;c+=128) {
            const unsigned char* b=wr+(c/64)*18;
            float d=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(b))));
            unsigned q=__ldg(b+2+(c%64)/4);float partial[4]={0,0,0,0};
            #pragma unroll
            for(int j=0;j<4;j++) {
                float value=(float)((int)((q>>(2*j))&3)-1);
                #pragma unroll
                for(int s=0;s<4;s++)if(s<count)partial[s]=fmaf(value,__ldg(inputs[s]+c+j),partial[s]);
            }
            #pragma unroll
            for(int s=0;s<4;s++)if(s<count)sums[s]=fmaf(d,partial[s],sums[s]);
        }
        #pragma unroll
        for(int s=0;s<4;s++)if(s<count){float sum=warp_sum(sums[s]);if(lane==0)y[(size_t)assignments[s]*rows+row]=sum;}
    }
}
extern "C" __global__ void unpack_jobs(const unsigned char* w,float* out,const int* expert,
    int first,int batch,int rows,int cols,int kind) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    size_t elements=(size_t)rows*cols;if((size_t)index>=batch*elements)return;
    int job=index/elements,position=index%elements;
    size_t rb=kind==0?(size_t)cols*4:(size_t)(cols/(kind==42?64:32))*(kind==42?18:34);
    const unsigned char* row=w+((size_t)expert[first+job]*rows+position/cols)*rb;
    out[index]=tile_weight(row,position%cols,kind);
}
extern "C" __global__ void gather_jobs(const float* x,float* out,const int* order,const int* start,
    const int* length,int first,int batch,int cols,int topk,int per_assignment) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=batch*8*cols)return;
    int job=index/(8*cols),slot=(index/cols)%8,c=index%cols;
    float value=0;
    if(slot<length[first+job]){int a=order[start[first+job]+slot];int input=per_assignment?a:a/topk;value=__ldg(x+(size_t)input*cols+c);}
    out[index]=value;
}
extern "C" __global__ void scatter_jobs(const float* input,float* out,const int* order,const int* start,
    const int* length,int first,int batch,int rows) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=batch*8*rows)return;
    int job=index/(8*rows),slot=(index/rows)%8,r=index%rows;
    if(slot<length[first+job]){int a=order[start[first+job]+slot];out[(size_t)a*rows+r]=input[index];}
}

extern "C" __global__ void combine_batch(const float* expert,const float* weights,const float* shared,
    float* x,int tokens,int width,int topk) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=tokens*width)return;
    int token=index/width,c=index%width;float sum=0;
    for(int slot=0;slot<topk;slot++)sum=fmaf(weights[token*topk+slot],expert[((size_t)token*topk+slot)*width+c],sum);
    x[index]+=sum+shared[index];
}

// One CTA/channel, carry state remains in registers through a bounded token chunk.
// No padding is processed: tokens is the actual (possibly partial) chunk length.
extern "C" __global__ void conv_chunk(const float* projected,const float* w,const float* bias,
    float* history,float* out,int tokens,int channels,int kernel,int stride,int gate_width) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=channels)return;
    for(int token=0;token<tokens;token++) {
        float input=projected[token*stride+gate_width+c],sum=bias[c];
        for(int j=0;j<kernel-1;j++)sum=fmaf(history[c*(kernel-1)+j],w[c*kernel+j],sum);
        sum=fmaf(input,w[c*kernel+kernel-1],sum);
        for(int j=0;j<kernel-2;j++)history[c*(kernel-1)+j]=history[c*(kernel-1)+j+1];
        history[c*(kernel-1)+kernel-2]=input;out[token*channels+c]=sum*sigmoid(sum);
    }
}
extern "C" __global__ void ssm_chunk(const float* conv,const float* projected,const float* dtbias,
    const float* a,const float* d,float* state,float* y,int tokens,int heads,int headdim,int statesize,
    int groups,int stride) {
    int channel=blockIdx.x,h=channel/headdim,p=channel%headdim,t=threadIdx.x;
    int inner=heads*headdim,channels=inner+2*groups*statesize,g=h/(heads/groups);
    // The adapter currently supports state size 128; generic decode remains available separately.
    float carry=state[((size_t)h*headdim+p)*statesize+t];
    __shared__ float sums[4];
    for(int token=0;token<tokens;token++) {
        const float* cv=conv+(size_t)token*channels;
        float dt=softplus(projected[(size_t)token*stride+inner+channels+h]+dtbias[h]);
        carry=fmaf(dt*cv[channel],cv[inner+g*statesize+t],expf(dt*a[h])*carry);
        float sum=carry*cv[inner+groups*statesize+g*statesize+t];sum=warp_sum(sum);
        if((t&31)==0)sums[t>>5]=sum;__syncthreads();
        if(t<32){sum=t<4?sums[t]:0;sum=warp_sum(sum);if(t==0)y[token*inner+channel]=fmaf(d[h],cv[channel],sum);}
        __syncthreads();
    }
    state[((size_t)h*headdim+p)*statesize+t]=carry;
}

// 128-key tiles, bounded score scratch, stable online softmax. Each query sees only its prefix.
extern "C" __global__ void attention_tiled(const float* q,const float* k,const float* v,float* out,
    int tokens,int past,int qheads,int kvheads,int dim) {
    int h=blockIdx.x,token=blockIdx.y,t=threadIdx.x,lane=t&31,warp=t>>5;
    if(token>=tokens)return;
    int kv=h/(qheads/kvheads),length=past+token+1;
    const float* query=q+((size_t)token*qheads+h)*dim;
    __shared__ float score[128],reduction[128],prob[128],alpha,beta,total,maxval;
    if(t==0){total=0;maxval=-INF;}__syncthreads();
    float acc=0;
    for(int base=0;base<length;base+=128) {
        int count=min(128,length-base);
        for(int j=warp;j<count;j+=4) {
            const float* key=k+((size_t)(base+j)*kvheads+kv)*dim;
            float dot=0;for(int c=lane;c<dim;c+=32)dot=fmaf(query[c],key[c],dot);
            dot=warp_sum(dot);if(lane==0)score[j]=dot*rsqrtf((float)dim);
        }
        __syncthreads();
        reduction[t]=t<count?score[t]:-INF;__syncthreads();
        for(int d=64;d;d/=2){if(t<d)reduction[t]=fmaxf(reduction[t],reduction[t+d]);__syncthreads();}
        float tilemax=reduction[0];
        prob[t]=t<count?expf(score[t]-tilemax):0;__syncthreads();
        reduction[t]=prob[t];__syncthreads();
        for(int d=64;d;d/=2){if(t<d)reduction[t]+=reduction[t+d];__syncthreads();}
        if(t==0){float next=fmaxf(maxval,tilemax);alpha=expf(maxval-next);beta=expf(tilemax-next);total=total*alpha+reduction[0]*beta;maxval=next;}
        __syncthreads();
        if(t<dim){float sum=0;for(int j=0;j<count;j++)sum=fmaf(prob[j],v[((size_t)(base+j)*kvheads+kv)*dim+t],sum);acc=acc*alpha+sum*beta;}
        __syncthreads();
    }
    if(t<dim)out[((size_t)token*qheads+h)*dim+t]=acc/total;
}
