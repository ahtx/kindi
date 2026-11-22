/* moeaccel.cuh - the device side of Kindi's MoE mailbox (F9, see moeaccel.h): kernels that a whole decode pass can
   hold in one CUDA graph. Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.
   NVRTC-safe: no host headers. The layout of a layer's mailbox (bytes from its start) is the one of moeaccel.c
   (ma_box_layout); the host checks it against MA_* below at start. */
#ifndef KINDI_MOEACCEL_CUH
#define KINDI_MOEACCEL_CUH
#define MA_POSTED 0
#define MA_DONE 64
#define MA_ERR 128
#define MA_T 192
#define MA_IDS 256

/* once per pass, first in the graph: the pass number that the mailboxes of this pass carry */
extern "C" __global__ void k_ma_seq(unsigned* seq) {
    if (threadIdx.x == 0 && blockIdx.x == 0) seq[0] += 1u;
}

/* layer l's request to the CPU: the T x nsel picks and the T expert inputs (kin floats each), then the pass number
   in "posted" (after a system fence, so the host sees the data before the number) */
extern "C" __global__ void k_ma_post(unsigned char* box, const unsigned* seq, const int* ids, const float* x, int T,
                                     int nsel, int kin, long long xoff) {
    volatile int* bi = (volatile int*)(box + MA_IDS);
    volatile float* bx = (volatile float*)(box + xoff);
    for (int i = threadIdx.x; i < T * nsel; i += blockDim.x) bi[i] = ids[i];
    for (int i = threadIdx.x; i < T * kin; i += blockDim.x) bx[i] = x[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile int*)(box + MA_T) = T;
        __threadfence_system();
        *(volatile unsigned*)(box + MA_POSTED) = seq[0];
        __threadfence_system();
    }
}

/* wait for the CPU's answer to layer l (its "done" = this pass), then copy the rows of the slots that are not on
   this card (slotmap[ids] < 0) from the mailbox into dst [T][nsel][K] (per-pair rows). Grid: one block per (token,
   slot). A wait past ~20 s sets the mailbox's err word and goes on (the host reports it). */
extern "C" __global__ void k_ma_wait(unsigned char* box, const unsigned* seq, float* dst, const int* ids,
                                     const int* slotmap, int nsel, int K, long long rowoff) {
    const int ts = blockIdx.x;                     /* token * nsel + slot */
    __shared__ int go;
    if (threadIdx.x == 0) {
        const unsigned want = seq[0];
        const long long t0 = clock64();
        go = 1;
        while (*(volatile unsigned*)(box + MA_DONE) != want) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(200);                      /* sm_70+; older cards poll without the pause */
#endif
            if (clock64() - t0 > 40000000000LL) { *(volatile unsigned*)(box + MA_ERR) = 1u; go = 0; break; }
        }
        __threadfence_system();
    }
    __syncthreads();
    if (!go || slotmap[ids[ts]] >= 0) return;
    /* 16-byte reads of the mapped rows (K a multiple of 4; the rows 64-byte aligned): a quarter of the PCIe reads */
    const float4* src = (const float4*)((const float*)(box + rowoff) + (long long)ts * K);
    float4* d = (float4*)(dst + (long long)ts * K);
    for (int i = threadIdx.x; i < K / 4; i += blockDim.x) d[i] = __ldcv(src + i);   /* no cached copy: fresh */
}
#endif
