/* Inkling-2-Mini DSpark CPU drafter (Q8_0 attention/FFN + BF16 Markov head, zero VRAM).
 * Matches llama.cpp's dflash.cpp / dspark.cpp (pr32218):
 *   - 3 non-causal layers (d=2304, nh=24, nkv=8, hd=128, ff=9216), SWA=1024 on layers {0, 2}, full on {1}
 *   - NEOX RoPE (freq_base=1e6, n_rot=128) + per-head Q/K/K_ctx RMSNorm (eps=1e-6)
 *   - Target feature injection (dspark_fuse[4,4] softmax -> dflare_norm / dspark_norm)
 *   - 7-step Markov GRU head (rank=256) outputting effective hidden = hidden[i+1] + dspark_hidden_out @ latent + b
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define DS_AVX2 1
#else
#define DS_AVX2 0
#endif

#define DS_D 2304
#define DS_NH 24
#define DS_NKV 8
#define DS_HD 128
#define DS_QDIM 3072
#define DS_GQA 1024
#define DS_FF 9216
#define DS_NL 3
#define DS_B 8
#define DS_R 256
#define DS_TAPS 4
#define DS_SWA 1024
#define DS_EPS 1e-6f
#define DS_ROPE_BASE 1000000.0f
#define DS_Q8_BSZ 34

typedef int (*DsHeadArgmaxFn)(void* user, const float* x_2304);

typedef struct {
    const uint8_t* attn_k_ctx;
    const uint8_t* attn_v_ctx;
    const float* attn_k_ctx_norm;
    const float* dflare_norm;

    const float* attn_norm;
    const uint8_t* attn_q;
    const uint8_t* attn_k;
    const uint8_t* attn_v;
    const float* attn_q_norm;
    const float* attn_k_norm;
    const uint8_t* attn_output;

    const float* ffn_norm;
    const uint8_t* ffn_gate;
    const uint8_t* ffn_up;
    const uint8_t* ffn_down;
} DsLayerSpec;

typedef struct {
    const float* fuse;            /* [4, 4] */
    const float* dspark_norm;     /* [2304] */
    const float* mask_embd;       /* [2304] */
    const float* output_norm;     /* [2304] */
    const uint16_t* hidden_out_w; /* BF16 [2304, 256] */
    const float* hidden_out_b;    /* [2304] */
    const uint16_t* joint_proj_w; /* BF16 [768, 5120] */
    const float* joint_proj_b;    /* [768] */
    const uint16_t* markov_w1;    /* BF16 [V_pad, 256] */
    int vocab_pad;
    int max_ctx;
    DsLayerSpec layers[DS_NL];
} DsModelSpec;

typedef struct {
    float* hidden_out_f32;        /* [2304 * 256] */
    float* joint_proj_f32;        /* [768 * 5120] */
    const uint16_t* markov_w1;
    int vocab_pad;
    int max_ctx;
    int n_threads;

    float fuse[4][4];
    float dspark_norm[DS_D];
    float mask_embd[DS_D];
    float output_norm[DS_D];
    float hidden_out_b[DS_D];
    float joint_proj_b[3 * DS_R];

    struct {
        const uint8_t *attn_k_ctx, *attn_v_ctx;
        const uint8_t *attn_q, *attn_k, *attn_v, *attn_output;
        const uint8_t *ffn_gate, *ffn_up, *ffn_down;
        float attn_k_ctx_norm[DS_HD];
        float dflare_norm[DS_D];
        float attn_norm[DS_D];
        float attn_q_norm[DS_HD];
        float attn_k_norm[DS_HD];
        float ffn_norm[DS_D];
        float *kv_k; /* [max_ctx * DS_GQA] */
        float *kv_v; /* [max_ctx * DS_GQA] */
    } L[DS_NL];

    float* rope_inv;              /* [DS_HD / 2] */
    float ctx_last[DS_D];         /* latest injected target context vector */

    /* Scratch buffers for B <= DS_B */
    float h_buf[DS_B * DS_D];
    float x_buf[DS_B * DS_D];
    float q_buf[DS_B * DS_QDIM];
    float k_buf[DS_B * DS_GQA];
    float v_buf[DS_B * DS_GQA];
    float att_buf[DS_B * DS_QDIM];
    float ff_g[DS_B * DS_FF];
    float ff_u[DS_B * DS_FF];
    float ff_out[DS_B * DS_D];
    int8_t xq_buf[DS_B * DS_FF];
    float xd_buf[DS_B * (DS_FF / 32)];
} DsCtx;

static inline float fp16_to_f32(uint16_t h) {
    uint32_t s = ((uint32_t)h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x03FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s;
        else {
            e = 1;
            while ((m & 0x0400u) == 0) { m <<= 1; e--; }
            m &= 0x03FFu;
            f = s | ((e + 112u) << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112u) << 23) | (m << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float out;
    memcpy(&out, &u, 4);
    return out;
}

static void rmsnorm_rows(const float* x, const float* w, int D, int T, float* y) {
    for (int t = 0; t < T; t++) {
        const float* xt = x + (size_t)t * D;
        float* yt = y + (size_t)t * D;
        double ss = 0.0;
        for (int i = 0; i < D; i++) {
            double v = xt[i];
            ss += v * v;
        }
        float inv = 1.0f / sqrtf((float)(ss / (double)D) + DS_EPS);
        for (int i = 0; i < D; i++) {
            yt[i] = xt[i] * inv * w[i];
        }
    }
}

static void rope_neox_heads(const DsCtx* C, float* x, int n_heads, int T, int pos0) {
    const int half = DS_HD / 2;
    for (int t = 0; t < T; t++) {
        float pos = (float)(pos0 + t);
        for (int h = 0; h < n_heads; h++) {
            float* v = x + ((size_t)t * n_heads + h) * DS_HD;
            for (int j = 0; j < half; j++) {
                float ang = pos * C->rope_inv[j];
                float c = cosf(ang), s = sinf(ang);
                float a = v[j], b = v[j + half];
                v[j]        = a * c - b * s;
                v[j + half] = a * s + b * c;
            }
        }
    }
}

static void quant_q8_rows(const float* x, int K, int T, int8_t* xq, float* xd) {
    int nb = K / 32;
    for (int t = 0; t < T; t++) {
        const float* xt = x + (size_t)t * K;
        int8_t* qt = xq + (size_t)t * K;
        float* dt = xd + (size_t)t * nb;
        for (int b = 0; b < nb; b++) {
            const float* xb = xt + b * 32;
            int8_t* qb = qt + b * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; i++) {
                float a = fabsf(xb[i]);
                if (a > amax) amax = a;
            }
            if (amax == 0.0f) {
                dt[b] = 0.0f;
                memset(qb, 0, 32);
            } else {
                float d = amax / 127.0f;
                float inv = 127.0f / amax;
                dt[b] = d;
                for (int i = 0; i < 32; i++) {
                    int q = (int)lrintf(xb[i] * inv);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    qb[i] = (int8_t)q;
                }
            }
        }
    }
}

static inline float dot_q8_q8(const uint8_t* w_row, int nb, const int8_t* xq, const float* xd) {
#if DS_AVX2
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = w_row + (size_t)b * DS_Q8_BSZ;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        float sc = fp16_to_f32(dh) * xd[b];
        __m256i aw = _mm256_loadu_si256((const __m256i*)(blk + 2));
        __m256i ax = _mm256_loadu_si256((const __m256i*)(xq + b * 32));
        __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(aw, aw), _mm256_sign_epi8(ax, aw));
        __m256i p32 = _mm256_madd_epi16(p16, ones);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(sc), acc);
    }
    __m128 h4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    h4 = _mm_hadd_ps(h4, h4);
    h4 = _mm_hadd_ps(h4, h4);
    return _mm_cvtss_f32(h4);
#else
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = w_row + (size_t)b * DS_Q8_BSZ;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        float sc = fp16_to_f32(dh) * xd[b];
        const int8_t* wq = (const int8_t*)(blk + 2);
        const int8_t* xqb = xq + b * 32;
        int isum = 0;
        for (int i = 0; i < 32; i++) isum += (int)wq[i] * (int)xqb[i];
        sum += sc * (float)isum;
    }
    return sum;
#endif
}

static void q8_mm_rows(DsCtx* C, const uint8_t* w, int R, int K, const float* x, int T, float* y) {
    int nb = K / 32;
    size_t rb = (size_t)nb * DS_Q8_BSZ;
    quant_q8_rows(x, K, T, C->xq_buf, C->xd_buf);
    int r;
#pragma omp parallel for schedule(static) if (R >= 256)
    for (r = 0; r < R; r++) {
        const uint8_t* wr = w + (size_t)r * rb;
        if (T == 1) {
            y[r] = dot_q8_q8(wr, nb, C->xq_buf, C->xd_buf);
        } else {
            for (int t = 0; t < T; t++) {
                y[(size_t)t * R + r] = dot_q8_q8(wr, nb, C->xq_buf + (size_t)t * K, C->xd_buf + (size_t)t * nb);
            }
        }
    }
}

static void mix_features_row(const DsCtx* C, int row, const float* feats_4xD, const float* norm_w, float* out_D) {
    float tmp[DS_D];
    const float* w = C->fuse[row];
    for (int i = 0; i < DS_D; i++) {
        tmp[i] = w[0] * feats_4xD[0 * DS_D + i]
               + w[1] * feats_4xD[1 * DS_D + i]
               + w[2] * feats_4xD[2 * DS_D + i]
               + w[3] * feats_4xD[3 * DS_D + i];
    }
    rmsnorm_rows(tmp, norm_w, DS_D, 1, out_D);
}

void* ds_init(const DsModelSpec* spec, int n_threads) {
    DsCtx* C = (DsCtx*)calloc(1, sizeof(DsCtx));
    if (!C) return NULL;
    C->vocab_pad = spec->vocab_pad;
    C->max_ctx = spec->max_ctx > 0 ? spec->max_ctx : 4096;
    C->n_threads = n_threads > 0 ? n_threads : 4;
#ifdef _OPENMP
    omp_set_num_threads(C->n_threads);
#endif

    for (int r = 0; r < 4; r++) {
        const float* fr = spec->fuse + r * 4;
        float mx = fr[0];
        for (int k = 1; k < 4; k++) if (fr[k] > mx) mx = fr[k];
        float s = 0.0f;
        for (int k = 0; k < 4; k++) { C->fuse[r][k] = expf(fr[k] - mx); s += C->fuse[r][k]; }
        float inv = 1.0f / s;
        for (int k = 0; k < 4; k++) C->fuse[r][k] *= inv;
    }

    memcpy(C->dspark_norm, spec->dspark_norm, DS_D * sizeof(float));
    memcpy(C->mask_embd, spec->mask_embd, DS_D * sizeof(float));
    memcpy(C->output_norm, spec->output_norm, DS_D * sizeof(float));
    memcpy(C->hidden_out_b, spec->hidden_out_b, DS_D * sizeof(float));
    memcpy(C->joint_proj_b, spec->joint_proj_b, 3 * DS_R * sizeof(float));
    C->markov_w1 = spec->markov_w1;

    C->hidden_out_f32 = (float*)malloc((size_t)DS_D * DS_R * sizeof(float));
    C->joint_proj_f32 = (float*)malloc((size_t)(3 * DS_R) * (2 * DS_R + 2 * DS_D) * sizeof(float));
    C->rope_inv = (float*)malloc((DS_HD / 2) * sizeof(float));
    if (!C->hidden_out_f32 || !C->joint_proj_f32 || !C->rope_inv) { free(C); return NULL; }

    for (size_t i = 0; i < (size_t)DS_D * DS_R; i++) {
        C->hidden_out_f32[i] = bf16_to_f32(spec->hidden_out_w[i]);
    }
    for (size_t i = 0; i < (size_t)(3 * DS_R) * (2 * DS_R + 2 * DS_D); i++) {
        C->joint_proj_f32[i] = bf16_to_f32(spec->joint_proj_w[i]);
    }
    for (int j = 0; j < DS_HD / 2; j++) {
        C->rope_inv[j] = powf(DS_ROPE_BASE, -2.0f * (float)j / (float)DS_HD);
    }

    for (int l = 0; l < DS_NL; l++) {
        const DsLayerSpec* s = &spec->layers[l];
        C->L[l].attn_k_ctx = s->attn_k_ctx;
        C->L[l].attn_v_ctx = s->attn_v_ctx;
        C->L[l].attn_q     = s->attn_q;
        C->L[l].attn_k     = s->attn_k;
        C->L[l].attn_v     = s->attn_v;
        C->L[l].attn_output = s->attn_output;
        C->L[l].ffn_gate   = s->ffn_gate;
        C->L[l].ffn_up     = s->ffn_up;
        C->L[l].ffn_down   = s->ffn_down;
        memcpy(C->L[l].attn_k_ctx_norm, s->attn_k_ctx_norm, DS_HD * sizeof(float));
        memcpy(C->L[l].dflare_norm, s->dflare_norm, DS_D * sizeof(float));
        memcpy(C->L[l].attn_norm, s->attn_norm, DS_D * sizeof(float));
        memcpy(C->L[l].attn_q_norm, s->attn_q_norm, DS_HD * sizeof(float));
        memcpy(C->L[l].attn_k_norm, s->attn_k_norm, DS_HD * sizeof(float));
        memcpy(C->L[l].ffn_norm, s->ffn_norm, DS_D * sizeof(float));
        C->L[l].kv_k = (float*)calloc((size_t)C->max_ctx * DS_GQA, sizeof(float));
        C->L[l].kv_v = (float*)calloc((size_t)C->max_ctx * DS_GQA, sizeof(float));
        if (!C->L[l].kv_k || !C->L[l].kv_v) return NULL;
    }
    return C;
}

void ds_free(void* handle) {
    DsCtx* C = (DsCtx*)handle;
    if (!C) return;
    free(C->hidden_out_f32);
    free(C->joint_proj_f32);
    free(C->rope_inv);
    for (int l = 0; l < DS_NL; l++) {
        free(C->L[l].kv_k);
        free(C->L[l].kv_v);
    }
    free(C);
}

void ds_reset(void* handle) {
    DsCtx* C = (DsCtx*)handle;
    if (!C) return;
    memset(C->ctx_last, 0, sizeof(C->ctx_last));
}

int ds_inject(void* handle, int pos0, int T, const float* feats_flat) {
    DsCtx* C = (DsCtx*)handle;
    if (!C || T <= 0 || pos0 + T > C->max_ctx) return -1;
    /* Process in chunks <= DS_B to reuse scratch buffers cleanly */
    float ctx_chunk[DS_B * DS_D];
    for (int off = 0; off < T; off += DS_B) {
        int n = T - off;
        if (n > DS_B) n = DS_B;
        int p0 = pos0 + off;
        for (int l = 0; l < DS_NL; l++) {
            for (int t = 0; t < n; t++) {
                mix_features_row(C, l, feats_flat + (size_t)(off + t) * DS_TAPS * DS_D,
                                 C->L[l].dflare_norm, ctx_chunk + (size_t)t * DS_D);
            }
            q8_mm_rows(C, C->L[l].attn_k_ctx, DS_GQA, DS_D, ctx_chunk, n, C->k_buf);
            q8_mm_rows(C, C->L[l].attn_v_ctx, DS_GQA, DS_D, ctx_chunk, n, C->v_buf);
            rmsnorm_rows(C->k_buf, C->L[l].attn_k_ctx_norm, DS_HD, n * DS_NKV, C->k_buf);
            rope_neox_heads(C, C->k_buf, DS_NKV, n, p0);
            memcpy(C->L[l].kv_k + (size_t)p0 * DS_GQA, C->k_buf, (size_t)n * DS_GQA * sizeof(float));
            memcpy(C->L[l].kv_v + (size_t)p0 * DS_GQA, C->v_buf, (size_t)n * DS_GQA * sizeof(float));
        }
    }
    mix_features_row(C, DS_NL, feats_flat + (size_t)(T - 1) * DS_TAPS * DS_D,
                     C->dspark_norm, C->ctx_last);
    return 0;
}

static float dot_f32(const float* a, const float* b, int n) {
#if DS_AVX2
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    }
    __m128 h4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    h4 = _mm_hadd_ps(h4, h4);
    h4 = _mm_hadd_ps(h4, h4);
    float s = _mm_cvtss_f32(h4);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

static void f32_matvec(const float* W, int R, int K, const float* x, const float* bias, float* y) {
    int r;
#pragma omp parallel for schedule(static) if (R >= 256)
    for (r = 0; r < R; r++) {
        y[r] = dot_f32(W + (size_t)r * K, x, K) + (bias ? bias[r] : 0.0f);
    }
}

void ds_forward_block(void* handle, int anchor_tok, int pos0, const float* anchor_embd_2304, float* out_hidden_8xD) {
    DsCtx* C = (DsCtx*)handle;
    (void)anchor_tok;
    memcpy(C->h_buf, anchor_embd_2304, DS_D * sizeof(float));
    for (int i = 1; i < DS_B; i++) {
        memcpy(C->h_buf + (size_t)i * DS_D, C->mask_embd, DS_D * sizeof(float));
    }

    const float scale = 1.0f / sqrtf((float)DS_HD);
    static const int is_swa[DS_NL] = {1, 0, 1};

    for (int l = 0; l < DS_NL; l++) {
        rmsnorm_rows(C->h_buf, C->L[l].attn_norm, DS_D, DS_B, C->x_buf);
        q8_mm_rows(C, C->L[l].attn_q, DS_QDIM, DS_D, C->x_buf, DS_B, C->q_buf);
        q8_mm_rows(C, C->L[l].attn_k, DS_GQA,  DS_D, C->x_buf, DS_B, C->k_buf);
        q8_mm_rows(C, C->L[l].attn_v, DS_GQA,  DS_D, C->x_buf, DS_B, C->v_buf);

        rmsnorm_rows(C->q_buf, C->L[l].attn_q_norm, DS_HD, DS_B * DS_NH, C->q_buf);
        rmsnorm_rows(C->k_buf, C->L[l].attn_k_norm, DS_HD, DS_B * DS_NKV, C->k_buf);
        rope_neox_heads(C, C->q_buf, DS_NH,  DS_B, pos0);
        rope_neox_heads(C, C->k_buf, DS_NKV, DS_B, pos0);

        int k_start = (is_swa[l] && pos0 >= DS_SWA) ? (pos0 - (DS_SWA - 1)) : 0;
        const float* kv_k_past = C->L[l].kv_k;
        const float* kv_v_past = C->L[l].kv_v;

        /* Non-causal block_uniform attention across [k_start .. pos0-1] and block keys [0..7] */
        int qi;
#pragma omp parallel for schedule(static)
        for (qi = 0; qi < DS_B * DS_NH; qi++) {
            int i = qi / DS_NH;
            int h = qi % DS_NH;
            int kh = h / (DS_NH / DS_NKV);
            const float* qv = C->q_buf + ((size_t)i * DS_NH + h) * DS_HD;
            float* ov = C->att_buf + ((size_t)i * DS_NH + h) * DS_HD;

            float m = -1e30f;
            float d = 0.0f;
            memset(ov, 0, DS_HD * sizeof(float));

            for (int kp = k_start; kp < pos0; kp++) {
                const float* kv = kv_k_past + (size_t)kp * DS_GQA + kh * DS_HD;
                const float* vv = kv_v_past + (size_t)kp * DS_GQA + kh * DS_HD;
                float sc = dot_f32(qv, kv, DS_HD) * scale;
                float m_new = sc > m ? sc : m;
                float exp_old = expf(m - m_new);
                float exp_new = expf(sc - m_new);
                d = d * exp_old + exp_new;
                for (int j = 0; j < DS_HD; j++) ov[j] = ov[j] * exp_old + exp_new * vv[j];
                m = m_new;
            }
            for (int bk = 0; bk < DS_B; bk++) {
                const float* kv = C->k_buf + (size_t)bk * DS_GQA + kh * DS_HD;
                const float* vv = C->v_buf + (size_t)bk * DS_GQA + kh * DS_HD;
                float sc = dot_f32(qv, kv, DS_HD) * scale;
                float m_new = sc > m ? sc : m;
                float exp_old = expf(m - m_new);
                float exp_new = expf(sc - m_new);
                d = d * exp_old + exp_new;
                for (int j = 0; j < DS_HD; j++) ov[j] = ov[j] * exp_old + exp_new * vv[j];
                m = m_new;
            }
            float inv_d = 1.0f / d;
            for (int j = 0; j < DS_HD; j++) ov[j] *= inv_d;
        }

        q8_mm_rows(C, C->L[l].attn_output, DS_D, DS_QDIM, C->att_buf, DS_B, C->x_buf);
        for (int idx = 0; idx < DS_B * DS_D; idx++) C->h_buf[idx] += C->x_buf[idx];

        rmsnorm_rows(C->h_buf, C->L[l].ffn_norm, DS_D, DS_B, C->x_buf);
        q8_mm_rows(C, C->L[l].ffn_gate, DS_FF, DS_D, C->x_buf, DS_B, C->ff_g);
        q8_mm_rows(C, C->L[l].ffn_up,   DS_FF, DS_D, C->x_buf, DS_B, C->ff_u);
        for (int idx = 0; idx < DS_B * DS_FF; idx++) {
            float g = C->ff_g[idx];
            C->ff_g[idx] = (g / (1.0f + expf(-g))) * C->ff_u[idx];
        }
        q8_mm_rows(C, C->L[l].ffn_down, DS_D, DS_FF, C->ff_g, DS_B, C->ff_out);
        for (int idx = 0; idx < DS_B * DS_D; idx++) C->h_buf[idx] += C->ff_out[idx];
    }

    rmsnorm_rows(C->h_buf, C->output_norm, DS_D, DS_B, out_hidden_8xD);
}

int ds_draft_tokens(void* handle, int anchor_tok, int pos0, int n_draft,
                    const float* anchor_embd_2304,
                    DsHeadArgmaxFn head_fn, void* head_user,
                    int* out_toks) {
    DsCtx* C = (DsCtx*)handle;
    if (!C || pos0 + DS_B > C->max_ctx || n_draft <= 0 || n_draft > DS_B - 1) return -1;

    float hidden[DS_B * DS_D];
    ds_forward_block(C, anchor_tok, pos0, anchor_embd_2304, hidden);

    float state[DS_R];
    memset(state, 0, sizeof(state));
    float joint_in[2 * DS_R + 2 * DS_D];
    float joint_out[3 * DS_R];
    float eff_h[DS_D];

    memcpy(joint_in + 2 * DS_R + DS_D, C->ctx_last, DS_D * sizeof(float));

    int prev = anchor_tok;
    for (int i = 0; i < n_draft; i++) {
        if (prev < 0 || prev >= C->vocab_pad) return -2;
        memcpy(joint_in, state, DS_R * sizeof(float));
        const uint16_t* w1r = C->markov_w1 + (size_t)prev * DS_R;
        for (int j = 0; j < DS_R; j++) joint_in[DS_R + j] = bf16_to_f32(w1r[j]);
        const float* row = hidden + (size_t)(i + 1) * DS_D;
        memcpy(joint_in + 2 * DS_R, row, DS_D * sizeof(float));

        f32_matvec(C->joint_proj_f32, 3 * DS_R, 2 * DS_R + 2 * DS_D, joint_in, C->joint_proj_b, joint_out);
        for (int j = 0; j < DS_R; j++) {
            float gate = 1.0f / (1.0f + expf(-joint_out[j]));
            float cand = tanhf(joint_out[DS_R + j]);
            state[j] = cand + gate * (state[j] - cand);
            joint_out[2 * DS_R + j] = tanhf(joint_out[2 * DS_R + j]);
        }

        f32_matvec(C->hidden_out_f32, DS_D, DS_R, joint_out + 2 * DS_R, C->hidden_out_b, eff_h);
        for (int j = 0; j < DS_D; j++) eff_h[j] += row[j];

        prev = head_fn(head_user, eff_h);
        if (prev < 0) return -3;
        out_toks[i] = prev;
    }
    return n_draft;
}
