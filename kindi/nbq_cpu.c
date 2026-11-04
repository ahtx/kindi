/* nbq_cpu.c - IQ3_XXS matrix x vector on the CPU (AVX2, OpenMP), for the host layers of the Nanbeige engine
   (nanbeige.py, KINDI_NB_CPUF).

   Structure: the codebook is expanded into a table of signed words, wtab[n * 256 + g] = the 4 int8 values of
   codebook entry g with the 4 sign bits n applied (16 KB, it stays in L1). At load, nbq_cpu_expand_idx writes
   the rows that the CPU computes in an index form: per 4 weights one uint16 = g | n << 8 (the parity sign bit
   already applied), per block the fp16 d and the 8 run scales (4 bits each): 4.19 bits per weight. In a step,
   nbq_cpu_dot_idx takes 4 signed weights from one table load, widens them to int16, and multiplies them with
   the input (int16) with _mm256_madd_epi16 into int32: exact, like the int8 dot products of the card kernels.
   y = sum over runs of xs * s * (the int32 sums of the run); the float steps are in another order than on the
   card (about 1e-7 relative).

   The GGUF blocks are 98 bytes: fp16 d, 64 bytes of codebook indices, then 4 words of 7 sign bits and a 4-bit
   scale per run of 32; the 8th sign bit of each 8 weights makes the parity even.
   Built on first use with gcc -O3 -march=native -fopenmp. */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern const uint32_t nbq_iq3xxs_grid_c[256];

static uint32_t wtab[16 * 256];
static int ready = 0;

static void init(void) {
    for (int n = 0; n < 16; n++)
        for (int g = 0; g < 256; g++) {
            uint32_t e = nbq_iq3xxs_grid_c[g], o = 0;
            for (int k = 0; k < 4; k++) {
                int v = (int)((e >> (8 * k)) & 255u);
                if (n >> k & 1) v = -v;
                o |= (uint32_t)(uint8_t)(int8_t)v << (8 * k);
            }
            wtab[n * 256 + g] = o;
        }
    ready = 1;
}

static inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
    return _mm_cvtss_f32(a);
}

/* the index form: per 4 weights one uint16 = codebook entry | 4 sign bits << 8 (an index into wtab), per block
   the fp16 d and the 8 run scales (4 bits each) */
void nbq_cpu_expand_idx(const uint8_t* w, int r0, int r1, int cols, uint16_t* wi, uint8_t* wb) {
    if (!ready) init();
    const int nb = cols / 256;
#pragma omp parallel for schedule(static)
    for (int r = r0; r < r1; r++) {
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t* b = w + ((size_t)r * nb + blk) * 98;
            uint16_t* o = wi + ((size_t)(r - r0) * nb + blk) * 64;
            uint8_t* h = wb + ((size_t)(r - r0) * nb + blk) * 6;
            memcpy(h, b, 2);
            uint32_t scw = 0;
            for (int sub = 0; sub < 8; sub++) {
                uint32_t aux;
                memcpy(&aux, b + 66 + 4 * sub, 4);
                scw |= (aux >> 28) << (4 * sub);
                for (int l = 0; l < 4; l++) {
                    const uint32_t s7 = (aux >> (7 * l)) & 127u;
                    const uint32_t s8 = s7 | ((uint32_t)__builtin_parity(s7) << 7);
                    o[8 * sub + 2 * l] = (uint16_t)(b[2 + 8 * sub + 2 * l] | (s8 & 15u) << 8);
                    o[8 * sub + 2 * l + 1] = (uint16_t)(b[2 + 8 * sub + 2 * l + 1] | (s8 >> 4) << 8);
                }
            }
            memcpy(h + 2, &scw, 4);
        }
    }
}

void nbq_cpu_dot_idx(const uint16_t* wi, const uint8_t* wb, int n, int cols, const int8_t* xq, const float* xs, float* y) {
    if (!ready) init();
    int16_t* x16 = (int16_t*)aligned_alloc(32, (size_t)cols * 2);
    for (int i = 0; i < cols; i++) x16[i] = xq[i];
    const int nb = cols / 256;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < n; r++) {
        __m256 acc = _mm256_setzero_ps();
        for (int blk = 0; blk < nb; blk++) {
            const uint16_t* o = wi + ((size_t)r * nb + blk) * 64;
            const uint8_t* h = wb + ((size_t)r * nb + blk) * 6;
            uint16_t dh; uint32_t scw;
            memcpy(&dh, h, 2); memcpy(&scw, h + 2, 4);
            const float d = _cvtsh_ss(dh);
            for (int sub = 0; sub < 8; sub++) {
                const int g32 = blk * 8 + sub;
#ifdef NBQ_IDX_GATHER      /* AVX2 gather: not faster on the E5-2673 v4 */
                const __m256i ix = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(o + 8 * sub)));
                const __m256i wv = _mm256_i32gather_epi32((const int*)wtab, ix, 4);
#else
                const uint16_t* p8 = o + 8 * sub;
                const __m256i wv = _mm256_set_epi32((int)wtab[p8[7]], (int)wtab[p8[6]], (int)wtab[p8[5]], (int)wtab[p8[4]],
                                                    (int)wtab[p8[3]], (int)wtab[p8[2]], (int)wtab[p8[1]], (int)wtab[p8[0]]);
#endif
                const int16_t* xp = x16 + 32 * g32;
                const __m256i w0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(wv));
                const __m256i w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(wv, 1));
                const __m256i p = _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_load_si256((const __m256i*)xp)),
                                                   _mm256_madd_epi16(w1, _mm256_load_si256((const __m256i*)(xp + 16))));
                const float s = d * (0.5f + (float)((scw >> (4 * sub)) & 15u)) * 0.5f;
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(s * xs[g32]), acc);
            }
        }
        y[r] = hsum8(acc);
    }
    free(x16);
}
