/* moeaccel.h - Kindi's shared MoE acceleration layer (F9): the policies that every MoE engine shares (DeepSeek,
   Flash-Next, Lightning). Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.

   The layer knows experts as (layer, expert) keys and byte sizes, never their weight format: each engine keeps its
   kernels, its relayout and its CPU maths, and asks this layer what to do.

   - ma_cache: an adaptive card expert cache. Seeded from the release's usage order; the engine reports the experts
     each call routes to; at the end of a verify window, ma_cache_tick() returns the swaps to start (a missed expert
     that the recent tokens keep asking for replaces the cached one with the lowest aged count). The victim's slot is
     no longer used from that moment; the engine copies the new expert into the slot and calls ma_cache_admit() when
     the copy has landed, and only then is the expert used. So a window never waits on a copy.
   - ma_off: the share of a layer's cache misses that is copied to a card and computed there while the CPU computes
     the rest, adapted from the measured CPU time and copy time per expert so that both finish together.

   Which card or core computes a (token, expert) pair never changes its value in Kindi's engines (the card and CPU
   expert kernels give the same bits), so these policies cannot change the output. */
#ifndef KINDI_MOEACCEL_H
#define KINDI_MOEACCEL_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ma_cache ma_cache;
typedef struct { int slot, layer, expert; } ma_swap;

/* nl layers of ne experts each, nslot slots (one expert each; an engine with per-layer sizes keeps one cache per
   size class). window: verify windows between two swap rounds; swap: the most swaps per round; decay: the aging of
   the counts per round (0..1). Env overrides: KINDI_MA_WINDOW, KINDI_MA_SWAP, KINDI_MA_DECAY. */
ma_cache* ma_cache_new(int nl, int ne, int nslot, int window, int swap, float decay);
void ma_cache_free(ma_cache* c);
/* seed: the usage order (layer, expert pairs, most used first); fills the slots in order, skipping excluded keys.
   Returns the slots filled; the engine loads them (they are admitted at once: the load is before serving). */
int ma_cache_seed(ma_cache* c, const int* layer, const int* expert, int n);
/* keys the cache must never take (an expert held elsewhere, e.g. on another card) */
void ma_cache_exclude(ma_cache* c, int layer, int expert);
/* a whole layer that the cache never takes (all its experts resident elsewhere, e.g. a drafter's own layer) */
void ma_cache_exclude_layer(ma_cache* c, int layer);
/* slots [0, n) have card memory (a partly mapped elastic tail): the policy targets only those (0: all) */
void ma_cache_set_live(ma_cache* c, int n);
/* the slot of an admitted expert, or -1 */
int ma_cache_slot(const ma_cache* c, int layer, int expert);
/* the expert in a slot (layer * ne + expert), -1 if empty or not admitted */
int ma_cache_owner(const ma_cache* c, int slot);
/* the picks of one call of layer l (n ids); counts the uses and the hits. Note the verify passes only (not the
   drafter's steps): the cache follows what the main model asks for. */
void ma_cache_note(ma_cache* c, int layer, const int* ids, int n);
/* a swap that the engine could not start: the slot is empty again (the key can be chosen again later) */
void ma_cache_cancel(ma_cache* c, int slot);
/* the end of a verify window: on every window-th call, fills out[] with up to max swaps to start and returns how
   many. The slots in out[] are empty from now until ma_cache_admit(). */
int ma_cache_tick(ma_cache* c, ma_swap* out, int max);
/* the copy into this slot has landed: its expert is used from now on */
void ma_cache_admit(ma_cache* c, int slot);
/* counters: [0] picks noted, [1] picks on admitted slots, [2] swaps started, [3] swaps admitted, [4] windows */
void ma_cache_stats(const ma_cache* c, long long* out);

typedef struct ma_off ma_off;
/* share0: the starting share of the misses to copy (0..1); maxshare caps it. Env: KINDI_MA_OFF_SHARE */
ma_off* ma_off_new(float share0, float maxshare);
void ma_off_free(ma_off* o);
/* how many of n missed experts to copy this layer (the engine copies the first k of its own order) */
int ma_off_count(ma_off* o, int n);
/* measured: the CPU took cpu_ms for n_cpu experts and the copies + card took copy_ms for n_copy experts (in
   parallel); the share moves toward the balance point (exponential average) */
void ma_off_feed(ma_off* o, double cpu_ms, int n_cpu, double copy_ms, int n_copy);
float ma_off_share(const ma_off* o);

/* ---------------------------------------------------------------- the mailbox (card <-> CPU pool in one graph)
   A pass of a decode step is one CUDA graph. For each MoE layer the card writes the picks and the expert inputs into
   the layer's mailbox in page-locked, card-mapped host memory and sets "posted" to the pass number (k_ma_post in
   moeaccel.cuh); the host, which serves the pass layer by layer (ma_box_serve_pass), sees it, computes the experts
   that are not on a card into the mailbox's rows and sets "done"; the card (k_ma_wait) waits for "done" and copies
   the rows it needs. No driver call on the CPU path. The engine allocates the memory (cuMemHostAlloc with
   DEVICEMAP, in its own driver-API code) and gives this layer the host address; this file stays free of CUDA.

   Layout of one layer's mailbox (bytes): posted u32 at 0, done u32 at 64, err u32 at 128, T i32 at 192, the ids
   [tmax][nsel] i32 at 256, then the inputs [tmax][kin_bytes], then the rows: [tmax][nsel][row_bytes] (per_pair) or
   [tmax][row_bytes]; each part 64-byte aligned; the layers follow each other at the stride. */
/* off[0] ids, off[1] inputs, off[2] rows, off[3] stride (bytes) */
void ma_box_layout(int tmax, int nsel, long kin_bytes, long row_bytes, int per_pair, long* off);
/* the engine's CPU step for layer l: T tokens, ids [T][nsel], inputs, rows to fill; returns < 0 on an error */
typedef int (*ma_fn)(void* ctx, int layer, int T, const int* ids, const void* in, void* rows);
/* serve one pass: for layer 0 .. nl - 1 in order, wait for "posted" == seq, call fn, then set "done" = seq.
   Returns 0, or -(1 + layer) on a timeout (seconds) or an error of fn (the layer's err word is set). */
int ma_box_serve_pass(void* host, int nl, int tmax, int nsel, long kin_bytes, long row_bytes, int per_pair,
                      unsigned seq, ma_fn fn, void* ctx, double timeout_s);

#ifdef __cplusplus
}
#endif
#endif
