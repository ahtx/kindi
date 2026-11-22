/* xstore.c - routed experts streamed from the model file (Copyright (c) 2026 Amir Husain. SPDX-License-Identifier:
   AGPL-3.0-only). Kindi's expert store for MoE models whose experts do not fit in RAM (the DeepSeek-V4.1-Flash
   "nvme" profile; the code knows nothing of the model and serves any engine whose experts are GGUF tensors).

   What it does:
   - Reads ONLY the experts a call routes to, straight from the model file with direct I/O (O_DIRECT, aligned
     buffers, past the page cache), by a pool of pread threads with many reads in flight (each expert matrix is
     split into chunks of KINDI_XS_CHUNK bytes, so one expert keeps several requests on the drive at once).
   - Keeps a RAM cache of experts with a byte budget (KINDI_XS_GB). The slots hold the expert matrices byte for byte
     as the file has them (the 17-byte MXFP4 blocks), so the kernels of moe_cpu.c read them exactly as they read
     the mapped file: the results are the same bits.
   - Pins the most used experts (the engine passes the release's usage order and skips the experts that its cards
     hold), and evicts the others by an aged use count; experts used in the last few calls are kept first.
   - Prefetch: the engine names the experts it predicts for the next layers; they are read at a lower priority than
     the experts a call is waiting for, and a prefetch evicts only an expert used less than the predicted one
     (its count + KINDI_XS_PF_BIAS, default 1).
   - An expert that a call needs is promoted to the front of the queue if a prefetch already asked for it.
   If a slot cannot be found (every slot is in use by the call itself) or a read fails, the call reads that expert
   from the mapped file instead, so a call never fails; the counters report it.

   Not io_uring: liburing is not on every host, and a pool of blocking pread threads keeps as many reads in flight
   as it has threads (KINDI_XS_THREADS, default 32), which is what an NVMe drive needs.

   Threads: the engine's calls (acquire, release, prefetch) come from one thread at a time per store, or from a
   predictor thread (prefetch only); everything is under one mutex. The I/O threads never hold it during a read. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define XS_ALIGN 4096
#define XS_MAXF 64
#define XS_MAXL 256
#define XS_MAXT 256

typedef struct { const uint8_t* base; size_t size; int fd, direct, drive; char path[1024]; } XsFile;
#define XS_MAXD 4                 /* drives, each with its own read queues and threads */
typedef struct {
    const uint8_t* mm[3];         /* the matrices in the mapped file (gate, up, down) */
    int file[3]; long off[3];     /* their place in the file */
    size_t len[3];
    int slot;                     /* -1: not in RAM */
    uint8_t state;                /* 0 absent, 1 loading, 2 ready, 3 failed */
    uint8_t pinned, oncard, pf;   /* pinned: never evicted; oncard: a card holds it; pf: loaded by a prefetch, unused */
    uint8_t demand;               /* a call waits for it */
    uint8_t fb;                   /* a call read it through the mapping: its pages leave when the last call is done */
    int refs, pend;               /* calls using it now; chunk reads outstanding */
    float uses;                   /* aged use count */
    unsigned miss, miss_dec;      /* demand reads (all calls / calls of at most 8 tokens: decode), for KINDI_XS_MISS_LOG */
    unsigned last;                /* clock of the last use or prefetch */
} XsExp;
typedef struct { int key, mat; long off; size_t len; uint8_t* dst; } XsReq;
typedef struct { XsReq* q; int head, tail, cap; } XsRing;

static struct {
    int on;
    XsFile f[XS_MAXF]; int nf;
    int lbase[XS_MAXL], lne[XS_MAXL], nl;  /* first key and experts of each layer */
    XsExp* x; int nx, capx;
    size_t slot_bytes, mat_cap[3];
    uint8_t* arena; size_t arena_bytes;
    int* owner; int nslot, *freel, nfree, npinned, pin_max;
    XsRing dq[XS_MAXD], pq[XS_MAXD];   /* per drive: demand and prefetch reads */
    int ndrive;
    long long dbytes[XS_MAXD], dchunks[XS_MAXD];
    pthread_mutex_t mu;
    pthread_cond_t work, done;
    pthread_t th[XS_MAXD * XS_MAXT]; int nth;
    size_t chunk;
    unsigned clock, protect, age_every, age_n;
    int check, quit;
    float pf_bias;                /* a prefetch may evict an expert used less than its own count + pf_bias */
    /* counters */
    long long acq, hits, pf_hits, misses, waits_pf, fallbacks, pf_issued, pf_wasted, evictions, reads, rfail;
    long long bytes, chunks;
    double wait_s, read_s;
    /* instrumentation (F5/F9 desktop): reads by size, the queue depth over time, the drive's rate */
    long long hist[8];            /* chunk reads by size: <64K <256K <1M <4M <16M <64M <256M >=256M */
    int inflight, qd_max;         /* reads in the drive now, the most seen */
    double qd_area, qd_t0, qd_last;   /* integral of the reads in flight over time; its start and last update */
    long long busy_bytes; double busy_s;   /* bytes read while at least one read was in flight, and that time */
} XS = { .mu = PTHREAD_MUTEX_INITIALIZER, .work = PTHREAD_COND_INITIALIZER, .done = PTHREAD_COND_INITIALIZER };

static double xs_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static size_t up4k(size_t n) { return (n + XS_ALIGN - 1) & ~(size_t)(XS_ALIGN - 1); }

/* ---------------------------------------------------------------- registration (before xs_start) */
/* a mapped model file: base .. base + size is the mapping of path */
int xs_add_file(const void* base, long size, const char* path) {
    if (XS.nf == XS_MAXF) return -1;
    XsFile* f = &XS.f[XS.nf];
    f->base = base; f->size = (size_t)size; f->fd = -1;
    snprintf(f->path, sizeof f->path, "%s", path);
    return XS.nf++;
}
/* the registered file mapping that holds p (base and size), for page-locking by the expert cards: 0 = found */
int xs_file_range(const void* p, const void** base, long* size) {
    for (int i = 0; i < XS.nf; i++)
        if ((const uint8_t*)p >= XS.f[i].base && (const uint8_t*)p < XS.f[i].base + XS.f[i].size) {
            *base = XS.f[i].base; *size = (long)XS.f[i].size; return 0;
        }
    return -1;
}
static int xs_locate(const uint8_t* p, size_t len, int* file, long* off) {
    for (int i = 0; i < XS.nf; i++)
        if (XS.f[i].base && p >= XS.f[i].base && p + len <= XS.f[i].base + XS.f[i].size) { *file = i; *off = (long)(p - XS.f[i].base); return 0; }
    return -1;
}
/* a layer of E experts; gate/up/down point at expert 0 in the mapping, experts are gb (gate, up) and db (down) bytes
   apart. Returns the store's layer id. */
int xs_add_layer(int E, const void* gate, const void* up, const void* down, long gb, long db) {
    if (XS.nl == XS_MAXL) return -1;
    if (XS.nx + E > XS.capx) {
        XS.capx = (XS.nx + E) * 2;
        XS.x = realloc(XS.x, sizeof(XsExp) * XS.capx);
    }
    const uint8_t* b[3] = {gate, up, down};
    const size_t sz[3] = {(size_t)gb, (size_t)gb, (size_t)db};
    for (int e = 0; e < E; e++) {
        XsExp* X = &XS.x[XS.nx + e];
        memset(X, 0, sizeof *X);
        X->slot = -1;
        for (int m = 0; m < 3; m++) {
            X->mm[m] = b[m] + (size_t)e * sz[m];
            X->len[m] = sz[m];
            if (xs_locate(X->mm[m], sz[m], &X->file[m], &X->off[m])) {
                fprintf(stderr, "xstore: an expert matrix is outside every registered file\n");
                return -2;
            }
            size_t cap = up4k(sz[m]) + XS_ALIGN;
            if (cap > XS.mat_cap[m]) XS.mat_cap[m] = cap;
        }
    }
    XS.lbase[XS.nl] = XS.nx; XS.lne[XS.nl] = E;
    XS.nx += E;
    return XS.nl++;
}
static inline XsExp* xs_exp(int l, int e) { return &XS.x[XS.lbase[l] + e]; }
/* a second copy of some experts on another drive (F5, two NVMe drives): a file without a mapping, read on drive
   `drive` by its own threads. Returns the file id. */
int xs_add_mirror_file(const char* path, int drive) {
    if (XS.nf == XS_MAXF || drive < 0 || drive >= XS_MAXD) return -1;
    XsFile* f = &XS.f[XS.nf];
    memset(f, 0, sizeof *f);
    f->fd = -1; f->drive = drive;
    snprintf(f->path, sizeof f->path, "%s", path);
    return XS.nf++;
}
/* expert (l, e) is read from `file` at these offsets (gate, up, down: the same bytes as in the model file, so the
   results do not change; KINDI_XS_CHECK=1 compares every read with the mapping) */
int xs_set_mirror(int l, int e, int file, long og, long ou, long od) {
    if (l < 0 || l >= XS.nl || e < 0 || e >= XS.lne[l] || file < 0 || file >= XS.nf) return -1;
    XsExp* X = &XS.x[XS.lbase[l] + e];
    const long o[3] = {og, ou, od};
    for (int m = 0; m < 3; m++) { X->file[m] = file; X->off[m] = o[m]; }
    return 0;
}
/* bytes read so far on each drive (out[0..XS_MAXD-1]) */
void xs_drive_bytes(long long* out) {
    pthread_mutex_lock(&XS.mu);
    for (int d = 0; d < XS_MAXD; d++) out[d] = XS.dbytes[d];
    pthread_mutex_unlock(&XS.mu);
}
/* a card holds this expert: never pinned in RAM (calls do not ask for it) */
void xs_mark_oncard(int l, int e) { if (l >= 0 && l < XS.nl && e >= 0 && e < XS.lne[l]) xs_exp(l, e)->oncard = 1; }
/* the starting use count of an expert (the release's measured counts, any scale) */
void xs_seed(int l, int e, float uses) { if (l >= 0 && l < XS.nl && e >= 0 && e < XS.lne[l]) xs_exp(l, e)->uses = uses; }

/* ---------------------------------------------------------------- queues */
static void ring_push(XsRing* r, XsReq q) {
    if (r->tail == r->cap) {
        if (r->head > r->cap / 2) {               /* compact */
            memmove(r->q, r->q + r->head, sizeof(XsReq) * (r->tail - r->head));
            r->tail -= r->head; r->head = 0;
        } else {
            r->cap = r->cap ? 2 * r->cap : 4096;
            r->q = realloc(r->q, sizeof(XsReq) * r->cap);
        }
    }
    r->q[r->tail++] = q;
}
static int ring_pop(XsRing* r, XsReq* out) {
    while (r->head < r->tail) {
        XsReq q = r->q[r->head++];
        if (q.key >= 0) { *out = q; return 1; }
    }
    r->head = r->tail = 0;
    return 0;
}
/* a demanded expert's queued prefetch chunks move to the demand queue */
static void ring_promote(int key) {
    for (int d = 0; d < XS.ndrive; d++)
        for (int i = XS.pq[d].head; i < XS.pq[d].tail; i++)
            if (XS.pq[d].q[i].key == key) { ring_push(&XS.dq[d], XS.pq[d].q[i]); XS.pq[d].q[i].key = -1; }
}

static int xs_bucket(size_t n) {
    int b = 0;
    for (size_t lim = 64 << 10; b < 7 && n >= lim; lim <<= 2) b++;
    return b > 7 ? 7 : b;
}
static void xs_qd(int delta) {           /* under XS.mu: the reads in flight change by delta */
    const double t = xs_now();
    if (XS.qd_t0 == 0) XS.qd_t0 = XS.qd_last = t;
    XS.qd_area += XS.inflight * (t - XS.qd_last);
    if (XS.inflight > 0) XS.busy_s += t - XS.qd_last;
    XS.qd_last = t;
    XS.inflight += delta;
    if (XS.inflight > XS.qd_max) XS.qd_max = XS.inflight;
}
static void* xs_worker(void* a) {
    const int dv = (int)(intptr_t)a;          /* this thread's drive */
    pthread_mutex_lock(&XS.mu);
    for (;;) {
        XsReq q;
        while (!XS.quit && !ring_pop(&XS.dq[dv], &q) && !ring_pop(&XS.pq[dv], &q)) pthread_cond_wait(&XS.work, &XS.mu);
        if (XS.quit) break;
        xs_qd(+1);
        pthread_mutex_unlock(&XS.mu);
        XsExp* X = &XS.x[q.key];
        const XsFile* F = &XS.f[X->file[q.mat]];
        double t0 = xs_now();
        size_t got = 0; int bad = 0;
        while (got < q.len) {
            ssize_t r = pread(F->fd, q.dst + got, q.len - got, q.off + (long)got);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) { bad = r < 0 || got == 0; break; }   /* r == 0: the end of the file (the aligned tail) */
            got += (size_t)r;
        }
        double dt = xs_now() - t0;
        pthread_mutex_lock(&XS.mu);
        xs_qd(-1);
        XS.hist[xs_bucket(q.len)]++; XS.busy_bytes += (long long)got;
        XS.dbytes[dv] += (long long)got; XS.dchunks[dv]++;
        XS.read_s += dt; XS.chunks++; XS.bytes += (long long)got;
        if (bad) { X->state = 3; XS.rfail++; }
        if (--X->pend == 0) {
            if (X->state == 1) {
                X->state = 2; XS.reads++;
                if (XS.check) {
                    const uint8_t* s = XS.arena + (size_t)X->slot * XS.slot_bytes;
                    for (int m = 0; m < 3; m++) {
                        const uint8_t* p = s + (m ? XS.mat_cap[0] : 0) + (m == 2 ? XS.mat_cap[1] : 0) + (X->off[m] & (XS_ALIGN - 1));
                        if (memcmp(p, X->mm[m], X->len[m])) {
                            fprintf(stderr, "xstore: CHECK FAILED: expert %d matrix %d differs from the mapped file\n", q.key, m);
                            abort();
                        }
                    }
                }
            }
            pthread_cond_broadcast(&XS.done);
        }
    }
    pthread_mutex_unlock(&XS.mu);
    return NULL;
}

/* ---------------------------------------------------------------- slots */
static const uint8_t* mat_ptr(const XsExp* X, int m) {
    const uint8_t* s = XS.arena + (size_t)X->slot * XS.slot_bytes;
    size_t o = m == 0 ? 0 : m == 1 ? XS.mat_cap[0] : XS.mat_cap[0] + XS.mat_cap[1];
    return s + o + (X->off[m] & (XS_ALIGN - 1));
}
/* a free slot, or the slot of the expert with the lowest score that nobody uses (recent ones last). For a prefetch
   (limit >= 0), only a victim used less than `limit`. -1: none. */
static int take_slot(float limit) {
    if (XS.nfree) return XS.freel[--XS.nfree];
    int best = -1, best_recent = -1; float bs = 0, bsr = 0;
    for (int s = 0; s < XS.nslot; s++) {
        int k = XS.owner[s];
        if (k < 0) continue;
        XsExp* X = &XS.x[k];
        if (X->pinned || X->refs || X->state == 1 || X->demand) continue;
        float sc = X->uses;
        if (XS.clock - X->last < XS.protect) {
            if (best_recent < 0 || sc < bsr) { best_recent = s; bsr = sc; }
        } else if (best < 0 || sc < bs) { best = s; bs = sc; }
    }
    if (best < 0 && limit < 0) { best = best_recent; bs = bsr; }
    if (best < 0 || (limit >= 0 && bs >= limit)) return -1;
    XsExp* V = &XS.x[XS.owner[best]];
    if (V->pf) XS.pf_wasted++;
    V->slot = -1; V->state = 0; V->pf = 0;
    XS.owner[best] = -1;
    XS.evictions++;
    return best;
}
static void issue(int key, int slot, int prefetch) {
    XsExp* X = &XS.x[key];
    X->slot = slot; X->state = 1; X->pend = 0;
    XS.owner[slot] = key;
    uint8_t* s = XS.arena + (size_t)slot * XS.slot_bytes;
    for (int m = 0; m < 3; m++) {
        uint8_t* dst = s + (m == 0 ? 0 : m == 1 ? XS.mat_cap[0] : XS.mat_cap[0] + XS.mat_cap[1]);
        long a = X->off[m] & ~(long)(XS_ALIGN - 1);
        size_t n = up4k((size_t)(X->off[m] - a) + X->len[m]);
        for (size_t o = 0; o < n; o += XS.chunk) {
            XsReq q = {key, m, a + (long)o, n - o < XS.chunk ? n - o : XS.chunk, dst + o};
            const int dv = XS.f[X->file[m]].drive;
            ring_push(prefetch ? &XS.pq[dv] : &XS.dq[dv], q);
            X->pend++;
        }
    }
    pthread_cond_broadcast(&XS.work);
}
static void age(void) {
    if (!XS.age_every || ++XS.age_n < XS.age_every) return;
    XS.age_n = 0;
    for (int i = 0; i < XS.nx; i++) XS.x[i].uses *= 0.5f;
}

/* the cores of the store's threads (and the engine's predictor): KINDI_XS_CPUS ("a-b,c"), default every online CPU
   (not the core of the thread that starts the store, whose affinity they would inherit) */
void xs_thread_cpus(cpu_set_t* cs) {
    CPU_ZERO(cs);
    const char* s = getenv("KINDI_XS_CPUS");
    if (s && s[0]) {
        for (const char* p = s; *p;) {
            char* e; long a = strtol(p, &e, 10), b = a;
            if (e == p) break;
            if (*e == '-') b = strtol(e + 1, &e, 10);
            for (long c = a; c <= b && c < CPU_SETSIZE; c++) if (c >= 0) CPU_SET((int)c, cs);
            p = *e == ',' ? e + 1 : e;
            if (*e != ',') break;
        }
        if (CPU_COUNT(cs)) return;
    }
    for (int c = 0, n = (int)sysconf(_SC_NPROCESSORS_ONLN); c < n && c < CPU_SETSIZE; c++) CPU_SET(c, cs);
}

static void* xs_logger(void* a);
/* ---------------------------------------------------------------- start */
/* budget: bytes of RAM for the cache; threads: I/O threads; chunk: bytes per read */
int xs_start(double budget, int threads, long chunk) {
    if (!XS.nx) return -1;
    XS.slot_bytes = XS.mat_cap[0] + XS.mat_cap[1] + XS.mat_cap[2];
    XS.nslot = (int)(budget / XS.slot_bytes);
    if (XS.nslot > XS.nx) XS.nslot = XS.nx;
    if (XS.nslot < 1) { fprintf(stderr, "xstore: the budget (%.2f GB) is less than one expert\n", budget / 1e9); return -2; }
    XS.arena_bytes = (size_t)XS.nslot * XS.slot_bytes;
    XS.arena = mmap(NULL, XS.arena_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (XS.arena == MAP_FAILED) { XS.arena = NULL; fprintf(stderr, "xstore: cannot reserve %.2f GB\n", XS.arena_bytes / 1e9); return -3; }
    madvise(XS.arena, XS.arena_bytes, MADV_HUGEPAGE);
    XS.owner = malloc(sizeof(int) * XS.nslot); XS.freel = malloc(sizeof(int) * XS.nslot);
    for (int s = 0; s < XS.nslot; s++) { XS.owner[s] = -1; XS.freel[s] = XS.nslot - 1 - s; }
    XS.nfree = XS.nslot;
    int ndirect = 0;
    for (int i = 0; i < XS.nf; i++) {
        XsFile* F = &XS.f[i];
        int nod = getenv("KINDI_XS_DIRECT") && !atoi(getenv("KINDI_XS_DIRECT"));
        F->fd = nod ? -1 : open(F->path, O_RDONLY | O_DIRECT);
        F->direct = F->fd >= 0;
        if (F->fd < 0) F->fd = open(F->path, O_RDONLY);   /* a file system without O_DIRECT: plain reads */
        if (F->fd < 0) { fprintf(stderr, "xstore: cannot open %s\n", F->path); return -4; }
        ndirect += F->direct;
    }
    XS.chunk = chunk > 0 ? up4k((size_t)chunk) : (size_t)2 << 20;
    XS.pin_max = getenv("KINDI_XS_PIN") ? (int)(atof(getenv("KINDI_XS_PIN")) * XS.nslot) : XS.nslot / 2;
    XS.protect = getenv("KINDI_XS_PROTECT") ? (unsigned)atoi(getenv("KINDI_XS_PROTECT")) : 3 * (unsigned)XS.nl;
    XS.age_every = getenv("KINDI_XS_AGE") ? (unsigned)atoi(getenv("KINDI_XS_AGE")) : 64 * (unsigned)XS.nl;
    XS.pf_bias = getenv("KINDI_XS_PF_BIAS") ? (float)atof(getenv("KINDI_XS_PF_BIAS")) : 1.f;
    XS.check = getenv("KINDI_XS_CHECK") && atoi(getenv("KINDI_XS_CHECK"));
    XS.nth = threads < 1 ? 1 : threads > XS_MAXT ? XS_MAXT : threads;
    XS.ndrive = 1;
    for (int i = 0; i < XS.nf; i++) if (XS.f[i].drive + 1 > XS.ndrive) XS.ndrive = XS.f[i].drive + 1;
    pthread_attr_t at; pthread_attr_init(&at);
    cpu_set_t cs; xs_thread_cpus(&cs);
    pthread_attr_setaffinity_np(&at, sizeof cs, &cs);
    for (int d = 0; d < XS.ndrive; d++)            /* each drive its own threads (KINDI_XS_THREADS each) */
        for (int t = 0; t < XS.nth; t++) pthread_create(&XS.th[d * XS_MAXT + t], &at, xs_worker, (void*)(intptr_t)d);
    pthread_attr_destroy(&at);
    XS.on = 1;
    if (getenv("KINDI_XS_LOG_S") && atof(getenv("KINDI_XS_LOG_S")) > 0) {
        static double every; static pthread_t lt;
        every = atof(getenv("KINDI_XS_LOG_S"));
        pthread_create(&lt, NULL, xs_logger, &every);
    }
    fprintf(stderr, "xstore: %d of %d experts fit the cache (%.2f GB, %.1f MB per slot), %d I/O threads, %zu KB reads, "
            "%s I/O on %d of %d files%s\n", XS.nslot, XS.nx, XS.arena_bytes / 1e9, XS.slot_bytes / 1048576.0, XS.nth,
            XS.chunk >> 10, ndirect ? "direct" : "buffered", ndirect, XS.nf, XS.check ? ", every read checked" : "");
    return XS.nslot;
}
int xs_on(void) { return XS.on; }

/* pin (and start loading) an expert, as long as the pin share of the slots allows; 1 = pinned */
int xs_pin(int l, int e) {
    if (!XS.on || l < 0 || l >= XS.nl || e < 0 || e >= XS.lne[l]) return 0;
    pthread_mutex_lock(&XS.mu);
    XsExp* X = xs_exp(l, e);
    int ok = 0;
    if (!X->oncard && !X->pinned && XS.npinned < XS.pin_max) {
        if (X->state == 0) {
            int s = take_slot(-1);
            if (s >= 0) { issue(XS.lbase[l] + e, s, 1); ok = 1; }
        } else ok = 1;
        if (ok) { X->pinned = 1; XS.npinned++; }
    }
    pthread_mutex_unlock(&XS.mu);
    return ok;
}

/* ---------------------------------------------------------------- calls */
/* the experts of one call: wait until all are in RAM and give their matrices (ptr[i * 3 + m]); they stay until
   xs_release. Returns the experts read for this call (misses). */
int xs_acquire(int l, int n, const int* ids, const uint8_t** ptr) {
    pthread_mutex_lock(&XS.mu);
    XS.clock++; age();
    int miss = 0;
    double t0 = 0;
    for (int i = 0; i < n; i++) {
        const int key = XS.lbase[l] + ids[i];
        XsExp* X = &XS.x[key];
        X->refs++; X->uses += 1.f; X->last = XS.clock; X->demand = 1;
        XS.acq++;
        if (X->state == 2) { XS.hits++; if (X->pf) XS.pf_hits++; }
        else if (X->state == 1) { XS.waits_pf++; ring_promote(key); }
        else if (X->state == 0) {
            const int s = take_slot(-1);
            if (s >= 0) {
                issue(key, s, 0); miss++; XS.misses++;
                X->miss++; if (n <= 48) X->miss_dec++;   /* at most 48 experts: a decode or verify call (8 x 6) */
            }
        }
        X->pf = 0;
    }
    for (int i = 0; i < n; i++) {
        XsExp* X = &XS.x[XS.lbase[l] + ids[i]];
        if (X->state == 1 && !t0) t0 = xs_now();
        while (X->state == 1) pthread_cond_wait(&XS.done, &XS.mu);
        if (X->state == 2) for (int m = 0; m < 3; m++) ptr[i * 3 + m] = mat_ptr(X, m);
        else {                                    /* no slot or a failed read: the mapped file */
            for (int m = 0; m < 3; m++) ptr[i * 3 + m] = X->mm[m];
            XS.fallbacks++; X->fb = 1;
            if (X->state == 3) {                  /* give the slot back */
                if (X->slot >= 0) { XS.owner[X->slot] = -1; XS.freel[XS.nfree++] = X->slot; X->slot = -1; }
                X->state = 0;
            }
        }
        X->demand = 0;
    }
    if (t0) XS.wait_s += xs_now() - t0;
    pthread_mutex_unlock(&XS.mu);
    return miss;
}
/* an expert read through the mapping (no free slot) leaves this process and the page cache after its last call, so
   prompt chunks larger than the store do not grow the page cache (KINDI_XS_FB_KEEP=1: keep the pages) */
static void fb_drop(XsExp* X) {
    static int keep = -1;
    if (keep < 0) keep = getenv("KINDI_XS_FB_KEEP") && atoi(getenv("KINDI_XS_FB_KEEP"));
    X->fb = 0;
    if (keep) return;
    for (int m = 0; m < 3; m++) {
        const uintptr_t a = (uintptr_t)X->mm[m], a0 = a & ~(uintptr_t)4095, a1 = (a + X->len[m] + 4095) & ~(uintptr_t)4095;
        madvise((void*)a0, a1 - a0, MADV_DONTNEED);
        const XsFile* f = &XS.f[X->file[m]];
        if (f->fd >= 0) posix_fadvise(f->fd, X->off[m] & ~4095L, (off_t)(a1 - a0), POSIX_FADV_DONTNEED);
    }
}
void xs_release(int l, int n, const int* ids) {
    pthread_mutex_lock(&XS.mu);
    for (int i = 0; i < n; i++) {
        XsExp* X = &XS.x[XS.lbase[l] + ids[i]];
        if (--X->refs == 0 && X->fb) fb_drop(X);
    }
    pthread_mutex_unlock(&XS.mu);
}
/* read these experts ahead, most likely first; returns the reads started */
int xs_prefetch(int l, int n, const int* ids) {
    if (!XS.on || l < 0 || l >= XS.nl) return 0;
    pthread_mutex_lock(&XS.mu);
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= XS.lne[l]) continue;
        const int key = XS.lbase[l] + ids[i];
        XsExp* X = &XS.x[key];
        if (X->oncard) continue;
        X->last = XS.clock;                       /* recent: kept until it is used */
        if (X->state != 0) continue;
        const int s = take_slot(X->uses + XS.pf_bias);
        if (s < 0) continue;
        issue(key, s, 1); X->pf = 1; k++; XS.pf_issued++;
    }
    pthread_mutex_unlock(&XS.mu);
    return k;
}
/* the prefetch's outcome so far: [0] prefetches started, [1] used (a hit on a prefetched expert, or a call that
   waited on one: both were read ahead of the call) */
void xs_pf_counts(long long* out) {
    pthread_mutex_lock(&XS.mu);
    out[0] = XS.pf_issued; out[1] = XS.pf_hits + XS.waits_pf;
    pthread_mutex_unlock(&XS.mu);
}
/* wait until no read is queued or running (tests, warm start) */
void xs_drain(void) {
    pthread_mutex_lock(&XS.mu);
    for (;;) {
        int busy = 0;
        for (int i = 0; !busy && i < XS.nx; i++) busy = XS.x[i].state == 1;
        if (!busy) break;
        pthread_cond_wait(&XS.done, &XS.mu);
    }
    pthread_mutex_unlock(&XS.mu);
}
/* is this expert in RAM now (tests) */
/* start the demand reads of a call's experts without waiting (KINDI_XS_EARLY): every expert is held (demand) until a
   later xs_acquire of it, the absent ones are read now, and ready[i] says which are in RAM already, so the caller can
   compute those while the others load. Returns how many are ready. */
int xs_demand(int l, int n, const int* ids, uint8_t* ready) {
    if (!XS.on || l < 0 || l >= XS.nl) return -1;
    pthread_mutex_lock(&XS.mu);
    int k = 0;
    for (int i = 0; i < n; i++) {
        const int key = XS.lbase[l] + ids[i];
        XsExp* X = &XS.x[key];
        X->demand = 1;
        if (X->state == 0) {
            const int s = take_slot(-1);
            if (s >= 0) {
                issue(key, s, 0); XS.misses++;
                X->miss++; if (n <= 48) X->miss_dec++;
            }
        }
        ready[i] = X->state == 2;
        k += ready[i];
    }
    pthread_mutex_unlock(&XS.mu);
    return k;
}
int xs_resident(int l, int e) { return XS.on && l >= 0 && l < XS.nl && e >= 0 && e < XS.lne[l] && xs_exp(l, e)->state == 2; }

/* out: [0] acquires, [1] hits, [2] hits that a prefetch loaded, [3] misses (read on demand), [4] waits on a running
   prefetch, [5] mapped-file fallbacks, [6] prefetches started, [7] prefetches evicted unused, [8] evictions,
   [9] experts read, [10] read failures, [11] bytes read, [12] chunk reads, [13] slots, [14] pinned, [15] resident */
void xs_stats(long long* out, double* sec) {
    pthread_mutex_lock(&XS.mu);
    long long res = 0;
    for (int s = 0; s < XS.nslot; s++) res += XS.owner[s] >= 0 && XS.x[XS.owner[s]].state == 2;
    long long v[16] = {XS.acq, XS.hits, XS.pf_hits, XS.misses, XS.waits_pf, XS.fallbacks, XS.pf_issued, XS.pf_wasted,
                       XS.evictions, XS.reads, XS.rfail, XS.bytes, XS.chunks, XS.nslot, XS.npinned, res};
    memcpy(out, v, sizeof v);
    if (sec) { sec[0] = XS.wait_s; sec[1] = XS.read_s; }
    pthread_mutex_unlock(&XS.mu);
}
/* out[0..7] read-size histogram, [8] reads in flight now, [9] most, [10] mean in flight while the drive was busy,
   [11] the drive's rate while busy (GB/s), [12] IOPS while busy, [13] seconds busy, [14] bytes read (GB) */
void xs_stats2(double* out) {
    pthread_mutex_lock(&XS.mu);
    xs_qd(0);
    for (int i = 0; i < 8; i++) out[i] = (double)XS.hist[i];
    out[8] = XS.inflight; out[9] = XS.qd_max;
    out[10] = XS.busy_s > 0 ? XS.qd_area / XS.busy_s : 0;
    out[11] = XS.busy_s > 0 ? XS.busy_bytes / XS.busy_s / 1e9 : 0;
    out[12] = XS.busy_s > 0 ? XS.chunks / XS.busy_s : 0;
    out[13] = XS.busy_s; out[14] = XS.bytes / 1e9;
    pthread_mutex_unlock(&XS.mu);
}
/* KINDI_XS_LOG_S=N: a line every N seconds with the interval's reads (GB/s, IOPS, mean queue depth while busy),
   the RAM-cache hits and the prefetch outcome */
static void* xs_logger(void* a) {
    const double every = *(double*)a;
    long long b0 = 0, c0 = 0, acq0 = 0, hit0 = 0, pfh0 = 0, pfw0 = 0, pfi0 = 0, d0[XS_MAXD] = {0};
    double busy0 = 0, area0 = 0, t0 = xs_now();
    for (;;) {
        struct timespec ts = {(time_t)every, (long)((every - (time_t)every) * 1e9)};
        nanosleep(&ts, NULL);
        pthread_mutex_lock(&XS.mu);
        xs_qd(0);
        const double t = xs_now(), busy = XS.busy_s - busy0, area = XS.qd_area - area0;
        fprintf(stderr, "xstore %.0fs: %.2f GB/s (%.0f MB, %lld reads, %.0f IOPS while busy %.0f%%), queue depth %.1f "
                "(max %d), RAM hits %.1f%% of %lld uses, prefetch %lld started, %lld used, %lld evicted unused\n",
                t - t0, busy > 0 ? (XS.busy_bytes - b0) / busy / 1e9 : 0.0, (XS.busy_bytes - b0) / 1e6, XS.chunks - c0,
                busy > 0 ? (XS.chunks - c0) / busy : 0.0, 100.0 * busy / every, busy > 0 ? area / busy : 0.0, XS.qd_max,
                XS.acq - acq0 ? 100.0 * (XS.hits - hit0) / (XS.acq - acq0) : 0.0, XS.acq - acq0, XS.pf_issued - pfi0,
                XS.pf_hits - pfh0, XS.pf_wasted - pfw0);
        if (XS.ndrive > 1) {
            long long all = 0;
            for (int d = 0; d < XS.ndrive; d++) all += XS.dbytes[d] - d0[d];
            fprintf(stderr, "xstore %.0fs: drive share", t - t0);
            for (int d = 0; d < XS.ndrive; d++)
                fprintf(stderr, " %d: %.0f MB (%.0f%%)", d, (XS.dbytes[d] - d0[d]) / 1e6, all ? 100.0 * (XS.dbytes[d] - d0[d]) / all : 0.0);
            fprintf(stderr, "\n");
            for (int d = 0; d < XS.ndrive; d++) d0[d] = XS.dbytes[d];
        }
        b0 = XS.busy_bytes; c0 = XS.chunks; acq0 = XS.acq; hit0 = XS.hits; pfh0 = XS.pf_hits; pfw0 = XS.pf_wasted;
        pfi0 = XS.pf_issued; busy0 = XS.busy_s; area0 = XS.qd_area;
        pthread_mutex_unlock(&XS.mu);
    }
    return NULL;
}
void xs_print_stats(const char* tag) {
    if (!XS.on) return;
    long long v[16]; double s[2];
    xs_stats(v, s);
    fprintf(stderr, "xstore %s: %lld expert uses, %.1f%% in RAM (%lld loaded by prefetch), %lld read on demand, %lld waited on a "
            "prefetch, %lld from the mapped file; %lld prefetches (%lld evicted unused); %.2f GB read in %.1f s of "
            "thread time, calls waited %.2f s; %lld of %lld slots used, %lld pinned\n", tag ? tag : "", v[0],
            100.0 * v[1] / (v[0] ? v[0] : 1), v[2], v[3], v[4], v[5], v[6], v[7], v[11] / 1e9, s[1], s[0], v[15], v[13], v[14]);
    if (XS.ndrive > 1) {
        long long all = 0;
        for (int d = 0; d < XS.ndrive; d++) all += XS.dbytes[d];
        fprintf(stderr, "xstore %s: drive share", tag ? tag : "");
        for (int d = 0; d < XS.ndrive; d++)
            fprintf(stderr, " %d: %.2f GB (%.1f%%)", d, XS.dbytes[d] / 1e9, all ? 100.0 * XS.dbytes[d] / all : 0.0);
        fprintf(stderr, "\n");
    }
}
/* the demand reads per expert so far, as "layer expert misses decode_misses" lines, most missed first (input of
   tools/dsv41_mirror.py: the experts this machine really reads); returns the experts written */
static int miss_cmp(const void* a, const void* b) {
    const unsigned x = XS.x[*(const int*)a].miss, y = XS.x[*(const int*)b].miss;
    return x < y ? 1 : x > y ? -1 : *(const int*)a - *(const int*)b;
}
int xs_write_misses(const char* path) {
    if (!XS.on || !path || !path[0]) return -1;
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    pthread_mutex_lock(&XS.mu);
    int* ord = malloc(sizeof(int) * XS.nx), n = 0;
    for (int k = 0; k < XS.nx; k++) if (XS.x[k].miss) ord[n++] = k;
    qsort(ord, (size_t)n, sizeof(int), miss_cmp);
    for (int i = 0; i < n; i++) {
        const int k = ord[i];
        int l = 0;
        while (l + 1 < XS.nl && XS.lbase[l + 1] <= k) l++;
        fprintf(f, "%d %d %u %u\n", l, k - XS.lbase[l], XS.x[k].miss, XS.x[k].miss_dec);
    }
    pthread_mutex_unlock(&XS.mu);
    free(ord);
    fclose(f);
    return n;
}
