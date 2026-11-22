/* Kindi Monitor: a tray app for the Kindi servers on this PC (and on the LAN), in the style of Amir's Fluid monitor.

   The UI thread only paints. One worker thread polls: /health of each Kindi server, NVML for each NVIDIA card, PDH
   and Win32 for the host. A second, short-lived thread runs a control (start, stop, restart, model) when the user
   asks for one. The tray popup is tiny and closes when it loses focus; the dashboard is a normal resizable window.
   docs/dev/F8-WINDOWS.md has the design: the telemetry list and its sources, discovery, controls, update rates.

   Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md. */

#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0A00
#define _CRT_SECURE_NO_WARNINGS
#include <winsock2.h>
#include <windows.h>
#include <shellapi.h>
#include <winhttp.h>
#include <pdh.h>
#include <ws2ipdef.h>
#include <iphlpapi.h>
#include <psapi.h>
#include <dwmapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <math.h>
#include "json.h"

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "iphlpapi.lib")
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "dwmapi.lib")

#define APP_VERSION L"0.7.0"

/* ------------------------------------------------------------------ GDI+ flat API (C has no gdiplus.h) */
typedef struct { UINT32 version; ULONG_PTR callback; BOOL suppress_bg, suppress_codecs; } GdiStartup;
typedef struct { float x, y; } GPointF;
__declspec(dllimport) int __stdcall GdiplusStartup(ULONG_PTR *, const GdiStartup *, void *);
__declspec(dllimport) void __stdcall GdiplusShutdown(ULONG_PTR);
__declspec(dllimport) int __stdcall GdipCreateFromHDC(HDC, void **);
__declspec(dllimport) int __stdcall GdipDeleteGraphics(void *);
__declspec(dllimport) int __stdcall GdipSetSmoothingMode(void *, int);
__declspec(dllimport) int __stdcall GdipSetPixelOffsetMode(void *, int);
__declspec(dllimport) int __stdcall GdipCreatePen1(unsigned int, float, int, void **);
__declspec(dllimport) int __stdcall GdipDeletePen(void *);
__declspec(dllimport) int __stdcall GdipDrawEllipse(void *, void *, float, float, float, float);
__declspec(dllimport) int __stdcall GdipDrawArc(void *, void *, float, float, float, float, float, float);
__declspec(dllimport) int __stdcall GdipDrawLines(void *, void *, const GPointF *, int);
__declspec(dllimport) int __stdcall GdipCreateSolidFill(unsigned int, void **);
__declspec(dllimport) int __stdcall GdipDeleteBrush(void *);
__declspec(dllimport) int __stdcall GdipFillPolygon(void *, void *, const GPointF *, int, int);
__declspec(dllimport) int __stdcall GdipFillEllipse(void *, void *, float, float, float, float);

/* ------------------------------------------------------------------ look (Fluid's palette) */
#define INK   RGB(8, 8, 8)
#define PANEL RGB(17, 17, 17)
#define LIME  RGB(202, 243, 28)
#define EDGE  RGB(35, 35, 35)
#define GREY  RGB(154, 154, 154)
#define DIM   RGB(96, 96, 96)
#define PAPER RGB(255, 255, 255)
#define CYAN  RGB(64, 200, 224)
#define RED   RGB(255, 84, 84)
#define AMBER RGB(255, 180, 40)
#define ARGB_OF(c, a) (((unsigned)(a) << 24) | ((unsigned)GetRValue(c) << 16) | ((unsigned)GetGValue(c) << 8) | GetBValue(c))

/* ------------------------------------------------------------------ limits and rates */
#define MAX_SRV 16
#define MAX_CARD 8
#define MAX_SLOT 8
#define HIST 300                 /* points in each graph: 2.5 min at 500 ms, 25 min at 5 s */
#define FAST_MS 500              /* a window is open */
#define SLOW_MS 5000             /* tray only */
#define DISCOVER_OPEN_MS 30000
#define DISCOVER_TRAY_MS 60000
#define WM_TRAY (WM_APP + 2)
#define WM_DATA (WM_APP + 1)
#define LOG_CAP 40

enum { K_UNKNOWN, K_NATIVE, K_WSL_DOCKER, K_WSL_UNIT, K_DOCKER, K_LAN };
static const wchar_t *KIND_NAME[] = { L"local", L"native", L"WSL docker", L"WSL service", L"Docker Desktop", L"LAN" };
enum { SRC_PORT, SRC_PLAN, SRC_INSTALL, SRC_MANUAL, SRC_ROUTE };
static const wchar_t *SRC_NAME[] = { L"port", L"plan", L"installer", L"added", L"front door" };

typedef struct { float v[HIST]; int n, head; } Ring;

typedef struct {
    char phase[24];
    double read, to_read, reused, generated, elapsed;
} SlotInfo;

typedef struct {
    /* identity */
    wchar_t host[64];
    int port, source, kind;
    wchar_t name[64];            /* container, unit or plan server name */
    wchar_t distro[64];          /* WSL distro */
    DWORD pid;                   /* native: the process that owns the port */
    ULONGLONG born;              /* its creation time (PID reuse check) */
    wchar_t image[MAX_PATH];
    int started_here;            /* started by this monitor */
    /* state */
    int online, kindi, front, models_401, offline_polls;
    long long next_poll;
    char err[96];
    char model[64], version[32], commit[16], head[16], phase[24], device[8], gpu_uuid[48], gpu_pci[24];
    char draft_method[24], drafter[48];
    int draft_on, dspark;
    double ctx, uptime, gpu_free, gpu_total;
    double prompt_tokens, reused, read, to_read, generated, elapsed;
    double requests, prompt_total, completion_total, cached_total;
    double q_chat, q_bg, yields, accept;
    int slots_n, slots_busy;
    SlotInfo slot[MAX_SLOT];
    double last_ptps, last_wtps, last_tokens;
    double write_tps, read_tps;   /* observed between polls */
    Ring wtps, rtps;
    double prev_gen, prev_read, prev_ptok;
    long long prev_t;
    double proc_cpu, proc_mb;     /* native: the server process */
    ULONGLONG prev_cpu_100ns;
    int card;                     /* index into the card list, or -1 */
    int gone;                     /* removed by the user (an added server) */
    int has_stats, has_uptime, has_head, has_device, has_gpu;
    char experts[96], calib[96];  /* the DeepSeek server's experts field, in words */
    char unit_state[24];          /* a WSL unit that does not answer: systemctl is-active (activating, inactive, failed) */
    wchar_t wsl_user[32];         /* the Linux user of a WSL systemd unit */
    char life[32];                /* "starting (warming model)", "stopping": what the monitor is doing to it */
    char fields[96][96];          /* every scalar of /health, flattened: "state.phase = reading prompt" */
    int nfields;
    char model_ids[16][48];       /* `kindi models --json`: ids for the selector */
    wchar_t model_label[16][96];  /* "id  5.9 GB, one card, downloaded" */
    int model_n;
} Server;

typedef struct {
    char name[64], uuid[48], pci[24];
    double used_mib, total_mib, util, mem_util, temp, power_w, limit_w, fan, sm_mhz, mem_mhz;
    int gen, width, max_gen, max_width;
    unsigned long long throttle;
    double rx_kbs, tx_kbs;
    Ring util_h, power_h, temp_h;
} Card;

typedef struct {
    wchar_t name[64];
    double cpu, ram_used_gb, ram_total_gb, ram_pct, commit_gb, commit_limit_gb, commit_pct, disk_pct, disk_free_gb,
           net_in_kbs, net_out_kbs;
    Ring cpu_h, ram_h;
} Host;

enum { CK_OK, CK_WARN, CK_FAIL, CK_INFO };
enum { WS_NONE, WS_CHECKING, WS_CHECKED, WS_INSTALLING, WS_STARTING, WS_ATTACHED };
typedef struct { int state; wchar_t name[48]; wchar_t why[240]; } WslCheck;
typedef struct {
    int stage, ncheck, wsl_off, need_install, gpu_ok, disk_ok, installed, can_install, can_start;
    WslCheck check[12];
    wchar_t model[48], distro[64], user[32], status[160], install_step[160], install_last[160];
    DWORD install_pid;
    HANDLE install_proc;
} WslFlow;

typedef struct {
    Server srv[MAX_SRV];
    int nsrv;
    Card card[MAX_CARD];
    int ncard;
    int nvml;
    Host host;
    wchar_t log[LOG_CAP][120];
    int nlog;
    int busy_control;            /* a control is running */
    double self_cpu, self_mb;    /* this monitor */
    WslFlow wsl;                 /* a model that runs in WSL: its checks, install and start */
} State;

static CRITICAL_SECTION g_lock;
#define DSV_PORT 8101         /* the DeepSeek server (tools/install_dsv41.sh) */
static void wsl_install_progress(void);
static void wsl_check(void);
static void open_dash(void);
static void draw_wsl_panel(HDC mem, int x, int y, int w, int H, const State *st);
static State g_state;            /* written by the worker under g_lock, copied by the UI */
static HINSTANCE g_inst;
static HWND g_main;              /* hidden owner of the tray icon */
static HWND g_popup, g_dash;
static HANDLE g_stop, g_wake;
static int g_dpi = 96, g_dash_dpi = 96;
static int g_poll_ms_fast = FAST_MS;
static wchar_t g_dir[MAX_PATH];  /* the monitor's log folder: %LOCALAPPDATA%\Kindi */
static wchar_t g_kdir[MAX_PATH]; /* %LOCALAPPDATA%\Kindi, where install.ps1 keeps state.json and install.log */
static wchar_t g_exe_dir[MAX_PATH];
static ULONG_PTR g_gdi;
static HICON g_icon_idle, g_icon_busy, g_icon_off;
static int g_icon_state = -1;
static int g_selected;           /* the server shown in the dashboard */
static long long g_popup_hid;    /* when the popup last hid (a tray click right after must not reopen it) */
static HINTERNET g_http;

#define S(v) MulDiv((v), g_dpi, 96)

/* ------------------------------------------------------------------ tiny helpers */
static long long now_ms(void) { return (long long)GetTickCount64(); }

static void ring_push(Ring *r, double v) {
    r->v[r->head] = (float)v;
    r->head = (r->head + 1) % HIST;
    if (r->n < HIST) r->n++;
}

static double ring_at(const Ring *r, int i) {  /* i = 0 is the oldest */
    int start = (r->head - r->n + HIST) % HIST;
    return r->v[(start + i) % HIST];
}

static double ring_max(const Ring *r) {
    double m = 0;
    int i;
    for (i = 0; i < r->n; i++) if (ring_at(r, i) > m) m = ring_at(r, i);
    return m;
}

static void utf8_to_w(const char *s, wchar_t *out, int cap) {
    if (!s) s = "";
    if (MultiByteToWideChar(CP_UTF8, 0, s, -1, out, cap) <= 0) out[0] = 0;
    out[cap - 1] = 0;
}

static void copy_a(char *dst, int cap, const char *src) {
    if (!src) src = "";
    strncpy(dst, src, (size_t)cap - 1);
    dst[cap - 1] = 0;
}

static void log_file(const wchar_t *text) {
    wchar_t path[MAX_PATH];
    char buf[600];
    SYSTEMTIME st;
    DWORD done;
    HANDLE f;
    int n;
    swprintf(path, MAX_PATH, L"%s\\monitor.log", g_dir);
    f = CreateFileW(path, FILE_APPEND_DATA, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) return;
    GetLocalTime(&st);
    n = snprintf(buf, sizeof buf, "%04u-%02u-%02u %02u:%02u:%02u %ls\n", st.wYear, st.wMonth, st.wDay, st.wHour,
                 st.wMinute, st.wSecond, text);
    if (n > 0) WriteFile(f, buf, (DWORD)min(n, (int)sizeof buf - 1), &done, NULL);
    CloseHandle(f);
}

/* an action line: kept for the dashboard's log and written to %LOCALAPPDATA%\Kindi\monitor.log */
static void remember(const wchar_t *fmt, ...) {
    wchar_t body[100], line[120];
    SYSTEMTIME st;
    va_list ap;
    va_start(ap, fmt);
    _vsnwprintf(body, 99, fmt, ap);
    va_end(ap);
    body[99] = 0;
    GetLocalTime(&st);
    swprintf(line, 120, L"%02u:%02u:%02u  %s", st.wHour, st.wMinute, st.wSecond, body);
    EnterCriticalSection(&g_lock);
    if (g_state.nlog == LOG_CAP) {
        MoveMemory(g_state.log[0], g_state.log[1], sizeof g_state.log[0] * (LOG_CAP - 1));
        g_state.nlog--;
    }
    wcscpy(g_state.log[g_state.nlog++], line);
    LeaveCriticalSection(&g_lock);
    log_file(body);
    if (g_main) PostMessageW(g_main, WM_DATA, 0, 0);
}

/* ------------------------------------------------------------------ settings (%APPDATA%\Kindi\monitor.ini) */
static wchar_t g_ini[MAX_PATH];

static int ini_int(const wchar_t *key, int dflt) { return (int)GetPrivateProfileIntW(L"monitor", key, dflt, g_ini); }
static void ini_set_int(const wchar_t *key, int v) {
    wchar_t b[24];
    swprintf(b, 24, L"%d", v);
    WritePrivateProfileStringW(L"monitor", key, b, g_ini);
}

/* ------------------------------------------------------------------ HTTP (WinHTTP, no proxy) */
static int http_get(const wchar_t *host, int port, const wchar_t *path, int timeout_ms, char **out, int *status) {
    HINTERNET c = NULL, r = NULL;
    DWORD code = 0, len = sizeof code, avail = 0, got = 0;
    char *buf = NULL;
    int size = 0, cap = 0, ok = 0;
    *out = NULL;
    *status = 0;
    if (!g_http) return 0;
    c = WinHttpConnect(g_http, host, (INTERNET_PORT)port, 0);
    if (!c) return 0;
    r = WinHttpOpenRequest(c, L"GET", path, NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
    if (!r) goto done;
    WinHttpSetTimeouts(r, timeout_ms, timeout_ms, timeout_ms, timeout_ms);
    if (!WinHttpSendRequest(r, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) goto done;
    if (!WinHttpReceiveResponse(r, NULL)) goto done;
    WinHttpQueryHeaders(r, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, NULL, &code, &len, NULL);
    *status = (int)code;
    for (;;) {
        if (!WinHttpQueryDataAvailable(r, &avail) || !avail) break;
        if (size + (int)avail + 1 > cap) {
            char *nb;
            cap = (size + (int)avail + 1) * 2;
            if (cap > (4 << 20)) break;              /* a /health is a few KB */
            nb = (char *)realloc(buf, (size_t)cap);
            if (!nb) break;
            buf = nb;
        }
        if (!WinHttpReadData(r, buf + size, avail, &got) || !got) break;
        size += (int)got;
    }
    if (buf) buf[size] = 0;
    *out = buf;
    buf = NULL;
    ok = size > 0;
done:
    free(buf);
    if (r) WinHttpCloseHandle(r);
    if (c) WinHttpCloseHandle(c);
    return ok;
}

static JNode *get_json(const wchar_t *host, int port, const wchar_t *path, int timeout_ms, int *status, char *err, int errcap) {
    char *body = NULL;
    JNode *j = NULL;
    if (http_get(host, port, path, timeout_ms, &body, status)) j = json_parse(body, (int)strlen(body));
    else if (err) snprintf(err, (size_t)errcap, *status ? "HTTP %d" : "no answer", *status);
    free(body);
    return j;
}

/* ------------------------------------------------------------------ run a command (hidden), capture its output */
static int run_capture(const wchar_t *cmd, char *out, int cap, DWORD timeout_ms, DWORD *exit_code) {
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE rd = NULL, wr = NULL;
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    wchar_t *line;
    int n = 0;
    DWORD got, start = GetTickCount();
    if (out && cap) out[0] = 0;
    if (exit_code) *exit_code = (DWORD)-1;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return 0;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    ZeroMemory(&si, sizeof si);
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError = wr;
    si.hStdInput = NULL;
    line = _wcsdup(cmd);
    if (!line || !CreateProcessW(NULL, line, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        free(line);
        CloseHandle(rd);
        CloseHandle(wr);
        return 0;
    }
    free(line);
    CloseHandle(wr);
    for (;;) {
        DWORD avail = 0;
        if (PeekNamedPipe(rd, NULL, 0, NULL, &avail, NULL) && avail) {
            char tmp[4096];
            if (!ReadFile(rd, tmp, min(avail, (DWORD)sizeof tmp), &got, NULL) || !got) break;
            if (out && n + (int)got < cap) { memcpy(out + n, tmp, got); n += (int)got; out[n] = 0; }
            continue;
        }
        if (WaitForSingleObject(pi.hProcess, 50) == WAIT_OBJECT_0) {
            while (PeekNamedPipe(rd, NULL, 0, NULL, &avail, NULL) && avail) {
                char tmp[4096];
                if (!ReadFile(rd, tmp, min(avail, (DWORD)sizeof tmp), &got, NULL) || !got) break;
                if (out && n + (int)got < cap) { memcpy(out + n, tmp, got); n += (int)got; out[n] = 0; }
            }
            break;
        }
        if (GetTickCount() - start > timeout_ms) { TerminateProcess(pi.hProcess, 1); break; }
    }
    if (exit_code) GetExitCodeProcess(pi.hProcess, exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(rd);
    return 1;
}

/* ------------------------------------------------------------------ NVML (nvml.dll from the driver, loaded at run time) */
typedef struct { unsigned long long total, free, used; } NvmlMemory;
typedef struct { unsigned int gpu, memory; } NvmlUtil;
typedef struct { char busIdLegacy[16]; unsigned int domain, bus, device, pciDeviceId, pciSubSystemId; char busId[32]; } NvmlPci;
typedef int (*Fn0)(void);
typedef int (*FnCount)(unsigned int *);
typedef int (*FnHandle)(unsigned int, void **);
typedef int (*FnMem)(void *, NvmlMemory *);
typedef int (*FnUtil)(void *, NvmlUtil *);
typedef int (*FnStr)(void *, char *, unsigned int);
typedef int (*FnPci)(void *, NvmlPci *);
typedef int (*FnU1)(void *, unsigned int *);
typedef int (*FnU2)(void *, int, unsigned int *);
typedef int (*FnU64)(void *, unsigned long long *);
static struct {
    HMODULE dll;
    Fn0 shutdown;
    FnCount count;
    FnHandle handle;
    FnMem mem;
    FnUtil util;
    FnStr name, uuid;
    FnPci pci;
    FnU2 temp, clock, pcie_thru;
    FnU1 power, limit, fan, gen, width, max_gen, max_width;
    FnU64 throttle;
    unsigned int n;
    void *dev[MAX_CARD];
} nv;

static int nvml_open(void) {
    Fn0 init;
    unsigned int i;
    wchar_t path[MAX_PATH];
    nv.dll = LoadLibraryW(L"nvml.dll");
    if (!nv.dll && GetEnvironmentVariableW(L"ProgramW6432", path, MAX_PATH)) {
        wcscat(path, L"\\NVIDIA Corporation\\NVSMI\\nvml.dll");
        nv.dll = LoadLibraryW(path);
    }
    if (!nv.dll) return 0;
#define NV(field, type, name) nv.field = (type)GetProcAddress(nv.dll, name)
    init = (Fn0)GetProcAddress(nv.dll, "nvmlInit_v2");
    NV(shutdown, Fn0, "nvmlShutdown");
    NV(count, FnCount, "nvmlDeviceGetCount_v2");
    NV(handle, FnHandle, "nvmlDeviceGetHandleByIndex_v2");
    NV(mem, FnMem, "nvmlDeviceGetMemoryInfo");
    NV(util, FnUtil, "nvmlDeviceGetUtilizationRates");
    NV(name, FnStr, "nvmlDeviceGetName");
    NV(uuid, FnStr, "nvmlDeviceGetUUID");
    NV(pci, FnPci, "nvmlDeviceGetPciInfo_v3");
    NV(temp, FnU2, "nvmlDeviceGetTemperature");
    NV(clock, FnU2, "nvmlDeviceGetClockInfo");
    NV(pcie_thru, FnU2, "nvmlDeviceGetPcieThroughput");
    NV(power, FnU1, "nvmlDeviceGetPowerUsage");
    NV(limit, FnU1, "nvmlDeviceGetEnforcedPowerLimit");
    NV(fan, FnU1, "nvmlDeviceGetFanSpeed");
    NV(gen, FnU1, "nvmlDeviceGetCurrPcieLinkGeneration");
    NV(width, FnU1, "nvmlDeviceGetCurrPcieLinkWidth");
    NV(max_gen, FnU1, "nvmlDeviceGetMaxPcieLinkGeneration");
    NV(max_width, FnU1, "nvmlDeviceGetMaxPcieLinkWidth");
    NV(throttle, FnU64, "nvmlDeviceGetCurrentClocksThrottleReasons");
#undef NV
    if (!init || !nv.count || !nv.handle || init() != 0) { FreeLibrary(nv.dll); nv.dll = NULL; return 0; }
    if (nv.count(&nv.n) != 0) nv.n = 0;
    if (nv.n > MAX_CARD) nv.n = MAX_CARD;
    for (i = 0; i < nv.n; i++) if (nv.handle(i, &nv.dev[i]) != 0) nv.dev[i] = NULL;
    return 1;
}

static void nvml_sample(Card *cards, int *ncard, int detail) {
    unsigned int i;
    for (i = 0; i < nv.n; i++) {
        Card *c = &cards[i];
        void *d = nv.dev[i];
        NvmlMemory m;
        NvmlUtil u;
        NvmlPci p;
        unsigned int v;
        if (!d) continue;
        if (!c->name[0] && nv.name) nv.name(d, c->name, sizeof c->name);
        if (!c->uuid[0] && nv.uuid) nv.uuid(d, c->uuid, sizeof c->uuid);
        if (!c->pci[0] && nv.pci && nv.pci(d, &p) == 0) {
            /* "00000000:02:00.0" -> "02:00.0", the form Kindi's /health and nvidia-smi -L use */
            const char *s = strchr(p.busId, ':');
            copy_a(c->pci, sizeof c->pci, s ? s + 1 : p.busId);
        }
        if (nv.mem && nv.mem(d, &m) == 0) { c->used_mib = m.used / 1048576.0; c->total_mib = m.total / 1048576.0; }
        if (nv.util && nv.util(d, &u) == 0) { c->util = u.gpu; c->mem_util = u.memory; }
        if (nv.temp && nv.temp(d, 0, &v) == 0) c->temp = v;
        if (nv.power && nv.power(d, &v) == 0) c->power_w = v / 1000.0;
        if (nv.limit && nv.limit(d, &v) == 0) c->limit_w = v / 1000.0;
        if (nv.fan && nv.fan(d, &v) == 0) c->fan = v;
        if (nv.clock && nv.clock(d, 1, &v) == 0) c->sm_mhz = v;   /* NVML_CLOCK_SM */
        if (nv.clock && nv.clock(d, 2, &v) == 0) c->mem_mhz = v;  /* NVML_CLOCK_MEM */
        if (nv.gen && nv.gen(d, &v) == 0) c->gen = (int)v;
        if (nv.width && nv.width(d, &v) == 0) c->width = (int)v;
        if (nv.max_gen && nv.max_gen(d, &v) == 0) c->max_gen = (int)v;
        if (nv.max_width && nv.max_width(d, &v) == 0) c->max_width = (int)v;
        if (nv.throttle) nv.throttle(d, &c->throttle);
        if (detail && nv.pcie_thru) {   /* each call samples for about 20 ms: the dashboard only */
            if (nv.pcie_thru(d, 1, &v) == 0) c->rx_kbs = v;  /* NVML_PCIE_UTIL_RX_BYTES, KB/s */
            if (nv.pcie_thru(d, 0, &v) == 0) c->tx_kbs = v;
        }
        ring_push(&c->util_h, c->util);
        ring_push(&c->power_h, c->power_w);
        ring_push(&c->temp_h, c->temp);
    }
    *ncard = (int)nv.n;
}

/* ------------------------------------------------------------------ host: CPU (PDH), RAM, disk, network */
static PDH_HQUERY g_pdh;
static PDH_HCOUNTER g_pdh_cpu;
static ULONG64 g_net_in, g_net_out;
static long long g_net_t;

static void host_open(Host *h) {
    DWORD n = 64;
    GetComputerNameW(h->name, &n);
    if (PdhOpenQueryW(NULL, 0, &g_pdh) == ERROR_SUCCESS) {
        /* "% Processor Utility" is what Task Manager shows; older systems have only "% Processor Time" */
        if (PdhAddEnglishCounterW(g_pdh, L"\\Processor Information(_Total)\\% Processor Utility", 0, &g_pdh_cpu) != ERROR_SUCCESS)
            PdhAddEnglishCounterW(g_pdh, L"\\Processor(_Total)\\% Processor Time", 0, &g_pdh_cpu);
        PdhCollectQueryData(g_pdh);
    }
}

static void host_sample(Host *h) {
    MEMORYSTATUSEX ms;
    ULARGE_INTEGER avail, total, freeb;
    wchar_t home[MAX_PATH];
    PMIB_IF_TABLE2 t = NULL;
    PDH_FMT_COUNTERVALUE v;
    if (g_pdh && PdhCollectQueryData(g_pdh) == ERROR_SUCCESS &&
        PdhGetFormattedCounterValue(g_pdh_cpu, PDH_FMT_DOUBLE, NULL, &v) == ERROR_SUCCESS)
        h->cpu = v.doubleValue > 100 ? 100 : v.doubleValue;
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) {
        h->ram_total_gb = ms.ullTotalPhys / 1073741824.0;
        h->ram_used_gb = (ms.ullTotalPhys - ms.ullAvailPhys) / 1073741824.0;
        h->ram_pct = ms.dwMemoryLoad;
        /* commit charge, as Task Manager and Fluid show it: the page-file-backed limit and what is committed */
        h->commit_limit_gb = ms.ullTotalPageFile / 1073741824.0;
        h->commit_gb = (ms.ullTotalPageFile - ms.ullAvailPageFile) / 1073741824.0;
        h->commit_pct = ms.ullTotalPageFile ? 100.0 * (double)(ms.ullTotalPageFile - ms.ullAvailPageFile) / (double)ms.ullTotalPageFile : 0;
    }
    if (GetEnvironmentVariableW(L"USERPROFILE", home, MAX_PATH) && GetDiskFreeSpaceExW(home, &avail, &total, &freeb) && total.QuadPart)
    {
        h->disk_pct = 100.0 * (double)(total.QuadPart - freeb.QuadPart) / (double)total.QuadPart;
        h->disk_free_gb = freeb.QuadPart / 1073741824.0;
    }
    if (GetIfTable2(&t) == NO_ERROR) {
        ULONG64 in = 0, out = 0;
        ULONG i;
        long long now = now_ms();
        for (i = 0; i < t->NumEntries; i++) {
            MIB_IF_ROW2 *r = &t->Table[i];
            if (r->Type == IF_TYPE_SOFTWARE_LOOPBACK || r->OperStatus != IfOperStatusUp || !r->InterfaceAndOperStatusFlags.HardwareInterface)
                continue;
            in += r->InOctets;
            out += r->OutOctets;
        }
        if (g_net_t && now > g_net_t) {
            double s = (now - g_net_t) / 1000.0;
            h->net_in_kbs = (in >= g_net_in ? (double)(in - g_net_in) : 0) / 1024.0 / s;
            h->net_out_kbs = (out >= g_net_out ? (double)(out - g_net_out) : 0) / 1024.0 / s;
        }
        g_net_in = in;
        g_net_out = out;
        g_net_t = now;
        FreeMibTable(t);
    }
    ring_push(&h->cpu_h, h->cpu);
    ring_push(&h->ram_h, h->ram_pct);
}

/* ------------------------------------------------------------------ processes: who owns a port */
static ULONGLONG process_born(HANDLE p) {
    FILETIME c, e, k, u;
    if (!GetProcessTimes(p, &c, &e, &k, &u)) return 0;
    return ((ULONGLONG)c.dwHighDateTime << 32) | c.dwLowDateTime;
}

static DWORD port_owner(int port) {
    DWORD size = 0, pid = 0, i;
    MIB_TCPTABLE_OWNER_PID *t;
    GetExtendedTcpTable(NULL, &size, FALSE, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0);
    t = (MIB_TCPTABLE_OWNER_PID *)malloc(size);
    if (!t) return 0;
    if (GetExtendedTcpTable(t, &size, FALSE, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0) == NO_ERROR) {
        for (i = 0; i < t->dwNumEntries; i++) {
            if (ntohs((u_short)t->table[i].dwLocalPort) == port) {
                DWORD a = t->table[i].dwLocalAddr;
                pid = t->table[i].dwOwningPid;
                if (a == htonl(INADDR_LOOPBACK) || a == 0) break;   /* prefer the loopback or wildcard listener */
            }
        }
    }
    free(t);
    return pid;
}

static void process_image(DWORD pid, wchar_t *out, int cap, ULONGLONG *born) {
    HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    DWORD n = (DWORD)cap;
    out[0] = 0;
    if (born) *born = 0;
    if (!p) return;
    if (!QueryFullProcessImageNameW(p, 0, out, &n)) out[0] = 0;
    if (born) *born = process_born(p);
    CloseHandle(p);
}

static const wchar_t *base_name(const wchar_t *path) {
    const wchar_t *s = wcsrchr(path, L'\\');
    return s ? s + 1 : path;
}

/* the kind of a local server, from the process that owns its port */
static void classify_local(Server *s) {
    DWORD pid = port_owner(s->port);
    wchar_t img[MAX_PATH];
    ULONGLONG born = 0;
    const wchar_t *b;
    if (!pid) return;
    process_image(pid, img, MAX_PATH, &born);
    b = base_name(img);
    if (!_wcsicmp(b, L"wslrelay.exe")) {
        if (s->kind != K_WSL_DOCKER && s->kind != K_WSL_UNIT) s->kind = K_WSL_DOCKER;   /* refined by wsl_scan */
    } else if (!_wcsicmp(b, L"com.docker.backend.exe") || !_wcsicmp(b, L"vpnkit.exe") || !_wcsicmp(b, L"com.docker.proxy.exe")) {
        s->kind = K_DOCKER;
    } else if (!_wcsicmp(b, L"python.exe") || !_wcsicmp(b, L"pythonw.exe") || !_wcsicmp(b, L"kindi.exe")) {
        s->kind = K_NATIVE;
        s->pid = pid;
        s->born = born;
        wcsncpy(s->image, img, MAX_PATH - 1);
    }
}

/* ------------------------------------------------------------------ servers: the list, polls, telemetry */
static Server *find_server(State *st, const wchar_t *host, int port) {
    int i;
    for (i = 0; i < st->nsrv; i++)
        if (st->srv[i].port == port && !_wcsicmp(st->srv[i].host, host)) return &st->srv[i];
    return NULL;
}

static Server *add_server(State *st, const wchar_t *host, int port, int source) {
    Server *s = find_server(st, host, port);
    if (s) {
        if (source < s->source || s->source == SRC_PORT) s->source = source;
        return s;
    }
    if (st->nsrv >= MAX_SRV) return NULL;
    s = &st->srv[st->nsrv++];
    ZeroMemory(s, sizeof *s);
    wcsncpy(s->host, host, 63);
    s->port = port;
    s->source = source;
    s->card = -1;
    if (_wcsicmp(host, L"127.0.0.1") && _wcsicmp(host, L"localhost")) s->kind = K_LAN;
    return s;
}

/* /health -> the server's fields. A server is Kindi only when it says so: a "kindi" version in /health (the
   plain server) or owned_by "kindi" in /v1/models (the DeepSeek server). {"status":"ok"} alone is llama.cpp. */
static void read_health(Server *s, JNode *h, long long now) {
    JNode *busy = json_get(h, "busy"), *slots, *each, *e;
    double gen, rd;
    int i;
    s->front = json_get(h, "servers") != NULL && json_get(h, "model") == NULL;
    if (s->front) s->kindi = 1;   /* kindi route, the front door: listed, no telemetry of its own */
    if (json_is(h, "kindi", J_STR)) {
        s->kindi = 1;
        copy_a(s->version, sizeof s->version, json_str(h, "kindi", ""));
    }
    s->has_stats = json_get(h, "stats") != NULL;
    s->has_uptime = json_get(h, "uptime_s") != NULL;
    s->has_head = json_is(h, "head", J_STR);
    s->has_device = json_is(h, "device", J_STR);
    s->has_gpu = json_get(h, "gpu_total_mib") != NULL;
    s->experts[0] = s->calib[0] = 0;
    if (json_is(h, "experts", J_OBJ)) {   /* the DeepSeek server (0.7): where the experts are, and the calibration */
        JNode *ex = json_get(h, "experts"), *cards = json_get(ex, "cards"), *c;
        char list[48] = "";
        for (c = cards ? cards->child : NULL; c; c = c->next)
            snprintf(list + strlen(list), sizeof list - strlen(list), "%s%d", list[0] ? "," : "", (int)c->num);
        snprintf(s->experts, sizeof s->experts, "%s%s%s%s%s", json_str(ex, "mode", "?"),
                 json_num(ex, "cache_gb", 0) > 0 ? ", RAM cache" : "", json_num(ex, "main_card", 0) ? ", main card" : "",
                 list[0] ? " + expert cards " : ", no expert card", list);
        if (json_is(ex, "calibration", J_OBJ)) {
            JNode *cal = json_get(ex, "calibration");
            double mp = json_num(cal, "max_pairs", -1);
            char cap[32];
            if (mp < 0) copy_a(cap, sizeof cap, "default pair cap");
            else if (mp == 0) copy_a(cap, sizeof cap, "no pair cap");
            else snprintf(cap, sizeof cap, "at most %d pair%s", (int)mp, mp == 1 ? "" : "s");
            snprintf(s->calib, sizeof s->calib, "%s, %s (%s)", cap, json_num(cal, "adaptive", 0) ? "adaptive cache" : "static",
                     json_str(cal, "source", "?"));
        } else if (json_is(ex, "calibration", J_NULL)) copy_a(s->calib, sizeof s->calib, "none (no expert card)");
    }
    copy_a(s->commit, sizeof s->commit, json_str(h, "commit", ""));
    copy_a(s->model, sizeof s->model, json_str(h, "model", s->model));
    copy_a(s->head, sizeof s->head, json_str(h, "head", ""));
    copy_a(s->device, sizeof s->device, json_str(h, "device", ""));
    copy_a(s->gpu_uuid, sizeof s->gpu_uuid, json_str(h, "gpu_uuid", ""));
    copy_a(s->gpu_pci, sizeof s->gpu_pci, json_str(h, "gpu_pci", ""));
    s->ctx = json_num(h, "ctx", s->ctx);
    s->uptime = json_num(h, "uptime_s", 0);
    s->gpu_free = json_num(h, "gpu_free_mib", 0);
    s->gpu_total = json_num(h, "gpu_total_mib", 0);
    s->requests = json_num(h, "stats.requests", 0);
    s->prompt_total = json_num(h, "stats.prompt_tokens", 0);
    s->completion_total = json_num(h, "stats.completion_tokens", 0);
    s->cached_total = json_num(h, "stats.cached_tokens", 0);
    s->q_chat = json_num(h, "queue.chat", 0);
    s->q_bg = json_num(h, "queue.background", 0);
    s->yields = json_num(h, "queue.yields", 0);
    s->draft_on = (int)json_num(h, "drafting.enabled", 0);
    copy_a(s->draft_method, sizeof s->draft_method, json_str(h, "drafting.method", ""));
    copy_a(s->drafter, sizeof s->drafter, json_str(h, "drafting.drafter", ""));
    s->accept = json_num(h, "drafting.last.acceptance", -1);
    s->dspark = (int)json_num(h, "dspark", 0);
    if (s->dspark) s->draft_on = 1;
    /* the last request: the plain server's last {prompt_tps, write_tps, tokens}; DeepSeek's last {prefill_tok_s,
       decode_tok_s, generated} */
    s->last_ptps = json_num(h, "last.prompt_tps", json_num(h, "last.prefill_tok_s", 0));
    s->last_wtps = json_num(h, "last.write_tps", json_num(h, "last.decode_tok_s", 0));
    s->last_tokens = json_num(h, "last.tokens", json_num(h, "last.generated", 0));
    /* progress: the plain server's "busy", or the DeepSeek server's "state" */
    if (busy && busy->type == J_OBJ) {
        const char *pr = json_str(busy, "prompt_read", "");
        copy_a(s->phase, sizeof s->phase, json_str(busy, "phase", "busy"));
        s->prompt_tokens = json_num(busy, "prompt_tokens", 0);
        s->reused = json_num(busy, "reused", 0);
        s->generated = json_num(busy, "generated", 0);
        s->elapsed = json_num(busy, "elapsed_s", 0);
        s->read = s->to_read = 0;
        if (pr[0]) sscanf(pr, "%lf/%lf", &s->read, &s->to_read);
    } else if (json_is(h, "state", J_OBJ)) {
        /* DeepSeek: state {busy, phase, prompt, computed, generated, start_pos, prefill_s} */
        JNode *st = json_get(h, "state");
        int on = (int)json_num(st, "busy", 0);
        copy_a(s->phase, sizeof s->phase, on ? json_str(st, "phase", "busy") : "idle");
        s->prompt_tokens = json_num(st, "prompt", 0);
        s->reused = json_num(st, "start_pos", 0);
        s->read = json_num(st, "computed", 0);
        s->to_read = s->prompt_tokens > s->reused ? s->prompt_tokens - s->reused : s->prompt_tokens;
        s->generated = json_num(st, "generated", 0);
        s->elapsed = json_num(st, "prefill_s", 0);
        if (!on) s->prompt_tokens = s->reused = s->read = s->to_read = s->generated = s->elapsed = 0;
    } else {
        const char *state = json_str(h, "state", "");
        copy_a(s->phase, sizeof s->phase, state[0] && strcmp(state, "idle") ? state : "idle");
        s->prompt_tokens = s->reused = s->generated = s->elapsed = s->read = s->to_read = 0;
    }
    /* decode slots (F7): {"n", "busy", "each": [...]}, or a count (the DeepSeek server) */
    slots = json_get(h, "slots");
    s->slots_n = 1;
    s->slots_busy = busy && busy->type == J_OBJ ? 1 : 0;
    if (slots && slots->type == J_NUM && slots->num >= 1) s->slots_n = (int)slots->num;
    if (slots && slots->type == J_OBJ) {
        s->slots_n = (int)json_num(slots, "n", 1);
        s->slots_busy = (int)json_num(slots, "busy", 0);
        each = json_get(slots, "each");
        for (i = 0, e = each ? each->child : NULL; e && i < MAX_SLOT; e = e->next, i++) {
            const char *pr = json_str(e, "prompt_read", "");
            copy_a(s->slot[i].phase, sizeof s->slot[i].phase, json_str(e, "phase", "idle"));
            s->slot[i].reused = json_num(e, "reused", 0);
            s->slot[i].generated = json_num(e, "generated", 0);
            s->slot[i].elapsed = json_num(e, "elapsed_s", 0);
            s->slot[i].read = s->slot[i].to_read = 0;
            if (pr[0]) sscanf(pr, "%lf/%lf", &s->slot[i].read, &s->slot[i].to_read);
        }
    }
    if (s->slots_n > MAX_SLOT) s->slots_n = MAX_SLOT;
    if (s->slots_n < 1) s->slots_n = 1;
    /* observed rates, as Bridge computes them: the change between two polls of the same request */
    gen = s->generated;
    rd = s->read;
    s->write_tps = s->read_tps = 0;
    if (s->prev_t && now > s->prev_t && s->prompt_tokens == s->prev_ptok && strcmp(s->phase, "idle")) {
        double dt = (now - s->prev_t) / 1000.0;
        if (gen >= s->prev_gen) s->write_tps = (gen - s->prev_gen) / dt;
        if (rd >= s->prev_read) s->read_tps = (rd - s->prev_read) / dt;
    }
    s->prev_gen = gen;
    s->prev_read = rd;
    s->prev_ptok = s->prompt_tokens;
    s->prev_t = now;
    ring_push(&s->wtps, s->write_tps);
    ring_push(&s->rtps, s->read_tps);
}

static void flatten(Server *s, JNode *n, const char *prefix) {
    JNode *c;
    char key[64];
    int idx = 0;
    for (c = n ? n->child : NULL; c && s->nfields < 96; c = c->next, idx++) {
        if (n->type == J_OBJ) snprintf(key, sizeof key, "%s%s%s", prefix, prefix[0] ? "." : "", c->key ? c->key : "?");
        else snprintf(key, sizeof key, "%s[%d]", prefix, idx);
        if (c->type == J_OBJ || c->type == J_ARR) {
            if (!c->count) snprintf(s->fields[s->nfields++], 96, "%s = %s", key, c->type == J_OBJ ? "{}" : "[]");
            else flatten(s, c, key);
            continue;
        }
        switch (c->type) {
        case J_NULL: snprintf(s->fields[s->nfields++], 96, "%s = null", key); break;
        case J_BOOL: snprintf(s->fields[s->nfields++], 96, "%s = %s", key, c->num ? "true" : "false"); break;
        case J_NUM: snprintf(s->fields[s->nfields++], 96, "%s = %.10g", key, c->num); break;
        default: snprintf(s->fields[s->nfields++], 96, "%s = %s", key, c->str ? c->str : ""); break;
        }
    }
}

static void read_models(Server *s, JNode *m) {
    JNode *d = json_get(m, "data"), *first = json_at(d, 0);
    if (!first) return;
    if (!strcmp(json_str(first, "owned_by", ""), "kindi")) s->kindi = 1;
    if (!s->model[0]) copy_a(s->model, sizeof s->model, json_str(first, "id", ""));
    if (!s->ctx) s->ctx = json_num(first, "context_length", 0);
}

static void process_usage(Server *s, long long dt_ms) {
    HANDLE p;
    FILETIME c, e, k, u;
    PROCESS_MEMORY_COUNTERS pm;
    if (s->kind != K_NATIVE || !s->pid) return;
    p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, s->pid);
    if (!p) return;
    if (process_born(p) == s->born && GetProcessTimes(p, &c, &e, &k, &u)) {
        ULONGLONG t = (((ULONGLONG)k.dwHighDateTime << 32) | k.dwLowDateTime) + (((ULONGLONG)u.dwHighDateTime << 32) | u.dwLowDateTime);
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        if (s->prev_cpu_100ns && dt_ms > 0)
            s->proc_cpu = (double)(t - s->prev_cpu_100ns) / (dt_ms * 10000.0) / si.dwNumberOfProcessors * 100.0;
        s->prev_cpu_100ns = t;
        pm.cb = sizeof pm;
        if (GetProcessMemoryInfo(p, &pm, sizeof pm)) s->proc_mb = pm.WorkingSetSize / 1048576.0;
    }
    CloseHandle(p);
}

/* poll one server (outside the lock: the network call can take up to the timeout) */
static void poll_server(Server *s, int with_models, long long dt_ms) {
    int status = 0;
    char err[96] = "";
    long long now = now_ms();
    JNode *h = get_json(s->host, s->port, L"/health", s->kind == K_LAN ? 1500 : 800, &status, err, sizeof err);
    if (!h || status != 200) {
        s->online = 0;
        s->offline_polls++;
        copy_a(s->err, sizeof s->err, h ? "/health is not 200" : err);
        copy_a(s->phase, sizeof s->phase, "offline");
        s->unit_state[0] = 0;
        if (s->kind == K_WSL_UNIT && s->distro[0] && s->name[0]) {   /* why it does not answer: the unit's state */
            wchar_t cmd[300];
            char out[64] = "";
            DWORD code;
            swprintf(cmd, 300, L"wsl.exe -d %s%s%s -e systemctl --user is-active %s", s->distro, s->wsl_user[0] ? L" -u " : L"", s->wsl_user, s->name);
            if (run_capture(cmd, out, sizeof out, 8000, &code)) {
                char *nl = strpbrk(out, "\r\n");
                if (nl) *nl = 0;
                copy_a(s->unit_state, sizeof s->unit_state, out);
            }
        }
        s->write_tps = s->read_tps = 0;
        ring_push(&s->wtps, 0);
        ring_push(&s->rtps, 0);
        json_free(h);
        return;
    }
    s->online = 1;
    s->offline_polls = 0;
    s->err[0] = 0;
    read_health(s, h, now);
    s->nfields = 0;
    flatten(s, h, "");
    json_free(h);
    if (with_models || !s->model[0]) {
        JNode *m = get_json(s->host, s->port, L"/v1/models", 800, &status, NULL, 0);
        s->models_401 = status == 401;
        if (m && status == 200) read_models(s, m);
        json_free(m);
    }
    process_usage(s, dt_ms);
}

/* ------------------------------------------------------------------ discovery */
static void home_path(wchar_t *out, const wchar_t *tail) {
    wchar_t home[MAX_PATH];
    if (!GetEnvironmentVariableW(L"USERPROFILE", home, MAX_PATH)) home[0] = 0;
    swprintf(out, MAX_PATH, L"%s\\.kindi\\%s", home, tail);
}

static char *read_file(const wchar_t *path, int *len) {
    HANDLE f = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
    DWORD size, got = 0;
    char *buf;
    *len = 0;
    if (f == INVALID_HANDLE_VALUE) return NULL;
    size = GetFileSize(f, NULL);
    if (size == INVALID_FILE_SIZE || size > (1 << 20)) { CloseHandle(f); return NULL; }
    buf = (char *)malloc(size + 1);
    if (buf && ReadFile(f, buf, size, &got, NULL)) { buf[got] = 0; *len = (int)got; }
    CloseHandle(f);
    return buf;
}

/* the plan file (F6) and the installer state, on Windows: their servers and ports */
static void discover_files(State *st) {
    wchar_t path[MAX_PATH];
    const wchar_t *files[2] = { L"plan.json", L"install.json" };
    int f;
    for (f = 0; f < 2; f++) {
        int len;
        char *text;
        JNode *j, *list, *e;
        home_path(path, files[f]);
        text = read_file(path, &len);
        if (!text) continue;
        j = json_parse(text, len);
        list = json_get(j, "servers");
        for (e = list ? list->child : NULL; e; e = e->next) {
            int port = (int)json_num(e, "port", 0);
            Server *s;
            if (port <= 0) continue;
            s = add_server(st, L"127.0.0.1", port, f == 0 ? SRC_PLAN : SRC_INSTALL);
            if (s && !s->name[0]) utf8_to_w(json_str(e, "name", json_str(e, "container", "")), s->name, 64);
        }
        if (f == 0 && json_num(j, "front.port", 0) > 0) add_server(st, L"127.0.0.1", (int)json_num(j, "front.port", 0), SRC_ROUTE);
        json_free(j);
        free(text);
    }
}

/* servers added by hand: [servers] in monitor.ini, lines "name=host:port" */
static void discover_manual(State *st) {
    wchar_t buf[2048], *p;
    GetPrivateProfileSectionW(L"servers", buf, 2048, g_ini);
    for (p = buf; *p; p += wcslen(p) + 1) {
        wchar_t *eq = wcschr(p, L'='), *colon, host[64];
        int port;
        Server *s;
        if (!eq) continue;
        colon = wcsrchr(eq + 1, L':');
        if (!colon) continue;
        port = _wtoi(colon + 1);
        wcsncpy(host, eq + 1, min(63, (int)(colon - eq - 1)));
        host[min(63, (int)(colon - eq - 1))] = 0;
        if (port <= 0 || !host[0]) continue;
        s = add_server(st, host, port, SRC_MANUAL);
        if (s && !s->name[0]) { wcsncpy(s->name, p, min(63, (int)(eq - p))); s->name[min(63, (int)(eq - p))] = 0; }
    }
}

/* the usual Kindi ports on this PC: a listener that answers /health is listed (Kindi or not) */
static const int PORTS[] = { 8090, 8091, 8093, 8094, 8095, 8096, 8097, 8098, 8099, 8101 };

static void discover_ports(State *st) {
    int i;
    for (i = 0; i < (int)(sizeof PORTS / sizeof PORTS[0]); i++) {
        int status = 0;
        JNode *h;
        Server *s = NULL;
        EnterCriticalSection(&g_lock);
        if (find_server(st, L"127.0.0.1", PORTS[i])) { LeaveCriticalSection(&g_lock); continue; }
        LeaveCriticalSection(&g_lock);
        if (!port_owner(PORTS[i])) continue;      /* nothing listens: no HTTP call at all */
        h = get_json(L"127.0.0.1", PORTS[i], L"/health", 600, &status, NULL, 0);
        if (h && status == 200) {
            EnterCriticalSection(&g_lock);
            s = add_server(st, L"127.0.0.1", PORTS[i], SRC_PORT);
            LeaveCriticalSection(&g_lock);
        }
        json_free(h);
    }
}

/* the running WSL distributions (wsl.exe writes UTF-8: WSL_UTF8=1 is set at start) */
static int wsl_running(wchar_t names[][64], int cap) {
    char out[2048], *line;
    DWORD code = 1;
    int n = 0;
    if (!run_capture(L"wsl.exe -l -q --running", out, sizeof out, 8000, &code) || code) return 0;
    for (line = strtok(out, "\r\n"); line && n < cap; line = strtok(NULL, "\r\n")) {
        if (!line[0] || !_strnicmp(line, "docker-desktop", 14)) continue;
        utf8_to_w(line, names[n], 64);
        n++;
    }
    return n;
}

/* WSL: for each running distro, the Kindi containers and units and their ports (only while a server is in WSL) */
static void wsl_scan(State *st) {
    char out[8192];
    wchar_t distros[8][64], cmd[1024];
    int nd, i;
    DWORD code;
    nd = wsl_running(distros, 8);
    for (i = 0; i < nd; i++) {
        char *line;
        /* as root: Docker, and the Kindi units of every user (~/.config/systemd/user/kindi*.service) with their port */
        swprintf(cmd, 1024,
                 L"wsl.exe -d %s -u root -e sh -c \"docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | sed 's/^/DOCKER /';"
                 L" for f in /root/.config/systemd/user/kindi*.service /home/*/.config/systemd/user/kindi*.service; do"
                 L" [ -f $f ] || continue; u=$(echo $f | cut -d/ -f3); [ $u = .config ] && u=root;"
                 L" echo UNIT $u $(basename $f) $(grep -o -- '--port [0-9]*' $f | head -1); done\"",
                 distros[i]);
        if (!run_capture(cmd, out, sizeof out, 15000, &code)) continue;
        for (line = strtok(out, "\n"); line; line = strtok(NULL, "\n")) {
            char name[64] = "", user[32] = "";
            int port = 0, kind = 0;
            if (!strncmp(line, "DOCKER ", 7)) {
                const char *arrow = strstr(line, "->"), *colon;
                if (!arrow) continue;
                for (colon = arrow; colon > line && *colon != ':'; colon--) {}
                port = atoi(colon + 1);
                sscanf(line + 7, "%63s", name);
                kind = K_WSL_DOCKER;
            } else if (!strncmp(line, "UNIT ", 5)) {
                const char *p = strstr(line, "--port ");
                port = p ? atoi(p + 7) : 0;
                if (!p && strstr(line, "kindi-dsv41")) port = DSV_PORT;   /* the DeepSeek server's default port */
                sscanf(line + 5, "%31s %63s", user, name);
                kind = K_WSL_UNIT;
            }
            if (port > 0) {
                Server *s;
                EnterCriticalSection(&g_lock);
                s = find_server(st, L"127.0.0.1", port);
                if (!s && kind == K_WSL_UNIT) {   /* a kindi*.service unit is Kindi's, running or not */
                    s = add_server(st, L"127.0.0.1", port, SRC_INSTALL);
                    if (s) s->kindi = 1;
                }
                if (s && (s->kind == K_UNKNOWN || s->kind == K_WSL_DOCKER || s->kind == K_WSL_UNIT)) {
                    s->kind = kind;
                    utf8_to_w(name, s->name, 64);
                    utf8_to_w(user, s->wsl_user, 32);
                    wcscpy(s->distro, distros[i]);
                }
                LeaveCriticalSection(&g_lock);
            }
        }
    }
}

/* Docker Desktop: container names for the ports it publishes */
static void docker_scan(State *st) {
    char out[8192], *line;
    DWORD code;
    if (!run_capture(L"docker.exe ps --format \"{{.Names}} {{.Ports}}\"", out, sizeof out, 5000, &code) || code) return;
    for (line = strtok(out, "\n"); line; line = strtok(NULL, "\n")) {
        const char *arrow = strstr(line, "->"), *colon;
        char name[64] = "";
        Server *s;
        if (!arrow) continue;
        for (colon = arrow; colon > line && *colon != ':'; colon--) {}
        sscanf(line, "%63s", name);
        EnterCriticalSection(&g_lock);
        s = find_server(st, L"127.0.0.1", atoi(colon + 1));
        if (s && s->kind == K_DOCKER) utf8_to_w(name, s->name, 64);
        LeaveCriticalSection(&g_lock);
    }
}

static void discover(State *st, int full) {
    /* full: at start and when asked (a window opens, "Find servers again", after a control): also the WSL scan,
       which finds the Kindi units in WSL that are stopped (no port answers for them) */
    int i, need_wsl = full, need_docker = 0;
    EnterCriticalSection(&g_lock);
    discover_files(st);
    discover_manual(st);
    LeaveCriticalSection(&g_lock);
    discover_ports(st);
    EnterCriticalSection(&g_lock);
    for (i = 0; i < st->nsrv; i++) {
        Server *s = &st->srv[i];
        if (s->kind != K_LAN) classify_local(s);
        if ((s->kind == K_WSL_DOCKER || s->kind == K_WSL_UNIT) && s->kindi && !s->distro[0]) need_wsl = 1;
        if (s->kind == K_DOCKER && s->kindi && !s->name[0]) need_docker = 1;
    }
    LeaveCriticalSection(&g_lock);
    if (need_wsl) wsl_scan(st);
    if (need_docker) docker_scan(st);
}

/* ------------------------------------------------------------------ the worker */
static volatile LONG g_ui_open;      /* a window is visible: poll fast */
static volatile LONG g_want_discover;

static void self_usage(State *st, long long dt_ms) {
    static ULONGLONG prev;
    FILETIME c, e, k, u;
    PROCESS_MEMORY_COUNTERS pm;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) {
        ULONGLONG t = (((ULONGLONG)k.dwHighDateTime << 32) | k.dwLowDateTime) + (((ULONGLONG)u.dwHighDateTime << 32) | u.dwLowDateTime);
        if (prev && dt_ms > 0) st->self_cpu = (double)(t - prev) / (dt_ms * 10000.0) * 100.0;   /* % of one core */
        prev = t;
    }
    pm.cb = sizeof pm;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pm, sizeof pm)) st->self_mb = pm.WorkingSetSize / 1048576.0;
}

static DWORD WINAPI worker(void *arg) {
    long long last = now_ms(), last_discover = 0, last_models = 0;
    (void)arg;
    int use_nvml = ini_int(L"nvml", 1);   /* nvml=0 in monitor.ini: no card telemetry */
    int nvml_seen = 0, nvml_failed = 0;
    host_open(&g_state.host);
    for (;;) {
        long long now = now_ms(), dt = now - last;
        int open = g_ui_open != 0, i, n, interval = open ? g_poll_ms_fast : SLOW_MS;
        int with_models = now - last_models > 30000;
        Server copy[MAX_SRV];
        Card cards[MAX_CARD];
        int ncard = 0;
        Host host;
        last = now;
        {
            int asked = InterlockedExchange(&g_want_discover, 0) != 0;
            if (asked || !last_discover || now - last_discover > (open ? DISCOVER_OPEN_MS : DISCOVER_TRAY_MS)) {
                discover(&g_state, asked || !last_discover);
                last_discover = now;
            }
        }
        EnterCriticalSection(&g_lock);
        n = g_state.nsrv;
        memcpy(copy, g_state.srv, sizeof(Server) * (size_t)n);
        memcpy(cards, g_state.card, sizeof cards);
        host = g_state.host;
        LeaveCriticalSection(&g_lock);
        for (i = 0; i < n; i++) {
            /* an offline server is asked every 5 s, not at every poll */
            if (copy[i].gone || (copy[i].offline_polls > 2 && now < copy[i].next_poll)) continue;
            poll_server(&copy[i], with_models, dt);
            copy[i].next_poll = now + 5000;
        }
        if (with_models) last_models = now;
        wsl_install_progress();
        for (i = 0; i < n; i++)
            if (copy[i].kindi && copy[i].online && (copy[i].kind == K_WSL_DOCKER || copy[i].kind == K_WSL_UNIT) && !copy[i].distro[0]) {
                EnterCriticalSection(&g_lock);
                if (i < g_state.nsrv && !g_state.srv[i].distro[0]) InterlockedExchange(&g_want_discover, 1);
                LeaveCriticalSection(&g_lock);
                break;
            }
        /* NVML from the start, so the card graphs have their history when a window opens (its driver DLLs take
           about 19 MB; nvml=0 in monitor.ini leaves it out) */
        if (use_nvml && !nv.dll && !nvml_failed) {
            if (nvml_open()) nvml_seen = 1; else nvml_failed = 1;
        }
        if (nv.dll) nvml_sample(cards, &ncard, g_dash && IsWindowVisible(g_dash));
        else ncard = g_state.ncard;
        host_sample(&host);
        EnterCriticalSection(&g_lock);
        /* keep what discovery added meanwhile; the polled fields replace the old ones */
        for (i = 0; i < n && i < g_state.nsrv; i++) {
            int kind = g_state.srv[i].kind, started = g_state.srv[i].started_here, model_n = g_state.srv[i].model_n;
            int gone = g_state.srv[i].gone;
            char model_ids[16][48];
            wchar_t model_label[16][96];
            DWORD pid = g_state.srv[i].pid;
            ULONGLONG born = g_state.srv[i].born;
            wchar_t name[64], distro[64], wsl_user[32];
            char life[32];
            wcscpy(name, g_state.srv[i].name);
            wcscpy(distro, g_state.srv[i].distro);
            wcscpy(wsl_user, g_state.srv[i].wsl_user);
            memcpy(life, g_state.srv[i].life, sizeof life);
            memcpy(model_ids, g_state.srv[i].model_ids, sizeof model_ids);
            memcpy(model_label, g_state.srv[i].model_label, sizeof model_label);
            g_state.srv[i] = copy[i];
            g_state.srv[i].model_n = model_n;
            memcpy(g_state.srv[i].model_ids, model_ids, sizeof model_ids);
            memcpy(g_state.srv[i].model_label, model_label, sizeof model_label);
            g_state.srv[i].kind = kind;
            g_state.srv[i].gone = gone;
            g_state.srv[i].started_here = started;
            g_state.srv[i].pid = pid ? pid : copy[i].pid;
            g_state.srv[i].born = pid ? born : copy[i].born;
            wcscpy(g_state.srv[i].name, name);
            wcscpy(g_state.srv[i].distro, distro);
            wcscpy(g_state.srv[i].wsl_user, wsl_user);
            memcpy(g_state.srv[i].life, life, sizeof life);
            if (g_state.srv[i].online && strcmp(life, "stopping")) g_state.srv[i].life[0] = 0;
        }
        memcpy(g_state.card, cards, sizeof cards);
        g_state.ncard = ncard;
        g_state.nvml = nvml_seen;
        g_state.host = host;
        /* each server's card: by the PCI bus id or UUID from /health */
        for (i = 0; i < g_state.nsrv; i++) {
            Server *s = &g_state.srv[i];
            int c;
            s->card = -1;
            if (s->kind == K_LAN) continue;
            for (c = 0; c < ncard; c++) {
                size_t a = strlen(s->gpu_pci), b = strlen(cards[c].pci);
                if ((s->gpu_uuid[0] && !_stricmp(s->gpu_uuid, cards[c].uuid)) ||
                    (a >= 7 && b >= 7 && !_stricmp(s->gpu_pci + a - 7, cards[c].pci + b - 7)))
                    s->card = c;
            }
        }
        self_usage(&g_state, dt);
        LeaveCriticalSection(&g_lock);
        PostMessageW(g_main, WM_DATA, 0, 0);
        {
            HANDLE w[2] = { g_stop, g_wake };
            long long spent = now_ms() - now;
            DWORD wait = spent >= interval ? 1 : (DWORD)(interval - spent);
            if (WaitForMultipleObjects(2, w, FALSE, wait) == WAIT_OBJECT_0) break;
        }
    }
    if (nv.dll) { if (nv.shutdown) nv.shutdown(); FreeLibrary(nv.dll); }
    if (g_pdh) PdhCloseQuery(g_pdh);
    return 0;
}

/* ------------------------------------------------------------------ controls (a short-lived thread each) */
enum { C_START, C_STOP, C_RESTART, C_MODEL, C_MODELS, C_LOGS };
static char g_logs_text[32768];   /* the last C_LOGS output, for the logs window */
static wchar_t g_logs_title[96];
typedef struct { int index, op; char model[48]; } Control;

static void kindi_exe(wchar_t *out) { swprintf(out, MAX_PATH, L"%s\\kindi.exe", g_exe_dir); }

static int wait_health(Server *s, int want_online, int seconds) {
    int i;
    for (i = 0; i < seconds * 2; i++) {
        int status = 0;
        char *body = NULL;
        int up = http_get(s->host, s->port, L"/health", 500, &body, &status) && status == 200;
        free(body);
        if (up == want_online) return 1;
        if (WaitForSingleObject(g_stop, 500) == WAIT_OBJECT_0) return 0;
    }
    return 0;
}

static int native_stop(Server *s) {
    HANDLE p;
    wchar_t img[MAX_PATH];
    ULONGLONG born = 0;
    int ok = 0;
    if (!s->pid) { remember(L"stop: no process recorded for :%d", s->port); return 0; }
    process_image(s->pid, img, MAX_PATH, &born);
    /* only the recorded process: the same PID, the same creation time, the same image */
    if (born != s->born || _wcsicmp(img, s->image)) { remember(L"stop: PID %lu changed; not stopped", s->pid); return 0; }
    p = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, s->pid);
    if (p) {
        ok = TerminateProcess(p, 0) && WaitForSingleObject(p, 10000) == WAIT_OBJECT_0;
        CloseHandle(p);
    }
    remember(ok ? L"stopped PID %lu (:%d)" : L"could not stop PID %lu (:%d)", s->pid, s->port);
    return ok;
}

static int native_start(Server *s) {
    wchar_t exe[MAX_PATH], cmd[MAX_PATH * 2], logp[MAX_PATH];
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE logf;
    kindi_exe(exe);
    if (GetFileAttributesW(exe) == INVALID_FILE_ATTRIBUTES) { remember(L"start: %s is missing", exe); return 0; }
    swprintf(logp, MAX_PATH, L"%s\\serve-%d.log", g_dir, s->port);
    logf = CreateFileW(logp, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_ALWAYS, 0, NULL);
    swprintf(cmd, MAX_PATH * 2, L"\"%s\" serve --port %d", exe, s->port);
    ZeroMemory(&si, sizeof si);
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = si.hStdError = logf;
    remember(L"start: kindi.exe serve --port %d", s->port);
    if (!CreateProcessW(NULL, cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        if (logf != INVALID_HANDLE_VALUE) CloseHandle(logf);
        remember(L"start failed (error %lu)", GetLastError());
        return 0;
    }
    if (logf != INVALID_HANDLE_VALUE) CloseHandle(logf);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    /* the port's owner is python.exe (kindi.exe runs it): classify once /health answers */
    s->started_here = 1;
    return 1;
}

static int run_logged(const wchar_t *cmd, const wchar_t *what, DWORD timeout_ms) {
    char out[2048];
    DWORD code = 1;
    remember(L"%s", what);
    if (!run_capture(cmd, out, sizeof out, timeout_ms, &code)) { remember(L"could not run the command"); return 0; }
    if (code) {
        wchar_t w[100];
        char *nl = strchr(out, '\n');
        if (nl) *nl = 0;
        utf8_to_w(out, w, 100);
        remember(L"failed (exit %lu): %s", code, w);
    }
    return code == 0;
}

/* the command of one control for a WSL or Docker server; NULL when the kind has none */
static int kind_command(const Server *s, const wchar_t *verb, wchar_t *cmd, int cap) {
    if (s->kind == K_WSL_DOCKER && s->name[0])
        return swprintf(cmd, cap, L"wsl.exe -d %s -e docker %s %s", s->distro, verb, s->name) > 0;
    if (s->kind == K_WSL_UNIT && s->name[0])
        return swprintf(cmd, cap, L"wsl.exe -d %s%s%s -e systemctl --user %s %s", s->distro, s->wsl_user[0] ? L" -u " : L"",
                        s->wsl_user, verb, s->name) > 0;
    if (s->kind == K_DOCKER && s->name[0])
        return swprintf(cmd, cap, L"docker.exe %s %s", verb, s->name) > 0;
    return 0;
}

static void set_life(int index, const char *what) {
    EnterCriticalSection(&g_lock);
    if (index >= 0 && index < g_state.nsrv) copy_a(g_state.srv[index].life, sizeof g_state.srv[index].life, what);
    LeaveCriticalSection(&g_lock);
    PostMessageW(g_main, WM_DATA, 0, 0);
}

static DWORD WINAPI control_thread(void *arg) {
    Control c = *(Control *)arg;
    Server s;
    wchar_t cmd[512], what[160];
    int ok = 0, started_here = 0;
    free(arg);
    EnterCriticalSection(&g_lock);
    s = g_state.srv[c.index];
    LeaveCriticalSection(&g_lock);
    if (s.kind == K_LAN || !s.kindi) { remember(L"no controls for this server"); goto done; }
    if (c.op == C_LOGS) {
        static char out[32768];
        DWORD code = 1;
        wchar_t logp[MAX_PATH];
        int n;
        if (s.kind == K_WSL_UNIT)
            swprintf(cmd, 512, L"wsl.exe -d %s%s%s -e journalctl --user -u %s -n 80 --no-pager", s.distro, s.wsl_user[0] ? L" -u " : L"", s.wsl_user, s.name);
        else if (s.kind == K_WSL_DOCKER) swprintf(cmd, 512, L"wsl.exe -d %s -e docker logs --tail 80 %s", s.distro, s.name);
        else if (s.kind == K_DOCKER) swprintf(cmd, 512, L"docker.exe logs --tail 80 %s", s.name);
        else if (s.kind == K_NATIVE) {
            char *text;
            swprintf(logp, MAX_PATH, L"%s\\serve-%d.log", g_dir, s.port);
            text = read_file(logp, &n);
            copy_a(out, sizeof out, text ? (n > 30000 ? text + n - 30000 : text) : "(no log: the monitor did not start this server)");
            free(text);
            cmd[0] = 0;
        } else goto done;
        if (cmd[0] && !run_capture(cmd, out, sizeof out, 30000, &code)) copy_a(out, sizeof out, "(the log command did not run)");
        EnterCriticalSection(&g_lock);
        memcpy(g_logs_text, out, sizeof g_logs_text);
        swprintf(g_logs_title, 96, L"Log of :%d (%s)", s.port, KIND_NAME[s.kind]);
        LeaveCriticalSection(&g_lock);
        PostMessageW(g_main, WM_APP + 20, 0, 0);
        goto done;
    }
    if (c.op == C_MODELS) {   /* the selector's list: `kindi models --json` where the server runs */
        char out[8192];
        DWORD code = 1;
        wchar_t exe[MAX_PATH];
        JNode *j, *e;
        if (s.kind == K_NATIVE) { kindi_exe(exe); swprintf(cmd, 512, L"\"%s\" models --json", exe); }
        else if (s.kind == K_WSL_DOCKER) swprintf(cmd, 512, L"wsl.exe -d %s -e docker exec %s kindi models --json", s.distro, s.name);
        else if (s.kind == K_DOCKER) swprintf(cmd, 512, L"docker.exe exec %s kindi models --json", s.name);
        else goto done;
        if (!run_capture(cmd, out, sizeof out, 60000, &code) || code) { remember(L"kindi models failed"); goto done; }
        j = json_parse(strchr(out, '[') ? strchr(out, '[') : out, (int)strlen(strchr(out, '[') ? strchr(out, '[') : out));
        EnterCriticalSection(&g_lock);
        g_state.srv[c.index].model_n = 0;
        for (e = j ? j->child : NULL; e && g_state.srv[c.index].model_n < 16; e = e->next) {
            Server *v = &g_state.srv[c.index];
            wchar_t id[48], fit[32];
            const char *f = json_str(e, "fits", "");
            copy_a(v->model_ids[v->model_n], 48, json_str(e, "id", "?"));
            utf8_to_w(v->model_ids[v->model_n], id, 48);
            utf8_to_w(f[0] ? f : "too large", fit, 32);
            swprintf(v->model_label[v->model_n], 96, L"%s\t%.1f GB, %s%s", id, json_num(e, "size_gb", 0), fit,
                     json_num(e, "downloaded", 0) ? L", downloaded" : L"");
            v->model_n++;
        }
        LeaveCriticalSection(&g_lock);
        json_free(j);
        goto done;
    }
    if (c.op == C_STOP || c.op == C_RESTART) {
        set_life(c.index, "stopping");
        if (s.kind == K_NATIVE) ok = native_stop(&s);
        else if (kind_command(&s, c.op == C_STOP ? L"stop" : L"restart", cmd, 512)) {
            swprintf(what, 160, L"%s %s (:%d)", c.op == C_STOP ? L"stop" : L"restart", s.name, s.port);
            ok = run_logged(cmd, what, 120000);
        } else remember(L"the monitor does not know how to control :%d", s.port);
        if (ok && c.op == C_STOP) wait_health(&s, 0, 30);
    }
    if (c.op == C_MODEL) {
        wchar_t model[48], exe[MAX_PATH];
        utf8_to_w(c.model, model, 48);
        if (s.kind == K_NATIVE) { kindi_exe(exe); swprintf(cmd, 512, L"\"%s\" use %s", exe, model); }
        else if (s.kind == K_WSL_DOCKER) swprintf(cmd, 512, L"wsl.exe -d %s -e docker exec %s kindi use %s", s.distro, s.name, model);
        else if (s.kind == K_DOCKER) swprintf(cmd, 512, L"docker.exe exec %s kindi use %s", s.name, model);
        else { remember(L"no model selector for this server"); goto done; }
        swprintf(what, 160, L"kindi use %s (:%d); a download can take a while", model, s.port);
        if (!run_logged(cmd, what, 6 * 3600 * 1000)) goto done;
        if (s.kind == K_NATIVE) ok = native_stop(&s);
        else if (kind_command(&s, L"restart", cmd, 512)) ok = run_logged(cmd, L"restart", 120000);
        c.op = C_RESTART;
    }
    if ((c.op == C_START || c.op == C_RESTART) && s.kind == K_NATIVE) {
        wait_health(&s, 0, 30);
        ok = native_start(&s);
        started_here = ok;
    } else if (c.op == C_START && kind_command(&s, L"start", cmd, 512)) {
        swprintf(what, 160, L"start %s (:%d)", s.name, s.port);
        ok = run_logged(cmd, what, 120000);
    }
    if (ok && (c.op == C_START || c.op == C_RESTART)) {
        long long t0 = now_ms();
        /* the DeepSeek server reads its model into memory first: minutes, longer from a hard disk */
        set_life(c.index, strstr(s.model, "deepseek") || wcsstr(s.name, L"dsv41") ? "starting (warming model)" : "starting");
        if (wait_health(&s, 1, 1800)) remember(L":%d answers again after %.0f s", s.port, (now_ms() - t0) / 1000.0);
        else remember(L":%d did not answer within 30 minutes", s.port);
    }
done:
    set_life(c.index, "");
    EnterCriticalSection(&g_lock);
    g_state.busy_control = 0;
    if (started_here) {
        g_state.srv[c.index].started_here = 1;
        g_state.srv[c.index].pid = 0;      /* the new process: classify_local records it */
    }
    LeaveCriticalSection(&g_lock);
    InterlockedExchange(&g_want_discover, 1);
    SetEvent(g_wake);
    return 0;
}

static void control(int index, int op, const char *model) {
    Control *c;
    HANDLE t;
    EnterCriticalSection(&g_lock);
    if (g_state.busy_control || index < 0 || index >= g_state.nsrv) { LeaveCriticalSection(&g_lock); return; }
    g_state.busy_control = 1;
    LeaveCriticalSection(&g_lock);
    c = (Control *)calloc(1, sizeof *c);
    if (!c) return;
    c->index = index;
    c->op = op;
    if (model) copy_a(c->model, sizeof c->model, model);
    t = CreateThread(NULL, 0, control_thread, c, 0, NULL);
    if (t) CloseHandle(t);
    else { free(c); EnterCriticalSection(&g_lock); g_state.busy_control = 0; LeaveCriticalSection(&g_lock); }
}

/* ------------------------------------------------------------------ painting helpers */
static int g_wsl_view;   /* the dashboard shows the WSL panel */
typedef struct { HFONT light, mid, label, num, big, body, bold, tiny; } Fonts;
static Fonts g_fp, g_fd;   /* popup fonts (g_dpi), dashboard fonts (g_dash_dpi) */
static const Fonts *F;     /* the set the current paint uses */
static int g_sdpi = 96;    /* the DPI the current paint uses */
#define D(v) MulDiv((v), g_sdpi, 96)

static HFONT make_font(const wchar_t *face, int at96, int weight, int dpi) {
    return CreateFontW(-MulDiv(at96, dpi, 96), 0, 0, 0, weight, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                       CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, face);
}

static void free_fonts(Fonts *f) {
    HFONT *all = (HFONT *)f;
    int i;
    for (i = 0; i < (int)(sizeof *f / sizeof(HFONT)); i++) if (all[i]) { DeleteObject(all[i]); all[i] = NULL; }
}

/* Segoe UI ships with Windows 10 and 11; no font file is shipped (Fluid's DIN OT is licensed) */
static void build_fonts(Fonts *f, int dpi) {
    free_fonts(f);
    f->light = make_font(L"Segoe UI Light", 20, FW_LIGHT, dpi);
    f->mid = make_font(L"Segoe UI Semibold", 13, FW_SEMIBOLD, dpi);
    f->label = make_font(L"Segoe UI Semibold", 10, FW_SEMIBOLD, dpi);
    f->num = make_font(L"Segoe UI Light", 26, FW_LIGHT, dpi);
    f->big = make_font(L"Segoe UI Light", 34, FW_LIGHT, dpi);
    f->body = make_font(L"Segoe UI", 12, FW_NORMAL, dpi);
    f->bold = make_font(L"Segoe UI", 12, FW_BOLD, dpi);
    f->tiny = make_font(L"Segoe UI", 10, FW_NORMAL, dpi);
}

static void fill(HDC dc, int l, int t, int r, int b, COLORREF c) {
    RECT rc = { l, t, r, b };
    HBRUSH br = CreateSolidBrush(c);
    FillRect(dc, &rc, br);
    DeleteObject(br);
}

static int text_at(HDC dc, HFONT f, COLORREF c, int x, int y, const wchar_t *s) {
    SIZE sz;
    SelectObject(dc, f);
    SetTextColor(dc, c);
    TextOutW(dc, x, y, s, (int)wcslen(s));
    GetTextExtentPoint32W(dc, s, (int)wcslen(s), &sz);
    return sz.cx;
}

static int text_w(HDC dc, HFONT f, const wchar_t *s) {
    SIZE sz;
    SelectObject(dc, f);
    GetTextExtentPoint32W(dc, s, (int)wcslen(s), &sz);
    return sz.cx;
}

static void text_right(HDC dc, HFONT f, COLORREF c, int right, int y, const wchar_t *s) {
    text_at(dc, f, c, right - text_w(dc, f, s), y, s);
}

/* an uppercase label with letter spacing, as Fluid draws its captions */
static int label_at(HDC dc, int x, int y, COLORREF c, const wchar_t *s) {
    wchar_t up[96];
    int w, old;
    wcsncpy(up, s, 95);
    up[95] = 0;
    CharUpperBuffW(up, (DWORD)wcslen(up));
    old = SetTextCharacterExtra(dc, D(1));
    w = text_at(dc, F->label, c, x, y, up) + (int)wcslen(up) * D(1);
    SetTextCharacterExtra(dc, old);
    return w;
}

static void hline(HDC dc, int x0, int x1, int y, COLORREF c) { fill(dc, x0, y, x1, y + 1, c); }

static void bar(HDC dc, int x, int y, int w, int h, double frac, COLORREF c) {
    fill(dc, x, y, x + w, y + h, EDGE);
    if (frac > 0) fill(dc, x, y, x + (int)(w * (frac > 1 ? 1 : frac)), y + h, c);
}

/* a graph of the last n points: a line and a faint fill, scaled to top (or the largest value) */
static void spark(HDC dc, int x, int y, int w, int h, const Ring *r, double top, COLORREF c) {
    void *g = NULL, *pen = NULL, *brush = NULL;
    GPointF pts[HIST + 2];
    int i, n = r->n, np = 0, show = w / D(2);
    double m = top > 0 ? top : ring_max(r);
    fill(dc, x, y, x + w, y + h, PANEL);
    if (n < 2 || m <= 0) { hline(dc, x, x + w, y + h - 1, EDGE); return; }
    if (show > n) show = n;
    if (show > HIST) show = HIST;
    if (show < 2) show = 2;
    for (i = 0; i < show; i++) {
        double v = ring_at(r, n - show + i) / m;
        if (v > 1) v = 1;
        if (v < 0) v = 0;
        pts[np].x = (float)x + (float)w * (float)i / (float)(show - 1);
        pts[np].y = (float)(y + h - 1) - (float)(h - 2) * (float)v;
        np++;
    }
    if (GdipCreateFromHDC(dc, &g) != 0) return;
    GdipSetSmoothingMode(g, 4);
    GdipSetPixelOffsetMode(g, 4);
    if (GdipCreateSolidFill(ARGB_OF(c, 40), &brush) == 0) {
        pts[np].x = (float)(x + w); pts[np].y = (float)(y + h);
        pts[np + 1].x = (float)x; pts[np + 1].y = (float)(y + h);
        GdipFillPolygon(g, brush, pts, np + 2, 0);
        GdipDeleteBrush(brush);
    }
    if (GdipCreatePen1(ARGB_OF(c, 255), (float)D(1) + 0.5f, 2, &pen) == 0) {
        GdipDrawLines(g, pen, pts, np);
        GdipDeletePen(pen);
    }
    GdipDeleteGraphics(g);
}

/* Fluid's ring: a dark track and a lime arc for a percentage */
static void ring(HDC dc, int cx, int cy, int outer, int inner, double pct, COLORREF c) {
    void *g = NULL, *track = NULL, *val = NULL;
    float mid = (outer + inner) / 2.f, wid = (float)(outer - inner);
    if (pct < 0) pct = 0;
    if (pct > 100) pct = 100;
    if (GdipCreateFromHDC(dc, &g) != 0) return;
    GdipSetSmoothingMode(g, 4);
    GdipSetPixelOffsetMode(g, 4);
    if (GdipCreatePen1(0xFF232323u, wid, 2, &track) == 0) {
        GdipDrawEllipse(g, track, cx - mid, cy - mid, mid * 2, mid * 2);
        GdipDeletePen(track);
    }
    if (pct > 0.3 && GdipCreatePen1(ARGB_OF(c, 255), wid, 2, &val) == 0) {
        GdipDrawArc(g, val, cx - mid, cy - mid, mid * 2, mid * 2, -90.f, (float)(pct * 3.6));
        GdipDeletePen(val);
    }
    GdipDeleteGraphics(g);
}

static void dot(HDC dc, int cx, int cy, int r, COLORREF c) {
    void *g = NULL, *b = NULL;
    if (GdipCreateFromHDC(dc, &g) != 0) return;
    GdipSetSmoothingMode(g, 4);
    if (GdipCreateSolidFill(ARGB_OF(c, 255), &b) == 0) {
        GdipFillEllipse(g, b, (float)(cx - r), (float)(cy - r), (float)(2 * r), (float)(2 * r));
        GdipDeleteBrush(b);
    }
    GdipDeleteGraphics(g);
}

static void fmt_int(wchar_t *out, int cap, double v) {   /* 12,345 */
    wchar_t raw[32], *o = out;
    int len, i, lead;
    swprintf(raw, 32, L"%.0f", v < 0 ? 0 : v);
    len = (int)wcslen(raw);
    lead = len % 3 ? len % 3 : 3;
    for (i = 0; i < len && o - out < cap - 2; i++) {
        if (i && (i - lead) % 3 == 0 && i >= lead) *o++ = L',';
        *o++ = raw[i];
    }
    *o = 0;
}

static COLORREF phase_color(const Server *s) {
    if (!s->online) return RED;
    if (strstr(s->phase, "generat") || !strcmp(s->phase, "decode")) return LIME;
    if (strcmp(s->phase, "idle")) return CYAN;
    return GREY;
}

static void phase_text(const Server *s, wchar_t *out, int cap) {
    if (s->life[0] && (!s->online || !strcmp(s->life, "stopping"))) { utf8_to_w(s->life, out, cap); CharUpperBuffW(out, (DWORD)wcslen(out)); return; }
    if (!s->online && s->unit_state[0]) {
        if (!strcmp(s->unit_state, "active") || !strcmp(s->unit_state, "activating") || !strcmp(s->unit_state, "reloading"))
            wcsncpy(out, strstr(s->model, "deepseek") || wcsstr(s->name, L"dsv41") ? L"STARTING (WARMING MODEL)" : L"STARTING", cap);
        else if (!strcmp(s->unit_state, "failed")) wcsncpy(out, L"FAILED (SEE LOGS)", cap);
        else wcsncpy(out, L"STOPPED", cap);
        return;
    }
    if (!s->online) { wcsncpy(out, L"UNREACHABLE", cap); return; }
    if (!s->kindi) { wcsncpy(out, L"NOT KINDI", cap); return; }
    if (strstr(s->phase, "generat")) wcsncpy(out, L"GENERATING", cap);
    else if (!strcmp(s->phase, "prompt")) wcsncpy(out, L"READING PROMPT", cap);
    else if (!strcmp(s->phase, "idle")) wcsncpy(out, L"IDLE", cap);
    else { utf8_to_w(s->phase, out, cap); CharUpperBuffW(out, (DWORD)wcslen(out)); }
}

static void server_title(const Server *s, wchar_t *out, int cap) {
    wchar_t m[64];
    utf8_to_w(s->model, m, 64);
    if (s->front) swprintf(out, cap, L"front door :%d", s->port);
    else if (!m[0]) swprintf(out, cap, L"%s:%d", s->host, s->port);   /* not answered yet: its address */
    else swprintf(out, cap, L"%s", m);
}

/* the "speed now" of a server: the observed rate while it writes, else the last request's */
static double speed_now(const Server *s) {
    if (s->write_tps > 0) return s->write_tps;
    if (s->read_tps > 0) return s->read_tps;
    return 0;
}

/* ------------------------------------------------------------------ the tray popup */
#define POPUP_W 380
typedef struct { RECT r; int what, index; } Hit;
static Hit g_hits[64];
static int g_nhits;
static void add_hit(int l, int t, int r, int b, int what, int index) {
    if (g_nhits < 64) { RECT rc = { l, t, r, b }; g_hits[g_nhits].r = rc; g_hits[g_nhits].what = what; g_hits[g_nhits].index = index; g_nhits++; }
}
enum { H_DASH = 1, H_RESTART, H_MODEL, H_SERVER, H_START, H_STOP, H_CLOSE, H_ADD, H_REMOVE, H_LOGS,
       H_WSL_CHECK, H_WSL_INSTALL, H_WSL_START, H_WSL_CLOSE, H_WSL_OPEN };

static int popup_height(const State *st) {
    int h = 64, i;
    for (i = 0; i < st->nsrv; i++) if (!st->srv[i].gone) h += st->srv[i].kindi && st->srv[i].online ? 112 : 30;
    if (!st->nsrv) h += 48;
    h += st->ncard * 46 + (st->nvml ? 0 : 26) + 136;
    return h;
}

static void button(HDC dc, int x, int y, int w, int h, const wchar_t *t, int on, int what, int index) {
    RECT rc = { x, y, x + w, y + h };
    fill(dc, x, y, x + w, y + h, on ? EDGE : PANEL);
    SelectObject(dc, F->bold);
    SetTextColor(dc, on ? LIME : DIM);
    DrawTextW(dc, t, -1, &rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX);
    if (on) add_hit(x, y, x + w, y + h, what, index);
}

static void draw_popup(HDC mem, int W, int H, const State *st) {
    RECT rc = { 0, 0, W, H };
    int y, i, pad, cur = -1;
    wchar_t t[160], t2[64];
    F = &g_fp;
    g_sdpi = g_dpi;
    g_nhits = 0;
    pad = D(18);
    fill(mem, 0, 0, rc.right, rc.bottom, INK);
    SetBkMode(mem, TRANSPARENT);
    /* masthead: KINDI and a lime kicker, as Fluid's "AMIR HUSAIN  FLUID" */
    SetTextCharacterExtra(mem, D(3));
    i = text_at(mem, F->light, PAPER, pad, D(14), L"KINDI");
    SetTextCharacterExtra(mem, D(3));
    text_at(mem, F->mid, LIME, pad + i + D(26), D(20), L"MONITOR");
    SetTextCharacterExtra(mem, 0);
    {
        int busy = 0, online = 0;
        for (i = 0; i < st->nsrv; i++) if (st->srv[i].kindi) { online += st->srv[i].online; busy += st->srv[i].online && strcmp(st->srv[i].phase, "idle"); }
        swprintf(t, 160, L"%d ONLINE", online);
        text_right(mem, F->label, GREY, W - pad, D(24), t);
    }
    y = D(52);
    hline(mem, 0, W, y, EDGE);
    y += D(10);
    if (!st->nsrv) {
        text_at(mem, F->body, GREY, pad, y + D(8), L"No server found on this PC.");
        y += D(48);
    }
    for (i = 0; i < st->nsrv; i++) {
        const Server *s = &st->srv[i];
        COLORREF pc = phase_color(s);
        if (s->gone) continue;
        server_title(s, t, 160);
        if (!s->kindi || !s->online) {
            text_at(mem, F->body, DIM, pad, y + D(4), t);
            {
                wchar_t ph[48];
                phase_text(s, ph, 48);
                swprintf(t2, 64, L":%d  %s", s->port, ph);
            }
            text_right(mem, F->label, DIM, W - pad, y + D(6), t2);
            y += D(30);
            continue;
        }
        if (cur < 0) cur = i;
        dot(mem, pad + D(4), y + D(11), D(4), pc);
        text_at(mem, F->mid, PAPER, pad + D(16), y + D(2), t);
        phase_text(s, t2, 64);
        label_at(mem, W - pad - text_w(mem, F->label, t2) - (int)wcslen(t2) * D(1), y + D(5), pc, t2);
        add_hit(0, y, W, y + D(24), H_SERVER, i);
        y += D(26);
        /* the big number: tok/s now, and the graph of the last minutes */
        swprintf(t, 160, L"%.1f", speed_now(s));
        text_at(mem, F->num, s->write_tps > 0 ? PAPER : GREY, pad, y, t);
        label_at(mem, pad, y + D(36), GREY, s->read_tps > 0 && s->write_tps <= 0 ? L"read tok/s" : L"write tok/s");
        spark(mem, pad + D(110), y + D(2), W - 2 * pad - D(110), D(44), &s->wtps, 0, LIME);
        y += D(54);
        if (!strcmp(s->phase, "prompt") && s->to_read > 0) {
            bar(mem, pad, y, W - 2 * pad, D(4), s->read / s->to_read, CYAN);
        } else if (s->card >= 0 || s->gpu_total > 0) {
            double used = s->card >= 0 ? st->card[s->card].used_mib : s->gpu_total - s->gpu_free;
            double total = s->card >= 0 ? st->card[s->card].total_mib : s->gpu_total;
            bar(mem, pad, y, W - 2 * pad, D(4), total > 0 ? used / total : 0, used > total * 0.95 ? RED : DIM);
        }
        y += D(8);
        {
            wchar_t ctx[32];
            fmt_int(ctx, 32, s->ctx);
            swprintf(t, 160, L"ctx %s   slots %d/%d   :%d   %s", ctx, s->slots_busy, s->slots_n, s->port, KIND_NAME[s->kind]);
            text_at(mem, F->tiny, GREY, pad, y, t);
        }
        y += D(24);
    }
    hline(mem, 0, W, y, EDGE);
    y += D(8);
    if (!st->nvml) { text_at(mem, F->tiny, DIM, pad, y + D(4), L"No NVIDIA driver (NVML) on this PC"); y += D(26); }
    for (i = 0; i < st->ncard; i++) {
        const Card *c = &st->card[i];
        wchar_t nm[64];
        utf8_to_w(strncmp(c->name, "NVIDIA ", 7) ? c->name : c->name + 7, nm, 64);
        swprintf(t, 160, L"GPU %d  %s", i, nm);
        text_at(mem, F->bold, PAPER, pad, y, t);
        swprintf(t, 160, L"%.0f%%", c->util);
        text_right(mem, F->bold, LIME, W - pad, y, t);
        spark(mem, W - pad - D(150), y + D(2), D(110), D(14), &c->util_h, 100, LIME);
        swprintf(t, 160, L"%.1f / %.0f GiB   %.0f\u00B0C   %.0f W   %.0f MHz   PCIe %d x%d",
                 c->used_mib / 1024, c->total_mib / 1024, c->temp, c->power_w, c->sm_mhz, c->gen, c->width);
        text_at(mem, F->tiny, GREY, pad, y + D(20), t);
        y += D(46);
    }
    hline(mem, 0, W, y, EDGE);
    y += D(10);
    swprintf(t, 160, L"CPU %.0f%%", st->host.cpu);
    text_at(mem, F->tiny, GREY, pad, y, t);
    spark(mem, pad + D(64), y + D(1), W / 2 - pad - D(76), D(14), &st->host.cpu_h, 100, CYAN);
    swprintf(t, 160, L"RAM %.0f / %.0f GiB", st->host.ram_used_gb, st->host.ram_total_gb);
    text_at(mem, F->tiny, GREY, W / 2 + D(4), y, t);
    bar(mem, W / 2 + D(4), y + D(17), W / 2 - pad - D(4), D(3), st->host.ram_pct / 100, CYAN);
    y += D(26);
    swprintf(t, 160, L"COMMIT %.0f / %.0f GiB", st->host.commit_gb, st->host.commit_limit_gb);
    text_at(mem, F->tiny, GREY, pad, y, t);
    bar(mem, pad, y + D(17), W / 2 - pad - D(12), D(3), st->host.commit_pct / 100, st->host.commit_pct >= 90 ? RED : CYAN);
    swprintf(t, 160, L"DISK %.0f%%   %.0f GiB free", st->host.disk_pct, st->host.disk_free_gb);
    text_at(mem, F->tiny, GREY, W / 2 + D(4), y, t);
    bar(mem, W / 2 + D(4), y + D(17), W / 2 - pad - D(4), D(3), st->host.disk_pct / 100, DIM);
    y += D(34);
    {
        int bw = (W - 2 * pad - 2 * D(8)) / 3;
        button(mem, pad, y, bw, D(30), L"DASHBOARD", 1, H_DASH, 0);
        button(mem, pad + bw + D(8), y, bw, D(30), L"RESTART", cur >= 0 && st->srv[cur].kind != K_LAN && !st->busy_control, H_RESTART, cur);
        button(mem, pad + 2 * (bw + D(8)), y, bw, D(30), L"MODEL", cur >= 0 && st->srv[cur].kind != K_LAN && st->srv[cur].kind != K_WSL_UNIT && !st->busy_control, H_MODEL, cur);
    }
    hline(mem, 0, W, 0, EDGE);
    {
        HPEN pen = CreatePen(PS_SOLID, 1, EDGE);
        HGDIOBJ op = SelectObject(mem, pen), ob = SelectObject(mem, GetStockObject(NULL_BRUSH));
        Rectangle(mem, 0, 0, rc.right, rc.bottom);
        SelectObject(mem, op);
        SelectObject(mem, ob);
        DeleteObject(pen);
    }
}

/* paint a window through a back buffer: draw(mem, width, height, state) */
typedef void (*DrawFn)(HDC, int, int, const State *);
static void paint_with(HWND hwnd, DrawFn draw, const State *st) {
    PAINTSTRUCT ps;
    HDC dc = BeginPaint(hwnd, &ps), mem;
    HBITMAP bmp, oldb;
    RECT rc;
    GetClientRect(hwnd, &rc);
    if (rc.right > 1 && rc.bottom > 1) {
        mem = CreateCompatibleDC(dc);
        bmp = CreateCompatibleBitmap(dc, rc.right, rc.bottom);
        oldb = (HBITMAP)SelectObject(mem, bmp);
        draw(mem, rc.right, rc.bottom, st);
        BitBlt(dc, 0, 0, rc.right, rc.bottom, mem, 0, 0, SRCCOPY);
        SelectObject(mem, oldb);
        DeleteObject(bmp);
        DeleteDC(mem);
    }
    EndPaint(hwnd, &ps);
}

/* ------------------------------------------------------------------ the dashboard */
static Hit g_dhits[96];
static int g_ndhits;
static void add_dhit(int l, int t, int r, int b, int what, int index) {
    if (g_ndhits < 96) { RECT rc = { l, t, r, b }; g_dhits[g_ndhits].r = rc; g_dhits[g_ndhits].what = what; g_dhits[g_ndhits].index = index; g_ndhits++; }
}

static void dbutton(HDC dc, int x, int y, int w, int h, const wchar_t *t, int on, int what, int index) {
    RECT rc = { x, y, x + w, y + h };
    fill(dc, x, y, x + w, y + h, on ? EDGE : PANEL);
    SelectObject(dc, F->bold);
    SetTextColor(dc, on ? LIME : DIM);
    DrawTextW(dc, t, -1, &rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX);
    if (on) add_dhit(x, y, x + w, y + h, what, index);
}

/* one labelled graph with its current and peak value */
static void graph_box(HDC dc, int x, int y, int w, int h, const wchar_t *name, const Ring *r, double top, COLORREF c,
                      const wchar_t *now_text) {
    wchar_t t[48];
    label_at(dc, x, y, GREY, name);
    text_right(dc, F->bold, c, x + w, y - D(2), now_text);
    spark(dc, x, y + D(18), w, h - D(18), r, top, c);
    swprintf(t, 48, L"peak %.1f", ring_max(r));
    text_right(dc, F->tiny, DIM, x + w - D(4), y + D(20), t);
}

static void kv(HDC dc, int x, int y, int w, const wchar_t *k, const wchar_t *v) {
    text_at(dc, F->tiny, GREY, x, y, k);
    text_right(dc, F->tiny, PAPER, x + w, y, v);
}

static void draw_dash(HDC mem, int W, int H, const State *st) {
    int pad, left_w, x, y, i, sel;
    wchar_t t[200], t2[64];
    const Server *s = NULL;
    F = &g_fd;
    g_sdpi = g_dash_dpi;
    g_ndhits = 0;
    pad = D(18);
    fill(mem, 0, 0, W, H, INK);
    SetBkMode(mem, TRANSPARENT);
    SetTextCharacterExtra(mem, D(3));
    i = text_at(mem, F->light, PAPER, pad, D(14), L"KINDI");
    text_at(mem, F->mid, LIME, pad + i + D(26), D(20), L"MONITOR");
    SetTextCharacterExtra(mem, 0);
    swprintf(t, 200, L"%s   monitor %.1f MB, %.2f%% of a core", st->host.name, st->self_mb, st->self_cpu);
    text_right(mem, F->tiny, DIM, W - pad, D(22), t);
    y = D(52);
    hline(mem, 0, W, y, EDGE);
    /* the left column: servers, then cards */
    left_w = D(250);
    fill(mem, 0, y + 1, left_w, H, PANEL);
    fill(mem, left_w, y + 1, left_w + 1, H, EDGE);
    sel = g_selected < st->nsrv ? g_selected : 0;
    x = D(14);
    y += D(12);
    label_at(mem, x, y, GREY, L"Servers");
    dbutton(mem, left_w - D(70), y - D(4), D(58), D(20), L"+ ADD", 1, H_ADD, 0);
    y += D(20);
    for (i = 0; i < st->nsrv; i++) {
        const Server *v = &st->srv[i];
        COLORREF col = v->kindi && v->online ? PAPER : DIM;
        if (v->gone) continue;
        if (i == sel) fill(mem, 0, y - D(2), left_w, y + D(38), EDGE);
        dot(mem, x + D(4), y + D(9), D(4), v->kindi ? phase_color(v) : DIM);
        server_title(v, t, 200);
        text_at(mem, F->body, col, x + D(16), y, t);
        swprintf(t, 200, L"%s:%d  %s%s%s", v->kind == K_LAN ? L"" : L"", v->port, KIND_NAME[v->kind],
                 v->name[0] ? L"  " : L"", v->name);
        if (v->kind == K_LAN) swprintf(t, 200, L"%s:%d  LAN", v->host, v->port);
        text_at(mem, F->tiny, GREY, x + D(16), y + D(18), t);
        if (v->online && !v->kindi) text_right(mem, F->label, DIM, left_w - D(10), y + D(3), L"NOT KINDI");
        add_dhit(0, y - D(2), left_w, y + D(38), H_SERVER, i);
        y += D(42);
    }
    y += D(8);
    label_at(mem, x, y, GREY, L"Cards");
    y += D(20);
    if (!st->nvml) { text_at(mem, F->tiny, DIM, x, y, L"No NVIDIA driver (NVML)"); y += D(20); }
    for (i = 0; i < st->ncard; i++) {
        const Card *c = &st->card[i];
        wchar_t nm[64];
        utf8_to_w(strncmp(c->name, "NVIDIA ", 7) ? c->name : c->name + 7, nm, 64);
        swprintf(t, 200, L"GPU %d  %s", i, nm);
        text_at(mem, F->bold, PAPER, x, y, t);
        swprintf(t, 200, L"%.0f%%", c->util);
        text_right(mem, F->bold, LIME, left_w - D(12), y, t);
        spark(mem, x, y + D(20), left_w - x - D(12), D(22), &c->util_h, 100, LIME);
        swprintf(t, 200, L"%.1f/%.0f GiB  %.0f\u00B0C  %.0f/%.0f W", c->used_mib / 1024, c->total_mib / 1024, c->temp, c->power_w, c->limit_w);
        text_at(mem, F->tiny, GREY, x, y + D(46), t);
        swprintf(t, 200, L"SM %.0f  MEM %.0f MHz  PCIe %d x%d (max %d x%d)", c->sm_mhz, c->mem_mhz, c->gen, c->width, c->max_gen, c->max_width);
        text_at(mem, F->tiny, GREY, x, y + D(62), t);
        utf8_to_w(c->pci, nm, 64);
        swprintf(t, 200, L"bus %s  fan %.0f%%%s", nm, c->fan, c->throttle & 0x6C ? L"  throttled" : L"");
        text_at(mem, F->tiny, c->throttle & 0x6C ? AMBER : DIM, x, y + D(78), t);
        y += D(102);
    }
    /* the main area: the selected server */
    x = left_w + pad;
    y = D(52) + D(14);
    {
        int w = W - x - pad;
        if (st->nsrv) s = &st->srv[sel];
        if (s && s->gone) s = NULL;
        if (g_wsl_view) {
            draw_wsl_panel(mem, x, y, w, H, st);
        } else if (!s) {   /* no server: the machine itself, as Fluid shows it */
            int cy = y + D(46), gw = (w - D(16)) / 2, k;
            wchar_t a[16];
            ring(mem, x + D(46), cy, D(42), D(32), st->host.cpu, CYAN);
            ring(mem, x + D(146), cy, D(42), D(32), st->host.ram_pct, CYAN);
            ring(mem, x + D(246), cy, D(42), D(32), st->host.commit_pct, st->host.commit_pct >= 90 ? RED : CYAN);
            swprintf(a, 16, L"%.0f", st->host.cpu); text_at(mem, F->mid, PAPER, x + D(46) - text_w(mem, F->mid, a) / 2, cy - D(10), a);
            swprintf(a, 16, L"%.0f", st->host.ram_pct); text_at(mem, F->mid, PAPER, x + D(146) - text_w(mem, F->mid, a) / 2, cy - D(10), a);
            swprintf(a, 16, L"%.0f", st->host.commit_pct); text_at(mem, F->mid, PAPER, x + D(246) - text_w(mem, F->mid, a) / 2, cy - D(10), a);
            label_at(mem, x + D(46) - D(22), cy + D(48), GREY, L"CPU");
            label_at(mem, x + D(146) - D(22), cy + D(48), GREY, L"RAM");
            label_at(mem, x + D(246) - D(36), cy + D(48), GREY, L"COMMIT");
            text_at(mem, F->body, GREY, x + D(320), y + D(10), L"No Kindi server is running or added.");
            text_at(mem, F->tiny, GREY, x + D(320), y + D(34), L"Servers on this PC are found from ~/.kindi/plan.json, the installer");
            text_at(mem, F->tiny, GREY, x + D(320), y + D(50), L"state and the ports 8090-8101. + ADD adds one by its URL (LAN too).");
            y += D(130);
            for (k = 0; k < st->ncard; k++) {
                const Card *c = &st->card[k];
                wchar_t nm[64], name[96];
                utf8_to_w(strncmp(c->name, "NVIDIA ", 7) ? c->name : c->name + 7, nm, 64);
                swprintf(name, 96, L"GPU %d %s utilization", k, nm);
                swprintf(t, 200, L"%.0f%%   %.1f / %.0f GiB", c->util, c->used_mib / 1024, c->total_mib / 1024);
                graph_box(mem, x, y, gw, D(80), name, &c->util_h, 100, LIME, t);
                swprintf(name, 96, L"GPU %d power", k);
                swprintf(t, 200, L"%.0f W   %.0f\u00B0C", c->power_w, c->temp);
                graph_box(mem, x + gw + D(16), y, gw, D(80), name, &c->power_h, c->limit_w > 0 ? c->limit_w : 0, AMBER, t);
                y += D(96);
            }
            graph_box(mem, x, y, w, D(70), L"Host CPU", &st->host.cpu_h, 100, CYAN, L"");
            y += D(86);
            dbutton(mem, x, y, D(240), D(30), L"DEEPSEEK-V4.1-FLASH (WSL)", 1, H_WSL_OPEN, 0);
        } else if (0) {
            text_at(mem, F->body, GREY, x, y, L"No server found. Kindi servers on this PC are found from ~/.kindi/plan.json,");
            text_at(mem, F->body, GREY, x, y + D(20), L"the installer state, the ports 8090-8101, and [servers] in %APPDATA%\\Kindi\\monitor.ini.");
        } else {
            const Card *c = s->card >= 0 ? &st->card[s->card] : NULL;
            int gw, gh, cy;
            wchar_t a[32], b[32], d[32];
            server_title(s, t, 200);
            text_at(mem, F->light, PAPER, x, y, t);
            phase_text(s, t2, 64);
            label_at(mem, x + w - text_w(mem, F->label, t2) - (int)wcslen(t2) * D(1), y + D(8), phase_color(s), t2);
            y += D(32);
            {
                wchar_t v[40], m[48];
                utf8_to_w(s->version, v, 40);
                utf8_to_w(s->commit, m, 48);
                swprintf(t, 200, L"%s:%d   %s%s%s   found by %s%s%s", s->host, s->port, s->kindi ? L"Kindi " : L"",
                         s->kindi ? (v[0] ? v : L"(the DeepSeek server reports no version)") : s->online ? L"not Kindi" : L"not known yet",
                         s->commit[0] ? L"" : L"", SRC_NAME[s->source], s->err[0] ? L"   " : L"", L"");
                if (s->commit[0]) { wcscat(t, L"   commit "); wcscat(t, m); }
                if (s->err[0]) { wchar_t e[96]; utf8_to_w(s->err, e, 96); wcscat(t, L"   "); wcscat(t, e); }
            }
            text_at(mem, F->tiny, GREY, x, y, t);
            y += D(26);
            if (!s->online) {
                wchar_t e[96];
                utf8_to_w(s->err[0] ? s->err : "no answer", e, 96);
                if (s->unit_state[0]) {   /* a Kindi service in WSL: say what systemd says */
                    wchar_t us[24];
                    utf8_to_w(s->unit_state, us, 24);
                    swprintf(t, 200, L"The service %s in '%s' is %s; /health gives %s.", s->name, s->distro, us, e);
                    text_at(mem, F->body, GREY, x, y, t);
                    text_at(mem, F->body, GREY, x, y + D(20), !strcmp(s->unit_state, "active") || !strcmp(s->unit_state, "activating")
                                ? L"It is starting: the DeepSeek server reads its model into memory first (minutes from an SSD)."
                                : L"START starts it; LOGS shows its journal.");
                } else {
                    swprintf(t, 200, L"Unreachable: /health gives %s. The monitor asks again every 5 seconds.", e);
                    text_at(mem, F->body, GREY, x, y, t);
                    text_at(mem, F->body, GREY, x, y + D(20), s->kindi ? L"It is a Kindi server; its values come back when it answers."
                                                                       : L"It has not answered yet, so the monitor does not know whether it is Kindi.");
                }
                y += D(56);
                goto controls;
            }
            if (!s->kindi && s->online) {
                text_at(mem, F->body, GREY, x, y, L"This server answers /health, but it is not Kindi: its /health has no \"kindi\" version, and");
                text_at(mem, F->body, GREY, x, y + D(20), L"/v1/models does not say owned_by \"kindi\". The monitor shows no telemetry and no controls for it.");
                y += D(56);
                goto action_log;
            }
            /* rings: card utilization, card memory, host RAM; then the two speeds */
            cy = y + D(46);
            ring(mem, x + D(46), cy, D(42), D(32), c ? c->util : 0, LIME);
            ring(mem, x + D(146), cy, D(42), D(32), c ? 100.0 * c->used_mib / (c->total_mib > 0 ? c->total_mib : 1) :
                 (s->gpu_total > 0 ? 100.0 * (s->gpu_total - s->gpu_free) / s->gpu_total : 0), LIME);
            ring(mem, x + D(246), cy, D(42), D(32), st->host.ram_pct, CYAN);
            swprintf(a, 32, c ? L"%.0f" : L"--", c ? c->util : 0);
            if (c || s->gpu_total > 0)
                swprintf(b, 32, L"%.0f", c ? 100.0 * c->used_mib / (c->total_mib > 0 ? c->total_mib : 1) :
                         100.0 * (s->gpu_total - s->gpu_free) / s->gpu_total);
            else wcscpy(b, L"--");
            swprintf(d, 32, L"%.0f", st->host.ram_pct);
            text_at(mem, F->mid, c ? PAPER : GREY, x + D(46) - text_w(mem, F->mid, a) / 2, cy - D(10), a);
            text_at(mem, F->mid, c || s->gpu_total > 0 ? PAPER : GREY, x + D(146) - text_w(mem, F->mid, b) / 2, cy - D(10), b);
            text_at(mem, F->mid, PAPER, x + D(246) - text_w(mem, F->mid, d) / 2, cy - D(10), d);
            label_at(mem, x + D(46) - D(24), cy + D(48), GREY, L"GPU");
            label_at(mem, x + D(146) - D(26), cy + D(48), GREY, L"VRAM");
            label_at(mem, x + D(246) - D(22), cy + D(48), GREY, L"RAM");
            {
                int sx = x + D(320);
                swprintf(t, 200, L"%.1f", s->write_tps);
                text_at(mem, F->big, s->write_tps > 0 ? PAPER : GREY, sx, y + D(4), t);
                label_at(mem, sx, y + D(52), GREY, L"write tok/s now");
                swprintf(t, 200, L"%.0f", s->read_tps);
                text_at(mem, F->big, s->read_tps > 0 ? PAPER : GREY, sx + D(150), y + D(4), t);
                label_at(mem, sx + D(150), y + D(52), GREY, L"read tok/s now");
                if (s->last_wtps > 0 || s->last_ptps > 0) {
                    swprintf(t, 200, L"last request: write %.1f tok/s, read %.0f tok/s", s->last_wtps, s->last_ptps);
                    text_at(mem, F->tiny, GREY, sx, y + D(72), t);
                }
            }
            y += D(118);
            /* graphs: write, read; card utilization, power */
            gw = (w - D(16)) / 2;
            gh = D(86);
            swprintf(t, 200, L"%.1f", s->write_tps);
            graph_box(mem, x, y, gw, gh, L"Write tok/s", &s->wtps, 0, LIME, t);
            swprintf(t, 200, L"%.0f", s->read_tps);
            graph_box(mem, x + gw + D(16), y, gw, gh, L"Read tok/s (prompt)", &s->rtps, 0, CYAN, t);
            y += gh + D(14);
            if (c) {
                swprintf(t, 200, L"%.0f%%", c->util);
                graph_box(mem, x, y, gw, gh, L"GPU utilization", &c->util_h, 100, LIME, t);
                swprintf(t, 200, L"%.0f W  %.0f\u00B0C", c->power_w, c->temp);
                graph_box(mem, x + gw + D(16), y, gw, gh, L"Power", &c->power_h, c->limit_w > 0 ? c->limit_w : 0, AMBER, t);
            } else {
                text_at(mem, F->tiny, DIM, x, y + D(20), s->kind == K_LAN ? (s->has_gpu ? L"Card telemetry: this PC's cards only (NVML); the server reports its GPU memory."
                                                                                     : L"Card telemetry: this PC's cards only (NVML); this server reports no GPU fields in /health.")
                                                                         : L"Card: the server does not say which card it uses (gpu_pci in /health).");
                gh = D(40);
            }
            y += gh + D(16);
            /* details: two columns of the /health fields */
            {
                int cw = (w - D(24)) / 2, x2 = x + cw + D(24), y0 = y;
                wchar_t m[64];
                label_at(mem, x, y, GREY, L"Request");
                label_at(mem, x2, y, GREY, L"Server");
                y += D(20);
                fmt_int(a, 32, s->read); fmt_int(b, 32, s->to_read);
                swprintf(t, 200, L"%s / %s", a, b);
                kv(mem, x, y, cw, L"prompt read", t);
                fmt_int(a, 32, s->reused);
                kv(mem, x, y + D(17), cw, L"reused from the cache", a);
                fmt_int(a, 32, s->generated);
                kv(mem, x, y + D(34), cw, L"tokens written", a);
                swprintf(t, 200, L"%.0f s", s->elapsed);
                kv(mem, x, y + D(51), cw, L"elapsed", t);
                swprintf(t, 200, L"%.0f chat, %.0f background, %.0f yields", s->q_chat, s->q_bg, s->yields);
                kv(mem, x, y + D(68), cw, L"queue", t);
                if (s->draft_on) {
                    utf8_to_w(s->draft_method[0] ? s->draft_method : (s->dspark ? "dspark" : "on"), m, 64);
                    if (s->accept >= 0) swprintf(t, 200, L"%s, %.0f%% accepted", m, s->accept * 100);
                    else swprintf(t, 200, L"%s", m);
                } else wcscpy(t, L"off");
                kv(mem, x, y + D(85), cw, L"drafting", t);
                fmt_int(a, 32, s->ctx);
                kv(mem, x2, y, cw, L"context", a);
                utf8_to_w(s->head, m, 64);
                kv(mem, x2, y + D(17), cw, L"output head", s->has_head ? m : L"not reported");
                utf8_to_w(s->device, m, 64);
                kv(mem, x2, y + D(34), cw, L"device", s->has_device ? m : L"not reported");
                swprintf(t, 200, L"%.0f h %02.0f min", floor(s->uptime / 3600), fmod(floor(s->uptime / 60), 60));
                kv(mem, x2, y + D(51), cw, L"uptime", s->has_uptime ? t : L"not reported");
                fmt_int(a, 32, s->requests);
                fmt_int(b, 32, s->completion_total);
                swprintf(t, 200, L"%s requests, %s tokens written", a, b);
                kv(mem, x2, y + D(68), cw, L"since start", s->has_stats ? t : L"not reported");
                fmt_int(a, 32, s->cached_total);
                fmt_int(b, 32, s->prompt_total);
                swprintf(t, 200, L"%s of %s prompt tokens", a, b);
                kv(mem, x2, y + D(85), cw, L"from the cache", s->has_stats ? t : L"not reported");
                if (s->gpu_total > 0) {
                    swprintf(t, 200, L"%.0f / %.0f MiB", s->gpu_total - s->gpu_free, s->gpu_total);
                    kv(mem, x2, y + D(102), cw, L"GPU memory (the server)", t);
                }
                if (s->experts[0]) {
                    utf8_to_w(s->experts, m, 64);
                    kv(mem, x2, y + D(102), cw, L"experts", m);
                    utf8_to_w(s->calib[0] ? s->calib : "not reported", m, 64);
                    kv(mem, x2, y + D(119), cw, L"expert-card calibration", m);
                }
                if (s->kind == K_NATIVE && s->pid) {
                    swprintf(t, 200, L"PID %lu, %.0f MB, %.0f%% CPU", s->pid, s->proc_mb, s->proc_cpu);
                    kv(mem, x, y + D(102), cw, L"process", t);
                }
                if (s->models_401) kv(mem, x, y + D(119), cw, L"/v1/models", L"needs an API key");
                y = y0 + D(146);
            }
            /* slots */
            if (s->slots_n > 1) {
                int k;
                swprintf(t, 200, L"Slots  %d of %d busy", s->slots_busy, s->slots_n);
                label_at(mem, x, y, GREY, t);
                y += D(20);
                for (k = 0; k < s->slots_n && k < MAX_SLOT; k++) {
                    const SlotInfo *q = &s->slot[k];
                    wchar_t ph[24];
                    utf8_to_w(q->phase[0] ? q->phase : "idle", ph, 24);
                    swprintf(t, 200, L"slot %d   %s   read %.0f/%.0f   written %.0f   %.0f s", k, ph, q->read, q->to_read, q->generated, q->elapsed);
                    text_at(mem, F->tiny, strcmp(q->phase, "idle") && q->phase[0] ? PAPER : DIM, x, y, t);
                    y += D(17);
                }
                y += D(8);
            }
            /* controls */
        controls:
            {
                int can = s->kindi && s->kind != K_LAN && s->kind != K_UNKNOWN && !st->busy_control;
                int bw = D(96);
                dbutton(mem, x, y, bw, D(30), L"START", can && !s->online, H_START, sel);
                dbutton(mem, x + bw + D(8), y, bw, D(30), L"STOP", can && s->online, H_STOP, sel);
                dbutton(mem, x + 2 * (bw + D(8)), y, bw, D(30), L"RESTART", can && s->online, H_RESTART, sel);
                dbutton(mem, x + 3 * (bw + D(8)), y, D(120), D(30), L"MODEL \x25BE", can, H_MODEL, sel);
                dbutton(mem, x + 3 * (bw + D(8)) + D(128), y, D(80), D(30), L"LOGS", can, H_LOGS, sel);
                bw += D(44);   /* the note after the buttons moves right of LOGS */
                if (!s->kindi) text_at(mem, F->tiny, DIM, x + 3 * (bw + D(8)) + D(132), y + D(8), s->online ? L"not Kindi: no controls" : L"no controls until it answers as Kindi");
                else if (s->kind == K_LAN) text_at(mem, F->tiny, DIM, x + 3 * (bw + D(8)) + D(132), y + D(8), L"LAN server: telemetry only");
                else if (st->busy_control) text_at(mem, F->tiny, LIME, x + 3 * (bw + D(8)) + D(132), y + D(8), L"working...");
                if (s->source == SRC_MANUAL) dbutton(mem, x + w - D(96), y, D(96), D(30), L"REMOVE", 1, H_REMOVE, sel);
                y += D(42);
            }
            /* every field of /health, as the server sends it (DeepSeek's state, last and drafting fields too) */
            if (s->online && s->nfields) {
                int cols = w >= D(760) ? 3 : 2, cw = (w - (cols - 1) * D(16)) / cols, rows = (s->nfields + cols - 1) / cols;
                int room = (H - D(48) - D(16) * 4 - y - D(20)) / D(15), k;
                label_at(mem, x, y, GREY, L"/health, every field");
                y += D(18);
                if (rows > room) rows = room > 0 ? room : 0;
                for (k = 0; k < s->nfields && k < rows * cols; k++) {
                    wchar_t f[96];
                    RECT r;
                    utf8_to_w(s->fields[k], f, 96);
                    r.left = x + (k / rows) * (cw + D(16));
                    r.top = y + (k % rows) * D(15);
                    r.right = r.left + cw;
                    r.bottom = r.top + D(15);
                    SelectObject(mem, F->tiny);
                    SetTextColor(mem, GREY);
                    DrawTextW(mem, f, -1, &r, DT_LEFT | DT_SINGLELINE | DT_NOPREFIX | DT_END_ELLIPSIS);
                }
                if (rows) y += rows * D(15) + D(10);
                if (k < s->nfields) { swprintf(t, 200, L"(%d more fields: make the window taller)", s->nfields - k); text_at(mem, F->tiny, DIM, x, y - D(8), t); y += D(10); }
            }
            /* the action log */
        action_log:
            {
                int k, first = st->nlog - (H - y - D(56)) / D(16);
                if (first < 0) first = 0;
                for (k = first; k < st->nlog; k++) { text_at(mem, F->tiny, GREY, x, y, st->log[k]); y += D(16); }
            }
        }
    }
    /* the bottom strip: the host */
    {
        int by = H - D(40), bx = left_w + pad, w = W - bx - pad, cw = w / 5;
        fill(mem, left_w + 1, by - D(8), W, H, PANEL);
        hline(mem, left_w + 1, W, by - D(8), EDGE);
        swprintf(t, 200, L"CPU %.0f%%", st->host.cpu);
        text_at(mem, F->tiny, GREY, bx, by, t);
        spark(mem, bx + D(64), by + D(1), cw - D(76), D(16), &st->host.cpu_h, 100, CYAN);
        swprintf(t, 200, L"RAM %.1f / %.0f GiB", st->host.ram_used_gb, st->host.ram_total_gb);
        text_at(mem, F->tiny, GREY, bx + cw, by, t);
        bar(mem, bx + cw, by + D(19), cw - D(16), D(3), st->host.ram_pct / 100, CYAN);
        swprintf(t, 200, L"COMMIT %.0f / %.0f GiB", st->host.commit_gb, st->host.commit_limit_gb);
        text_at(mem, F->tiny, GREY, bx + 2 * cw, by, t);
        bar(mem, bx + 2 * cw, by + D(19), cw - D(16), D(3), st->host.commit_pct / 100, st->host.commit_pct >= 90 ? RED : CYAN);
        swprintf(t, 200, L"DISK %.0f%%  %.0f GiB free", st->host.disk_pct, st->host.disk_free_gb);
        text_at(mem, F->tiny, GREY, bx + 3 * cw, by, t);
        bar(mem, bx + 3 * cw, by + D(19), cw - D(16), D(3), st->host.disk_pct / 100, DIM);
        swprintf(t, 200, L"NET \x2193 %.0f  \x2191 %.0f KiB/s", st->host.net_in_kbs, st->host.net_out_kbs);
        text_at(mem, F->tiny, GREY, bx + 4 * cw, by, t);
    }
}

/* ------------------------------------------------------------------ windows */
static void snapshot(State *out) {
    EnterCriticalSection(&g_lock);
    memcpy(out, &g_state, sizeof *out);
    LeaveCriticalSection(&g_lock);
}

static State *g_snap;   /* the UI thread's copy (heap: State is large) */

static int monitor_dpi(HWND hwnd) {
    UINT d = GetDpiForWindow(hwnd);
    return d >= 96 ? (int)d : 96;
}

static void update_ui_open(void) {
    LONG open = (g_popup && IsWindowVisible(g_popup)) || (g_dash && IsWindowVisible(g_dash) && !IsIconic(g_dash));
    if (InterlockedExchange(&g_ui_open, open) != open && open) SetEvent(g_wake);
}

static void wsl_open(const wchar_t *model) {
    EnterCriticalSection(&g_lock);
    if (g_state.wsl.stage != WS_INSTALLING && g_state.wsl.stage != WS_STARTING) {
        ZeroMemory(&g_state.wsl, sizeof g_state.wsl);
        wcsncpy(g_state.wsl.model, model, 47);
    }
    LeaveCriticalSection(&g_lock);
    g_wsl_view = 1;
    open_dash();
    wsl_check();
}

static void model_menu(HWND hwnd, int index) {
    HMENU m;
    POINT pt;
    int k, cmd, n = 0;
    char chosen[48] = "";
    snapshot(g_snap);
    if (index >= g_snap->nsrv) return;
    if (index >= 0 && !g_snap->srv[index].model_n && g_snap->srv[index].kind != K_WSL_UNIT) {   /* not read yet: ask for it */
        control(index, C_MODELS, NULL);
        remember(L"reading the model list; click MODEL again in a few seconds");
        return;
    }
    m = CreatePopupMenu();
    if (index >= 0) {
        n = g_snap->srv[index].model_n;
        for (k = 0; k < n; k++)
            AppendMenuW(m, MF_STRING | (strstr(g_snap->srv[index].model, g_snap->srv[index].model_ids[k]) ? MF_CHECKED : 0), 100 + k,
                        g_snap->srv[index].model_label[k]);
        if (n) AppendMenuW(m, MF_SEPARATOR, 0, NULL);
    }
    /* Kindi 0.7 runs these only in WSL2: the monitor installs or starts them there and attaches */
    AppendMenuW(m, MF_STRING | (index >= 0 && strstr(g_snap->srv[index].model, "deepseek") ? MF_CHECKED : 0), 300,
                L"deepseek-v4.1-flash\t552B MoE, runs in WSL");
    AppendMenuW(m, MF_STRING | MF_GRAYED, 301, L"flash-next\truns in WSL (later in 0.7)");
    GetCursorPos(&pt);
    SetForegroundWindow(hwnd);
    cmd = (int)TrackPopupMenu(m, TPM_RETURNCMD | TPM_RIGHTBUTTON, pt.x, pt.y, 0, hwnd, NULL);
    DestroyMenu(m);
    if (cmd == 300) { wsl_open(L"DeepSeek-V4.1-Flash"); return; }
    if (cmd >= 100 && cmd < 100 + n) {
        wchar_t q[160], w[48];
        copy_a(chosen, 48, g_snap->srv[index].model_ids[cmd - 100]);
        utf8_to_w(chosen, w, 48);
        swprintf(q, 160, L"Switch :%d to %s? The server restarts; a model that is not on this PC is downloaded first.",
                 g_snap->srv[index].port, w);
        if (MessageBoxW(hwnd, q, L"Kindi Monitor", MB_OKCANCEL | MB_ICONQUESTION) == IDOK) control(index, C_MODEL, chosen);
    }
}

static void confirm_control(HWND hwnd, int index, int op) {
    wchar_t q[200], title[64];
    const wchar_t *verb = op == C_STOP ? L"Stop" : op == C_START ? L"Start" : L"Restart";
    snapshot(g_snap);
    if (index < 0 || index >= g_snap->nsrv) return;
    server_title(&g_snap->srv[index], title, 64);
    swprintf(q, 200, L"%s %s on :%d (%s)?", verb, title, g_snap->srv[index].port, KIND_NAME[g_snap->srv[index].kind]);
    if (MessageBoxW(hwnd, q, L"Kindi Monitor", MB_OKCANCEL | MB_ICONQUESTION) == IDOK) control(index, op, NULL);
}

/* ------------------------------------------------------------------ WSL models (DeepSeek-V4.1-Flash; Flash-Next later)
   These models do not run natively on Windows in 0.7. The monitor checks what WSL needs, offers the Windows installer
   (install.ps1, which turns on WSL2 and runs the Linux installer with the DeepSeek plan), starts the server in WSL with
   its systemd user unit, and attaches to it on 127.0.0.1 (WSL2 forwards localhost). docs/dev/F8-WINDOWS.md, section 11. */
#define DSV_RAM_GIB 496      /* tools/install_dsv41.sh: the ram profile needs this much MemTotal */
#define DSV_DISK_GB 530      /* tools/install_dsv41.sh NEED_GB: the download and the converted drafter */
#define DSV_UNIT L"kindi-dsv41.service"

static void wsl_cmd(wchar_t *out, int cap, const wchar_t *distro, const wchar_t *user, const wchar_t *script) {
    /* wsl.exe -d D -u U -e sh -c "script" (the script must not contain double quotes) */
    swprintf(out, cap, L"wsl.exe -d %s%s%s -e sh -c \"%s\"", distro, user && user[0] ? L" -u " : L"", user && user[0] ? user : L"", script);
}

static int wsl_distros(wchar_t names[][64], int cap, int running_only) {
    char out[4096], *line;
    DWORD code = 1;
    int n = 0;
    if (!run_capture(running_only ? L"wsl.exe -l -q --running" : L"wsl.exe -l -q", out, sizeof out, 8000, &code) || code) return -1;
    for (line = strtok(out, "\r\n"); line && n < cap; line = strtok(NULL, "\r\n")) {
        if (!line[0] || !_strnicmp(line, "docker-desktop", 14)) continue;
        utf8_to_w(line, names[n], 64);
        n++;
    }
    return n;
}

/* the Kindi installer's choice: %LOCALAPPDATA%\Kindi\state.json {distro, user} */
static void installer_state(wchar_t *distro, wchar_t *user) {
    wchar_t path[MAX_PATH];
    char *text;
    int len;
    JNode *j;
    distro[0] = user[0] = 0;
    swprintf(path, MAX_PATH, L"%s\\state.json", g_kdir);
    text = read_file(path, &len);
    if (!text) return;
    /* PowerShell writes UTF-8 with a byte-order mark */
    j = json_parse(len >= 3 && (unsigned char)text[0] == 0xEF ? text + 3 : text, len >= 3 && (unsigned char)text[0] == 0xEF ? len - 3 : len);
    utf8_to_w(json_str(j, "distro", ""), distro, 64);
    utf8_to_w(json_str(j, "user", ""), user, 32);
    json_free(j);
    free(text);
}

static void add_check(WslFlow *w, int state, const wchar_t *name, const wchar_t *fmt, ...) {
    va_list ap;
    if (w->ncheck >= 12) return;
    w->check[w->ncheck].state = state;
    wcsncpy(w->check[w->ncheck].name, name, 47);
    va_start(ap, fmt);
    _vsnwprintf(w->check[w->ncheck].why, 239, fmt, ap);
    va_end(ap);
    w->check[w->ncheck].why[239] = 0;
    w->ncheck++;
}

static DWORD WINAPI wsl_check_thread(void *arg) {
    WslFlow w;
    wchar_t distros[16][64], probe[64] = L"", cmd[1024], user[32] = L"";
    char out[4096];
    DWORD code = 1;
    int nd, i, have_target = 0, gpu_ok = 0, disk_ok = 1, installed = 0;
    MEMORYSTATUSEX ms;
    (void)arg;
    ZeroMemory(&w, sizeof w);
    w.stage = WS_CHECKING;
    EnterCriticalSection(&g_lock);
    wcscpy(w.model, g_state.wsl.model);
    g_state.wsl.stage = WS_CHECKING;
    g_state.wsl.ncheck = 0;
    LeaveCriticalSection(&g_lock);
    /* 1. WSL2 */
    if (!run_capture(L"wsl.exe --status", out, sizeof out, 10000, &code) || code) {
        add_check(&w, CK_FAIL, L"WSL2", L"WSL is off or missing. The installer turns it on: Windows asks for administrator approval, then a restart.");
        w.wsl_off = 1;
    } else add_check(&w, CK_OK, L"WSL2", L"WSL is on.");
    /* 2. the distribution: the installer's (state.json), else 'Kindi' */
    installer_state(w.distro, w.user);
    nd = w.wsl_off ? 0 : wsl_distros(distros, 16, 0);
    if (!w.distro[0]) wcscpy(w.distro, L"Kindi");
    for (i = 0; i < nd; i++) if (!_wcsicmp(distros[i], w.distro)) have_target = 1;
    if (have_target) {
        add_check(&w, CK_OK, L"Distribution", L"'%s' (the Kindi installer's)%s%s.", w.distro, w.user[0] ? L", user " : L"", w.user);
        wcscpy(probe, w.distro);
        wcscpy(user, w.user);
    } else {
        w.need_install = 1;
        add_check(&w, CK_FAIL, L"Distribution", L"No '%s' distribution yet: the installer makes it (Ubuntu 24.04 from Canonical, sha256-checked)%s%s.",
                  w.distro, nd > 0 ? L". Your " : L"", nd > 0 ? L"other distributions stay as they are" : L"");
        if (nd > 0) wcscpy(probe, distros[0]);   /* the checks below look through an existing one */
    }
    /* 3. the GPU in WSL, 4. RAM, 5. disk, 6. installed: one call into the distribution (it starts it if stopped) */
    ms.dwLength = sizeof ms;
    GlobalMemoryStatusEx(&ms);
    if (probe[0]) {
        wsl_cmd(cmd, 1024, probe, user, L"echo GPU; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -8;"
                                        L" echo MEM $(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo);"
                                        L" echo DISK $(df -BG --output=avail ~ | tail -1 | tr -dc 0-9);"
                                        L" echo UNIT $(test -f ~/.config/systemd/user/kindi-dsv41.service && echo yes || echo no)");
        if (run_capture(cmd, out, sizeof out, 60000, &code)) {
            char *gpu = strstr(out, "GPU\n"), *mem = strstr(out, "MEM "), *disk = strstr(out, "DISK "), *unit = strstr(out, "UNIT ");
            int wsl_ram = mem ? atoi(mem + 4) : 0, wsl_disk = disk ? atoi(disk + 5) : 0;
            double win_ram = ms.ullTotalPhys / 1073741824.0;
            wchar_t g[200] = L"";
            if (gpu && mem && mem > gpu + 4) {
                int n = (int)(mem - gpu - 4);
                char tmp[200];
                if (n > 199) n = 199;
                memcpy(tmp, gpu + 4, n);
                tmp[n] = 0;
                while (n > 0 && (tmp[n - 1] == '\n' || tmp[n - 1] == '\r')) tmp[--n] = 0;
                for (i = 0; tmp[i]; i++) if (tmp[i] == '\n') tmp[i] = ';';
                utf8_to_w(tmp, g, 200);
                gpu_ok = strstr(tmp, "NVIDIA") != NULL || strstr(tmp, "MiB") != NULL;
            }
            if (gpu_ok) add_check(&w, CK_OK, L"GPU in WSL", L"nvidia-smi in '%s': %s", probe, g);
            else add_check(&w, CK_FAIL, L"GPU in WSL", L"nvidia-smi in '%s' sees no NVIDIA card (%s). Update the Windows NVIDIA driver; WSL2 GPU needs Windows 10 21H2 or 11.", probe, g[0] ? g : L"no output");
            if (wsl_ram >= DSV_RAM_GIB)
                add_check(&w, CK_OK, L"RAM", L"WSL has %d GiB: the ram profile (the whole model in the page cache, about 18 tok/s) needs %d.", wsl_ram, DSV_RAM_GIB);
            else
                add_check(&w, CK_WARN, L"RAM", L"Not enough RAM for the ram profile: WSL has %d GiB (Windows %.0f GiB), the ram profile needs %d. "
                          L"Only the nvme profile fits: the experts are streamed from an NVMe SSD, much slower.", wsl_ram, win_ram, DSV_RAM_GIB);
            disk_ok = wsl_disk >= DSV_DISK_GB;
            if (have_target || !w.need_install)
                add_check(&w, disk_ok ? CK_OK : CK_FAIL, L"Disk", L"%d GB free in the WSL home folder; the download and the drafter need about %d GB.%s",
                          wsl_disk, DSV_DISK_GB, disk_ok ? L"" : L" Not enough disk.");
            installed = unit && !strncmp(unit + 5, "yes", 3) && have_target;
        } else add_check(&w, CK_FAIL, L"GPU in WSL", L"could not run a command in '%s'", probe);
    } else {
        add_check(&w, CK_INFO, L"GPU in WSL", L"checked after the installer made the distribution");
        add_check(&w, (ms.ullTotalPhys / 1073741824.0) >= DSV_RAM_GIB ? CK_OK : CK_WARN, L"RAM",
                  L"Windows has %.0f GiB; the ram profile needs %d GiB in WSL, else the nvme profile (slower).", ms.ullTotalPhys / 1073741824.0, DSV_RAM_GIB);
    }
    {   /* the Windows drive that holds the distributions' disks (the installer's default: %LOCALAPPDATA%\Kindi\wsl) */
        ULARGE_INTEGER avail, total, freeb;
        if (GetDiskFreeSpaceExW(g_kdir, &avail, &total, &freeb)) {
            double gb = avail.QuadPart / 1e9;
            int ok = gb >= DSV_DISK_GB;
            if (!ok) disk_ok = 0;
            add_check(&w, ok ? CK_OK : CK_FAIL, L"Windows disk", L"%.0f GB free on the drive of %s; the WSL disk grows by about %d GB.%s",
                      gb, g_kdir, DSV_DISK_GB, ok ? L"" : L" Not enough disk: free space or install with -DistroDir on another drive.");
        }
    }
    if (installed) add_check(&w, CK_OK, L"DeepSeek", L"installed in '%s' (kindi-dsv41.service).", w.distro);
    else { w.need_install = 1; add_check(&w, CK_INFO, L"DeepSeek", L"not installed: the installer downloads about 494 GB (the model) and the drafter."); }
    {   /* native Kindi servers on this PC hold cards that the DeepSeek plan wants */
        int k, nat = 0;
        EnterCriticalSection(&g_lock);
        for (k = 0; k < g_state.nsrv; k++) if (g_state.srv[k].kind == K_NATIVE && g_state.srv[k].online && !g_state.srv[k].gone) nat++;
        LeaveCriticalSection(&g_lock);
        if (nat) add_check(&w, CK_WARN, L"Native servers", L"%d native Kindi server(s) run on the cards; they are stopped first (the monitor asks).", nat);
    }
    w.gpu_ok = gpu_ok;
    w.disk_ok = disk_ok;
    w.installed = installed;
    w.can_install = disk_ok && !installed;
    w.can_start = installed && gpu_ok;
    w.stage = WS_CHECKED;
    swprintf(w.status, 160, installed ? L"Ready to start in WSL." : disk_ok ? L"Not installed: INSTALL runs the Windows installer." : L"Cannot install here: not enough disk.");
    EnterCriticalSection(&g_lock);
    w.install_pid = g_state.wsl.install_pid;
    g_state.wsl = w;
    LeaveCriticalSection(&g_lock);
    PostMessageW(g_main, WM_DATA, 0, 0);
    return 0;
}

static void wsl_check(void) {
    HANDLE t;
    EnterCriticalSection(&g_lock);
    if (g_state.wsl.stage == WS_CHECKING || g_state.wsl.stage == WS_STARTING) { LeaveCriticalSection(&g_lock); return; }
    g_state.wsl.stage = WS_CHECKING;
    LeaveCriticalSection(&g_lock);
    t = CreateThread(NULL, 0, wsl_check_thread, NULL, 0, NULL);
    if (t) CloseHandle(t);
}

/* the installer, in its own console window (it asks questions, and Windows asks for approval when WSL is off) */
static void wsl_install(HWND hwnd) {
    wchar_t ps1[MAX_PATH], cmd[1024], q[900];
    WslFlow w;
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    snapshot(g_snap);
    w = g_snap->wsl;
    swprintf(ps1, MAX_PATH, L"%s\\install.ps1", g_exe_dir);
    if (hwnd && GetFileAttributesW(ps1) == INVALID_FILE_ATTRIBUTES) { remember(L"install.ps1 is missing next to the monitor"); return; }
    if (hwnd && !w.disk_ok) {
        MessageBoxW(hwnd, L"There is not enough disk for DeepSeek-V4.1-Flash: it needs about 530 GB free, in WSL and on the "
                          L"Windows drive that holds the WSL disk. Nothing was changed.", L"Kindi Monitor", MB_OK | MB_ICONWARNING);
        return;
    }
    swprintf(q, 900, L"Install DeepSeek-V4.1-Flash in WSL?\n\nThe Kindi Windows installer (install.ps1) runs in a new window:\n"
                     L"%s- it makes or reuses the '%s' WSL distribution, with Docker and the NVIDIA toolkit;\n"
                     L"- then the Linux installer with the DeepSeek plan (install.sh --place dsv41:0+) downloads about 530 GB "
                     L"and installs kindi-dsv41.service.\n\nIt asks before each change. Its log: %s\\install.log",
                 w.wsl_off ? L"- WSL is off: it turns WSL2 on. Windows asks for ADMINISTRATOR approval, and the PC restarts;\n" : L"",
                 w.distro, g_kdir);
    if (hwnd && MessageBoxW(hwnd, q, L"Kindi Monitor", MB_OKCANCEL | MB_ICONQUESTION) != IDOK) return;
    swprintf(cmd, 1024, L"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"%s\" -Distro %s -LinuxArgs \"--place dsv41:0+\"", ps1, w.distro);
    if (!hwnd) GetEnvironmentVariableW(L"KINDI_TEST_INSTALL_CMD", cmd, 1024);   /* tests: a stand-in for install.ps1 */
    ZeroMemory(&si, sizeof si);
    si.cb = sizeof si;
    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, g_exe_dir, &si, &pi)) {
        remember(L"could not start the installer (error %lu)", GetLastError());
        return;
    }
    CloseHandle(pi.hThread);
    EnterCriticalSection(&g_lock);
    g_state.wsl.stage = WS_INSTALLING;
    g_state.wsl.install_pid = pi.dwProcessId;
    g_state.wsl.install_proc = pi.hProcess;
    wcscpy(g_state.wsl.status, L"Installing: the installer runs in its own window.");
    LeaveCriticalSection(&g_lock);
    remember(L"installer started (PID %lu): install.ps1 -LinuxArgs '--place dsv41:0+'", pi.dwProcessId);
}

/* the installer's progress: its last step ("[n] ...") and last line, from install.log; the end of its process */
static void wsl_install_progress(void) {
    wchar_t path[MAX_PATH];
    char *text, *p, *step = NULL, *last = NULL;
    int len;
    HANDLE proc;
    EnterCriticalSection(&g_lock);
    proc = g_state.wsl.stage == WS_INSTALLING ? g_state.wsl.install_proc : NULL;
    LeaveCriticalSection(&g_lock);
    if (!proc) return;
    swprintf(path, MAX_PATH, L"%s\\install.log", g_kdir);
    text = read_file(path, &len);
    if (text) {
        for (p = strtok(text, "\r\n"); p; p = strtok(NULL, "\r\n")) {
            while (*p == ' ') p++;
            if (!*p || !strncmp(p, "****", 4)) continue;
            if (p[0] == '[' && p[1] >= '0' && p[1] <= '9') step = p;
            last = p;
        }
    }
    EnterCriticalSection(&g_lock);
    if (step) utf8_to_w(step, g_state.wsl.install_step, 160);
    if (last) utf8_to_w(last, g_state.wsl.install_last, 160);
    if (WaitForSingleObject(proc, 0) == WAIT_OBJECT_0) {
        DWORD code = 1;
        GetExitCodeProcess(proc, &code);
        CloseHandle(proc);
        g_state.wsl.install_proc = NULL;
        g_state.wsl.stage = WS_CHECKED;
        swprintf(g_state.wsl.status, 160, code ? L"The installer stopped (exit code %lu); its log is %s\\install.log." : L"The installer finished.", code, g_kdir);
        LeaveCriticalSection(&g_lock);
        remember(code ? L"installer ended with exit code %lu" : L"installer finished", code);
        wsl_check();
    } else LeaveCriticalSection(&g_lock);
    free(text);
}

static DWORD WINAPI wsl_start_thread(void *arg) {
    wchar_t cmd[512], distro[64], user[32];
    char out[2048];
    DWORD code = 1;
    Server *s;
    int idx = -1, k;
    (void)arg;
    EnterCriticalSection(&g_lock);
    wcscpy(distro, g_state.wsl.distro);
    wcscpy(user, g_state.wsl.user);
    s = add_server(&g_state, L"127.0.0.1", DSV_PORT, SRC_PLAN);
    if (s) {
        idx = (int)(s - g_state.srv);
        s->kind = K_WSL_UNIT;
        s->kindi = 1;
        s->gone = 0;
        wcscpy(s->name, DSV_UNIT);
        wcscpy(s->distro, distro);
        wcscpy(s->wsl_user, user);
        copy_a(s->life, sizeof s->life, "starting (warming model)");
    }
    LeaveCriticalSection(&g_lock);
    swprintf(cmd, 512, L"wsl.exe -d %s%s%s -e systemctl --user start %s", distro, user[0] ? L" -u " : L"", user, DSV_UNIT);
    remember(L"start %s in '%s'", DSV_UNIT, distro);
    if (!run_capture(cmd, out, sizeof out, 120000, &code) || code) {
        wchar_t e[100];
        utf8_to_w(out, e, 100);
        remember(L"start failed (exit %lu): %s", code, e);
        EnterCriticalSection(&g_lock);
        if (idx >= 0) g_state.srv[idx].life[0] = 0;
        g_state.wsl.stage = WS_CHECKED;
        wcscpy(g_state.wsl.status, L"The start failed; see the action log.");
        LeaveCriticalSection(&g_lock);
        return 0;
    }
    /* the warm-up reads the model into the page cache: minutes on an SSD, much longer from a hard disk */
    for (k = 0; k < 3600; k++) {
        int status = 0;
        char *body = NULL;
        int up = http_get(L"127.0.0.1", DSV_PORT, L"/health", 800, &body, &status) && status == 200;
        free(body);
        if (up) break;
        if (WaitForSingleObject(g_stop, 1000) == WAIT_OBJECT_0) return 0;
    }
    EnterCriticalSection(&g_lock);
    if (idx >= 0) g_state.srv[idx].life[0] = 0;
    g_state.wsl.stage = WS_ATTACHED;
    swprintf(g_state.wsl.status, 160, k < 3600 ? L"Running in WSL; attached on 127.0.0.1:%d." : L"It did not answer within an hour.", DSV_PORT);
    g_selected = idx >= 0 ? idx : g_selected;
    LeaveCriticalSection(&g_lock);
    remember(k < 3600 ? L"DeepSeek answers on :%d after %d s" : L"DeepSeek did not answer on :%d", DSV_PORT, k);
    InterlockedExchange(&g_want_discover, 1);
    SetEvent(g_wake);
    return 0;
}

static void wsl_start(HWND hwnd) {
    int k;
    HANDLE t;
    snapshot(g_snap);
    if (!g_snap->wsl.can_start) return;
    /* native servers first: they hold cards that the DeepSeek plan uses (one question each) */
    for (k = 0; k < g_snap->nsrv; k++) {
        const Server *v = &g_snap->srv[k];
        wchar_t q[200], title[64];
        if (v->kind != K_NATIVE || !v->online || v->gone) continue;
        server_title(v, title, 64);
        swprintf(q, 200, L"Stop the native server %s on :%d first? It uses a card that DeepSeek in WSL needs.", title, v->port);
        if (MessageBoxW(hwnd, q, L"Kindi Monitor", MB_YESNO | MB_ICONQUESTION) != IDYES) { remember(L"DeepSeek not started: the native server on :%d stays", v->port); return; }
        native_stop((Server *)v);
    }
    EnterCriticalSection(&g_lock);
    g_state.wsl.stage = WS_STARTING;
    wcscpy(g_state.wsl.status, L"Starting in WSL (warming the model: it reads the model into memory).");
    LeaveCriticalSection(&g_lock);
    t = CreateThread(NULL, 0, wsl_start_thread, NULL, 0, NULL);
    if (t) CloseHandle(t);
}

/* the dashboard panel: the checks, then INSTALL / START */
static void draw_wsl_panel(HDC mem, int x, int y, int w, int H, const State *st) {
    const WslFlow *f = &st->wsl;
    wchar_t t[300];
    int k, bw = D(120);
    static const COLORREF col[] = { LIME, AMBER, RED, CYAN };
    static const wchar_t *mark[] = { L"OK", L"WARN", L"NO", L"INFO" };
    swprintf(t, 300, L"%s in WSL", f->model[0] ? f->model : L"DeepSeek-V4.1-Flash");
    text_at(mem, F->light, PAPER, x, y, t);
    label_at(mem, x + w - D(120), y + D(8), f->stage == WS_INSTALLING ? AMBER : f->stage == WS_ATTACHED ? LIME : CYAN,
             f->stage == WS_CHECKING ? L"checking" : f->stage == WS_INSTALLING ? L"installing" : f->stage == WS_STARTING ? L"starting" :
             f->stage == WS_ATTACHED ? L"running" : L"checked");
    y += D(34);
    text_at(mem, F->tiny, GREY, x, y, L"This model does not run natively on Windows in Kindi 0.7: it runs in WSL2, and the monitor attaches to it.");
    y += D(26);
    if (f->stage == WS_CHECKING && !f->ncheck) { text_at(mem, F->body, GREY, x, y, L"Checking WSL, the GPU, RAM and disk..."); y += D(24); }
    for (k = 0; k < f->ncheck; k++) {
        const WslCheck *c = &f->check[k];
        RECT r = { x + D(170), y, x + w, y + D(36) };
        label_at(mem, x, y + D(2), col[c->state & 3], mark[c->state & 3]);
        text_at(mem, F->bold, PAPER, x + D(56), y, c->name);
        SelectObject(mem, F->tiny);
        SetTextColor(mem, GREY);
        DrawTextW(mem, c->why, -1, &r, DT_LEFT | DT_WORDBREAK | DT_NOPREFIX);
        y += D(38);
    }
    y += D(6);
    text_at(mem, F->body, PAPER, x, y, f->status);
    y += D(28);
    if (f->stage == WS_INSTALLING) {
        text_at(mem, F->tiny, GREY, x, y, f->install_step[0] ? f->install_step : L"(waiting for the installer's first step)");
        text_at(mem, F->tiny, DIM, x, y + D(16), f->install_last);
        y += D(40);
    }
    dbutton(mem, x, y, bw, D(30), L"CHECK AGAIN", f->stage == WS_CHECKED || f->stage == WS_ATTACHED, H_WSL_CHECK, 0);
    dbutton(mem, x + bw + D(8), y, bw, D(30), L"INSTALL", f->stage == WS_CHECKED && f->can_install, H_WSL_INSTALL, 0);
    dbutton(mem, x + 2 * (bw + D(8)), y, bw, D(30), L"START", f->stage == WS_CHECKED && f->can_start, H_WSL_START, 0);
    dbutton(mem, x + 3 * (bw + D(8)), y, bw, D(30), L"CLOSE", 1, H_WSL_CLOSE, 0);
    (void)H;
}

static void open_dash(void);

/* ---- add a server by its URL: a small dialog; the server goes to [servers] in monitor.ini */
static HWND g_add, g_add_edit;
static HFONT g_add_font;

static int parse_url(const wchar_t *in, wchar_t *host, int cap, int *port) {
    wchar_t buf[256], *p = buf, *colon, *slash;
    wcsncpy(buf, in, 255);
    buf[255] = 0;
    while (*p == L' ') p++;
    if (!_wcsnicmp(p, L"http://", 7)) p += 7;
    else if (!_wcsnicmp(p, L"https://", 8)) return 0;   /* the monitor speaks plain HTTP, as Kindi serves it */
    slash = wcschr(p, L'/');
    if (slash) *slash = 0;
    colon = wcsrchr(p, L':');
    *port = 8093;
    if (colon) { *port = _wtoi(colon + 1); *colon = 0; }
    if (!*p || *port <= 0 || *port > 65535 || wcslen(p) >= (size_t)cap) return 0;
    wcscpy(host, p);
    return 1;
}

static void add_commit(void) {
    wchar_t text[256], host[64], key[96], val[96];
    int port;
    GetWindowTextW(g_add_edit, text, 256);
    if (!parse_url(text, host, 64, &port)) {
        MessageBoxW(g_add, L"Type the URL of the server, for example http://myserver:8101 (plain HTTP).", L"Kindi Monitor",
                    MB_OK | MB_ICONINFORMATION);
        return;
    }
    swprintf(key, 96, L"%s_%d", host, port);
    swprintf(val, 96, L"%s:%d", host, port);
    WritePrivateProfileStringW(L"servers", key, val, g_ini);
    remember(L"added %s:%d", host, port);
    DestroyWindow(g_add);
    InterlockedExchange(&g_want_discover, 1);
    SetEvent(g_wake);
}

static LRESULT CALLBACK add_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_COMMAND:
        if (LOWORD(wp) == IDOK) add_commit();
        else if (LOWORD(wp) == IDCANCEL) DestroyWindow(hwnd);
        return 0;
    case WM_CTLCOLORSTATIC:
        SetTextColor((HDC)wp, GREY);
        SetBkColor((HDC)wp, RGB(0, 0, 0));
        return (LRESULT)GetStockObject(BLACK_BRUSH);
    case WM_DESTROY:
        g_add = NULL;
        if (g_add_font) { DeleteObject(g_add_font); g_add_font = NULL; }
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static void add_dialog(HWND owner) {
    int dpi = monitor_dpi(owner), w = MulDiv(440, dpi, 96), h = MulDiv(170, dpi, 96);
    RECT r;
    HWND label, ok, cancel;
    BOOL dark = TRUE;
    if (g_add) { SetForegroundWindow(g_add); return; }
    GetWindowRect(owner, &r);
    g_add = CreateWindowExW(WS_EX_DLGMODALFRAME, L"KindiMonitorAdd", L"Add a Kindi server", WS_POPUP | WS_CAPTION | WS_SYSMENU,
                            r.left + MulDiv(60, dpi, 96), r.top + MulDiv(60, dpi, 96), w, h, owner, NULL, g_inst, NULL);
    if (!g_add) return;
    DwmSetWindowAttribute(g_add, 20, &dark, sizeof dark);
    g_add_font = make_font(L"Segoe UI", 12, FW_NORMAL, dpi);
    label = CreateWindowExW(0, L"STATIC", L"Server URL (for example http://myserver:8101):", WS_CHILD | WS_VISIBLE,
                            MulDiv(16, dpi, 96), MulDiv(14, dpi, 96), w - MulDiv(32, dpi, 96), MulDiv(20, dpi, 96), g_add, NULL, g_inst, NULL);
    g_add_edit = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"http://", WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL,
                                 MulDiv(16, dpi, 96), MulDiv(40, dpi, 96), w - MulDiv(40, dpi, 96), MulDiv(26, dpi, 96), g_add, NULL, g_inst, NULL);
    ok = CreateWindowExW(0, L"BUTTON", L"Add", WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON,
                         w - MulDiv(204, dpi, 96), MulDiv(82, dpi, 96), MulDiv(88, dpi, 96), MulDiv(28, dpi, 96), g_add, (HMENU)IDOK, g_inst, NULL);
    cancel = CreateWindowExW(0, L"BUTTON", L"Cancel", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
                             w - MulDiv(108, dpi, 96), MulDiv(82, dpi, 96), MulDiv(88, dpi, 96), MulDiv(28, dpi, 96), g_add, (HMENU)IDCANCEL, g_inst, NULL);
    SendMessageW(label, WM_SETFONT, (WPARAM)g_add_font, TRUE);
    SendMessageW(g_add_edit, WM_SETFONT, (WPARAM)g_add_font, TRUE);
    SendMessageW(ok, WM_SETFONT, (WPARAM)g_add_font, TRUE);
    SendMessageW(cancel, WM_SETFONT, (WPARAM)g_add_font, TRUE);
    SendMessageW(g_add_edit, EM_SETSEL, 7, 7);
    ShowWindow(g_add, SW_SHOW);
    SetFocus(g_add_edit);
}

static void remove_server(HWND hwnd, int index) {
    wchar_t q[160], buf[2048], *p, want[96], key[96];
    snapshot(g_snap);
    if (index < 0 || index >= g_snap->nsrv || g_snap->srv[index].source != SRC_MANUAL) return;
    swprintf(q, 160, L"Remove %s:%d from the monitor?", g_snap->srv[index].host, g_snap->srv[index].port);
    if (MessageBoxW(hwnd, q, L"Kindi Monitor", MB_OKCANCEL | MB_ICONQUESTION) != IDOK) return;
    /* the [servers] entry with this host:port, whatever its name */
    swprintf(want, 96, L"%s:%d", g_snap->srv[index].host, g_snap->srv[index].port);
    GetPrivateProfileSectionW(L"servers", buf, 2048, g_ini);
    for (p = buf; *p; p += wcslen(p) + 1) {
        wchar_t *eq = wcschr(p, L'=');
        if (eq && !_wcsicmp(eq + 1, want)) {
            int n = min(95, (int)(eq - p));
            wcsncpy(key, p, n);
            key[n] = 0;
            WritePrivateProfileStringW(L"servers", key, NULL, g_ini);
        }
    }
    EnterCriticalSection(&g_lock);
    g_state.srv[index].gone = 1;
    LeaveCriticalSection(&g_lock);
    remember(L"removed %s:%d", g_snap->srv[index].host, g_snap->srv[index].port);
    g_selected = 0;
}

static void click(HWND hwnd, const Hit *hits, int n, int x, int y) {
    int i;
    for (i = 0; i < n; i++) {
        POINT p = { x, y };
        if (!PtInRect(&hits[i].r, p)) continue;
        switch (hits[i].what) {
        case H_DASH: ShowWindow(g_popup, SW_HIDE); open_dash(); break;
        case H_SERVER: g_selected = hits[i].index; g_wsl_view = 0; if (hwnd == g_popup) { ShowWindow(g_popup, SW_HIDE); open_dash(); } break;
        case H_RESTART: confirm_control(hwnd, hits[i].index, C_RESTART); break;
        case H_START: confirm_control(hwnd, hits[i].index, C_START); break;
        case H_STOP: confirm_control(hwnd, hits[i].index, C_STOP); break;
        case H_MODEL: model_menu(hwnd, hits[i].index); break;
        case H_ADD: add_dialog(hwnd); break;
        case H_REMOVE: remove_server(hwnd, hits[i].index); break;
        case H_LOGS: control(hits[i].index, C_LOGS, NULL); break;
        case H_WSL_OPEN: wsl_open(L"DeepSeek-V4.1-Flash"); break;
        case H_WSL_CHECK: wsl_check(); break;
        case H_WSL_INSTALL: wsl_install(hwnd); break;
        case H_WSL_START: wsl_start(hwnd); break;
        case H_WSL_CLOSE: g_wsl_view = 0; break;
        }
        InvalidateRect(hwnd, NULL, FALSE);
        return;
    }
}

static void place_popup(void) {
    NOTIFYICONIDENTIFIER id;
    RECT icon, work;
    MONITORINFO mi;
    HMONITOR mon;
    POINT pt;
    int w, h, x, y;
    ZeroMemory(&id, sizeof id);
    id.cbSize = sizeof id;
    id.hWnd = g_main;
    id.uID = 1;
    if (Shell_NotifyIconGetRect(&id, &icon) != S_OK) { GetCursorPos(&pt); SetRect(&icon, pt.x, pt.y, pt.x, pt.y); }
    pt.x = (icon.left + icon.right) / 2;
    pt.y = (icon.top + icon.bottom) / 2;
    mon = MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST);
    mi.cbSize = sizeof mi;
    GetMonitorInfoW(mon, &mi);
    work = mi.rcWork;
    {
        UINT dx = 96, dy = 96;
        HMODULE sh = LoadLibraryW(L"Shcore.dll");
        typedef HRESULT(WINAPI * Fn)(HMONITOR, int, UINT *, UINT *);
        Fn f = sh ? (Fn)GetProcAddress(sh, "GetDpiForMonitor") : NULL;
        if (f) f(mon, 0, &dx, &dy);
        if (sh) FreeLibrary(sh);
        if ((int)dx != g_dpi) { g_dpi = dx >= 96 ? (int)dx : 96; build_fonts(&g_fp, g_dpi); }
    }
    snapshot(g_snap);
    w = S(POPUP_W);
    h = S(popup_height(g_snap));
    if (h > work.bottom - work.top - S(16)) h = work.bottom - work.top - S(16);
    /* above the tray icon when the taskbar is at the bottom; otherwise next to it, inside the work area */
    x = pt.x - w / 2;
    y = icon.top >= work.bottom - S(4) ? work.bottom - h - S(8) : (icon.bottom <= work.top + S(4) ? work.top + S(8) : pt.y - h / 2);
    if (x + w > work.right - S(8)) x = work.right - w - S(8);
    if (x < work.left + S(8)) x = work.left + S(8);
    if (y < work.top + S(8)) y = work.top + S(8);
    SetWindowPos(g_popup, HWND_TOPMOST, x, y, w, h, SWP_SHOWWINDOW);
    SetForegroundWindow(g_popup);
    update_ui_open();
}

static LRESULT CALLBACK popup_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_PAINT: snapshot(g_snap); paint_with(hwnd, draw_popup, g_snap); return 0;
    case WM_ERASEBKGND: return 1;
    case WM_ACTIVATE:
        if (LOWORD(wp) == WA_INACTIVE && IsWindowVisible(hwnd)) { ShowWindow(hwnd, SW_HIDE); g_popup_hid = now_ms(); update_ui_open(); }
        return 0;
    case WM_LBUTTONUP: click(hwnd, g_hits, g_nhits, (short)LOWORD(lp), (short)HIWORD(lp)); return 0;
    case WM_KEYDOWN: if (wp == VK_ESCAPE) { ShowWindow(hwnd, SW_HIDE); update_ui_open(); } return 0;
    case WM_DPICHANGED: g_dpi = HIWORD(wp); build_fonts(&g_fp, g_dpi); InvalidateRect(hwnd, NULL, FALSE); return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static LRESULT CALLBACK dash_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_PAINT: snapshot(g_snap); paint_with(hwnd, draw_dash, g_snap); return 0;
    case WM_ERASEBKGND: return 1;
    case WM_SIZE: update_ui_open(); InvalidateRect(hwnd, NULL, FALSE); return 0;
    case WM_SHOWWINDOW: PostMessageW(hwnd, WM_APP + 9, 0, 0); break;
    case WM_APP + 9: update_ui_open(); return 0;
    case WM_GETMINMAXINFO: {
        MINMAXINFO *mm = (MINMAXINFO *)lp;
        mm->ptMinTrackSize.x = MulDiv(760, g_dash_dpi, 96);
        mm->ptMinTrackSize.y = MulDiv(560, g_dash_dpi, 96);
        return 0;
    }
    case WM_DPICHANGED: {
        RECT *r = (RECT *)lp;
        g_dash_dpi = HIWORD(wp);
        build_fonts(&g_fd, g_dash_dpi);
        SetWindowPos(hwnd, NULL, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
        InvalidateRect(hwnd, NULL, FALSE);
        return 0;
    }
    case WM_LBUTTONUP: click(hwnd, g_dhits, g_ndhits, (short)LOWORD(lp), (short)HIWORD(lp)); return 0;
    case WM_CLOSE: {   /* closing the dashboard hides it; the tray icon stays */
        WINDOWPLACEMENT wpl;
        wpl.length = sizeof wpl;
        if (GetWindowPlacement(hwnd, &wpl)) {
            ini_set_int(L"dash_x", wpl.rcNormalPosition.left);
            ini_set_int(L"dash_y", wpl.rcNormalPosition.top);
            ini_set_int(L"dash_w", wpl.rcNormalPosition.right - wpl.rcNormalPosition.left);
            ini_set_int(L"dash_h", wpl.rcNormalPosition.bottom - wpl.rcNormalPosition.top);
        }
        ShowWindow(hwnd, SW_HIDE);
        update_ui_open();
        return 0;
    }
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static void open_dash(void) {
    if (!g_dash) {
        int w = ini_int(L"dash_w", 0), h = ini_int(L"dash_h", 0), x = ini_int(L"dash_x", CW_USEDEFAULT), y = ini_int(L"dash_y", CW_USEDEFAULT);
        BOOL dark = TRUE;
        if (w <= 0 || h <= 0) { w = S(1100); h = S(780); }
        g_dash = CreateWindowExW(0, L"KindiMonitorDash", L"Kindi Monitor", WS_OVERLAPPEDWINDOW, x, y, w, h, NULL, NULL, g_inst, NULL);
        if (!g_dash) return;
        DwmSetWindowAttribute(g_dash, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &dark, sizeof dark);
        g_dash_dpi = monitor_dpi(g_dash);
        build_fonts(&g_fd, g_dash_dpi);
    }
    if (IsIconic(g_dash)) ShowWindow(g_dash, SW_RESTORE);
    ShowWindow(g_dash, SW_SHOW);
    SetForegroundWindow(g_dash);
    InterlockedExchange(&g_want_discover, 1);
    update_ui_open();
}

/* the tray icon: a lime K on ink, a lime dot while a server works, a red dot when one is offline */
static HICON make_icon(COLORREF state) {
    int size = GetSystemMetrics(SM_CXSMICON), i;
    HDC screen = GetDC(NULL), dc = CreateCompatibleDC(screen);
    BITMAPINFO bi;
    void *bits = NULL;
    HBITMAP color, mask, old;
    ICONINFO ii;
    HICON icon;
    HFONT f, of;
    RECT rc = { 0, 0, size, size };
    ZeroMemory(&bi, sizeof bi);
    bi.bmiHeader.biSize = sizeof bi.bmiHeader;
    bi.bmiHeader.biWidth = size;
    bi.bmiHeader.biHeight = -size;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    color = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &bits, NULL, 0);
    mask = CreateBitmap(size, size, 1, 1, NULL);
    old = (HBITMAP)SelectObject(dc, color);
    ZeroMemory(bits, (size_t)size * size * 4);   /* transparent: the mark's rounded corners keep their alpha */
    {   /* the Kindi mark (kindi.ico, resource 1) at the tray size; a K in Segoe UI if the resource is missing */
        HICON mark = (HICON)LoadImageW(g_inst, MAKEINTRESOURCEW(1), IMAGE_ICON, size, size, 0);
        if (mark) {
            DrawIconEx(dc, 0, 0, mark, size, size, 0, NULL, DI_NORMAL);
            DestroyIcon(mark);
        } else {
            f = CreateFontW(-size * 15 / 16, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                            CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH, L"Segoe UI");
            of = (HFONT)SelectObject(dc, f);
            SetBkMode(dc, TRANSPARENT);
            SetTextColor(dc, LIME);
            DrawTextW(dc, L"K", 1, &rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
            SelectObject(dc, of);
            DeleteObject(f);
            GdiFlush();
            for (i = 0; i < size * size; i++) ((DWORD *)bits)[i] |= 0xFF000000u;
        }
    }
    GdiFlush();
    if (state != INK) {   /* the state dot, bottom right, with an ink ring; written as pixels with their alpha */
        double r = size * 0.22, cx = size - r - 0.5, cy = size - r - 0.5;
        int x, y;
        for (y = 0; y < size; y++)
            for (x = 0; x < size; x++) {
                double d = sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                DWORD *px = (DWORD *)bits + y * size + x;
                if (d <= r - 1.0) *px = 0xFF000000u | ((DWORD)GetRValue(state) << 16) | ((DWORD)GetGValue(state) << 8) | GetBValue(state);
                else if (d <= r + 0.5) *px = 0xFF080808u;
            }
    }
    SelectObject(dc, old);
    ii.fIcon = TRUE;
    ii.xHotspot = ii.yHotspot = 0;
    ii.hbmMask = mask;
    ii.hbmColor = color;
    icon = CreateIconIndirect(&ii);
    DeleteObject(color);
    DeleteObject(mask);
    DeleteDC(dc);
    ReleaseDC(NULL, screen);
    return icon;
}

static void tray(int op, HICON icon, const wchar_t *tip) {
    NOTIFYICONDATAW n;
    ZeroMemory(&n, sizeof n);
    n.cbSize = sizeof n;
    n.hWnd = g_main;
    n.uID = 1;
    n.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP | NIF_SHOWTIP;
    n.uCallbackMessage = WM_TRAY;
    n.hIcon = icon;
    wcsncpy(n.szTip, tip ? tip : L"Kindi Monitor", 127);
    Shell_NotifyIconW(op, &n);
    if (op == NIM_ADD) { n.uVersion = NOTIFYICON_VERSION_4; Shell_NotifyIconW(NIM_SETVERSION, &n); }
}

static void tray_update(void) {
    int i, busy = 0, off = 0, online = 0, state;
    wchar_t tip[128] = L"Kindi Monitor";
    snapshot(g_snap);
    for (i = 0; i < g_snap->nsrv; i++) {
        const Server *s = &g_snap->srv[i];
        if (!s->kindi && s->online) continue;
        if (!s->online) { if (s->kindi || s->source != SRC_PORT) off++; continue; }
        online++;
        if (strcmp(s->phase, "idle")) busy++;
    }
    state = busy ? 1 : off ? 2 : 0;
    for (i = 0; i < g_snap->nsrv; i++) {
        const Server *s = &g_snap->srv[i];
        if (s->kindi && s->online) {
            wchar_t m[48];
            utf8_to_w(s->model, m, 48);
            swprintf(tip, 128, L"Kindi  %s  %.1f tok/s%s", m, speed_now(s), online > 1 ? L"  (+ more)" : L"");
            break;
        }
    }
    if (!online) swprintf(tip, 128, L"Kindi Monitor  no server online");
    if (state != g_icon_state) {
        g_icon_state = state;
        tray(NIM_MODIFY, state == 1 ? g_icon_busy : state == 2 ? g_icon_off : g_icon_idle, tip);
    } else {
        NOTIFYICONDATAW n;
        ZeroMemory(&n, sizeof n);
        n.cbSize = sizeof n;
        n.hWnd = g_main;
        n.uID = 1;
        n.uFlags = NIF_TIP | NIF_SHOWTIP;
        wcsncpy(n.szTip, tip, 127);
        Shell_NotifyIconW(NIM_MODIFY, &n);
    }
}

static UINT g_taskbar_created;
static HWND g_logs;
static HFONT g_logs_font;

static LRESULT CALLBACK logs_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_SIZE: { HWND e = GetWindow(hwnd, GW_CHILD); if (e) MoveWindow(e, 0, 0, LOWORD(lp), HIWORD(lp), TRUE); return 0; }
    case WM_CTLCOLORSTATIC: SetTextColor((HDC)wp, RGB(210, 210, 210)); SetBkColor((HDC)wp, INK); return (LRESULT)GetStockObject(BLACK_BRUSH);
    case WM_DESTROY: g_logs = NULL; if (g_logs_font) { DeleteObject(g_logs_font); g_logs_font = NULL; } return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static void show_logs(void) {
    static wchar_t w[32768];
    char *src;
    int i = 0, o = 0, dpi = g_dash ? monitor_dpi(g_dash) : g_dpi;
    HWND e;
    EnterCriticalSection(&g_lock);
    src = g_logs_text;
    /* LF -> CRLF for the text box */
    {
        static wchar_t tmp[32768];
        MultiByteToWideChar(CP_UTF8, 0, src, -1, tmp, 32768);
        tmp[32767] = 0;
        for (i = 0; tmp[i] && o < 32760; i++) {
            if (tmp[i] == L'\n' && (i == 0 || tmp[i - 1] != L'\r')) w[o++] = L'\r';
            w[o++] = tmp[i];
        }
        w[o] = 0;
    }
    LeaveCriticalSection(&g_lock);
    if (!g_logs) {
        BOOL dark = TRUE;
        g_logs = CreateWindowExW(0, L"KindiMonitorLogs", g_logs_title, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                                 MulDiv(900, dpi, 96), MulDiv(560, dpi, 96), NULL, NULL, g_inst, NULL);
        if (!g_logs) return;
        DwmSetWindowAttribute(g_logs, 20, &dark, sizeof dark);
        g_logs_font = CreateFontW(-MulDiv(12, dpi, 96), 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, FIXED_PITCH, L"Consolas");
        e = CreateWindowExW(0, L"EDIT", L"", WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_HSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | ES_AUTOHSCROLL,
                            0, 0, 10, 10, g_logs, NULL, g_inst, NULL);
        SendMessageW(e, WM_SETFONT, (WPARAM)g_logs_font, TRUE);
        {   RECT r; GetClientRect(g_logs, &r); MoveWindow(e, 0, 0, r.right, r.bottom, TRUE); }
    }
    SetWindowTextW(g_logs, g_logs_title);
    e = GetWindow(g_logs, GW_CHILD);
    SetWindowTextW(e, w);
    SendMessageW(e, EM_SETSEL, (WPARAM)o, (LPARAM)o);
    SendMessageW(e, EM_SCROLLCARET, 0, 0);
    ShowWindow(g_logs, SW_SHOW);
    SetForegroundWindow(g_logs);
}

/* --snapshot DIR: draw the popup and the dashboard into 24-bit BMP files (for tests on machines without a desktop) */
static int save_bmp(const wchar_t *path, DrawFn draw, int w, int h, const State *st) {
    HDC screen = GetDC(NULL), mem = CreateCompatibleDC(screen);
    BITMAPINFO bi;
    BITMAPFILEHEADER fh;
    void *bits = NULL;
    HBITMAP bmp;
    HGDIOBJ old;
    HANDLE f;
    DWORD done, size = (DWORD)(w * h * 4);
    ZeroMemory(&bi, sizeof bi);
    bi.bmiHeader.biSize = sizeof bi.bmiHeader;
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bmp = CreateDIBSection(mem, &bi, DIB_RGB_COLORS, &bits, NULL, 0);
    old = SelectObject(mem, bmp);
    draw(mem, w, h, st);
    GdiFlush();
    ZeroMemory(&fh, sizeof fh);
    fh.bfType = 0x4D42;
    fh.bfOffBits = sizeof fh + sizeof bi.bmiHeader;
    fh.bfSize = fh.bfOffBits + size;
    f = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 0, NULL);
    if (f != INVALID_HANDLE_VALUE) {
        WriteFile(f, &fh, sizeof fh, &done, NULL);
        WriteFile(f, &bi.bmiHeader, sizeof bi.bmiHeader, &done, NULL);
        WriteFile(f, bits, size, &done, NULL);
        CloseHandle(f);
    }
    SelectObject(mem, old);
    DeleteObject(bmp);
    DeleteDC(mem);
    ReleaseDC(NULL, screen);
    return f != INVALID_HANDLE_VALUE;
}

static void draw_icons(HDC dc, int w, int h, const State *st) {   /* the three tray icons, on dark and light */
    int s = GetSystemMetrics(SM_CXSMICON);
    (void)st;
    fill(dc, 0, 0, w, h / 2, RGB(32, 32, 32));
    fill(dc, 0, h / 2, w, h, RGB(243, 243, 243));
    DrawIconEx(dc, 8, 8, g_icon_idle, s, s, 0, NULL, DI_NORMAL);
    DrawIconEx(dc, 16 + s, 8, g_icon_busy, s, s, 0, NULL, DI_NORMAL);
    DrawIconEx(dc, 24 + 2 * s, 8, g_icon_off, s, s, 0, NULL, DI_NORMAL);
    DrawIconEx(dc, 8, h / 2 + 8, g_icon_idle, s, s, 0, NULL, DI_NORMAL);
    DrawIconEx(dc, 16 + s, h / 2 + 8, g_icon_busy, s, s, 0, NULL, DI_NORMAL);
    DrawIconEx(dc, 24 + 2 * s, h / 2 + 8, g_icon_off, s, s, 0, NULL, DI_NORMAL);
}

static void snapshot_files(const wchar_t *dir) {
    wchar_t path[MAX_PATH];
    swprintf(path, MAX_PATH, L"%s\\tray-icons.bmp", dir);
    save_bmp(path, draw_icons, 32 + 3 * GetSystemMetrics(SM_CXSMICON), 2 * (16 + GetSystemMetrics(SM_CYSMICON)), g_snap);
    snapshot(g_snap);
    build_fonts(&g_fd, g_dash_dpi);
    swprintf(path, MAX_PATH, L"%s\\popup.bmp", dir);
    save_bmp(path, draw_popup, S(POPUP_W), S(popup_height(g_snap)), g_snap);
    {   /* one dashboard per server (dashboard-0.bmp, ...), and dashboard-none.bmp with no server selected */
        int i;
        for (i = 0; i < g_snap->nsrv; i++) {
            g_selected = i;
            swprintf(path, MAX_PATH, L"%s\\dashboard-%d.bmp", dir, i);
            save_bmp(path, draw_dash, S(1100), S(900), g_snap);
        }
        g_snap->nsrv = 0;
        swprintf(path, MAX_PATH, L"%s\\dashboard-none.bmp", dir);
        save_bmp(path, draw_dash, S(1100), S(900), g_snap);
    }
}

static LRESULT CALLBACK main_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == g_taskbar_created && msg) { tray(NIM_ADD, g_icon_idle, NULL); g_icon_state = -1; return 0; }
    switch (msg) {
    case WM_APP + 20: show_logs(); return 0;
    case WM_DATA:
        tray_update();
        if (g_popup && IsWindowVisible(g_popup)) InvalidateRect(g_popup, NULL, FALSE);
        if (g_dash && IsWindowVisible(g_dash)) InvalidateRect(g_dash, NULL, FALSE);
        return 0;
    case WM_TRAY:
        switch (LOWORD(lp)) {
        case NIN_SELECT: case NIN_KEYSELECT: case WM_LBUTTONUP:
            if (IsWindowVisible(g_popup)) { ShowWindow(g_popup, SW_HIDE); update_ui_open(); }
            else if (now_ms() - g_popup_hid > 400) place_popup();
            return 0;
        case WM_LBUTTONDBLCLK: open_dash(); return 0;
        case WM_CONTEXTMENU: case WM_RBUTTONUP: {
            HMENU m = CreatePopupMenu();
            POINT pt;
            UINT cmd;
            AppendMenuW(m, MF_STRING, 1, L"Open dashboard");
            AppendMenuW(m, MF_STRING, 2, L"Find servers again");
            AppendMenuW(m, MF_STRING, 5, L"Add a server by URL...");
            AppendMenuW(m, MF_STRING, 6, L"Run DeepSeek-V4.1-Flash in WSL...");
            AppendMenuW(m, MF_STRING, 3, L"Open the log folder");
            AppendMenuW(m, MF_SEPARATOR, 0, NULL);
            AppendMenuW(m, MF_STRING, 4, L"Exit");
            GetCursorPos(&pt);
            SetForegroundWindow(hwnd);
            cmd = (UINT)TrackPopupMenu(m, TPM_RETURNCMD | TPM_RIGHTBUTTON, pt.x, pt.y, 0, hwnd, NULL);
            DestroyMenu(m);
            PostMessageW(hwnd, WM_NULL, 0, 0);
            if (cmd == 1) open_dash();
            else if (cmd == 2) { InterlockedExchange(&g_want_discover, 1); SetEvent(g_wake); }
            else if (cmd == 5) { open_dash(); add_dialog(g_dash); }
            else if (cmd == 6) wsl_open(L"DeepSeek-V4.1-Flash");
            else if (cmd == 3) ShellExecuteW(NULL, L"open", g_dir, NULL, NULL, SW_SHOWNORMAL);
            else if (cmd == 4) DestroyWindow(hwnd);
            return 0;
        }
        }
        return 0;
    case WM_DESTROY:
        tray(NIM_DELETE, NULL, NULL);
        SetEvent(g_stop);
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE prev, PWSTR cmdline, int show) {
    WNDCLASSEXW wc;
    HANDLE single, thread;
    MSG msg;
    GdiStartup gs;
    int argc = 0, i, open_dash_now = 0, snap_s = 0, test_port = 0, wsl_panel_test = 0;
    wchar_t test_op[16] = L"";
    char test_model[48] = "";
    wchar_t snap_dir[MAX_PATH] = L"";
    wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc), appdata[MAX_PATH];
    (void)prev; (void)cmdline; (void)show;
    g_inst = inst;
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    SetEnvironmentVariableW(L"WSL_UTF8", L"1");   /* wsl.exe writes UTF-8, not UTF-16, to the monitor's pipes */
    if (!GetEnvironmentVariableW(L"LOCALAPPDATA", g_kdir, MAX_PATH)) wcscpy(g_kdir, L".");
    wcscat(g_kdir, L"\\Kindi");                  /* the Kindi folder: install.ps1's state.json and install.log */
    GetEnvironmentVariableW(L"KINDI_INSTALLER_HOME", g_kdir, MAX_PATH);   /* tests: a stand-in installer's folder */
    wcscpy(g_dir, g_kdir);
    CreateDirectoryW(g_dir, NULL);
    if (!GetEnvironmentVariableW(L"APPDATA", appdata, MAX_PATH)) wcscpy(appdata, g_dir);
    swprintf(g_ini, MAX_PATH, L"%s\\Kindi", appdata);
    CreateDirectoryW(g_ini, NULL);
    wcscat(g_ini, L"\\monitor.ini");
    /* KINDI_MONITOR_HOME: the monitor's own log and settings somewhere else (tests next to a running monitor) */
    if (GetEnvironmentVariableW(L"KINDI_MONITOR_HOME", appdata, MAX_PATH)) {
        wcscpy(g_dir, appdata);
        CreateDirectoryW(g_dir, NULL);
        swprintf(g_ini, MAX_PATH, L"%s\\monitor.ini", g_dir);
    }
    GetModuleFileNameW(NULL, g_exe_dir, MAX_PATH);
    if (wcsrchr(g_exe_dir, L'\\')) *wcsrchr(g_exe_dir, L'\\') = 0;
    g_poll_ms_fast = ini_int(L"poll_ms", FAST_MS);
    for (i = 1; argv && i < argc; i++) {
        if (!wcscmp(argv[i], L"--poll-ms") && i + 1 < argc) g_poll_ms_fast = _wtoi(argv[++i]);
        else if (!wcscmp(argv[i], L"--dashboard")) open_dash_now = 1;
        else if (!wcscmp(argv[i], L"--test-control") && i + 2 < argc) {
            test_port = _wtoi(argv[++i]);
            wcsncpy(test_op, argv[++i], 15);
            if (i + 1 < argc && argv[i + 1][0] != L'-') WideCharToMultiByte(CP_UTF8, 0, argv[++i], -1, test_model, 48, NULL, NULL);
        }
        else if (!wcscmp(argv[i], L"--snapshot") && i + 2 < argc) { wcsncpy(snap_dir, argv[++i], MAX_PATH - 1); snap_s = _wtoi(argv[++i]); }
        else if (!wcscmp(argv[i], L"--wsl-panel")) wsl_panel_test = 1;
        else if (!wcscmp(argv[i], L"--wsl-install-test")) wsl_panel_test = 2;
        else if (!wcscmp(argv[i], L"--version")) { MessageBoxW(NULL, L"Kindi Monitor " APP_VERSION, L"Kindi Monitor", MB_OK); return 0; }
    }
    if (argv) LocalFree(argv);
    /* one monitor per user session; the test modes (--snapshot, --test-control) run next to it */
    single = NULL;
    if (!snap_dir[0] && !test_port) {
        single = CreateMutexW(NULL, FALSE, L"Local\\KindiMonitor");
        if (!single || GetLastError() == ERROR_ALREADY_EXISTS) {
            /* a second start opens the running monitor's dashboard */
            HWND other = FindWindowW(L"KindiMonitorMain", NULL);
            if (other) PostMessageW(other, WM_TRAY, 1, WM_LBUTTONDBLCLK);
            return 0;
        }
    }
    if (g_poll_ms_fast < 100) g_poll_ms_fast = 100;
    InitializeCriticalSection(&g_lock);
    g_snap = (State *)calloc(1, sizeof(State));
    g_stop = CreateEventW(NULL, TRUE, FALSE, NULL);
    g_wake = CreateEventW(NULL, FALSE, FALSE, NULL);
    g_http = WinHttpOpen(L"KindiMonitor/" APP_VERSION, WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    ZeroMemory(&gs, sizeof gs);
    gs.version = 1;
    GdiplusStartup(&g_gdi, &gs, NULL);
    ZeroMemory(&wc, sizeof wc);
    wc.cbSize = sizeof wc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hIcon = (HICON)LoadImageW(inst, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXICON), GetSystemMetrics(SM_CYICON), 0);
    wc.hIconSm = (HICON)LoadImageW(inst, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), 0);
    wc.lpfnWndProc = main_proc;
    wc.lpszClassName = L"KindiMonitorMain";
    RegisterClassExW(&wc);
    wc.lpfnWndProc = popup_proc;
    wc.lpszClassName = L"KindiMonitorPopup";
    wc.style = CS_DROPSHADOW;
    RegisterClassExW(&wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = dash_proc;
    wc.lpszClassName = L"KindiMonitorDash";
    RegisterClassExW(&wc);
    wc.style = 0;
    wc.lpfnWndProc = logs_proc;
    wc.lpszClassName = L"KindiMonitorLogs";
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    RegisterClassExW(&wc);
    wc.lpfnWndProc = add_proc;
    wc.lpszClassName = L"KindiMonitorAdd";
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    RegisterClassExW(&wc);
    g_taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");
    g_main = CreateWindowExW(0, L"KindiMonitorMain", L"Kindi Monitor", 0, 0, 0, 0, 0, NULL, NULL, inst, NULL);
    g_popup = CreateWindowExW(WS_EX_TOOLWINDOW | WS_EX_TOPMOST, L"KindiMonitorPopup", L"Kindi", WS_POPUP, 0, 0, 10, 10, NULL, NULL, inst, NULL);
    if (!g_main || !g_popup) return 1;
    g_dpi = monitor_dpi(g_popup);
    build_fonts(&g_fp, g_dpi);
    g_icon_idle = make_icon(INK);
    g_icon_busy = make_icon(LIME);
    g_icon_off = make_icon(RED);
    tray(NIM_ADD, g_icon_idle, NULL);
    thread = CreateThread(NULL, 0, worker, NULL, 0, NULL);
    log_file(L"started");
    if (open_dash_now) open_dash();
    if (test_port) {   /* a control from the command line (tests): the same code the buttons run */
        int k, idx = -1, op = !wcscmp(test_op, L"start") ? C_START : !wcscmp(test_op, L"stop") ? C_STOP :
                                 !wcscmp(test_op, L"model") ? C_MODEL : !wcscmp(test_op, L"models") ? C_MODELS :
                                 !wcscmp(test_op, L"logs") ? C_LOGS : C_RESTART;
        g_ui_open = 1;
        if (op == C_START) {   /* a server that is not running is not found by the port scan: list it */
            Sleep(2000);
            EnterCriticalSection(&g_lock);
            add_server(&g_state, L"127.0.0.1", test_port, SRC_MANUAL);
            LeaveCriticalSection(&g_lock);
        }
        for (k = 0; k < 600 && idx < 0; k++) {
            int j;
            Sleep(100);
            EnterCriticalSection(&g_lock);
            for (j = 0; j < g_state.nsrv; j++) {
                const Server *v = &g_state.srv[j];
                int known = v->kind == K_NATIVE || v->kind == K_DOCKER ||
                            ((v->kind == K_WSL_DOCKER || v->kind == K_WSL_UNIT) && v->distro[0]) || (op == C_START && k > 300);
                if (v->port == test_port && v->kind != K_LAN && (v->kindi || op == C_START) && known) idx = j;
            }
            LeaveCriticalSection(&g_lock);
        }
        if (idx < 0) { log_file(L"test-control: no such server"); return 2; }
        if (op == C_START && g_state.srv[idx].kind == K_UNKNOWN) {   /* nothing known about it: a native server */
            EnterCriticalSection(&g_lock); g_state.srv[idx].kind = K_NATIVE; g_state.srv[idx].kindi = 1; LeaveCriticalSection(&g_lock);
        }
        control(idx, op, test_model[0] ? test_model : NULL);
        do { Sleep(500); } while (g_state.busy_control);
        if (op == C_LOGS) {
            wchar_t lp[MAX_PATH];
            HANDLE f;
            DWORD done;
            swprintf(lp, MAX_PATH, L"%s\\test-logs.txt", g_dir);
            f = CreateFileW(lp, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 0, NULL);
            if (f != INVALID_HANDLE_VALUE) { WriteFile(f, g_logs_text, (DWORD)strlen(g_logs_text), &done, NULL); CloseHandle(f); }
        }
        EnterCriticalSection(&g_lock);
        for (k = 0; k < g_state.nlog; k++) log_file(g_state.log[k]);
        LeaveCriticalSection(&g_lock);
        DestroyWindow(g_main);
    }
    if (snap_dir[0]) {   /* poll with a window "open" for snap_s seconds, write the two pictures, quit */
        int k;
        g_ui_open = 1;
        SetEvent(g_wake);
        if (wsl_panel_test) {   /* the WSL checks, then dashboard-wsl.bmp */
            wchar_t path[MAX_PATH];
            EnterCriticalSection(&g_lock);
            wcscpy(g_state.wsl.model, L"DeepSeek-V4.1-Flash");
            LeaveCriticalSection(&g_lock);
            wsl_check();
            for (k = 0; k < 1200 && g_state.wsl.stage == WS_CHECKING; k++) Sleep(100);
            if (wsl_panel_test == 2) {   /* the stand-in installer (KINDI_TEST_INSTALL_CMD) runs; draw it halfway */
                wsl_install(NULL);
                for (k = 0; k < snap_s * 10; k++) {
                    if (k % 5 == 0) wsl_install_progress();
                    Sleep(100);
                }
                wsl_install_progress();
            }
            snapshot(g_snap);
            build_fonts(&g_fd, g_dash_dpi);
            g_wsl_view = 1;
            swprintf(path, MAX_PATH, L"%s\\dashboard-wsl.bmp", snap_dir);
            save_bmp(path, draw_dash, S(1100), S(900), g_snap);
            g_wsl_view = 0;
        }
        for (k = 0; k < snap_s * 10; k++) {
            while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
            Sleep(100);
        }
        snapshot_files(snap_dir);
        DestroyWindow(g_main);
    }
    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        if (g_add && IsDialogMessageW(g_add, &msg)) continue;
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    if (thread) { WaitForSingleObject(thread, 5000); CloseHandle(thread); }
    free_fonts(&g_fp);
    free_fonts(&g_fd);
    DestroyIcon(g_icon_idle);
    DestroyIcon(g_icon_busy);
    DestroyIcon(g_icon_off);
    if (g_http) WinHttpCloseHandle(g_http);
    GdiplusShutdown(g_gdi);
    log_file(L"stopped");
    CloseHandle(g_stop);
    CloseHandle(g_wake);
    if (single) CloseHandle(single);
    return 0;
}
