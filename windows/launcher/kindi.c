/* kindi.exe: the Kindi command on Windows without WSL. It runs the bundled Python with `-m kindi` and the same
   arguments, after it points CUDA_PATH at the bundled NVRTC (CuPy compiles Kindi's kernels with it at run time).

     Kindi\kindi.exe          this launcher
     Kindi\python\python.exe  the python.org embeddable package; python3XX._pth adds ..\app and site-packages
     Kindi\app\kindi\         the Kindi source
     NVRTC (nvrtc64_120_0.dll) comes from the nvidia-cuda-nvrtc-cu12 wheel, in
     python\Lib\site-packages\nvidia\cuda_nvrtc\bin; CUDA_PATH points at its parent, as in B6 (docs/INSTALL.md).

   The NVIDIA driver is the only requirement (nvcuda.dll). docs/dev/F8-WINDOWS.md, section 7.
   Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

/* the arguments after the program name, as typed (quotes kept) */
static const wchar_t *rest_of(const wchar_t *cmd) {
    int quoted = 0;
    while (*cmd && (quoted || (*cmd != L' ' && *cmd != L'\t'))) {
        if (*cmd == L'"') quoted = !quoted;
        cmd++;
    }
    while (*cmd == L' ' || *cmd == L'\t') cmd++;
    return cmd;
}

int wmain(void) {
    wchar_t dir[MAX_PATH], py[MAX_PATH], nvrtc[MAX_PATH], *cmd, *slash;
    const wchar_t *args = rest_of(GetCommandLineW());
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    DWORD code = 1;
    size_t n;
    if (!GetModuleFileNameW(NULL, dir, MAX_PATH)) return 1;
    slash = wcsrchr(dir, L'\\');
    if (slash) *slash = 0;
    swprintf(py, MAX_PATH, L"%s\\python\\python.exe", dir);
    swprintf(nvrtc, MAX_PATH, L"%s\\python\\Lib\\site-packages\\nvidia\\cuda_nvrtc", dir);
    if (GetFileAttributesW(py) == INVALID_FILE_ATTRIBUTES) {
        fwprintf(stderr, L"kindi: %s is missing; unpack the whole Kindi folder again\n", py);
        return 1;
    }
    SetEnvironmentVariableW(L"CUDA_PATH", nvrtc);
    {   /* nvrtc64_120_0.dll loads nvrtc-builtins64_*.dll by name, through the PATH search: put its folder first */
        static wchar_t path[32768];
        int k = swprintf(path, 32768, L"%s\\bin;", nvrtc);
        if (k > 0) GetEnvironmentVariableW(L"PATH", path + k, 32768 - k);
        SetEnvironmentVariableW(L"PATH", path);
    }
    SetEnvironmentVariableW(L"PYTHONNOUSERSITE", L"1");
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
    SetEnvironmentVariableW(L"KINDI_LAUNCHER", L"kindi.exe");
    n = wcslen(py) + wcslen(args) + 32;
    cmd = (wchar_t *)malloc(n * sizeof(wchar_t));
    if (!cmd) return 1;
    swprintf(cmd, n, L"\"%s\" -m kindi %s", py, args);
    ZeroMemory(&si, sizeof si);
    si.cb = sizeof si;
    /* Ctrl+C goes to the Python process (the same console); the launcher waits for it and returns its exit code */
    SetConsoleCtrlHandler(NULL, TRUE);
    if (!CreateProcessW(py, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        fwprintf(stderr, L"kindi: could not start %s (error %lu)\n", py, GetLastError());
        free(cmd);
        return 1;
    }
    free(cmd);
    WaitForSingleObject(pi.hProcess, INFINITE);
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)code;
}
