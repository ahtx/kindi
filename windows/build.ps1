# Build Kindi Monitor (kindi-monitor.exe) and the kindi.exe launcher with MSVC (Visual Studio 2022 Build Tools).
#   powershell -NoProfile -File windows\build.ps1 [-Out DIR]
# The output goes to windows\build\ (not tracked). Static CRT (/MT): the exes need nothing but Windows.
# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
param([string]$Out = "")
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Out) { $Out = Join-Path $here "build" }
New-Item -ItemType Directory -Force $Out | Out-Null
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "MSVC not found: install Visual Studio 2022 Build Tools with the C++ workload" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$m = Join-Path $here "monitor"
$l = Join-Path $here "launcher"
$cmd = @"
call "$vcvars" >nul && cd /d "$Out" ^
 && rc /nologo /fo kindi-monitor.res "$m\kindi-monitor.rc" ^
 && cl /nologo /O1 /MT /W3 /DUNICODE /D_UNICODE "$m\kindi-monitor.c" "$m\json.c" kindi-monitor.res /Fe:kindi-monitor.exe /link /SUBSYSTEM:WINDOWS /MANIFEST:NO ^
 && rc /nologo /fo kindi.res "$l\kindi.rc" ^
 && cl /nologo /O1 /MT /W3 /DUNICODE /D_UNICODE "$l\kindi.c" kindi.res /Fe:kindi.exe /link /SUBSYSTEM:CONSOLE /MANIFEST:NO
"@
$bat = Join-Path $Out "build.cmd"
Set-Content -Path $bat -Value $cmd -Encoding ascii
cmd /c "`"$bat`""
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }
Remove-Item (Join-Path $Out "*.obj"), (Join-Path $Out "*.res"), $bat -ErrorAction SilentlyContinue
Get-Item (Join-Path $Out "kindi-monitor.exe"), (Join-Path $Out "kindi.exe") | ForEach-Object { "{0}  {1:N0} bytes" -f $_.Name, $_.Length }
