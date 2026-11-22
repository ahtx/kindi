<#
Kindi for Windows: the per-user setup (no administrator prompt). It ships inside the runtime folder.

  powershell -ExecutionPolicy Bypass -File setup.ps1 [-Dir DIR] [-NoAutostart] [-NoPath]
  powershell -ExecutionPolicy Bypass -File setup.ps1 -Uninstall [-Purge]

Install: copies this folder to %LOCALAPPDATA%\Programs\Kindi (or -Dir), adds the Start menu items (Kindi Monitor,
Kindi Shell), starts Kindi Monitor at sign-in (unless -NoAutostart), puts kindi.exe on the user PATH (unless -NoPath)
and adds an entry in Settings > Apps. Everything is per user: HKCU and the user's own folders.
Uninstall: stops Kindi Monitor, removes all of that again. The models and settings (%USERPROFILE%\.kindi,
%APPDATA%\Kindi) stay unless -Purge. Kindi servers started by the monitor are stopped by the user first (the monitor's
STOP), not by the uninstaller.
Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
#>
[CmdletBinding()]
param([string]$Dir = "", [switch]$Uninstall, [switch]$Purge, [switch]$NoAutostart, [switch]$NoPath)
$ErrorActionPreference = "Stop"
if (-not $Dir) { $Dir = Join-Path $env:LOCALAPPDATA "Programs\Kindi" }
$src = Split-Path -Parent $MyInvocation.MyCommand.Path
$menu = Join-Path ([Environment]::GetFolderPath("Programs")) "Kindi"
$runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$appKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Kindi"
function Say($m) { Write-Host "kindi setup: $m" }

function UserPath([string]$add, [string]$remove) {
    # the user PATH from the registry, unexpanded, so %VARIABLES% in it stay as they are
    $k = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey("Environment", $true)
    $cur = [string]$k.GetValue("Path", "", [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    $parts = @($cur -split ";" | Where-Object { $_ -and $_.TrimEnd("\") -ne $remove.TrimEnd("\") })
    if ($add) { $parts += $add }
    $new = $parts -join ";"
    if ($new -ne $cur) { $k.SetValue("Path", $new, [Microsoft.Win32.RegistryValueKind]::ExpandString) }
    $k.Close()
}

function StopMonitor {
    # only the monitor that runs from the install folder (by its path, never by name alone)
    Get-Process kindi-monitor -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($Dir, "OrdinalIgnoreCase") } |
        ForEach-Object { Say "stopping Kindi Monitor (PID $($_.Id))"; $_.CloseMainWindow() | Out-Null; if (-not $_.WaitForExit(3000)) { $_.Kill() } }
}

function Shortcut($path, $target, $arguments, $desc) {
    $sh = New-Object -ComObject WScript.Shell
    $l = $sh.CreateShortcut($path)
    $l.TargetPath = $target
    $l.Arguments = $arguments
    $l.WorkingDirectory = $Dir
    $l.IconLocation = (Join-Path $Dir "kindi-monitor.exe") + ",0"
    $l.Description = $desc
    $l.Save()
}

if ($Uninstall) {
    StopMonitor
    Remove-ItemProperty -Path $runKey -Name "KindiMonitor" -ErrorAction SilentlyContinue
    UserPath "" $Dir
    if (Test-Path $menu) { Remove-Item -Recurse -Force $menu }
    if (Test-Path $appKey) { Remove-Item -Recurse -Force $appKey }
    if (Test-Path $Dir) { Remove-Item -Recurse -Force $Dir; Say "removed $Dir" }
    if ($Purge) {
        foreach ($p in (Join-Path $env:USERPROFILE ".kindi"), (Join-Path $env:APPDATA "Kindi"), (Join-Path $env:LOCALAPPDATA "Kindi")) {
            if (Test-Path $p) { Remove-Item -Recurse -Force $p; Say "removed $p" }
        }
    } else { Say "kept the models and settings (%USERPROFILE%\.kindi, %APPDATA%\Kindi); -Purge removes them" }
    Say "Kindi is uninstalled"
    exit 0
}

foreach ($f in "kindi.exe", "kindi-monitor.exe", "python\python.exe") {
    if (-not (Test-Path (Join-Path $src $f))) { throw "$f is missing next to setup.ps1: unpack the whole zip first" }
}
StopMonitor
if ((Resolve-Path $src).Path.TrimEnd("\") -ne $Dir.TrimEnd("\")) {
    if (Test-Path $Dir) { Remove-Item -Recurse -Force $Dir }
    New-Item -ItemType Directory -Force $Dir | Out-Null
    Say "copying to $Dir"
    Copy-Item -Recurse -Force (Join-Path $src "*") $Dir
}
New-Item -ItemType Directory -Force $menu | Out-Null
Shortcut (Join-Path $menu "Kindi Monitor.lnk") (Join-Path $Dir "kindi-monitor.exe") "--dashboard" "Kindi Monitor: the Kindi servers, cards and host"
Shortcut (Join-Path $menu "Kindi Shell.lnk") "$env:SystemRoot\System32\cmd.exe" "/k `"set PATH=$Dir;%PATH%&& kindi.exe --help`"" "A command prompt with kindi.exe"
Shortcut (Join-Path $menu "Uninstall Kindi.lnk") "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    "-NoProfile -ExecutionPolicy Bypass -File `"$Dir\setup.ps1`" -Uninstall" "Remove Kindi (the models stay)"
if (-not $NoAutostart) { Set-ItemProperty -Path $runKey -Name "KindiMonitor" -Value "`"$Dir\kindi-monitor.exe`"" }
if (-not $NoPath) { UserPath $Dir $Dir }
$ver = (Get-Item (Join-Path $Dir "kindi.exe")).VersionInfo.ProductVersion
New-Item -Force $appKey | Out-Null
$size = [int]((Get-ChildItem -Recurse -File $Dir | Measure-Object -Sum Length).Sum / 1KB)
foreach ($kv in @{ DisplayName = "Kindi"; DisplayVersion = $ver; Publisher = "Amir Husain"; InstallLocation = $Dir;
                   DisplayIcon = "$Dir\kindi-monitor.exe"; NoModify = 1; NoRepair = 1; EstimatedSize = $size;
                   UninstallString = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Dir\setup.ps1`" -Uninstall" }.GetEnumerator()) {
    $type = if ($kv.Value -is [int]) { "DWord" } else { "String" }
    New-ItemProperty -Path $appKey -Name $kv.Key -Value $kv.Value -PropertyType $type -Force | Out-Null
}
Start-Process (Join-Path $Dir "kindi-monitor.exe")
Say "installed Kindi $ver in $Dir"
Say "Kindi Monitor is in the tray; Start > Kindi > Kindi Shell has kindi.exe (kindi use bonsai-2b, kindi serve)"
