<#
Kindi installer for Windows: one command from a Windows PC to a verified, running Kindi server in WSL2.

  irm https://amirhusain.com/kindi/install.ps1 | iex                       # install (or upgrade / repair)
  & ([scriptblock]::Create((irm https://amirhusain.com/kindi/install.ps1))) -DryRun     # with options
  powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall [-Purge]

What it does, in order (every step prints what it is about to do and why):
  1. checks the PC: x86-64, the Windows build, virtualization, RAM, disk, the GPU and its Windows driver, the proxy;
  2. turns on WSL2 if it is off (administrator approval; a restart; the installer continues after the restart);
  3. makes a WSL2 Linux system for Kindi ("Kindi", Ubuntu 24.04 from Canonical, sha256-checked), or reuses it.
     It never changes or removes your other WSL distributions;
  4. prepares it: systemd, the proxy, Docker (Docker Desktop if you have it, else Docker Engine inside the
     distribution) and the NVIDIA container toolkit; checks that the GPU reaches WSL;
  5. runs the Linux installer (install.sh) inside it: models, checks, the server;
  6. checks that the server answers on this PC at http://localhost:8093, and adds the Start-menu items
     (Kindi Console, Kindi Shell, Uninstall Kindi), a desktop shortcut and an entry in Settings > Apps.
Run it again to upgrade or repair. It never asks for or stores a password or a key.
Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.
#>
[CmdletBinding()]
param(
  [switch]$DryRun,            # print every step and command, change nothing
  [switch]$Check,             # only check the PC and say what the install would do
  [switch]$Yes,               # answer yes to every question
  [switch]$Uninstall,         # remove Kindi (asks before it deletes the Kindi distribution and its models)
  [switch]$Purge,             # with -Uninstall: also delete the Kindi distribution and its models
  [string]$Distro = 'Kindi',  # the WSL distribution to use; an existing one is reused, never replaced
  [string]$DistroDir = '',    # where a new distribution keeps its disk (default: %LOCALAPPDATA%\Kindi\wsl)
  [ValidateSet('auto', 'engine', 'desktop', 'skip')][string]$DockerMode = 'auto',
  [string]$LinuxArgs = '',    # options for install.sh, for example '--model 27b' or '--dry-run'
  [string]$InstallShUrl = 'https://amirhusain.com/kindi/install.sh',
  [string]$InstallShPath = '',  # a local install.sh instead of the download (for tests)
  [int]$WslMemoryGB = 0,      # set the memory of the WSL virtual machine (needs 'wsl --shutdown'; asks first)
  [switch]$WithBridge,        # also install Kindi Bridge (the agent workspace); interactive runs ask (default no)
  [switch]$NoBridge,          # do not install Kindi Bridge and do not ask
  [switch]$NoDesktopShortcut,
  [switch]$NoReboot,          # after WSL is turned on, do not restart: you restart, the install continues then
  [string]$AppName = 'Kindi', # the name of the shortcuts, the Settings entry and the data folder (tests use another)
  [string]$ScriptUrl = 'https://amirhusain.com/kindi/install.ps1',
  [switch]$Resume,            # internal: the run after the restart
  [switch]$ElevatedEnableWsl  # internal: the administrator step that turns on WSL
)

$KindiWinVersion = '0.6.0'
$RootfsBase = 'https://cloud-images.ubuntu.com/wsl/releases/24.04/current'
$RootfsFile = 'ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz'
$MinBuild = 19041      # Windows 10 2004: WSL2
$MinGpuBuild = 19044   # Windows 10 21H2 or Windows 11: GPU compute in WSL2
$MinNvidiaDriver = 470.0

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Windows PowerShell's progress bar makes downloads many times slower
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }
try { [System.Net.WebRequest]::DefaultWebProxy.Credentials = [System.Net.CredentialCache]::DefaultNetworkCredentials } catch { }

$ScriptParams = $PSBoundParameters           # this run's options (for the run after the restart)
$SelfBlock = $MyInvocation.MyCommand.ScriptBlock
$WslExe = Join-Path $env:WINDIR 'System32\wsl.exe'
$KHome = Join-Path $env:LOCALAPPDATA $AppName
$StateFile = Join-Path $KHome 'state.json'
if (-not $DistroDir) { $DistroDir = Join-Path $KHome 'wsl' }
$script:StepNo = 0

# ------------------------------------------------------------------------------------------------ helpers
function Say([string]$t) { Write-Host "    $t" }
function Ok([string]$t) { Write-Host '    ' -NoNewline; Write-Host 'ok' -ForegroundColor Green -NoNewline; Write-Host "  $t" }
function Note([string]$t) { Write-Host '    ' -NoNewline; Write-Host 'note' -ForegroundColor Yellow -NoNewline; Write-Host " $t" }
function Step([string]$t, [string]$why = '') {
  $script:StepNo++
  Write-Host ''
  Write-Host "[$($script:StepNo)] $t" -ForegroundColor White
  if ($why) { Say $why }
}
function Stop-Kindi([string]$t) { throw "KINDI-STOP: $t" }
function Ask([string]$q, [switch]$DefaultNo) {
  if ($Yes) { Say "$q yes (-Yes)"; return $true }
  if ($DryRun -or $Check) { Say "$q (dry run: no)"; return $false }
  $hint = '[Y/n]'; if ($DefaultNo) { $hint = '[y/N]' }
  $a = Read-Host "    $q $hint"
  if (-not $a) { return (-not $DefaultNo) }
  return ($a -match '^(y|yes)$')
}
function Act([string]$what, [scriptblock]$do) {   # print, then run unless -DryRun / -Check
  Write-Host "    > $what" -ForegroundColor DarkGray
  if ($DryRun -or $Check) { return $null }
  return (& $do)
}
function Test-Admin {
  $p = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
  return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}
function Save-State($s) {
  if ($DryRun -or $Check) { return }
  New-Item -ItemType Directory -Force -Path $KHome | Out-Null
  ($s | ConvertTo-Json -Depth 5) | Set-Content -Path $StateFile -Encoding UTF8
}
function Get-State {
  if (Test-Path $StateFile) { try { return (Get-Content $StateFile -Raw | ConvertFrom-Json) } catch { } }
  return [pscustomobject]@{ distro = $Distro; createdDistro = $false; dockerMode = ''; user = ''; keepAlivePid = 0 }
}
function Set-Prop($o, [string]$n, $v) {
  if ($o.PSObject.Properties[$n]) { $o.$n = $v } else { $o | Add-Member -NotePropertyName $n -NotePropertyValue $v }
}

function Join-WinArgs([string[]]$A) {   # a Windows command line from separate arguments
  return (($A | ForEach-Object { if ($_ -match '[\s"]' -or $_ -eq '') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' ')
}
# wsl.exe with its output captured: UTF-8 (WSL_UTF8=1; older versions write UTF-16, so the NUL bytes are removed).
# Its input is closed: when WSL is not installed, the wsl.exe of new Windows builds waits for "any key" to install
# WSL itself, and a captured call would then wait for ever.
function Invoke-Wsl([string[]]$A, [switch]$Capture) {
  $ErrorActionPreference = 'Continue'
  $env:WSL_UTF8 = '1'
  if ($Capture) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo $WslExe
    $psi.Arguments = Join-WinArgs $A
    $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
    $psi.RedirectStandardInput = $true; $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [Text.Encoding]::UTF8; $psi.StandardErrorEncoding = [Text.Encoding]::UTF8
    $p = [System.Diagnostics.Process]::Start($psi)
    $p.StandardInput.Close()
    $o = $p.StandardOutput.ReadToEndAsync(); $e = $p.StandardError.ReadToEndAsync()
    $p.WaitForExit()
    $script:WslCode = $p.ExitCode
    # stdout only: WSL writes notes to stderr (for example "Failed to start the systemd user session for 'root'"),
    # which must not end up in a value such as a path; on a failure the caller gets stderr too, for the message
    $r = $o.Result; if ($p.ExitCode -ne 0) { $r = $r + "`n" + $e.Result }
    return (($r -replace "`0", '').Trim())
  }
  & $WslExe @A
  $script:WslCode = $LASTEXITCODE
}
# wsl.exe on this console (not through a pipe), so that install.sh can ask questions and print its progress live
function Invoke-WslConsole([string[]]$A) {
  $q = Join-WinArgs $A
  $env:WSL_UTF8 = '1'
  $p = Start-Process -FilePath $WslExe -ArgumentList $q -NoNewWindow -Wait -PassThru
  $script:WslCode = $p.ExitCode
}
function Get-Web([string]$Uri, [string]$OutFile = '') {
  $a = @{ UseBasicParsing = $true; Uri = $Uri }
  if ($OutFile) { $a.OutFile = $OutFile }
  foreach ($v in 'HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy') {
    $x = [Environment]::GetEnvironmentVariable($v); if ($x) { $a.Proxy = $x; $a.ProxyUseDefaultCredentials = $true; break }
  }
  return (Invoke-WebRequest @a)
}
function Get-Distros {
  $o = Invoke-Wsl @('--list', '--quiet') -Capture
  if ($script:WslCode -ne 0) { return @() }
  return @($o -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
function Get-DistroVersion([string]$d) {   # 1 or 2, from 'wsl -l -v'
  $o = Invoke-Wsl @('--list', '--verbose') -Capture
  foreach ($l in ($o -split "`n")) {
    $c = ($l -replace '^\s*\*?\s*', '') -split '\s+'
    if ($c.Count -ge 3 -and $c[0] -eq $d) { return [int]$c[-1] }
  }
  return 0
}
# Run a bash script (text) in the distribution as USER; the file goes to the data folder and WSL reads it from the
# Windows drive (wslpath gives its Linux path).
function Invoke-WslScript([string]$Name, [string]$Body, [string]$User = 'root', [string[]]$ScriptArgs = @(), [switch]$Capture) {
  $dir = Join-Path $KHome 'wsl-scripts'
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $f = Join-Path $dir $Name
  [IO.File]::WriteAllText($f, ($Body -replace "`r`n", "`n"), (New-Object Text.UTF8Encoding($false)))
  $lp = Invoke-Wsl @('-d', $Distro, '-u', 'root', '--exec', 'wslpath', '-a', $f) -Capture
  $lp = @($lp -split "`n" | Where-Object { $_.StartsWith('/') })[0]
  if ($script:WslCode -ne 0 -or -not $lp) { Stop-Kindi "WSL cannot read $f (is the Windows drive mounted in '$Distro'? automount in /etc/wsl.conf): $lp" }
  # wsl.exe refuses an empty argument (E_INVALIDARG): '-' stands for an empty value, and the scripts read it so
  $sa = @($ScriptArgs | ForEach-Object { if ("$_" -eq '') { '-' } else { "$_" } })
  $a = @('-d', $Distro, '-u', $User, '--cd', '~', '--exec', 'bash', $lp) + $sa
  if ($Capture) { return (Invoke-Wsl $a -Capture) }
  Invoke-WslConsole $a
}
function Invoke-Root([string[]]$PhaseArgs) {   # a phase of the root setup; stops the install if it fails
  Invoke-WslScript 'kindi-root.sh' $RootScript 'root' $PhaseArgs
  if ($script:WslCode -ne 0) { Stop-Kindi "the Linux setup step '$($PhaseArgs[0])' in '$Distro' stopped (exit code $($script:WslCode)); its message is above" }
}
function Test-LinuxUser([string]$u) {
  if (-not $u) { return $false }
  Invoke-Wsl @('-d', $Distro, '-u', 'root', '--exec', 'id', '-u', $u) -Capture | Out-Null
  return ($script:WslCode -eq 0)
}

function Get-KindiProxy {
  foreach ($v in 'HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy') {
    $x = [Environment]::GetEnvironmentVariable($v); if ($x) { return $x.TrimEnd('/') }
  }
  try {
    $t = [Uri]'https://huggingface.co'
    $p = [System.Net.WebRequest]::GetSystemWebProxy().GetProxy($t)   # also resolves PAC / WPAD
    if ($p -and $p.AbsoluteUri -ne $t.AbsoluteUri) { return $p.AbsoluteUri.TrimEnd('/') }
  } catch { }
  return ''
}
function Get-NvidiaVersion([string]$dv) {   # Windows driver 32.0.15.6094 -> NVIDIA 560.94
  $p = $dv -split '\.'
  if ($p.Count -lt 4) { return 0.0 }
  $d = ($p[2] + $p[3].PadLeft(4, '0'))
  if ($d.Length -lt 5) { return 0.0 }
  $d = $d.Substring($d.Length - 5)
  return [double]("{0}.{1}" -f $d.Substring(0, 3), $d.Substring(3))
}
function Save-Self {
  New-Item -ItemType Directory -Force -Path $KHome | Out-Null
  $dst = Join-Path $KHome 'install.ps1'
  if ($PSCommandPath -and (Test-Path $PSCommandPath)) {
    if ((Resolve-Path $PSCommandPath).Path -ne $dst) { Copy-Item -Force $PSCommandPath $dst }
    return $dst
  }
  if ($SelfBlock -and $SelfBlock.ToString() -match 'Kindi installer for Windows') {
    [IO.File]::WriteAllText($dst, $SelfBlock.ToString(), (New-Object Text.UTF8Encoding($false))); return $dst
  }
  Get-Web $ScriptUrl $dst | Out-Null
  return $dst
}
function Get-PassArgs {   # this run's options, for the run after the restart and for the shortcuts
  $a = @()
  foreach ($k in $ScriptParams.Keys) {
    if ($k -in 'Resume', 'ElevatedEnableWsl', 'Check') { continue }
    $v = $ScriptParams[$k]
    if ($v -is [switch]) { if ($v.IsPresent) { $a += "-$k" } } else { $a += "-$k"; $a += "`"$v`"" }
  }
  return ($a -join ' ')
}

# ------------------------------------------------------------------------------------------------ 1. the PC
function Test-Pc {
  Step 'Check the PC' 'Kindi runs in WSL2 (a Linux virtual machine that Windows includes) on an x86-64 PC.'
  $arch = ''
  try { $arch = [string][System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture } catch { }
  if (-not $arch) { $a = (Get-CimInstance Win32_Processor | Select-Object -First 1).Architecture; if ($a -eq 12) { $arch = 'Arm64' } elseif ($a -eq 9) { $arch = 'X64' } }
  if ($arch -match 'Arm') {
    Stop-Kindi "this PC has an ARM processor ($arch). Kindi needs an x86-64 (Intel or AMD) PC: its engines are built for x86-64 Linux with CUDA or ROCm. Windows on ARM cannot run them."
  }
  $os = Get-CimInstance Win32_OperatingSystem
  $build = [int]$os.BuildNumber
  if ($build -lt $MinBuild) {
    Stop-Kindi "Windows build $build is too old for WSL2. Kindi needs Windows 10 version 2004 (build $MinBuild) or newer; Windows 11 is best. Open Settings > Windows Update, install the updates, and run this command again."
  }
  $script:GpuBuildOk = ($build -ge $MinGpuBuild)
  Ok "$($os.Caption) (build $build), x86-64"
  if (-not $script:GpuBuildOk) { Note "GPU compute in WSL2 needs Windows 11 or Windows 10 21H2 (build $MinGpuBuild) or newer: Kindi uses the CPU engine until you update Windows" }

  $cs = Get-CimInstance Win32_ComputerSystem
  $script:WinRamGB = [math]::Round($cs.TotalPhysicalMemory / 1GB)
  $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
  if ($cs.HypervisorPresent) { Ok 'virtualization is on (a hypervisor is running)' }
  elseif ($cpu.VirtualizationFirmwareEnabled -eq $false) {
    $script:VirtOff = $true
    Note 'virtualization is OFF in the BIOS/UEFI. WSL2 needs it.'
  } else { Ok 'virtualization is available' }
  $drv = Split-Path -Qualifier $DistroDir
  $free = [math]::Round((Get-PSDrive -Name $drv.TrimEnd(':')).Free / 1GB)
  Ok "$($cs.NumberOfLogicalProcessors) CPU threads, $($script:WinRamGB) GB RAM, $free GB free on $drv"
  if ($free -lt 40) { Note "Kindi needs about 15-40 GB on $drv for Linux, Docker, the models and their packs (free: $free GB). Use -DistroDir to put it on another drive." }

  $script:HasNvidia = $false; $script:HasAmd = $false
  foreach ($g in (Get-CimInstance Win32_VideoController)) {
    if ($g.Name -match 'NVIDIA') {
      $v = Get-NvidiaVersion $g.DriverVersion
      $script:HasNvidia = $true
      if ($v -gt 0 -and $v -lt $MinNvidiaDriver) { Note "$($g.Name): NVIDIA driver $v is too old for CUDA in WSL2: install a current Windows driver from nvidia.com (Game Ready or Studio), then run this again" }
      else { Ok "$($g.Name), NVIDIA driver $v" }
    } elseif ($g.Name -match 'AMD|Radeon') {
      $script:HasAmd = $true
      Note "$($g.Name): Kindi's AMD engine needs the Linux amdgpu compute driver (/dev/kfd), which WSL2 does not provide. On Windows, Kindi uses the CPU engine on this GPU. Native Linux gives the AMD GPU speed."
    } elseif ($g.Name -notmatch 'Microsoft|Remote|Virtual|Parsec|Basic') {
      Say "$($g.Name): not used by Kindi"
    }
  }
  if (-not $script:HasNvidia -and -not $script:HasAmd) { Say 'no NVIDIA GPU: Kindi uses the CPU engine' }

  $script:Proxy = Get-KindiProxy
  if ($script:Proxy) {
    Ok "proxy $($script:Proxy): Linux, Docker and the downloads use it"
    if ($script:Proxy -match '//(localhost|127\.0\.0\.1)[:/]') { Note "the proxy runs on this PC's localhost. WSL2 reaches it only with networkingMode=mirrored in %UserProfile%\.wslconfig; or give the proxy's network address with -LinuxArgs and the HTTPS_PROXY variable" }
  } else { Say 'no proxy' }
}

# ------------------------------------------------------------------------------------------------ 2. WSL2
function Test-WslReady {
  if (-not (Test-Path $WslExe)) { return $false }
  $o = Invoke-Wsl @('--status') -Capture
  if ($script:WslCode -eq 0) { return $true }
  if ($o -match 'not installed|wsl(\.exe)? --install') { return $false }   # the stub of new Windows builds
  Invoke-Wsl @('--list', '--quiet') -Capture | Out-Null
  return ($script:WslCode -eq 0)
}
function Enable-WslElevated {   # runs as administrator
  Step 'Turn on WSL2 (administrator)'
  Invoke-Wsl @('--install', '--no-distribution')
  if ($script:WslCode -ne 0) {
    Say 'wsl --install is not available: turning on the two Windows features directly'
    & dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart | Out-Host
    $c1 = $LASTEXITCODE
    & dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart | Out-Host
    $c2 = $LASTEXITCODE
    foreach ($c in $c1, $c2) { if ($c -ne 0 -and $c -ne 3010) { Stop-Kindi "DISM could not turn on WSL (exit code $c). Turn on 'Windows Subsystem for Linux' and 'Virtual Machine Platform' in 'Turn Windows features on or off', restart, and run this again." } }
  }
}
function Enable-Wsl {
  Step 'Turn on WSL2' 'WSL2 is a part of Windows. Turning it on needs administrator approval and a restart.'
  if ($script:VirtOff) {
    Stop-Kindi 'virtualization is off in the BIOS/UEFI, so WSL2 cannot start. Restart the PC, open the firmware setup (often with F2, F10, F12 or Del during start), turn on Intel Virtualization Technology (VT-x) or AMD SVM Mode, save, and run this command again.'
  }
  if (-not (Ask 'Turn on WSL2 now?')) { Stop-Kindi 'Kindi on Windows needs WSL2. Run this command again when you are ready.' }
  $self = Act 'save this installer, so that it can continue after the restart' { Save-Self }
  if ($DryRun) { return }
  if (Test-Admin) { Enable-WslElevated }
  else {
    Say 'Windows asks for administrator approval now (only for this step)'
    try {
      $p = Start-Process powershell.exe -Verb RunAs -Wait -PassThru -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$self`"", '-ElevatedEnableWsl', '-AppName', $AppName)
    } catch { Stop-Kindi 'the administrator approval was cancelled. WSL2 cannot be turned on without it.' }
    if ($p.ExitCode -ne 0) { Stop-Kindi "the administrator step stopped (exit code $($p.ExitCode)); see the messages in its window" }
  }
  $cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$self`" -Resume $(Get-PassArgs)"
  Act 'continue the install automatically after the restart (RunOnce)' {
    New-Item -Force -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce' | Out-Null
    Set-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce' -Name "$AppName-install" -Value $cmd
  } | Out-Null
  Ok 'WSL2 is turned on. Windows must restart before it can run.'
  if (-not $NoReboot -and (Ask 'Restart the PC now? (save your work first)')) { Restart-Computer }
  else { Say 'Restart the PC when you are ready: the install continues by itself after you sign in.' }
  $script:Paused = $true   # Main stops here; the run after the restart continues
}
function Update-WslIfOld {
  Invoke-Wsl @('--version') -Capture | Out-Null
  if ($script:WslCode -eq 0) { return }
  Note 'this is the older WSL that comes with Windows: it needs the update from Microsoft (systemd and GPU support)'
  Act 'wsl --update' { Invoke-Wsl @('--update') } | Out-Null
}

# ------------------------------------------------------------------------------------------------ 3. distribution
function Get-LinuxUser {
  $u = ($env:USERNAME.ToLower() -replace '[^a-z0-9_-]', '')
  if ($u -notmatch '^[a-z_]') { $u = "k$u" }
  if ($u.Length -gt 31) { $u = $u.Substring(0, 31) }
  if (-not $u -or $u -eq 'root') { $u = 'kindi' }
  return $u
}
function New-KindiDistro {
  $cache = Join-Path $KHome 'download'
  $tar = Join-Path $cache $RootfsFile
  Act "download Ubuntu 24.04 for WSL from Canonical ($RootfsBase)" {
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    $sums = (Get-Web "$RootfsBase/SHA256SUMS").Content
    if ($sums -is [byte[]]) { $sums = [Text.Encoding]::ASCII.GetString($sums) }
    $line = ($sums -split "`n" | Where-Object { $_ -match [regex]::Escape($RootfsFile) } | Select-Object -First 1)
    if (-not $line) { Stop-Kindi "SHA256SUMS at $RootfsBase does not list $RootfsFile" }
    $want = ($line -split '\s+')[0].ToLower()
    $have = ''
    if (Test-Path $tar) { $have = (Get-FileHash -Algorithm SHA256 $tar).Hash.ToLower() }
    if ($have -ne $want) { Say 'downloading (about 350 MB)'; Get-Web "$RootfsBase/$RootfsFile" $tar | Out-Null; $have = (Get-FileHash -Algorithm SHA256 $tar).Hash.ToLower() }
    if ($have -ne $want) { Remove-Item -Force $tar; Stop-Kindi "the Ubuntu download has the wrong sha256 ($have, expected $want): try again; if it repeats, a proxy or filter changes the file" }
    Ok "Ubuntu 24.04 rootfs, sha256 $($want.Substring(0, 16))... matches Canonical's SHA256SUMS"
  } | Out-Null
  Act "wsl --import $Distro `"$DistroDir`" (WSL2)" {
    New-Item -ItemType Directory -Force -Path $DistroDir | Out-Null
    $o = Invoke-Wsl @('--import', $Distro, $DistroDir, $tar, '--version', '2') -Capture
    if ($script:WslCode -ne 0) {
      if ($o -match '0x80370102|virtual machine|HCS_E_HYPERV_NOT_INSTALLED') { Stop-Kindi 'WSL2 cannot start a virtual machine: virtualization is off in the BIOS/UEFI. Turn on Intel VT-x or AMD SVM there, then run this again.' }
      Stop-Kindi "wsl --import failed: $o"
    }
    Remove-Item -Force $tar
  } | Out-Null
}
function Select-Distro($st) {
  Step "Prepare the Linux system '$Distro'" 'A separate WSL2 distribution only for Kindi. Your other WSL distributions are not changed.'
  $names = @(Get-Distros)
  $script:NewDistro = $false
  if ($names -contains $Distro) {
    $v = Get-DistroVersion $Distro
    if ($v -eq 1) { Stop-Kindi "'$Distro' is a WSL 1 distribution: Kindi needs WSL2 for the GPU and Docker. Use another name (-Distro KindiWsl2), or convert it yourself with 'wsl --set-version $Distro 2'." }
    if ($st.createdDistro -and $st.distro -eq $Distro) { Ok "'$Distro' exists (made by this installer): it is reused (upgrade or repair)" }
    else {
      Note "'$Distro' is an existing WSL distribution that this installer did not make. It is reused and never replaced."
      Say 'The installer adds to it: the proxy settings (if any), Docker and the NVIDIA container toolkit (if they are missing),'
      Say 'and a Linux user for Kindi. It turns on systemd only if you agree.'
      if (-not (Ask "Use '$Distro' for Kindi?")) { Stop-Kindi "not changed. Run again without -Distro to make a separate 'Kindi' distribution." }
    }
  } else {
    if ($names.Count -gt 0) { Say "your WSL distributions ($($names -join ', ')) stay as they are" }
    New-KindiDistro
    $script:NewDistro = $true
    Set-Prop $st 'createdDistro' $true
  }
  Set-Prop $st 'distro' $Distro
  Save-State $st   # now, so that a later failure does not leave a distribution that the next run takes for a foreign one
}

# ------------------------------------------------------------------------------------------------ 4. Linux setup (root)
$RootScript = @'
#!/bin/bash
# Kindi for Windows: root setup inside the WSL2 distribution. Phases: user, proxy, docker, gpu, mem, systemd.
set -euo pipefail
PH="$1"; U="${2:-}"; ARG="${3:-}"; NEW="${4:-0}"
[ "$U" = - ] && U=""; [ "$ARG" = - ] && ARG=""      # '-' = empty (wsl.exe refuses empty arguments)
say() { echo "    $*"; }
export DEBIAN_FRONTEND=noninteractive
# a stalled mirror must not hang the install: time out, retry, then stop with apt's message
apt-get() { command apt-get -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o Acquire::Retries=3 "$@"; }
[ -f /etc/profile.d/kindi-proxy.sh ] && . /etc/profile.d/kindi-proxy.sh
case "$PH" in
user)
  id "$U" >/dev/null 2>&1 || { useradd -m -s /bin/bash -G sudo "$U"; say "Linux user $U made"; }
  groupadd -f docker; usermod -aG docker "$U"     # the Docker socket (Docker Engine or Docker Desktop's integration)
  h=$(getent passwd "$U" | cut -d: -f6); [ -e "$h/.sudo_as_admin_successful" ] || install -o "$U" -g "$U" -m 644 /dev/null "$h/.sudo_as_admin_successful"   # no "man sudo_root" hint in every new shell
  if [ "$NEW" = 1 ]; then          # a distribution that this installer made: its own /etc/wsl.conf
    printf '[boot]\nsystemd=true\n\n[user]\ndefault=%s\n\n[interop]\nappendWindowsPath=false\n' "$U" > /etc/wsl.conf
    say "/etc/wsl.conf: systemd on, default user $U"
  fi ;;
systemd-on)                         # an existing distribution, with the user's consent
  if grep -qs '^\s*systemd\s*=\s*true' /etc/wsl.conf; then say "systemd is already on"; exit 0; fi
  [ -f /etc/wsl.conf ] && cp /etc/wsl.conf "/etc/wsl.conf.kindi-$(date +%Y%m%d%H%M%S)"
  if grep -qs '^\[boot\]' /etc/wsl.conf; then sed -i '/^\[boot\]/a systemd=true' /etc/wsl.conf
  else printf '\n[boot]\nsystemd=true\n' >> /etc/wsl.conf; fi
  say "systemd turned on in /etc/wsl.conf (a copy of the old file is beside it)" ;;
proxy)
  P="$ARG"
  if [ -z "$P" ]; then rm -f /etc/apt/apt.conf.d/95kindi-proxy /etc/profile.d/kindi-proxy.sh /etc/systemd/system/docker.service.d/kindi-proxy.conf; exit 0; fi
  printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' "$P" "$P" > /etc/apt/apt.conf.d/95kindi-proxy
  printf 'export http_proxy=%s https_proxy=%s HTTP_PROXY=%s HTTPS_PROXY=%s no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1\n' "$P" "$P" "$P" "$P" > /etc/profile.d/kindi-proxy.sh
  mkdir -p /etc/systemd/system/docker.service.d
  printf '[Service]\nEnvironment="HTTP_PROXY=%s" "HTTPS_PROXY=%s" "NO_PROXY=localhost,127.0.0.1"\n' "$P" "$P" > /etc/systemd/system/docker.service.d/kindi-proxy.conf
  if [ -n "$U" ] && id "$U" >/dev/null 2>&1; then
    h=$(getent passwd "$U" | cut -d: -f6); mkdir -p "$h/.docker"
    printf '{"proxies": {"default": {"httpProxy": "%s", "httpsProxy": "%s", "noProxy": "localhost,127.0.0.1"}}}\n' "$P" "$P" > "$h/.docker/config.json"
    chown -R "$U:$U" "$h/.docker"
  fi
  say "proxy $P for apt, Docker, docker build and the shell" ;;
docker)                             # Docker Engine and the NVIDIA container toolkit inside this distribution
  [ "$(ps -p 1 -o comm=)" = systemd ] || { echo "systemd is not running in this distribution (PID 1 is $(ps -p 1 -o comm=))"; exit 3; }
  command -v curl >/dev/null && command -v gpg >/dev/null || { apt-get update -qq; apt-get install -y -qq curl ca-certificates gnupg; }
  if ! command -v docker >/dev/null; then
    say "installing Docker Engine (Ubuntu package docker.io)"
    apt-get update -qq; apt-get install -y -qq docker.io
  fi
  usermod -aG docker "$U"
  systemctl enable --now docker >/dev/null 2>&1 || systemctl start docker
  export PATH="$PATH:/usr/lib/wsl/lib"
  if [ "$ARG" = nvidia ] && ! command -v nvidia-ctk >/dev/null; then
    say "installing the NVIDIA container toolkit (NVIDIA's apt repository)"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq; apt-get install -y -qq nvidia-container-toolkit
  fi
  if [ "$ARG" = nvidia ]; then nvidia-ctk runtime configure --runtime=docker >/dev/null; systemctl restart docker; fi
  docker info >/dev/null && say "Docker $(docker --version | awk '{print $3}' | tr -d ,) runs" ;;
bridge-deps)                        # Kindi Bridge needs python3-venv, tmux 3.2+ and Node 22.19+ (pi 1.3.0: engines node >=22.19.0)
  apt-get update -qq
  apt-get install -y -qq python3 python3-venv tmux curl ca-certificates gnupg ripgrep fd-find
  # pi uses fd and rg (Bridge runs it offline, so it cannot download them); Ubuntu names fd "fdfind"
  [ -x /usr/bin/fdfind ] && [ ! -e /usr/local/bin/fd ] && ln -s /usr/bin/fdfind /usr/local/bin/fd
  nv=0.0.0; command -v node >/dev/null && nv=$(node -v | sed 's/^v//')   # (no node yet: 0.0.0)
  if [ "$(printf '%s\n%s\n' 22.19.0 "$nv" | sort -V | head -1)" != 22.19.0 ]; then
    say "installing Node.js 22 (NodeSource's signed apt repository; Ubuntu's own nodejs is older; found $nv)"
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor --yes -o /etc/apt/keyrings/nodesource.gpg
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" > /etc/apt/sources.list.d/nodesource.list
    apt-get update -qq; apt-get install -y -qq nodejs
  fi
  say "for Kindi Bridge: $(python3 --version), tmux $(tmux -V | awk '{print $2}'), Node $(node -v)" ;;
gpu)                                # does the Windows driver give the GPU to WSL?
  [ -e /dev/dxg ] && echo "dxg=yes" || echo "dxg=no"
  ls /usr/lib/wsl/lib/libcuda.so* >/dev/null 2>&1 && echo "libcuda=yes" || echo "libcuda=no"
  if [ -x /usr/lib/wsl/lib/nvidia-smi ]; then /usr/lib/wsl/lib/nvidia-smi -L 2>&1 | sed 's/^/smi: /' || true; fi ;;
mem)
  awk '/MemTotal/ {print int($2 / 1048576 + 0.5)}' /proc/meminfo ;;
esac
'@

$UserScript = @'
#!/bin/bash
# Kindi for Windows: run the Linux installer as the Kindi user. $1 = a file or a URL; the rest goes to install.sh.
set -eo pipefail
SRC="$1"; shift
export PATH="$PATH:$HOME/.local/bin:/usr/lib/wsl/lib"
[ -f /etc/profile.d/kindi-proxy.sh ] && . /etc/profile.d/kindi-proxy.sh
if [ -f "$SRC" ]; then cp "$SRC" "$HOME/kindi-install.sh"
else curl -fsSL "$SRC" -o "$HOME/kindi-install.sh.new" && mv "$HOME/kindi-install.sh.new" "$HOME/kindi-install.sh"; fi
# an install.sh without the Kindi Bridge options (before 0.6.0) does not get them
if ! grep -q -- '--with-bridge' "$HOME/kindi-install.sh"; then
  A=(); for a in "$@"; do case "$a" in --with-bridge|--no-bridge) echo "    note this install.sh has no Kindi Bridge option: $a is not used" ;; *) A+=("$a") ;; esac; done
  set -- ${A[@]+"${A[@]}"}
fi
exec bash "$HOME/kindi-install.sh" "$@"
'@

function Initialize-Linux($st) {
  $u = $st.user
  if (-not $u) { $u = Get-LinuxUser; Set-Prop $st 'user' $u }
  if ($script:NewDistro -or $st.createdDistro) {
    Act "Linux user '$u', systemd on (/etc/wsl.conf)" { Invoke-Root @('user', $u, '', '1') } | Out-Null
  } else {
    $o = Invoke-Wsl @('-d', $Distro, '-u', 'root', '--exec', 'ps', '-p', '1', '-o', 'comm=') -Capture
    Act "Linux user '$u'" { Invoke-Root @('user', $u, '', '0') } | Out-Null
    if ($o -ne 'systemd' -and $script:Mode -eq 'engine') {
      Note "systemd is off in '$Distro'. Docker Engine inside WSL needs it."
      if (Ask "Turn on systemd in '$Distro'? (this restarts only '$Distro')") { Act 'systemd on in /etc/wsl.conf' { Invoke-Root @('systemd-on') } | Out-Null }
      else { Stop-Kindi "Docker Engine cannot run in '$Distro' without systemd. Use Docker Desktop's WSL integration, or run again without -Distro." }
    }
  }
  Act "restart '$Distro' so that the settings take effect (wsl --terminate $Distro; other distributions keep running)" { Invoke-Wsl @('--terminate', $Distro) -Capture | Out-Null } | Out-Null
  Act "proxy settings in '$Distro'" { Invoke-Root @('proxy', $u, $script:Proxy) } | Out-Null
  if ($DryRun -or $Check) { return }
  $p1 = Invoke-Wsl @('-d', $Distro, '-u', 'root', '--exec', 'ps', '-p', '1', '-o', 'comm=') -Capture
  if ($script:Mode -eq 'engine' -and $p1 -ne 'systemd') { Stop-Kindi "systemd did not start in '$Distro' (PID 1 is '$p1'). Update WSL ('wsl --update', WSL 0.67.6 or newer is needed), then run this again." }
  $memGB = [int](Invoke-WslScript 'kindi-root.sh' $RootScript 'root' @('mem') -Capture)
  Ok "'$Distro' runs; Linux user '$u'; the WSL virtual machine has $memGB GB of this PC's $($script:WinRamGB) GB RAM"
  if ($script:WinRamGB -ge 24 -and $memGB -lt [math]::Floor($script:WinRamGB * 0.6)) {
    Say "Windows gives WSL2 about half of the RAM by default. The 27B and the larger models need more: to give WSL2 more,"
    Say "run this installer with -WslMemoryGB $([math]::Floor($script:WinRamGB * 0.75)) (it stops all WSL distributions once)."
  }
}

function Set-WslMemory {
  if ($WslMemoryGB -le 0) { return }
  Step "Give the WSL2 virtual machine $WslMemoryGB GB of RAM" 'This edits %UserProfile%\.wslconfig and needs ''wsl --shutdown'': ALL WSL distributions and Docker Desktop stop once.'
  $f = Join-Path $env:USERPROFILE '.wslconfig'
  $old = ''; if (Test-Path $f) { $old = Get-Content $f -Raw }
  if (-not (Ask "Set memory=${WslMemoryGB}GB and run 'wsl --shutdown' now?" -DefaultNo)) { Say 'not changed'; return }
  Act "write $f (a copy of the old file is beside it) and wsl --shutdown" {
    if ($old) { Copy-Item $f "$f.kindi-$(Get-Date -Format yyyyMMddHHmmss)" }
    $lines = @(); if ($old) { $lines = $old -split "`r?`n" }
    $out = New-Object System.Collections.Generic.List[string]; $sec = ''; $done = $false
    foreach ($l in $lines) {
      if ($l -match '^\s*\[(.+)\]') {
        if ($sec -eq 'wsl2' -and -not $done) { $out.Add("memory=${WslMemoryGB}GB"); $done = $true }
        $sec = $Matches[1].Trim().ToLower()
      }
      if ($sec -eq 'wsl2' -and $l -match '^\s*memory\s*=') { if (-not $done) { $out.Add("memory=${WslMemoryGB}GB"); $done = $true }; continue }
      $out.Add($l)
    }
    if ($sec -eq 'wsl2' -and -not $done) { $out.Add("memory=${WslMemoryGB}GB"); $done = $true }
    if (-not $done) { $out.Add('[wsl2]'); $out.Add("memory=${WslMemoryGB}GB") }
    Set-Content -Path $f -Value $out -Encoding ASCII
    Invoke-Wsl @('--shutdown') -Capture | Out-Null
  } | Out-Null
}

function Select-DockerMode {
  $dd = (Test-Path (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'))
  $m = $DockerMode
  if ($m -eq 'auto') { if ($dd) { $m = 'desktop' } else { $m = 'engine' } }
  $script:Mode = $m
  switch ($m) {
    'desktop' { Ok 'Docker Desktop is installed: Kindi uses it (its WSL integration). No second Docker is installed.' }
    'engine' { Ok "Docker Engine runs inside '$Distro' (no Docker Desktop needed)" }
    'skip' { Note 'Docker setup skipped (-DockerMode skip)' }
  }
}
function Enable-DockerDesktopIntegration {
  $o = Invoke-Wsl @('-d', $Distro, '--exec', 'docker', 'info', '--format', '{{.OperatingSystem}}') -Capture
  if ($script:WslCode -eq 0 -and $o -match 'Docker Desktop') { Ok "Docker Desktop answers in '$Distro'"; return }
  Note "Docker Desktop's WSL integration is not on for '$Distro'"
  $sf = Join-Path $env:APPDATA 'Docker\settings-store.json'; $key = 'IntegratedWslDistros'
  if (-not (Test-Path $sf)) { $sf = Join-Path $env:APPDATA 'Docker\settings.json'; $key = 'integratedWslDistros' }
  $auto = (Test-Path $sf) -and (Ask "Turn on Docker Desktop's integration for '$Distro' (edits $sf; Docker Desktop restarts)?")
  if ($auto) {
    Act "add '$Distro' to $key in $sf (a copy is beside it), restart Docker Desktop" {
      Copy-Item $sf "$sf.kindi-$(Get-Date -Format yyyyMMddHHmmss)"
      $j = Get-Content $sf -Raw | ConvertFrom-Json
      $cur = @(); if ($j.PSObject.Properties[$key]) { $cur = @($j.$key) }
      if ($cur -notcontains $Distro) { Set-Prop $j $key (@($cur) + $Distro) }
      ($j | ConvertTo-Json -Depth 20) | Set-Content -Path $sf -Encoding UTF8
      $ErrorActionPreference = 'Continue'
      & docker desktop restart 2>$null | Out-Null
      if ($LASTEXITCODE -ne 0) { Note 'restart Docker Desktop now (right-click its tray icon > Restart): the installer waits' }
    } | Out-Null
  } else {
    Say "In Docker Desktop: Settings > Resources > WSL integration > turn on '$Distro' > Apply & restart. The installer waits."
  }
  if ($DryRun -or $Check) { return }
  for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 5
    $o = Invoke-Wsl @('-d', $Distro, '--exec', 'docker', 'info', '--format', '{{.OperatingSystem}}') -Capture
    if ($script:WslCode -eq 0 -and $o -match 'Docker Desktop') { Ok "Docker Desktop answers in '$Distro'"; return }
  }
  Stop-Kindi "Docker Desktop does not answer in '$Distro' after 5 minutes. Turn on its WSL integration for '$Distro' (Settings > Resources > WSL integration), then run this again. Or use -DockerMode engine."
}

function Install-Docker($st) {
  Step 'Docker and GPU access' 'Kindi runs in a Docker container. The Windows GPU driver gives the GPU to WSL2; Linux needs no GPU driver of its own.'
  Set-Prop $st 'dockerMode' $script:Mode
  $nv = ''; if ($script:HasNvidia -and $script:GpuBuildOk) { $nv = 'nvidia' }
  switch ($script:Mode) {
    'engine' {
      Act "Docker Engine$(if ($nv) { ' and the NVIDIA container toolkit' }) in '$Distro' (apt)" { Invoke-Root @('docker', $st.user, $nv) } | Out-Null
    }
    'desktop' { Enable-DockerDesktopIntegration }
    default { }
  }
  if (-not $nv) { return }
  $g = ''
  if (-not ($DryRun -or $Check)) { $g = Invoke-WslScript 'kindi-root.sh' $RootScript 'root' @('gpu') -Capture }
  if ($DryRun -or $Check) { Say '(dry run: the GPU check runs nvidia-smi -L inside WSL)'; return }
  $smi = @($g -split "`n" | Where-Object { $_ -match '^smi: GPU \d' } | ForEach-Object { $_.Substring(5) })
  if ($g -notmatch 'dxg=yes' -or $smi.Count -eq 0) {
    Stop-Kindi ("the NVIDIA GPU does not reach WSL2 (" + (($g -split "`n") -join '; ') + "). Install the current NVIDIA Windows driver from nvidia.com (Game Ready or Studio), restart, and run this again. Do not install a Linux NVIDIA driver inside WSL: the Windows driver serves WSL. If you only want the CPU engine, run again with -LinuxArgs '--gpu cpu'.")
  }
  foreach ($s in $smi) { Ok "WSL2 sees $s" }
}

function Select-Bridge($st) {   # Kindi Bridge, the optional agent workspace (a terminal app in WSL2)
  $script:Bridge = $false
  if ($WithBridge -and $NoBridge) { Stop-Kindi 'use -WithBridge or -NoBridge, not both' }
  if ($WithBridge) { $script:Bridge = $true }
  elseif ($NoBridge -or $Yes -or $DryRun -or $Check) { $script:Bridge = $false }   # like install.sh: -Yes alone never adds it
  elseif ($st.PSObject.Properties['bridge'] -and $st.bridge) { $script:Bridge = $true; Say 'Kindi Bridge is installed: it is upgraded too' }
  else {
    Step 'Kindi Bridge (optional)' 'An agent workspace in the terminal: it runs coding agents on your Kindi servers. You can add it later with -WithBridge.'
    $script:Bridge = (Ask 'Install Kindi Bridge too?' -DefaultNo)
  }
  Set-Prop $st 'bridge' $script:Bridge
  if ($script:Bridge) {
    Act "Python 3 venv, tmux and Node.js 22 for Kindi Bridge in '$Distro' (apt)" { Invoke-Root @('bridge-deps') } | Out-Null
  }
}

# ------------------------------------------------------------------------------------------------ 5. install.sh
function Invoke-LinuxInstaller($st) {
  Step 'Run the Kindi Linux installer inside WSL2' 'install.sh checks the GPU, lists the setups, downloads and checks the model, and starts the server.'
  $src = $InstallShUrl
  if ($InstallShPath) {
    $src = Invoke-Wsl @('-d', $Distro, '-u', 'root', '--exec', 'wslpath', '-a', (Resolve-Path $InstallShPath).Path) -Capture
  }
  $la = @(); if ($LinuxArgs) { $la = @($LinuxArgs.Trim() -split '\s+') }
  if ($script:Bridge) { $la += '--with-bridge' } else { $la += '--no-bridge' }   # this installer asked already
  if ($Yes -and ($la -notcontains '--yes') -and ($la -notcontains '-y') -and ($la -notcontains '--auto')) { $la += '--yes' }
  if ($DryRun -and ($la -notcontains '--dry-run')) { $la += '--dry-run' }
  if ($Check) { Say "(check only: install.sh $src $($la -join ' '))"; return }
  if ($DryRun -and -not (((Get-Distros) -contains $Distro) -and (Test-LinuxUser $st.user))) { Say "> bash install.sh $($la -join ' ')  (as '$($st.user)' in '$Distro')"; return }
  Say "install.sh: $src $($la -join ' ')"
  Invoke-WslScript 'kindi-run-install.sh' $UserScript $st.user (@($src) + $la)
  if ($script:WslCode -ne 0) { Stop-Kindi "install.sh stopped (exit code $($script:WslCode)); its message is above. Run this command again after you fix it: the downloads are kept." }
  if ($script:Bridge -and -not $DryRun) {   # the shortcut only when install.sh really installed Bridge
    Invoke-Wsl @('-d', $Distro, '-u', $st.user, '--cd', '~', '--exec', 'test', '-x', '.local/bin/kindi-bridge') -Capture | Out-Null
    if ($script:WslCode -ne 0) {
      Note 'Kindi Bridge was not installed (this install.sh has no Bridge option, or its Bridge step stopped): no Bridge item in the Start menu'
      $script:Bridge = $false; Set-Prop $st 'bridge' $false
    }
  }
}

# ------------------------------------------------------------------------------------------------ 6. Windows side
function Get-Servers($st) {
  $j = Invoke-Wsl @('-d', $Distro, '-u', $st.user, '--cd', '~', '--exec', 'cat', '.kindi/install.json') -Capture
  if ($script:WslCode -ne 0 -or -not $j) { return @() }
  try { return @(($j | ConvertFrom-Json).servers) } catch { return @() }
}
function Test-Localhost($st) {
  Step 'Check the server from Windows' 'WSL2 forwards the server port to this PC, so Windows programs use http://localhost:PORT/v1.'
  if ($DryRun -or $Check -or ($LinuxArgs -match '--dry-run')) { Say '(dry run: no server to check)'; return }
  $sv = @(Get-Servers $st)
  if ($sv.Count -eq 0) { Note 'no server is listed in ~/.kindi/install.json'; return }
  foreach ($s in $sv) {
    $u = "http://127.0.0.1:$($s.port)/health"; $okk = $false
    for ($i = 0; $i -lt 12 -and -not $okk; $i++) {
      try { Invoke-RestMethod -Uri $u -TimeoutSec 10 | Out-Null; $okk = $true } catch { Start-Sleep -Seconds 5 }
    }
    if ($okk) { Ok "$($s.container): http://localhost:$($s.port)/v1 answers on Windows ($($s.model))" }
    else {
      $wc = Join-Path $env:USERPROFILE '.wslconfig'
      $hint = 'WSL2 forwards localhost by default.'
      if ((Test-Path $wc) -and ((Get-Content $wc -Raw) -match 'localhostForwarding\s*=\s*false')) { $hint = "localhostForwarding=false is set in $wc. Remove it, then run 'wsl --shutdown'." }
      Note "$($s.container) does not answer at $u from Windows. $hint Inside WSL it is at the same address: 'wsl -d $Distro -- curl -s $u'."
    }
  }
}

$ConsoleScript = @'
# Kindi Console for Windows: the status of the Kindi servers in WSL2, a chat window, the logs and the API settings.
param([string]$Distro = '__DISTRO__', [string]$User = '__USER__', [string]$AppName = '__APP__')
$ErrorActionPreference = 'Continue'
$env:WSL_UTF8 = '1'
$Host.UI.RawUI.WindowTitle = "$AppName Console"
function W([string[]]$A) { ((& wsl.exe @A 2>$null | ForEach-Object { "$_" }) -join "`n") -replace "`0", '' }   # stdout only (WSL notes go to stderr)
function Servers { try { @((W @('-d', $Distro, '-u', $User, '--cd', '~', '--exec', 'cat', '.kindi/install.json') | ConvertFrom-Json).servers) } catch { @() } }
function Show-Status {
  Write-Host ''; Write-Host "$AppName (WSL2 distribution '$Distro')" -ForegroundColor White
  $sv = @(Servers)
  if ($sv.Count -eq 0) { Write-Host '  no Kindi server is installed. Run the installer again.' -ForegroundColor Yellow; return $sv }
  $i = 0
  foreach ($s in $sv) {
    $i++
    $st = 'not answering'
    try { $h = Invoke-RestMethod "http://127.0.0.1:$($s.port)/health" -TimeoutSec 5; $st = 'running'; if ($h.busy) { $st = "busy ($($h.busy | ConvertTo-Json -Compress))" } } catch { }
    Write-Host ("  [{0}] {1}  http://localhost:{2}/v1  {3}, {4} tokens  - {5}" -f $i, $s.container, $s.port, $s.model, $s.ctx, $st)
  }
  return $sv
}
function Chat($s) {
  $u = "http://127.0.0.1:$($s.port)/v1/chat/completions"; $msgs = New-Object System.Collections.ArrayList; $think = $false
  Write-Host "Chat with $($s.model). /think turns thinking on or off, /clear starts again, /exit goes back." -ForegroundColor DarkGray
  while ($true) {
    $q = Read-Host 'you'
    if ($q -eq '/exit' -or $q -eq $null) { return }
    if ($q -eq '/clear') { $msgs.Clear(); Write-Host '(new chat)' -ForegroundColor DarkGray; continue }
    if ($q -eq '/think') { $think = -not $think; Write-Host "(thinking $(if ($think) {'on'} else {'off'}))" -ForegroundColor DarkGray; continue }
    if (-not $q.Trim()) { continue }
    [void]$msgs.Add(@{ role = 'user'; content = $q })
    $body = @{ messages = @($msgs); chat_template_kwargs = @{ enable_thinking = $think } } | ConvertTo-Json -Depth 6
    try {
      $t0 = Get-Date
      $r = Invoke-RestMethod -Method Post -Uri $u -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 3600
      $a = $r.choices[0].message.content
      [void]$msgs.Add(@{ role = 'assistant'; content = $a })
      Write-Host $a -ForegroundColor Cyan
      if ($r.usage) { Write-Host ("({0} tokens in {1:n1} s)" -f $r.usage.completion_tokens, ((Get-Date) - $t0).TotalSeconds) -ForegroundColor DarkGray }
    } catch { Write-Host "error: $($_.Exception.Message)" -ForegroundColor Red; $msgs.RemoveAt($msgs.Count - 1) }
  }
}
while ($true) {
  $sv = @(Show-Status)        # @(): PowerShell 5.1 unrolls a one-server list into an object without .Count
  Write-Host ''
  Write-Host '  c N  chat with server N     l N  logs of server N     r N  restart server N'
  Write-Host '  a    API settings for OpenAI clients     s  Linux shell     q  quit'
  $c = (Read-Host "$AppName").Trim() -split '\s+'
  $n = 1; if ($c.Count -gt 1) { [int]::TryParse($c[1], [ref]$n) | Out-Null }
  $s = $null; if ($sv.Count -ge $n -and $n -ge 1) { $s = $sv[$n - 1] }
  switch ($c[0]) {
    'c' { if ($s) { Chat $s } }
    'l' { if ($s) { & wsl.exe -d $Distro -u $User --exec docker logs --tail 60 $s.container } }
    'r' { if ($s) { & wsl.exe -d $Distro -u $User --exec docker restart $s.container | Out-Null; Write-Host 'restarted' } }
    'a' { foreach ($x in $sv) { Write-Host "  OPENAI_BASE_URL=http://localhost:$($x.port)/v1   OPENAI_API_KEY=none   model: any name ($($x.model))" } }
    's' { & wsl.exe -d $Distro --cd '~' }
    'q' { return }
    default { }
  }
}
'@

function Add-WindowsItems($st) {
  Step 'Add Kindi to Windows' 'Start-menu items, a desktop shortcut, the Settings > Apps entry, and a task that keeps WSL2 running for the server.'
  Act "save the installer and the console in $KHome" {
    Save-Self | Out-Null
    $c = $ConsoleScript.Replace('__DISTRO__', $Distro).Replace('__USER__', $st.user).Replace('__APP__', $AppName)
    [IO.File]::WriteAllText((Join-Path $KHome 'kindi-console.ps1'), $c, (New-Object Text.UTF8Encoding($false)))
  } | Out-Null
  $ps = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
  $con = Join-Path $KHome 'kindi-console.ps1'
  $un = "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $KHome 'install.ps1')`" -Uninstall -AppName `"$AppName`" -Distro `"$Distro`""
  $menu = Join-Path ([Environment]::GetFolderPath('Programs')) $AppName
  $items = @(
    @{ f = (Join-Path $menu "$AppName Console.lnk"); t = $ps; a = "-NoProfile -ExecutionPolicy Bypass -NoExit -File `"$con`""; d = 'Status, chat, logs and API settings of the Kindi servers' },
    @{ f = (Join-Path $menu "$AppName Shell (WSL).lnk"); t = $WslExe; a = "-d $Distro --cd ~"; d = "A Linux shell in the '$Distro' WSL2 distribution" },
    @{ f = (Join-Path $menu "Uninstall $AppName.lnk"); t = $ps; a = $un; d = 'Remove Kindi' }
  )
  if ($script:Bridge) {   # Bridge is a tmux/curses app: Windows Terminal if it is there, else the console
    $wt = ''; $c = Get-Command wt.exe -ErrorAction SilentlyContinue; if ($c) { $wt = $c.Source }
    $alias = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\wt.exe'   # the app alias of Windows Terminal
    if (-not $wt -and (Test-Path $alias)) { $wt = $alias }
    if ($wt) { $items += @{ f = (Join-Path $menu 'Kindi Bridge.lnk'); t = $wt; a = "`"$WslExe`" -d $Distro -- bash -lc kindi-bridge"; d = 'Kindi Bridge: the agent workspace' } }
    else { $items += @{ f = (Join-Path $menu 'Kindi Bridge.lnk'); t = $WslExe; a = "-d $Distro -- bash -lc kindi-bridge"; d = 'Kindi Bridge: the agent workspace' } }
  }
  if (-not $NoDesktopShortcut) { $items += @{ f = (Join-Path ([Environment]::GetFolderPath('Desktop')) "$AppName Console.lnk"); t = $ps; a = "-NoProfile -ExecutionPolicy Bypass -NoExit -File `"$con`""; d = 'Kindi Console' } }
  Act "Start menu: $menu (Console, Shell, Uninstall)$(if (-not $NoDesktopShortcut) { '; desktop: Console' })" {
    New-Item -ItemType Directory -Force -Path $menu | Out-Null
    $sh = New-Object -ComObject WScript.Shell
    foreach ($i in $items) { $l = $sh.CreateShortcut($i.f); $l.TargetPath = $i.t; $l.Arguments = $i.a; $l.Description = $i.d; $l.WorkingDirectory = $KHome; $l.Save() }
  } | Out-Null
  Act "Settings > Apps entry '$AppName (WSL2)'" {
    $k = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$AppName"
    New-Item -Force -Path $k | Out-Null
    $v = @{ DisplayName = "$AppName (WSL2)"; DisplayVersion = $KindiWinVersion; Publisher = 'Amir Husain'; InstallLocation = $KHome
            UninstallString = "`"$ps`" $un"; URLInfoAbout = 'https://amirhusain.com/kindi'; NoModify = 1; NoRepair = 1 }
    foreach ($n in $v.Keys) { Set-ItemProperty -Path $k -Name $n -Value $v[$n] }
  } | Out-Null
  if ($script:Mode -eq 'engine') {
    # WSL2 stops an idle distribution; this hidden process keeps it (and the Docker containers in it) running
    $startup = Join-Path ([Environment]::GetFolderPath('Startup')) "$AppName WSL keep-alive.lnk"
    $conhost = Join-Path $env:WINDIR 'System32\conhost.exe'
    $ka = "--headless `"$WslExe`" -d $Distro --exec /bin/sleep infinity"
    Act "keep '$Distro' running: a hidden process at sign-in ($startup), started now" {
      $sh = New-Object -ComObject WScript.Shell
      $l = $sh.CreateShortcut($startup); $l.TargetPath = $conhost; $l.Arguments = $ka; $l.WindowStyle = 7; $l.Description = "Keeps the $AppName WSL2 distribution running"; $l.Save()
      $old = [int]$st.keepAlivePid
      $alive = $false
      if ($old -gt 0) { $pr = Get-CimInstance Win32_Process -Filter "ProcessId=$old" -ErrorAction SilentlyContinue; $alive = ($pr -and $pr.CommandLine -match [regex]::Escape("-d $Distro --exec /bin/sleep infinity")) }
      if (-not $alive) { $p = Start-Process -FilePath $conhost -ArgumentList $ka -WindowStyle Hidden -PassThru; Set-Prop $st 'keepAlivePid' $p.Id }
    } | Out-Null
  }
}

function Show-Done($st) {
  Write-Host ''
  if ($DryRun -or ($LinuxArgs -match '--dry-run')) { Write-Host 'Dry run done: the steps above show what the install does.' -ForegroundColor Green; return }
  Write-Host "$AppName is installed." -ForegroundColor Green
  Write-Host ''
  $sv = @(); if (-not ($DryRun -or $Check)) { $sv = @(Get-Servers $st) }
  foreach ($s in $sv) { Say "$($s.container)  http://localhost:$($s.port)/v1  $($s.model), $($s.ctx) tokens" }
  Say "Start menu > $AppName > $AppName Console: status, chat, logs and the API settings."
  if ($script:Bridge) { Say "Start menu > $AppName > Kindi Bridge: the agent workspace (or 'wsl -d $Distro -- bash -lc kindi-bridge')." }
  Say "OpenAI clients on Windows: OPENAI_BASE_URL=http://localhost:$(if ($sv.Count) { $sv[0].port } else { 8093 })/v1  OPENAI_API_KEY=none"
  Say "Linux shell: wsl -d $Distro      Run this installer again to upgrade or repair; Settings > Apps > $AppName to remove it."
}

# ------------------------------------------------------------------------------------------------ uninstall
function Remove-Kindi {
  $st = Get-State
  $d = $Distro; if ($st.distro) { $d = $st.distro }
  Write-Host "$AppName for Windows: uninstall" -ForegroundColor White
  $names = @(Get-Distros)
  if ($names -contains $d -and $st.user) {
    $ua = @('--uninstall'); if ($Purge) { $ua += '--purge' }; if ($Yes) { $ua += '--yes' }
    Act "install.sh $($ua -join ' ') in '$d' (removes the Kindi containers and image)" {
      Invoke-WslConsole (@('-d', $d, '-u', $st.user, '--cd', '~', '--exec', 'bash', 'kindi-install.sh') + $ua)
    } | Out-Null
  }
  $startup = Join-Path ([Environment]::GetFolderPath('Startup')) "$AppName WSL keep-alive.lnk"
  Act 'stop and remove the keep-alive process' {
    $old = [int]$st.keepAlivePid
    if ($old -gt 0) {
      $pr = Get-CimInstance Win32_Process -Filter "ProcessId=$old" -ErrorAction SilentlyContinue
      if ($pr -and $pr.CommandLine -match [regex]::Escape("-d $d --exec /bin/sleep infinity")) { Stop-Process -Id $old -Force }
    }
    Remove-Item -Force -ErrorAction SilentlyContinue $startup
  } | Out-Null
  if ($names -contains $d) {
    if ($st.createdDistro) {
      if ($Purge -or (Ask "Delete the '$d' WSL distribution with its models and downloads?" -DefaultNo)) {
        Act "wsl --unregister $d (deletes it and its disk)" { Invoke-Wsl @('--unregister', $d) -Capture | Out-Null } | Out-Null
      } else { Act "stop '$d' (it stays; delete it later with 'wsl --unregister $d')" { Invoke-Wsl @('--terminate', $d) -Capture | Out-Null } | Out-Null }
    } else { Say "'$d' was not made by this installer: it stays (only Kindi was removed from it)" }
  }
  Act 'remove the Start-menu items, the desktop shortcut and the Settings entry' {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path ([Environment]::GetFolderPath('Programs')) $AppName)
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path ([Environment]::GetFolderPath('Desktop')) "$AppName Console.lnk")
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$AppName"
    Remove-ItemProperty -ErrorAction SilentlyContinue -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce' -Name "$AppName-install"
  } | Out-Null
  $gone = -not ((Get-Distros) -contains $d)
  if ($gone -or $Purge) {
    if ($script:logging -and -not $DryRun) { try { Stop-Transcript | Out-Null } catch { }; $script:logging = $false }   # the log is in $KHome
    Act "remove $KHome" {
      Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $KHome
      # this script runs from that folder, so the folder itself can stay: remove it a few seconds after the exit
      if (Test-Path $KHome) { Start-Process cmd.exe -WindowStyle Hidden -ArgumentList "/c timeout /t 3 /nobreak >nul & rmdir /s /q `"$KHome`"" }
    } | Out-Null
  }
  else { Act "remove $KHome (keeps state.json for a later uninstall of '$d')" { Get-ChildItem $KHome -Exclude 'state.json', 'install.ps1' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue } | Out-Null }
  Write-Host "$AppName for Windows is removed (Start menu, Settings entry, keep-alive$(if ($gone) { ', the WSL2 distribution' }))." -ForegroundColor Green
}

# ------------------------------------------------------------------------------------------------ main
function Main {
  if ($ElevatedEnableWsl) { Enable-WslElevated; return }
  Write-Host "Kindi for Windows ($KindiWinVersion)$(if ($DryRun) { ' (dry run: nothing changes)' } elseif ($Check) { ' (check only)' })" -ForegroundColor White
  if ($Uninstall) { Remove-Kindi; return }
  if ($Resume) { Remove-ItemProperty -ErrorAction SilentlyContinue -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce' -Name "$AppName-install"; Say 'continuing after the restart' }
  if ((Test-Admin) -and -not $Resume) { Note 'this window runs as administrator. That is not needed: WSL distributions belong to the Windows user who makes them.' }
  Test-Pc
  if (-not (Test-WslReady)) {
    if ($Check) { Note 'WSL2 is off: the install turns it on (administrator approval and a restart)'; return }
    Enable-Wsl
    if ($DryRun) { Say '(dry run: the other steps run after the restart, when WSL2 is on)'; return }
    if ($script:Paused) { return }
  }
  Ok 'WSL2 is on'
  Update-WslIfOld
  Set-WslMemory
  $st = Get-State
  Select-DockerMode
  if ($Check) {
    $n = @(Get-Distros)
    if ($n -contains $Distro) { Say "the install would reuse '$Distro'" } else { Say "the install would make the WSL2 distribution '$Distro' in $DistroDir" }
    Say "then: Docker ($($script:Mode)), install.sh $LinuxArgs, the Start-menu items. Run without -Check to install."
    return
  }
  Select-Distro $st
  Initialize-Linux $st
  Save-State $st
  Install-Docker $st
  Select-Bridge $st
  Save-State $st
  Invoke-LinuxInstaller $st
  Test-Localhost $st
  Add-WindowsItems $st
  Save-State $st
  Show-Done $st
}

# every run writes a log (also the run that continues after the restart, which has no window to read later)
$Log = Join-Path $KHome 'install.log'
$logging = $false
if (-not ($ElevatedEnableWsl -or $DryRun -or $Check)) {   # -DryRun and -Check change nothing, not even a log
  try { New-Item -ItemType Directory -Force -Path $KHome | Out-Null; Start-Transcript -Path $Log -Append | Out-Null; $logging = $true } catch { }
}
try { Main }
catch {
  $m = "$($_.Exception.Message)"
  Write-Host ''
  if ($m.StartsWith('KINDI-STOP: ')) { Write-Host 'stopped: ' -ForegroundColor Red -NoNewline; Write-Host $m.Substring(12) }
  else { Write-Host "stopped: $m" -ForegroundColor Red; Write-Host "    at $($_.InvocationInfo.PositionMessage)" -ForegroundColor DarkGray }
  if ($logging) { Write-Host "    the log of this run: $Log"; try { Stop-Transcript | Out-Null } catch { } }
  if ($PSCommandPath) { exit 1 }
  return
}
if ($logging) { try { Stop-Transcript | Out-Null } catch { } }
