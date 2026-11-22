# Build the native Kindi runtime for Windows: one folder (and a zip) with kindi.exe, kindi-monitor.exe, the
# python.org embeddable Python, the wheels of runtime.lock and the Kindi source. docs/dev/F8-WINDOWS.md, section 7.
#
#   powershell -NoProfile -File windows\package.ps1 [-Lock] [-Out DIR] [-NoZip]
#
#   -Lock    resolve runtime.in for CPython 3.12 / win_amd64 and write runtime.lock (the wheels and their sha256)
#   -Out     the build folder (default windows\build); the runtime is Out\Kindi, the zip Out\kindi-VERSION-windows-x64.zip
#
# Every download is checked against a sha256 in runtime.lock. The host needs Python 3.8+ with pip (only to download
# wheels; nothing is installed on the host) and MSVC for windows\build.ps1.
# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
param([switch]$Lock, [string]$Out = "", [switch]$NoZip)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
if (-not $Out) { $Out = Join-Path $here "build" }
New-Item -ItemType Directory -Force $Out | Out-Null
$PyVer = "3.12.10"
$PyTag = "312"
$PyUrl = "https://www.python.org/ftp/python/$PyVer/python-$PyVer-embed-amd64.zip"
$lockFile = Join-Path $here "runtime.lock"
$wheels = Join-Path $Out "wheels"
New-Item -ItemType Directory -Force $wheels | Out-Null

function Sha256($path) { (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower() }

if ($Lock) {
    # resolve for the target (not for the host's Python), wheels only
    & python -m pip download --quiet --only-binary=:all: --platform win_amd64 --python-version $PyTag `
        --implementation cp -d $wheels -r (Join-Path $here "runtime.in")
    if ($LASTEXITCODE -ne 0) { throw "pip download failed" }
    $py = Join-Path $Out "python-$PyVer-embed-amd64.zip"
    if (-not (Test-Path $py)) { Invoke-WebRequest -UseBasicParsing $PyUrl -OutFile $py }
    $lines = @("# windows/runtime.lock: written by package.ps1 -Lock from runtime.in; one file per line: sha256  name  url",
               "$(Sha256 $py)  python-$PyVer-embed-amd64.zip  $PyUrl")
    Get-ChildItem $wheels -Filter *.whl | Sort-Object Name | ForEach-Object { $lines += "$(Sha256 $_.FullName)  $($_.Name)  pypi" }
    [IO.File]::WriteAllText($lockFile, ($lines -join "`n") + "`n")
    "runtime.lock: $($lines.Count - 1) files"
}

# ---- the files of runtime.lock, checked
$entries = Get-Content $lockFile | Where-Object { $_ -and -not $_.StartsWith("#") } | ForEach-Object {
    $p = $_ -split "\s+"; [pscustomobject]@{ Sha = $p[0]; Name = $p[1]; Url = $p[2] } }
foreach ($e in $entries) {
    $dest = if ($e.Name -like "*.whl") { Join-Path $wheels $e.Name } else { Join-Path $Out $e.Name }
    if (-not (Test-Path $dest)) {
        if ($e.Url -eq "pypi") {
            $spec = ($e.Name -split "-")[0..1] -join "=="
            & python -m pip download --quiet --no-deps --only-binary=:all: --platform win_amd64 --python-version $PyTag `
                --implementation cp -d $wheels $spec
        } else { Invoke-WebRequest -UseBasicParsing $e.Url -OutFile $dest }
    }
    if ((Sha256 $dest) -ne $e.Sha) { throw "sha256 mismatch: $($e.Name)" }
}

# ---- the folder
& powershell -NoProfile -File (Join-Path $here "build.ps1") -Out $Out | Out-Host
$k = Join-Path $Out "Kindi"
if (Test-Path $k) { Remove-Item -Recurse -Force $k }
New-Item -ItemType Directory -Force $k, (Join-Path $k "python"), (Join-Path $k "app") | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::ExtractToDirectory((Join-Path $Out "python-$PyVer-embed-amd64.zip"), (Join-Path $k "python"))
$site = Join-Path $k "python\Lib\site-packages"
New-Item -ItemType Directory -Force $site | Out-Null
foreach ($e in $entries | Where-Object { $_.Name -like "*.whl" }) {
    # a wheel is a zip: its files go into site-packages as pip would put them (no scripts are needed)
    $z = [IO.Compression.ZipFile]::OpenRead((Join-Path $wheels $e.Name))
    try {
        foreach ($f in $z.Entries) {
            if (-not $f.Name) { continue }
            $rel = $f.FullName -replace "^[^/]+\.data/(purelib|platlib)/", ""
            if ($rel -match "^[^/]+\.data/") { continue }
            $dest = Join-Path $site $rel
            New-Item -ItemType Directory -Force (Split-Path -Parent $dest) | Out-Null
            [IO.Compression.ZipFileExtensions]::ExtractToFile($f, $dest, $true)
        }
    } finally { $z.Dispose() }
}
# CuPy loads nvrtc64_120_0.dll; the wheel's second copy (.alt.dll, 90 MB) is not used
Get-ChildItem (Join-Path $site "nvidia\cuda_nvrtc\bin") -Filter "*.alt.dll" | Remove-Item
# python312._pth: the embeddable Python ignores PYTHONPATH; this file is its whole search path
[IO.File]::WriteAllText((Join-Path $k "python\python$PyTag._pth"), "python$PyTag.zip`n.`nLib\site-packages`n..\app`nimport site`n")
# the Kindi source: the files git tracks under kindi/ (the .cu kernels compile with NVRTC on first use)
Push-Location $repo
try {
    git ls-files kindi | ForEach-Object {
        $dest = Join-Path (Join-Path $k "app") $_
        New-Item -ItemType Directory -Force (Split-Path -Parent $dest) | Out-Null
        Copy-Item $_ $dest
    }
    foreach ($f in "LICENSE", "NOTICE", "COMMERCIAL-LICENSE.md") { Copy-Item $f $k }
} finally { Pop-Location }
Copy-Item (Join-Path $Out "kindi.exe"), (Join-Path $Out "kindi-monitor.exe"), (Join-Path $here "setup.ps1") $k
# the Windows installer (WSL2 + the Linux installer): the monitor runs it for the models that run only in WSL
Copy-Item (Join-Path $repo "install.ps1") $k
# double-click to install for this user (no administrator prompt)
[IO.File]::WriteAllText((Join-Path $k "Install Kindi.cmd"), "@powershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0setup.ps1`" %*`r`n@pause`r`n")
# the commit of this build, for /health and kindi --version (the source has no .git)
Push-Location $repo
try { [IO.File]::WriteAllText((Join-Path $k "app\kindi\COMMIT"), (git rev-parse --short=12 HEAD).Trim() + "`n") } finally { Pop-Location }
$third = @("Kindi for Windows bundles these programs; each keeps its own license (see python\LICENSE.txt and the",
           "*.dist-info folders under python\Lib\site-packages):", "",
           "  Python $PyVer (PSF License)")
foreach ($e in $entries | Where-Object { $_.Name -like "*.whl" }) { $third += "  " + (($e.Name -split "-")[0..1] -join " ") }
$third += "", "nvidia-cuda-nvrtc-cu12: the NVIDIA CUDA runtime compiler (NVRTC), redistributed under the NVIDIA CUDA",
          "Toolkit EULA (its redistributable files); its license is in the nvidia_cuda_nvrtc_cu12 dist-info folder."
[IO.File]::WriteAllText((Join-Path $k "THIRD-PARTY.txt"), ($third -join "`n") + "`n")
$bytes = (Get-ChildItem -Recurse -File $k | Measure-Object -Sum Length).Sum
"Kindi folder: {0:N0} bytes ({1:N0} MB) in {2}" -f $bytes, ($bytes / 1MB), $k
if (-not $NoZip) {
    $ver = (Select-String -Path (Join-Path $repo "kindi\__init__.py") -Pattern '^VERSION = "(.*)"').Matches[0].Groups[1].Value
    $zip = Join-Path $Out "kindi-$ver-windows-x64.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    [IO.Compression.ZipFile]::CreateFromDirectory($k, $zip, [IO.Compression.CompressionLevel]::Optimal, $true)
    "zip: {0:N0} bytes ({1:N0} MB) {2}" -f (Get-Item $zip).Length, ((Get-Item $zip).Length / 1MB), $zip
}
