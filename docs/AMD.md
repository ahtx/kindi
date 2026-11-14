# Kindi on AMD graphics

Kindi 0.3.0 runs on AMD RDNA3 graphics through ROCm. We tested it on the **Radeon 780M**, the graphics unit inside
the AMD Ryzen 7 7840HS, Ryzen 9 7940HS and similar chips. Our test computer is a Minisforum UM790 Pro mini PC with a
Ryzen 9 7940HS and 64 GB of DDR5-5600.

The 780M has no memory of its own. It uses the main memory of the computer, so with 64 GB it can hold the 27B, its
context and other programs at the same time.

## What runs

| Model | Status on the Radeon 780M |
|---|---|
| Ternary-Bonsai-2 27B | runs; context 24,576 |
| NVIDIA Nemotron 3.5 Bolt E4B | runs; tool calls and the prompt cache |
| Ternary-Bonsai-2 2B | runs; long prompts need `amdgpu.cwsr_enable=0` (below) |

The speeds, against llama.cpp on the same computer with the same model files, are in [BENCHMARKS.md](BENCHMARKS.md).

Kindi 0.5.0 adds RDNA 3.5 graphics. The Radeon 840M (gfx1152, in the Ryzen AI 5 340) and the Radeon 8060S (Strix
Halo, gfx1151, in the Ryzen AI MAX+ 395) run the 2B, 4B, 9B and 27B, without `HSA_OVERRIDE_GFX_VERSION`. `kindi
verify` passes on both. The installer builds the `kindi-rocm7` image for them (ROCm 7, HIP 7.16). Other RDNA3 graphics
(gfx1100, gfx1101, gfx1102) and Strix Point (gfx1150) use the same code. We did not measure them.

## Experimental RDNA2 work

The core signed-dot helper selects `sdot4` on gfx1030–gfx1036 instead of the
RDNA3-only `sudot4` instruction. A native Windows TheRock build of the complete
core kernel module executed 256 signed-dot cases correctly on an RX6600.
The RX6600 default uses one lane-layout packed-weight copy to avoid two resident
copies of the 27B weights; explicit layout environment overrides remain available.

The initial native Windows attempt was blocked by CuPy's explicit Windows HIP
compiler rejection and missing profiling libraries. A separate CuPy source port
now implements Windows HIP compilation, runtime DLL discovery and optional
profiling exclusion. It retains the MIT license and reports unavailable profiling
APIs as errors, not fake successes. The existing Kindi installer does not supply
this dependency or claim native AMD support.

### Verified configuration

- RX6600, gfx1032; native Windows, Python3.12, Visual Studio2022 build tools,
  and TheRock HIP7.18.26446.
- Source-built CuPy14.3.0.dev0 from revision
  `694e66cd450de584a5587e5a636ed265dcfa866e` plus the Windows HIP patch.
- Real host/device transfers, runtime kernel compilation, matrix multiplication
  and reductions pass; reviewed dense/sparse Windows HIP suite:28passed,6skipped.
- Bonsai2B and Bonsai27B each pass all four reference prompts,40/40tokens each.
- Bonsai27B PTQ1_0 uses5.60GB of lane-layout GPU weights. With32,768allocated
  context tokens and a real8,192-token prompt, one measured run gives278tok/s
  prefill and20.1tok/s greedy decode. This is not a distribution or a coding score.
  The 32K allocation exceeded the observed local budget slightly; near-full
  generation later measured11.7tok/s.

### Running with a built CuPy HIP environment

Use a native Windows-compatible source checkout and virtual environment. Set
`ROCM_HOME` or `HIP_PATH` to the real toolkit, and retain its headers, import
libraries and DLLs. The CuPy patch's installation documentation describes the
Clang/MSVC-compatible build. From the Kindi checkout, with that environment active:

```powershell
$env:HIP_PATH = 'C:\ROCm'
$env:ROCM_HOME = $env:HIP_PATH
$env:KINDI_HIP_LIB = "$env:HIP_PATH\bin\amdhip64_7.dll"
$env:PATH = "$env:HIP_PATH\bin;$env:HIP_PATH\lib\llvm\bin;$env:PATH"
python -m kindi convert C:\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf --out C:\models\bonsai27-pack
python -m kindi --pack C:\models\bonsai27-pack verify
```

Replace the example paths with actual installed locations. Verify the exact
PTQ1_0 file checksum from the installation documentation before conversion.
The ordinary `kindi bench` measures synthetic positions; real filled-context
throughput must include actual prompt processing. No RX6600 quality advantage,
other-GPU certification, or automatic daily-driver promotion follows from this run.

### Native Windows tuning

The tested desktop's DXGI local budget was about6.6GiB, below nominal8GiB.
Opt-in `KINDI_WDDM_BUDGET=1`, a256MiB reserve and `KINDI_CHUNK=128` produced
roughly18K-class actual windows at24tok/s after8K and22tok/s near full.

| Profile | Actual tested window | Near-full decode |
|---|---:|---:|
| All-GPU, unbudgeted |32,768 |11.7tok/s |
| All-GPU, budget fit, chunk128 |18,688 |22.1tok/s |
| Mapped head, budget fit |25,088 |15.0tok/s |
| Large allocation, all-GPU |61,440 |1.5–1.8tok/s |
| Mapped head plus16FFN layers |65,536 |4.6tok/s |

Cache reuse cut two follow-ups on a5.6K-token conversation from22.5s each to
about1.5s. Pinned checkpoints added no benefit. Cold off/on summaries differed
with different actual windows/checkpoint partitioning; this comparison does not
establish bit-identical caching. The selected profile keeps the ordinary cache
enabled and pinned checkpoints disabled.

The optional RDNA2 TC/ngram path copied one file exactly at52tok/s, but cold
prefill was slower and general8K output differed from the full-fidelity baseline.
It is not the selected default. Linux `HSA_OVERRIDE_GFX_VERSION` setups are
untested; an RX6600 reporting gfx1030 gets earlier row/dual defaults unless
layout flags are set explicitly. Do not infer that setup from Windows evidence.

### Frozen coding comparison

The final profile used Q8 K/V, exact head, one G5 lane copy, chunk128,
`KINDI_WDDM_BUDGET=1`, a256MiB reserve, ordinary host checkpoints and cache on.
Actual served context was19,200tokens. All three tasks used seed1, medium effort,
256thinking tokens and a4096total output cap per request, with a300second episode.
The native API budget field is `thinking_budget_tokens`.

| Task | Result | Episode time |
|---|---|---:|
| Pagination | completed and correct |137s |
| Configuration | repeated failing calls, then context exceeded |291s |
| Streaming parser | timeout, incorrect code |300s |

This1/3 result does not displace the preserved IQ3 2/3 practical-profile result.
No profile was retuned against these answers. Performance, runtime, context,
templates and budget handling differ; this is not a pure model-quality ranking.

To reproduce the engine controls, set the following before starting native Kindi:

```powershell
$env:KINDI_G5L = '1'
$env:KINDI_G5_DUAL = '0'
$env:KINDI_WDDM_BUDGET = '1'
$env:KINDI_CHUNK = '128'
$env:KINDI_MEM_RESERVE_MB = '256'
$env:KINDI_CKPT_PINNED = '0'
$env:KINDI_PROMPT_CACHE = '1'
```

Use an explicit state/checkpoint budget for the server (the test used1024MiB),
and inspect its reported actual context rather than assuming the requested32K.
Exact observed split pins and all request/response records are retained in the
research study; other GPU/driver configurations require their own measurements.






## Install

The installer finds the 780M and lists the 27B first:

```bash
curl -fsSL https://amirhusain.com/kindi/install.sh | bash
```

It builds the `kindi-rocm` image (ROCm 6.3.3 and CuPy 14.2.0) for RDNA3 graphics, and the `kindi-rocm7` image (ROCm 7
with HIP 7.16 and CuPy 14.2.0) for RDNA 3.5 graphics. The computer needs only:

- Linux with the `amdgpu` driver (kernel 6.2 or newer, and the `linux-firmware` package): `/dev/kfd` and
  `/dev/dri/renderD*` must exist;
- Docker.

You do not install ROCm on the computer. To run the image by hand, see [DOCKER.md](DOCKER.md).

## The kernel option `amdgpu.cwsr_enable=0`

On the Radeon 780M, the driver can stop a running GPU program to let another one run (compute wave save and restore,
CWSR). On this chip, that stop can fail, and then the GPU resets. Long prompts of 8,000 tokens or more made it happen
in our tests, with Kindi and with llama.cpp. With `amdgpu.cwsr_enable=0`, the same tests passed three times out of
three.

The installer checks the option. If it is not set, the installer tells you what the option does and asks before it
adds it to `/etc/default/grub`. It never adds it with `--yes` or `--auto`. After the change, restart the computer
one time.

To set it by hand on Ubuntu:

```bash
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 amdgpu.cwsr_enable=0"/' /etc/default/grub
sudo update-grub
sudo reboot
cat /sys/module/amdgpu/parameters/cwsr_enable    # 0
```

To undo it, remove the option from `/etc/default/grub`, run `sudo update-grub` and restart.

Kindi also turns off transparent huge pages for its own process on AMD graphics (`prctl(PR_SET_THP_DISABLE)`),
because the kernel's page compaction was one of the triggers of the same reset.

## Settings that the installer uses

| Setting | Value | Why |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.2` | the 780M (gfx1103) runs ROCm's gfx1102 code |
| `HSA_ENABLE_SDMA` | `0` | the setting of the measured computer |
| `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` | the card number | one server for each graphics unit |

## One box, three jobs

The 780M and the CPU can work at the same time: the 27B on the graphics unit for agent work, and a second Kindi server
with the 2B on the CPU for short text jobs. Both use the same memory, so each one writes a little slower when both
write at the same time. The measured numbers are in [BENCHMARKS.md](BENCHMARKS.md).

To install both, select the 27B setup and the CPU setup together, for example `--config 1,4`.
