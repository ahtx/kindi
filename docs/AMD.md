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
