# DeepSeek-V4.1-Flash on Kindi

DeepSeek-V4.1-Flash has 552B parameters, and 16B of them are active for each token. Kindi runs the published 4-bit
weights (the UD-Q4_K_XL GGUF, about 494 GB) on one NVIDIA card and the host computer. The engine is Kindi's own code.
The dense layers and the attention run on the card. The experts run on the CPU, from host memory or from a fast drive.
An optional second card holds the most used experts. This card can be any NVIDIA card from sm_50 on, with 4 GB or
more. The DSpark drafter uses the model's own three MTP stages. It drafts up to 5 tokens for each step. The model
checks every draft, so the output does not change.

## Requirements

| Part | Requirement |
|---|---|
| OS and CPU | Linux x86-64, AVX2 |
| Card | NVIDIA, sm_50 or newer; 12 GB or more for the dense layers and the drafter (an RTX 3060 12 GB works) |
| Expert card (optional) | any NVIDIA card sm_50+ with 4 GB or more; Kindi fills it with the most used experts |
| Software | NVIDIA driver, CUDA toolkit (`nvcc`), `gcc`, `python3` with `numpy` |
| Disk | about 530 GB free for the download and the converted drafter |
| RAM | see the two profiles below |

## The two host profiles

**ram (about 560 GB of RAM or more).** The whole model stays in the Linux page cache. A host can have two CPU
sockets. Then Kindi puts the rows of each expert on the memory node of the cores that compute them. The service does
this before it starts (`python -m kindi.dsv41 numa MODEL --rowsplit`). So both memory controllers read at the same
time. The model file can be on any disk. Kindi reads the file one time at the first start, and then uses the copy in
RAM. On a hard disk, the first read takes 40 minutes or more.

**nvme (less RAM).** Kindi reads the experts from an NVMe drive. The speed depends on the read speed of the drive and
on the part of the experts that RAM holds. The installer does not accept this profile on a hard disk. Two or more NVMe
drives in RAID 0 increase the read speed. We did not measure this profile yet.

## Install

```
bash tools/install_dsv41.sh                 # checks the host, picks the profile, downloads, starts the service
bash tools/install_dsv41.sh --dry-run       # prints the plan, changes nothing
```

| Option | What it does |
|---|---|
| `--dir DIR` | The folder for the model |
| `--port 8101` | The port of the server |
| `--ctx 65536` | The context length |
| `--profile auto\|ram\|nvme` | The host profile |
| `--no-service` | Does not make the service |
| `--skip-download` | Uses the files that are in `DIR/gguf` and `DIR/dspark` |
| `--yes` | Does not ask questions |
| `--force` | Replaces an `kindi-dsv41.service` that you have. Without it, the installer never replaces the service. |

The installer compares each file with the published sha256 lists in `tests/data/dsv41/`. If a download stops, run
the installer again. It continues from the point where it stopped.

The service is a systemd user service, `kindi-dsv41.service`. It gives the OpenAI API on `127.0.0.1:PORT`. At the
first start, it compiles the CPU and CUDA kernels into `~/.cache/kindi`, and it reads the model. This takes a few
minutes.

## Check the install

```
curl -s 127.0.0.1:8101/health
python3 -m kindi.dsv41 verify MODEL-00001-of-00009.gguf [--sha]
```

`verify` is the release gate. It runs 20 prompts of 64 tokens, with the reference implementation's greedy tokens as
input. It uses the numerics of the reference (`--flags 160`: an f16 cache and the attention order of the
reference). It passes when the top-1 agreement is 98.5% or more and each difference is a near-tie. A near-tie is a
reference margin of 0.75 or less. Kindi's default numerics use DeepSeek's own FP8 and FP4 roundings. These roundings
are different from the reference build, so the gate does not use them. The prompt set is not in the repository.
`KINDI_DSV_GATE` names its folder. `--sha` also compares every shard with the published sha256. This is slow: it
reads 494 GB.

## Sampling

`temperature` 0 is greedy (the default). When `temperature` is more than 0, the server samples from one
distribution, in this order: temperature, `top_k`, `top_p` and `min_p`. DSpark stays on. The server keeps a draft
with the probability that the model gives it. When it refuses a draft, it takes a sample from the rest of the
distribution. So the output has the same distribution as plain sampling (`tools/test_dsv41_sampling.py` checks
this).

## Hot experts

`tools/dsv41_hotplan.py` makes the hot-expert list from measured counts of the expert picks. It uses Kindi's shared
MoE placement (`kindi/moe.py`), the same policy as the other MoE engines. The installer makes the plan for the VRAM
of the expert cards. With the counts in the release, 7 GB of VRAM on the expert card holds 399 experts. These experts
get 36% of the expert picks.

## Measured speeds

The test host: one GeForce RTX 3060 12 GB (PCIe 3.0 x16), a Quadro M4000 8 GB as the expert card, and two Xeon
E5-2673 v4 CPUs (AVX2) with all memory channels in use. The ram profile, with the model in the page cache, a context
of 65,536 and Kindi 0.6.0, with DSpark on (the default).

| Speed | Tokens/s |
|---|---|
| Write (the mean over agent prompts) | 18.7 |

Prompt reading on real text is about 95 to 100 tokens/s at 2K and at 8K tokens. The CPU expert work sets this speed.

The prompt chunk (`tmax`) is 512 by default. A chunk of 1024 reads an 8K prompt about 7% faster, but it uses 1 GB
more VRAM. With DSpark, a 12 GB card has no space for a chunk of 1024. The installer uses 1024 on cards with 16 GB
or more.
