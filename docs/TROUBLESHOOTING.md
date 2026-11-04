# Troubleshooting

**`OutOfMemoryError` at start.** Something else holds GPU memory. `nvidia-smi` lists GPU processes (a desktop
session uses a few MB, which is fine; another CUDA or Vulkan program is not). Free the card, or lower `ctx`
(each 1,000 tokens of context is ~12 MB).

**`no model pack at ...`.** Run `kindi convert <model.gguf>` first, or point `pack` at the right directory.

**`needs the PTQ1_0 GGUF`.** Kindi reads the PTQ1_0 files of Ternary-Bonsai-2-2B, -4B, -9B and -27B. The Q2_0 / PQ2_0
files hold the same models in another format, and Kindi does not support them yet.

**CUDA errors such as `no kernel image is available` or `CUDA driver version is insufficient`.** Use the NVIDIA 470
driver and the CUDA 10.2 stack in the Docker image.

**The first start is slow.** Kindi compiles its CUDA kernels with NVRTC on first use (about a minute) and caches them
under `state_dir/kernels`.

**A client times out or retries on the first request.** Kindi reads the first long prompt at about 25-33 tokens/s. Raise the
client's timeout and limit retries (see [AGENTS.md](AGENTS.md)); watch progress with `curl -s
localhost:8093/health`.

**Answers are slow on every turn of an agent.** Check that the agent sends the same tools and system prompt each
time. `/health` and the server log show `prompt: N reused, M read` per request.

**`401 invalid API key`.** The server has `api_key` set; send `Authorization: Bearer <key>`.

**`the prompt has N tokens; the context is M`.** The conversation outgrew `ctx`. Let the client summarise older turns,
or raise `ctx` if memory allows.

**`could not select device driver "" with capabilities: [[gpu]]`.** The Docker daemon started before you installed the NVIDIA
Container Toolkit. Restart the daemon: `sudo systemctl restart docker`, or `systemctl --user restart docker`
for rootless Docker. Then `docker run --rm --gpus all ubuntu nvidia-smi` must show the cards.

**`PermissionError: [Errno 13] Permission denied: '/data/...'` on rootless Docker.** Remove `-u $(id -u):$(id -g)` from
the `docker run` command. Rootless Docker runs the container's root as your user (see [DOCKER.md](DOCKER.md)).

**`kindi verify` fails.** The pack is from a different model file, or the GPU stack differs from the tested one.
Re-run `kindi convert --force` from the official file and check the driver.
