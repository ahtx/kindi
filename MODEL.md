# External model identity

No weights are distributed, downloaded by the build, or placed in the source archive.
Use an existing file matching:

- Repository: https://huggingface.co/boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF
- GGUF revision: `570610701b08e16ffcfaa899a0bdec94f0cda819`
- File: `Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf`
- Size: **9,472,656,320 bytes**
- SHA-256: `38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3`
- Nota checkpoint: https://huggingface.co/nota-ai/Nemotron-3.5-Lightning-30B-A3B-Nota-Ternary-MLX-1014/tree/85889ce1c6c89bc6c999e28cf01c2dce0d0c4ef4
- Original model: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

NVIDIA -> Nota AI ternary checkpoint -> boro64 GGUF conversion. boro64 is the
converter's account name. The backend was written for this specific checkpoint;
we did not train or alter its weights. Model license: OpenMDW 1.1, preserved
with NVIDIA's notice in `licenses/Nemotron-OpenMDW-1.1.txt` for attribution.
The AGPL license of this program does not relicense the model.

`kindi convert` creates a small manifest referring to the existing GGUF. It does
not duplicate the weights. Run it inside the same `/models` mount used by the
service so its saved path stays valid. `verify` rejects other checkpoint hashes.
