# Intel Arc A770 device capabilities for the decode hypotheses (2026-11-26)

`python -m kindi intel-caps --json` on the i7-8086K rig — a read-only probe; the full
report is [caps.json](caps.json). It records the capability fields the ROADMAP Intel A770
decode-hypothesis rows rest on, without launching a kernel, allocating a device buffer, or
loading a model. No model ran on the Arc, and nothing here is a support claim
([REQ-0020](../../development/requirements/REQ-0020.md),
[INTEL-ARC.md](../../INTEL-ARC.md)).

## Device

| item | value |
|---|---|
| device | Intel(R) Arc(TM) A770 Graphics, PCI `8086:56A0` |
| display driver | 32.0.101.9175 |
| device version | OpenCL 3.0 NEO, OpenCL C 1.2 profile |
| loader | the host OpenCL loader |

## What the report records

- The OpenCL extension list, including the fp16 and Intel-subgroup extensions. The profile
  itself exposes no DPAS/XMX path, so that work needs SYCL/ESIMD rather than another OpenCL
  kernel (roadmap row I1).
- The per-allocation cap against a 27B code buffer: a single buffer does not fit, so the
  step keeps chunking codes on tensor boundaries.
- ReBAR state is not exposed by the OS interfaces this probe reads; capture it with
  `lspci -vv` on Linux or Intel Arc Control / GPU-Z on Windows.

Result `pass` with the Arc selected and no warnings. Identity lands on the (name, memory,
driver) triple: both KHR identity extensions are advertised yet unusable on this driver — the
UUID query succeeds but writes only a stub, and the PCI bus id comes back empty — so the
report records them as null rather than guessing. The report carries no machine-local paths.
