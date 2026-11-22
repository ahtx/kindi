"""kindi cards: every GPU in the machine, with name, compute capability, VRAM (total and free), PCIe link, memory
bandwidth, driver, and whether Kindi supports it (and why not). Plain Python 3 (3.6 and later), no third-party
modules: the installers run it on the host before any image exists (python3 kindi/cards.py --json), and it runs in
the image as `kindi cards`.

Each NVIDIA card is asked on its own (nvidia-smi -i N): a card whose query fails (a corrupt infoROM: exit 14) loses
only the fields that failed, never the other cards (the B2 rule of install.sh). docs/dev/F6-PLACEMENT.md section 2.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import glob
import json
import os
import re
import subprocess
import sys

FIELDS = ("index", "uuid", "pci.bus_id", "name", "memory.total", "memory.used", "memory.free", "compute_cap",
          "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max",
          "driver_version")
FIELDS_470 = tuple(f for f in FIELDS if f != "compute_cap")       # the 470 driver has no compute_cap field

# memory bandwidth in GB/s by chip, when the driver library cannot give it (no libcuda, or a card it cannot open)
BW_NAME = [("RTX 3090", 936), ("RTX 3080 Ti", 912), ("RTX 3080", 760), ("RTX 3070", 448), ("RTX 3060 Ti", 448),
           ("RTX 3060", 360), ("RTX 4090", 1008), ("RTX 4080", 717), ("RTX 4070", 504), ("RTX 4060 Ti", 288),
           ("RTX 4060", 272), ("RTX 5060 Ti", 448), ("RTX 2080 Ti", 616), ("RTX 2080", 448), ("RTX 2070", 448),
           ("RTX 2060", 336), ("GTX 1660", 192), ("GTX 1650", 128), ("GTX 1080 Ti", 484), ("GTX 1080", 320),
           ("GTX 1070", 256), ("GTX 1060", 192), ("TITAN X", 480), ("Tesla P100", 732), ("Tesla P40", 346),
           ("Tesla V100", 900), ("V100S", 1134), ("Tesla T4", 320), ("A100", 1555), ("CMP 170HX", 1500),
           ("Quadro M4000", 192), ("Quadro M5000", 211), ("Quadro M6000", 317), ("Quadro K1200", 80),
           ("Quadro K420", 29), ("Quadro K4200", 173), ("Quadro K6000", 288), ("Tesla K40", 288), ("Tesla K80", 240)]


def cc_from_name(n):
    """the compute capability (major * 10 + minor) from the card name (install.sh's table): for the 470 driver and
    for a card whose compute_cap query fails"""
    rules = [
        (r"GTX TITAN X", 52), (r"Tesla P100|Quadro GP100", 60), (r"V100|TITAN V|Quadro GV100", 70),
        (r"A100|A30|CMP 170HX", 80), (r"H100|H200|GH200", 90),
        (r"K6000|K5200|K40(?![0-9])|K20(?![0-9])|GTX TITAN|GTX 780", 35), (r"K80", 37),
        (r"K420|K600|K2000|K4000|K4200|K5000|K520|K10(?![0-9])|GTX 6|GTX 7[0-4]|GT 7", 30),
        (r"Quadro M2000|Quadro M4000|Quadro M5000|Quadro M6000|Quadro M3000M|Quadro M4000M|Quadro M5000M|Tesla M4$|"
         r"Tesla M40|Tesla M6$|Tesla M60|GTX 950|GTX 960|GTX 970|GTX 980|GTX 965M", 52),
        (r"K620|K1200|K2200|GTX 75|GTX 9|M[0-9]", 50),
        (r"GTX 10|Quadro P|TITAN X|Tesla P", 61), (r"RTX 20|GTX 16|Quadro RTX|T4", 75),
        (r"RTX 30|A[0-9]", 86), (r"RTX 40|L4", 89), (r"RTX 50", 120)]
    for pat, cc in rules:
        if re.search(pat, n):
            return cc
    return 61


def _num(s):
    s = (s or "").strip()
    try:
        return int(float(s))
    except ValueError:
        return None


def parse_row(line, fields):
    """one CSV row of an nvidia-smi query (noheader, nounits) -> dict, or None"""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(fields) or not re.match(r"^\d+$", parts[0]):
        return None
    return dict(zip(fields, parts))


def _smi(args, runner):
    """(exit code, stdout) of nvidia-smi ARGS; runner: a test's stand-in for subprocess"""
    if runner is not None:
        return runner(args)
    try:
        p = subprocess.run(["nvidia-smi"] + args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           universal_newlines=True, timeout=30)
        return p.returncode, p.stdout
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""


def _bandwidth_from_driver():
    """{bus id (lower case, 8.2 form 'xx:yy.z' as nvidia-smi's tail): GB/s} from the driver library, or {}"""
    try:
        import ctypes
        cu = ctypes.CDLL("libcuda.so.1") if os.name != "nt" else ctypes.WinDLL("nvcuda.dll")
        if cu.cuInit(0) != 0:
            return {}
        n = ctypes.c_int()
        cu.cuDeviceGetCount(ctypes.byref(n))
        out = {}
        for i in range(n.value):
            d, clk, width = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
            bus = ctypes.create_string_buffer(64)
            if cu.cuDeviceGet(ctypes.byref(d), i):
                continue
            cu.cuDeviceGetAttribute(ctypes.byref(clk), 36, d)       # MEMORY_CLOCK_RATE (kHz)
            cu.cuDeviceGetAttribute(ctypes.byref(width), 37, d)     # GLOBAL_MEMORY_BUS_WIDTH (bits)
            if cu.cuDeviceGetPCIBusId(bus, 64, d) == 0 and clk.value and width.value:
                out[bus.value.decode().lower()[-12:]] = int(round(2.0 * clk.value * 1e3 * width.value / 8 / 1e9))
        return out
    except (OSError, AttributeError):
        return {}


def _smi_error():
    """what nvidia-smi -L printed (stdout and stderr), for a failed run"""
    try:
        p = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=30)
        return p.stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def nvidia_cards(runner=None, driver_bw=None):
    """[card dict] of every NVIDIA GPU, asked one by one"""
    rc, out = _smi(["-L"], runner)
    ids = re.findall(r"^GPU (\d+):", out or "", re.M)
    if not ids and rc not in (0, 127) and runner is None:
        err = _smi_error()
        if "Driver/library version mismatch" in err:
            raise SystemExit("kindi cards: nvidia-smi: %s\nNVIDIA driver and library versions differ (often after an "
                             "automatic update): reboot, then rerun this installer" % err.strip())
    cards = []
    bw_drv = _bandwidth_from_driver() if driver_bw is None else driver_bw
    for i in ids:
        note = []
        rc, out = _smi(["-i", i, "--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits"], runner)
        row = next((r for r in (parse_row(x, FIELDS) for x in (out or "").splitlines()) if r), None) if rc == 0 else None
        if row is None:
            rc2, out2 = _smi(["-i", i, "--query-gpu=" + ",".join(FIELDS_470), "--format=csv,noheader,nounits"], runner)
            row = next((r for r in (parse_row(x, FIELDS_470) for x in (out2 or "").splitlines()) if r), None)
            if row is None:
                rc3, out3 = _smi(["-L"], runner)
                m = re.search(r"^GPU %s: (.*?) \(UUID: (\S+)\)" % i, out3 or "", re.M)
                if not m:
                    cards.append({"index": int(i), "name": "?", "verdict": "unknown",
                                  "why": "nvidia-smi exited with code %d and gave no name" % rc})
                    continue
                row = {"index": i, "name": m.group(1), "uuid": m.group(2)}
                note.append("nvidia-smi exited with code %d%s: only the name is known" % (
                    rc2, " (infoROM is corrupted)" if rc2 == 14 else ""))
            elif rc != 0 and rc not in (2, 6):
                note.append("nvidia-smi exited with code %d%s; the compute capability comes from the name" % (
                    rc, " (infoROM is corrupted)" if rc == 14 else ""))
        name = row.get("name", "?")
        cc = row.get("compute_cap", "")
        cc = int(round(float(cc) * 10)) if re.match(r"^\d+\.\d+$", cc or "") else cc_from_name(name)
        bus = (row.get("pci.bus_id") or "").lower()
        gc, gm = _num(row.get("pcie.link.gen.current")), _num(row.get("pcie.link.gen.max"))
        wc, wm = _num(row.get("pcie.link.width.current")), _num(row.get("pcie.link.width.max"))
        bw = bw_drv.get(bus[-12:]) if bus else None
        if not bw:
            bw = next((b for pat, b in BW_NAME if pat in name), None)
        c = {"index": int(row["index"]), "uuid": row.get("uuid"), "bus": bus or None, "name": name, "vendor": "nvidia",
             "sm": cc, "vram_mib": _num(row.get("memory.total")), "used_mib": _num(row.get("memory.used")),
             "free_mib": _num(row.get("memory.free")), "pcie": None, "pcie_max": None,
             "bandwidth_gbs": bw, "driver": row.get("driver_version") or None}
        if gc and wc:
            c["pcie"] = "gen%d x%d" % (gc, wc)
        if gm and wm:
            c["pcie_max"] = "gen%d x%d" % (gm, wm)
        c.update(verdict(c))
        c.update(busy(c))
        if note:
            c["note"] = "; ".join(note)
        cards.append(c)
    return cards


# a card whose memory is more than this share in use by other programs is left out of placements by default
# (kindi plan --use / --place still take it): a display, another model server, a training job
BUSY_SHARE = float(os.environ.get("KINDI_BUSY_SHARE", "0.25"))


def busy(c):
    """{'busy': True, 'busy_why': ...} when other programs use more than BUSY_SHARE of the card's memory, else {}"""
    used, vram = c.get("used_mib"), c.get("vram_mib")
    if used is None or not vram or used <= BUSY_SHARE * vram:
        return {}
    return {"busy": True, "busy_why": "in use by another program: %d MiB" % used}


def verdict(c):
    """{'verdict': tc | dense | maxwell | kepler | expert | no, 'why': text} (install.sh's card classes, docs/CARDS.md)"""
    sm, vram = c.get("sm") or 0, c.get("vram_mib") or 0
    drv = _num((c.get("driver") or "").split(".")[0]) or 0
    if sm < 30:
        return {"verdict": "no", "why": "compute capability %d.%d: Kindi needs sm_30 or later" % (sm // 10, sm % 10)}
    if sm < 50 and drv > 470:
        return {"verdict": "no", "why": "Kepler (sm_%d) needs the NVIDIA 470 driver branch; this one is %s" % (sm, c.get("driver"))}
    if sm >= 80 and vram >= 9500:
        return {"verdict": "tc", "why": "tensor cores (sm_%d)" % sm}
    if sm >= 60:
        return {"verdict": "dense", "why": "sm_%d%s" % (sm, " (Tesla P100: no dp4a, fp16 prompt path)" if sm == 60 else "")}
    if sm >= 50:
        return {"verdict": "maxwell", "why": "Maxwell (sm_%d)" % sm}
    return {"verdict": "kepler", "why": "Kepler (sm_%d) with the %s driver" % (sm, c.get("driver"))}


def amd_cards(root="/sys/class/kfd/kfd/topology/nodes"):
    """the AMD GPUs in the KFD topology (install.sh's rules: RDNA2 gfx1030-1036, RDNA3 gfx1100-1103, RDNA 3.5 gfx1150-1152)"""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "properties"))):
        try:
            props = dict(line.split()[:2] for line in open(p) if len(line.split()) >= 2)
        except OSError:
            continue
        g = int(props.get("gfx_target_version", "0"))
        if g <= 0:
            continue
        gfx = "gfx%d%d%x" % (g // 10000, g // 100 % 100, g % 100)
        dv = "/sys/class/drm/renderD%s/device" % props.get("drm_render_minor", "?")

        def mib(f):
            try:
                return int(open(os.path.join(dv, f)).read()) >> 20
            except (OSError, ValueError):
                return 0
        vram, gtt = mib("mem_info_vram_total"), mib("mem_info_gtt_total")
        m = gtt if gtt > vram and vram < 8192 else vram          # an APU uses system memory (GTT)
        ok = re.match(r"^gfx(103[0-6]|110[0-3]|115[0-2])$", gfx) is not None
        out.append({"index": len(out), "vendor": "amd", "name": "AMD " + gfx, "gfx": gfx, "vram_mib": m,
                    "free_mib": None, "bus": os.path.basename(os.path.realpath(dv)), "sm": None,
                    "verdict": "amd" if ok else "no",
                    "why": ("RDNA2" if gfx.startswith("gfx103") else "RDNA3 / RDNA 3.5") if ok else
                    "Kindi supports AMD RDNA2 (gfx1030-gfx1036), RDNA3 (gfx1100-gfx1103) and RDNA 3.5 (gfx1150-gfx1152) GPUs only"})
    return out


def inventory(runner=None, driver_bw=None, amd_root="/sys/class/kfd/kfd/topology/nodes"):
    cards = nvidia_cards(runner, driver_bw)
    wsl = os.path.exists("/dev/dxg")
    if wsl:
        for c in cards:
            c["pcie"] = c["pcie"] or "wsl2"
    return {"nvidia": cards, "amd": amd_cards(amd_root), "wsl2": wsl}


def table(inv):
    rows = [("card", "name", "sm", "VRAM", "free", "PCIe", "bandwidth", "driver", "verdict")]
    for c in inv["nvidia"]:
        sm = c.get("sm")
        rows.append((str(c["index"]), c.get("name", "?"), "%d.%d" % (sm // 10, sm % 10) if sm else "?",
                     "%s MiB" % c["vram_mib"] if c.get("vram_mib") is not None else "?",
                     "%s MiB" % c["free_mib"] if c.get("free_mib") is not None else "-",
                     c.get("pcie") or "-", "%s GB/s" % c["bandwidth_gbs"] if c.get("bandwidth_gbs") else "-",
                     c.get("driver") or "-", ("supported: " if c["verdict"] not in ("no", "unknown") else "not supported: ")
                     + c.get("why", "") + ("; left out by default: %s" % c["busy_why"] if c.get("busy") and c["verdict"] != "no" else "")
                     + (" [%s]" % c["note"] if c.get("note") else "")))
    for c in inv["amd"]:
        rows.append(("amd%d" % c["index"], c["name"], c["gfx"], "%d MiB" % c["vram_mib"], "-", "-", "-", "-",
                     ("supported: " if c["verdict"] == "amd" else "not supported: ") + c["why"]))
    if len(rows) == 1:
        return "no GPU found (or no driver): Kindi runs the Bonsai models on the CPU"
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    return "\n".join("  ".join(r[i].ljust(w[i]) for i in range(len(w))) + "  " + r[-1] for r in rows)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    inv = inventory()
    if "--json" in argv:
        print(json.dumps(inv, indent=1))
    else:
        print(table(inv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
