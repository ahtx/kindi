"""kindi plan: place models on the cards of this machine and write the plan file that the installers and the services
read (docs/dev/F6-PLACEMENT.md sections 3 to 5). Plain Python 3.6+, no third-party modules.

  kindi plan                                   proposals for this machine, the best first
  kindi plan --model 27b,2b                    proposals for these models
  kindi plan --place "dsv41:0+1,2" "27b:3"     the user's own placement (checked)
  kindi plan ... --out ~/.kindi/plan.json      write the chosen placement (the first proposal, or --pick N)
  kindi plan --show FILE | --check FILE        print a plan / compare it with the cards that are here now

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import itertools
import json
import os
import re
import sys
import time

# The models a plan can place. one: MiB free on one card (weights, buffers and a short context); two: MiB free on
# each card of a tensor-parallel pair (0: the model does not run on two cards); split: (total MiB, MiB more for each
# card) for a model whose layers go over 1..N cards; size: the weight of the model in the score (billions of
# parameters, so a 27B counts more than a 2B); kinds: the card verdicts (kindi.cards.verdict) it runs on.
MODELS = {
    "2b": dict(id="bonsai2-2b", one=900, two=0, size=2, kinds="tc dense maxwell kepler"),
    "4b": dict(id="bonsai2-4b", one=1900, two=0, size=4, kinds="tc dense maxwell kepler"),
    "9b": dict(id="bonsai2-9b", one=3800, two=0, size=9, kinds="tc dense maxwell kepler"),
    "27b": dict(id="bonsai2-27b", one=7600, two=3800, size=27, kinds="tc dense maxwell kepler"),
    "nemotron": dict(id="nemotron-bolt-e4b", one=3300, two=2400, size=8, kinds="tc dense maxwell"),
    "lightning": dict(id="nemotron-lightning-30b", one=5800, two=0, size=30, kinds="tc dense kepler"),
    "nanbeige": dict(id="nanbeige-33b-iq3", one=16200, two=0, split=(15500, 700), size=33, kinds="tc p100"),
    # Qwen3.8-Flash-Next (docs/FLASHNEXT.md): the dense layers (3.6 GB), the head (0.5 GB), the MTP head and its
    # experts (0.7 GB compact) and a 1.2 GB reserve on the card, the hot experts in the rest; the other experts run on
    # the CPU from the 83 GB file in host RAM. Layers split over 1..N cards (each card then holds its layers' hot
    # experts); dp4a cards (sm_61 and later)
    "flashnext": dict(id="qwen38-flash-next", one=8000, two=0, split=(8000, 1500), size=125, kinds="tc dense",
                      ram_gb=64),
    # DeepSeek: the main card holds the dense layers, the drafter and the context (about 11.4 GB at 64K on an RTX
    # 3060); every other card in the server is an expert card (sm_50+, 4 GB+). Needs 560 GB of RAM or an NVMe drive.
    "dsv41": dict(id="deepseek-v4.1-flash", one=11000, two=0, size=552, kinds="tc dense", expert_min=4096,
                  ram_gb=560),
}
# the recommended model of a card, the best first (install.sh's card_models order)
AUTO_ORDER = ("27b", "lightning", "nemotron", "9b", "4b", "2b")
MAIN_NEED_DSV = 11700          # MiB the DeepSeek main card keeps for itself (dense + DSpark + 64K context)
HOT_RESERVE = 400              # MiB kept free on the main card next to its hot experts (KINDI_HOT_RESERVE_MB)
HOT_RESERVE2 = 300             # on each expert card (KINDI_HOT_RESERVE2_MB)
PORT_DSV, PORT_DENSE, PORT_FRONT = 8101, 8093, 8090


class PlanError(ValueError):
    pass


def _ok(c):
    return c.get("verdict") not in ("no", "unknown", None)


def _avail(c):
    """a card the search may take: supported and not in use by other programs (kindi.cards.busy); --use and
    --place take busy cards too"""
    return _ok(c) and not c.get("busy")


def _kind(c):
    v = c.get("verdict")
    return "p100" if v == "dense" and c.get("sm") == 60 else v


def _free(c):
    f = c.get("free_mib")
    return f if f is not None else (c.get("vram_mib") or 0)


def _fast_link(c):
    """a link that carries the tensor-parallel exchange well (gen3 x8 or better); unknown counts as fast"""
    m = re.match(r"gen(\d+) x(\d+)", c.get("pcie_max") or c.get("pcie") or "")
    return True if not m else int(m.group(1)) * int(m.group(2)) >= 24


def fits(model, cards, inv_ram_gb=None):
    """(ok, why): can MODEL run on CARDS (a list of card dicts; for dsv41 the first is the main card)?"""
    m = MODELS[model]
    kinds = m["kinds"].split()
    if not cards:
        return False, "no card"
    bad = [c for c in cards if not _ok(c)]
    if bad:
        return False, "card %d is not supported (%s)" % (bad[0]["index"], bad[0].get("why", ""))
    if model == "dsv41":
        main, exp = cards[0], cards[1:]
        if _kind(main) not in kinds:
            return False, "the main card %d is %s; DeepSeek's main card needs sm_61 or later" % (main["index"], main.get("why"))
        if _free(main) < m["one"]:
            return False, "the main card %d has %d MiB free; DeepSeek needs %d" % (main["index"], _free(main), m["one"])
        small = [c for c in exp if (c.get("sm") or 0) < 50 or _free(c) < m["expert_min"]]
        if small:
            return False, "card %d cannot be an expert card (sm_50 and 4 GB free needed)" % small[0]["index"]
        if inv_ram_gb is not None and inv_ram_gb < m["ram_gb"]:
            return True, "RAM %d GB: the nvme profile (experts streamed from an NVMe drive)" % inv_ram_gb
        return True, ""
    wrong = [c for c in cards if _kind(c) not in kinds and not (_kind(c) == "p100" and "dense" in kinds)]
    if wrong:
        return False, "%s does not run on card %d (%s)" % (model, wrong[0]["index"], wrong[0].get("why"))
    if "split" in m:
        need = m["split"][0] + m["split"][1] * len(cards)
        have = sum(_free(c) for c in cards)
        return (have >= need, "" if have >= need else "%d MiB free on the cards, %s needs %d" % (have, model, need))
    if len(cards) == 1:
        ok = _free(cards[0]) >= m["one"]
        return ok, "" if ok else "card %d has %d MiB free, %s needs %d" % (cards[0]["index"], _free(cards[0]), model, m["one"])
    if len(cards) == 2 and m["two"]:
        # F4: the two cards share the weights by free memory; each needs its half of the heads and some room
        a, b = _free(cards[0]), _free(cards[1])
        ok = min(a, b) >= m["two"] * 0.6 and a + b >= 2 * m["two"]
        return ok, "" if ok else "cards %d and %d have %d + %d MiB free, %s needs %d together" % (
            cards[0]["index"], cards[1]["index"], a, b, model, 2 * m["two"])
    return False, "%s runs on one card%s" % (model, " or two" if m["two"] else "")


def speed(model, cards):
    """a speed estimate for the score: the memory bandwidth that limits the server"""
    bw = [c.get("bandwidth_gbs") or 100 for c in cards]
    if model == "dsv41":
        return bw[0] * (1.0 + 0.05 * len(cards[1:]))           # expert cards add hot experts
    if "split" in MODELS[model]:
        return min(bw)                                          # layers in order: the slowest card counts
    if len(cards) == 2:
        # a pair costs the exchange (on an x1 link a lot): one card that holds the model is preferred
        return min(bw) * (0.95 if all(_fast_link(c) for c in cards) else 0.5)
    return bw[0]


def score(servers):
    return sum(MODELS[s["model"]]["size"] * speed(s["model"], s["cards"]) for s in servers)


def _sets(model, cards):
    """candidate card sets for MODEL from CARDS (free supported cards), the fewest cards first"""
    if model == "dsv41":
        return [[c] for c in cards]                             # the main card; expert cards are added at the end
    out = [[c] for c in cards]
    if MODELS[model]["two"]:
        out += [list(p) for p in itertools.combinations(cards, 2)]
    if "split" in MODELS[model]:
        srt = sorted(cards, key=lambda c: -_free(c))
        for n in range(2, len(srt) + 1):
            out += [list(p) for p in itertools.combinations(srt, n) if len(p) <= 4]
    return out


def search(inv, models, ram_gb=None, keep=5):
    """proposals for MODELS (each placed once) on the inventory's supported cards: [(score, servers)], best first.
    DeepSeek, when asked for, takes every card that no other server takes as an expert card."""
    cards = [c for c in inv["nvidia"] if _avail(c)]
    order = sorted(models, key=lambda m: -MODELS[m]["size"])
    found = []

    def rec(i, free, servers):
        if i == len(order):
            done = [dict(s) for s in servers]
            for s in done:
                if s["model"] == "dsv41":
                    s["cards"] = s["cards"] + [c for c in free if fits("dsv41", [s["cards"][0], c])[0]]
            found.append((score(done), done))
            return
        m = order[i]
        opts = []
        for cs in _sets(m, free):
            if fits(m, cs, ram_gb)[0]:
                opts.append(cs)
        # the fewest cards first, then the fastest
        opts.sort(key=lambda cs: (len(cs), -speed(m, cs)))
        seen = set()
        for cs in opts:
            key = tuple(sorted((c.get("name"), c.get("bandwidth_gbs"), _free(c) // 512) for c in cs))
            if key in seen:                                   # an equal card set (same kind of cards): try one
                continue
            seen.add(key)
            rest = [c for c in free if c not in cs]
            rec(i + 1, rest, servers + [{"model": m, "cards": cs}])
    rec(0, cards, [])
    found.sort(key=lambda x: -x[0])
    out, sig = [], set()
    for sc, servers in found:
        k = tuple(sorted((s["model"], tuple(sorted(c["index"] for c in s["cards"]))) for s in servers))
        if k not in sig:
            sig.add(k)
            out.append((sc, servers))
        if len(out) >= keep:
            break
    return out


def auto(inv, ram_gb=None):
    """the 'every card' proposal: the largest model each free card holds, one server per card (a model that needs
    two cards takes a pair only when no single card holds anything better); then the single best server"""
    cards = sorted([c for c in inv["nvidia"] if _avail(c)], key=lambda c: -(c.get("bandwidth_gbs") or 0))
    dense = list(AUTO_ORDER)
    servers, free = [], list(cards)
    for c in cards:
        if c not in free:
            continue
        best = next((m for m in dense if fits(m, [c])[0]), None)
        if best is None:
            pair = next(([c, d] for d in free if d is not c for m in dense if fits(m, [c, d])[0]), None)
            if pair:
                best = next(m for m in dense if fits(m, pair)[0])
                servers.append({"model": best, "cards": pair})
                free = [x for x in free if x not in pair]
            continue
        servers.append({"model": best, "cards": [c]})
        free.remove(c)
    out = [(score(servers), servers)] if servers else []
    if len(servers) > 1:
        out.append((score(servers[:1]), servers[:1]))
    return out


# ------------------------------------------------------------------------------------------------ --place
def parse_place(args, inv):
    """--place "dsv41:0+1,2" "27b:3@8095" -> servers (cards resolved by index or UUID); raises PlanError"""
    by = {}
    for c in inv["nvidia"]:
        by[str(c["index"])] = c
        if c.get("uuid"):
            by[c["uuid"]] = c
    servers, used = [], {}
    for a in args:
        m = re.match(r"^([a-z0-9]+):([^@]*)(?:@(\d+))?$", a.strip())
        if not m:
            raise PlanError("--place %r: write MODEL:CARDS, for example 27b:3 or dsv41:0+1,2" % a)
        model, spec, port = m.group(1), m.group(2), m.group(3)
        if model not in MODELS:
            raise PlanError("--place %r: unknown model %s (%s)" % (a, model, " ".join(sorted(MODELS))))
        rest_all = False
        if model == "dsv41":
            main, _, exp = spec.partition("+")
            names = [main] + [x for x in exp.split(",") if x]
            rest_all = spec.endswith("+")
        else:
            if "+" in spec:
                raise PlanError("--place %r: '+' (main card + expert cards) is for dsv41 only" % a)
            names = [x for x in spec.split(",") if x]
        if not names or not names[0]:
            raise PlanError("--place %r: no card" % a)
        cs = []
        for n in names:
            if n not in by:
                raise PlanError("--place %r: no card %s on this machine" % (a, n))
            c = by[n]
            if c["index"] in used:
                raise PlanError("--place: card %d is in two servers (%s and %s)" % (c["index"], used[c["index"]], model))
            used[c["index"]] = model
            cs.append(c)
        s = {"model": model, "cards": cs}
        if port:
            s["port"] = int(port)
        if rest_all:
            s["rest"] = True
        servers.append(s)
    for s in servers:
        if s.pop("rest", False):
            s["cards"] += [c for c in inv["nvidia"] if _avail(c) and c["index"] not in used
                           and fits("dsv41", [s["cards"][0], c])[0]]
            for c in s["cards"]:
                used[c["index"]] = "dsv41"
        ok, why = fits(s["model"], s["cards"])
        if not ok:
            raise PlanError("--place %s:%s: %s" % (s["model"], ",".join(str(c["index"]) for c in s["cards"]), why))
    return servers


# ------------------------------------------------------------------------------------------------ the plan file
def make_plan(inv, servers, ram_gb=None, cpus=None, name="kindi", port0=PORT_DENSE, front=PORT_FRONT):
    """the plan file's dict: names, ports, cards by UUID, and the engine variables each server needs"""
    plan = {"format": 1, "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": {"ram_gb": ram_gb, "cpus": cpus},
            "cards": [{k: c.get(k) for k in ("index", "uuid", "bus", "name", "sm", "vram_mib", "free_mib",
                                              "bandwidth_gbs", "pcie", "verdict")} for c in inv["nvidia"]],
            "servers": [], "front": {"port": front, "models": {}}}
    ports, n_dense = set(), 0
    for s in sorted(servers, key=lambda s: s["model"] != "dsv41"):
        m, cs = s["model"], s["cards"]
        ids = [c.get("uuid") or str(c["index"]) for c in cs]
        if m == "dsv41":
            sname, port = "%s-dsv41" % name, s.get("port", PORT_DSV)
            # hot experts on all spare VRAM: the main card's room past its dense part, and each expert card
            spare = {ids[0]: max(0, _free(cs[0]) - MAIN_NEED_DSV - HOT_RESERVE)}
            for c, u in zip(cs[1:], ids[1:]):
                spare[u] = max(0, _free(c) - HOT_RESERVE2)
            # CUDA_DEVICE_ORDER=PCI_BUS_ID: a card named by its index (no UUID) means nvidia-smi's index (CUDA's own
            # default order is fastest first); with UUIDs the order is the list's either way
            env = {"CUDA_VISIBLE_DEVICES": ",".join(ids), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "KINDI_DSV_GPU": "2",
                   "KINDI_HOT_RESERVE_MB": str(HOT_RESERVE), "KINDI_HOT_RESERVE2_MB": str(HOT_RESERVE2)}
            if len(cs) > 1:
                env["KINDI_DSV_GPU2_DEVS"] = ",".join(str(i) for i in range(1, len(cs)))
            else:
                env["KINDI_DSV_GPU2"] = "0"
            srv = {"name": sname, "model": m, "id": MODELS[m]["id"], "engine": "dsv41", "main": ids[0],
                   "experts": ids[1:], "hot_gb": {u: round(v / 1024.0, 1) for u, v in spare.items()},
                   "hot_gb_total": round(sum(spare.values()) / 1024.0, 1), "env": env}
        else:
            n_dense += 1
            sname = name if n_dense == 1 else "%s-%d" % (name, n_dense)
            port = s.get("port")
            if port is None:
                port = port0
                while port in ports or port in (PORT_DSV, front):
                    port += 1
            env = {"CUDA_VISIBLE_DEVICES": ",".join(ids), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
            engine = {"nanbeige": "nanbeige", "lightning": "lightning", "nemotron": "gemma4", "flashnext": "flashnext"}.get(m, "plan")
            if engine == "flashnext":     # the engine numbers the visible cards 0 .. n - 1, layers split in that order
                env["KINDI_FN_CARDS"] = ",".join(str(i) for i in range(len(cs)))
            if len(cs) == 2 and engine in ("plan", "gemma4"):
                env["KINDI_CARDS"] = "2"
                if engine == "plan":
                    env["KINDI_TP_SPLIT"] = "auto"
            srv = {"name": sname, "model": m, "id": MODELS[m]["id"], "engine": engine, "cards": ids, "env": env}
        if port in ports:
            raise PlanError("port %d is in two servers" % port)
        ports.add(port)
        srv["port"] = port
        plan["servers"].append(srv)
        plan["front"]["models"].setdefault(srv["id"], []).append(sname)     # one model on several servers: all
    check_claims(plan)
    return plan


def server_cards(srv):
    return [srv["main"]] + list(srv["experts"]) if srv.get("engine") == "dsv41" else list(srv["cards"])


def check_claims(plan):
    """no card in two servers, no port twice"""
    seen = {}
    for srv in plan["servers"]:
        for u in server_cards(srv):
            if u in seen:
                raise PlanError("card %s is in two servers (%s and %s)" % (u, seen[u], srv["name"]))
            seen[u] = srv["name"]


def check(plan, inv):
    """[problems] of a plan against the cards here now (gone, or with less free memory than the plan needs)"""
    here = {c.get("uuid") or str(c["index"]): c for c in inv["nvidia"]}
    out = []
    try:
        check_claims(plan)
    except PlanError as e:
        out.append(str(e))
    for srv in plan["servers"]:
        cs = []
        for u in server_cards(srv):
            if u not in here:
                out.append("%s: card %s is not in this machine" % (srv["name"], u))
            else:
                cs.append(here[u])
        if len(cs) == len(server_cards(srv)):
            ok, why = fits(srv["model"], cs)
            if not ok:
                out.append("%s: %s" % (srv["name"], why))
    return out


def describe(servers):
    parts = []
    for s in servers:
        cs = s["cards"]
        if s["model"] == "dsv41":
            parts.append("dsv41 on GPU %d (main)%s" % (cs[0]["index"], " + expert GPU%s %s" % (
                "s" if len(cs) > 2 else "", ",".join(str(c["index"]) for c in cs[1:])) if len(cs) > 1 else ""))
        else:
            parts.append("%s on GPU%s %s%s" % (s["model"], "s" if len(cs) > 1 else "", ",".join(str(c["index"]) for c in cs),
                                              " (split by free memory)" if len(cs) == 2 and s["model"] in ("27b", "2b", "4b", "9b") else
                                              " (layers over the cards)" if len(cs) > 1 else ""))
    return " + ".join(parts)


def show(plan):
    lines = ["plan of %s: %d server%s" % (plan.get("made", "?"), len(plan["servers"]), "s" if len(plan["servers"]) != 1 else "")]
    idx = {c.get("uuid") or str(c["index"]): c for c in plan["cards"]}
    for s in plan["servers"]:
        names = ["GPU %s %s" % (idx[u]["index"], idx[u]["name"]) if u in idx else u for u in server_cards(s)]
        extra = ""
        if s["engine"] == "dsv41":
            extra = "; main %s, hot experts %.1f GB" % (names[0], s["hot_gb_total"])
        lines.append("  %-12s %-24s port %d  %s%s" % (s["name"], s["id"], s["port"], ", ".join(names), extra))
    lines.append("  front door: port %d, %s" % (plan["front"]["port"], ", ".join(sorted(plan["front"]["models"]))))
    return "\n".join(lines)


def _host():
    ram = cpus = None
    try:
        ram = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) // 1048576
    except (OSError, StopIteration):
        pass
    try:
        cpus = os.cpu_count()
    except AttributeError:
        pass
    return ram, cpus


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="kindi plan")
    ap.add_argument("--model", help="models to place, comma separated (%s); default: every card" % " ".join(sorted(MODELS)))
    ap.add_argument("--place", nargs="+", help="your own placement: MODEL:CARDS[@PORT] ... (dsv41:MAIN+EXPERT,...)")
    ap.add_argument("--pick", type=int, default=1, help="the proposal to write with --out (default 1)")
    ap.add_argument("--out", help="write the plan file")
    ap.add_argument("--show", help="print a plan file")
    ap.add_argument("--check", help="compare a plan file with the cards here now")
    ap.add_argument("--cards", help="read the card inventory from this JSON file (kindi cards --json) instead")
    ap.add_argument("--use", help="place on these cards only (numbers or UUIDs, comma separated); the others are left alone")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    ap.add_argument("--name", default="kindi", help="the servers' names: NAME, NAME-2, ... and NAME-dsv41")
    ap.add_argument("--port-base", type=int, default=PORT_DENSE, help="the first port of the dense servers")
    ap.add_argument("--front-port", type=int, default=PORT_FRONT, help="the front door's port")
    a = ap.parse_args(argv)
    if a.show:
        print(show(json.load(open(a.show))))
        return 0
    from . import cards as _cards
    inv = json.load(open(a.cards)) if a.cards else _cards.inventory()
    if a.use:
        keep = set(x.strip() for x in a.use.split(","))
        inv = dict(inv, nvidia=[dict(c, busy=False) for c in inv["nvidia"] if str(c["index"]) in keep or c.get("uuid") in keep])
    ram, cpus = _host()
    if a.check:
        probs = check(json.load(open(a.check)), inv)
        print("\n".join(probs) if probs else "the plan fits the cards in this machine")
        return 1 if probs else 0
    try:
        if a.place:
            props = [(None, parse_place(a.place, inv))]
        elif a.model:
            ms = [m.strip() for m in a.model.split(",") if m.strip()]
            unk = [m for m in ms if m not in MODELS]
            if unk:
                raise PlanError("unknown model %s (%s)" % (", ".join(unk), " ".join(sorted(MODELS))))
            props = search(inv, ms, ram)
            if not props:
                raise PlanError("%s do%s not fit the free cards of this machine (kindi cards; cards in use by other "
                                "programs are left out unless --use names them)" % (", ".join(ms), "es" if len(ms) == 1 else ""))
        else:
            props = auto(inv, ram)
            if not props:
                raise PlanError("no card here runs a Kindi model (kindi cards says why); the CPU engine runs the 2B")
    except PlanError as e:
        print("kindi plan: %s" % e, file=sys.stderr)
        return 2
    if not a.out and not a.json:
        for i, (sc, servers) in enumerate(props):
            print("%d) %s%s" % (i + 1, describe(servers), "" if sc is None else "   (score %.0f)" % sc))
        return 0
    if not 1 <= a.pick <= len(props):
        print("kindi plan: --pick %d: there are %d proposals" % (a.pick, len(props)), file=sys.stderr)
        return 2
    plan = make_plan(inv, props[a.pick - 1][1], ram, cpus, a.name, a.port_base, a.front_port)
    if a.out:
        d = os.path.dirname(os.path.abspath(os.path.expanduser(a.out)))
        os.makedirs(d, exist_ok=True)
        tmp = os.path.expanduser(a.out) + ".tmp"
        with open(tmp, "w", newline="") as f:
            json.dump(plan, f, indent=1)
        os.replace(tmp, os.path.expanduser(a.out))
        print(show(plan))
    else:
        print(json.dumps(plan, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
