"""Turbo: the speed settings that are measured faster but do not give bit-identical output.

`kindi serve --turbo`, KINDI_TURBO=1 or `turbo = true` in the config turns them on. Each setting's own environment
variable still wins over turbo. `kindi turbo` (or `kindi serve --turbo --explain`) prints the table below.
The default settings stay exact where 0.8 measured them exact; turbo is the opt-in for the rest.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.
"""
import os

TRUE = ("1", "true", "yes", "on")

# One entry per setting. engine: which server reads it. env/value: what turbo sets when the variable is not set.
# speed / delta: measured (docs/TURBO.md, docs/dev/bench-0.8.0/MATRIX.md), not estimated.
SETTINGS = [
    dict(engine="deepseek", env="KINDI_SPLIT_CARD", value="16",
         what="16 streamed prompt experts per layer on the main card (default 12)",
         cards="RTX 3060 12 GB main card + Quadro M4000 8 GB, dual Xeon E5-2673 v4",
         speed="read 90.7 -> 96.0 tok/s at 4,284 prompt tokens, 92.7 -> 97.2 at 13,779; write flat",
         delta="greedy tokens identical; a seeded T=0.7 13,779-token reply differed; kindi verify PASS "
               "1266 of 1280 top-1 (98.91%), the same as without turbo"),
]


def active(cfg=None):
    """turbo on: KINDI_TURBO in the environment, else the config key turbo"""
    v = os.environ.get("KINDI_TURBO")
    if v is None and cfg is not None:
        v = str(cfg.get("turbo", "0"))
    return str(v or "0").strip().lower() in TRUE


def env_settings(engine):
    """the environment-variable settings of an engine"""
    return [s for s in SETTINGS if s["engine"] == engine and s["value"] is not None]


def apply(engine, log=print, cfg=None):
    """turbo on: set each of the engine's settings whose variable is not already set, and log one line for each.
    Returns {variable: value} of what turbo set."""
    if not active(cfg):
        return {}
    out = {}
    for s in env_settings(engine):
        if s["env"] in os.environ:
            log("turbo: %s=%s set in the environment, kept (turbo would use %s)" % (s["env"], os.environ[s["env"]],
                                                                                   s["value"]))
            continue
        os.environ[s["env"]] = s["value"]
        out[s["env"]] = s["value"]
        log("turbo: %s=%s - %s; %s; output: %s" % (s["env"], s["value"], s["what"], s["speed"], s["delta"]))
    if not env_settings(engine) and engine != "deepseek":
        log("turbo: no turbo settings for this model (its fast settings are already the defaults)")
    return out


def dsv_prefill(a, log=print):
    """DeepSeek's prompt chunks: --tmax 512 unless set. Turbo leaves them alone: the 1,024-token adaptive chunks need
    KINDI_DSV_CKV4 on a 12 GB card, and CKV4 itself is below the verify gate (1260 of 1280, with or without turbo)."""
    if a.tmax is None:
        a.tmax = 512
    if a.adaptive_prefill is None:
        a.adaptive_prefill = False


def table():
    """the turbo table (kindi turbo)"""
    rows = ["Kindi turbo: faster settings with slightly different output (same quality gates passed).",
            "Turn on: kindi serve --turbo, KINDI_TURBO=1, or turbo = true in the config. A setting's own variable wins.",
            ""]
    for s in SETTINGS:
        rows += ["%s: %s%s" % ({"deepseek": "DeepSeek V4.1 Flash"}.get(s["engine"], s["engine"]), s["env"],
                                "=" + s["value"] if s["value"] else ""),
                 "  what:   " + s["what"],
                 "  cards:  " + s["cards"],
                 "  speed:  " + s["speed"] + " (normal -> turbo)",
                 "  output: " + s["delta"],
                 ""]
    rows.append("Bonsai, Flash-Next on cards, Lightning, Nanbeige and Inkling: no turbo settings; their fast settings "
                "are already the defaults.")
    return "\n".join(rows)
