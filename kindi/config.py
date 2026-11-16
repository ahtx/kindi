"""Kindi configuration.

Precedence (later wins): built-in defaults < config file < environment (KINDI_<KEY>) < command-line flags.
The config file is INI with a [kindi] section; it is looked for at --config, $KINDI_CONFIG, ./kindi.conf and
~/.kindi/kindi.conf, in that order. See kindi.conf.example for every key.
"""
import configparser
import json
import os
import re

DEFAULTS = {
    # where things live
    "pack": "~/.kindi/pack",            # the converted model (kindi convert)
    "state_dir": "~/.kindi/state",      # saved prompt openings, head mode, kernel cache
    # server
    "host": "127.0.0.1",                # 0.0.0.0 to serve the LAN
    "port": "8093",
    "api_key": "",                      # if set, clients must send Authorization: Bearer <api_key>
    "model_name": "auto",              # infer the reported id from the selected pack
    # engine
    "ctx": "24576",                     # requested context; each backend clamps it to available device memory
    "head": "exact",                    # exact | fast (shortlist head, ~15% faster, approximate)
    "fast_clusters": "256",             # clusters scored by the fast head
    # prompt cache
    "checkpoint_every": "256",          # recurrent-state checkpoint spacing (tokens)
    "prefix_max": "8",                  # saved prompt openings kept on disk
    "prefix_min": "512",                # smallest opening worth saving (tokens)
    "park_max": "3",                    # other conversations kept in host RAM
    "checkpoint_budget_mb": "0",        # host RAM for checkpoints and parked conversations (0: 20% of the RAM)
    "turn_save": "1",                   # 1: save the state at the end of each turn to disk, in the background
    "turn_keep": "4",                   # saved turns: conversations kept on disk (the newest turn of each)
    "turn_disk_mb": "4096",             # saved turns: disk space they may use
    "turn_min": "64",                   # saved turns: shortest conversation worth saving (tokens)
    "repeat_stop": "0",                 # 1: stop a reply caught in a repeating loop (close it while thinking)
    "lookup": "5",                      # Nanbeige: prompt-lookup drafting, up to this many tokens per check (0: off)
    "mtp_head": "",                     # Nanbeige: an MTP head file, a second draft source (empty: none)
    "kv_cache": "auto",                 # Nanbeige: the K/V cache, auto, f16 or q8_0 (auto: f16 when it fits, else q8_0)
    "reasoning_effort": "",             # 27B template: xhigh | medium | low (empty: the template default, xhigh)
    # generation defaults (a request's own values win)
    "max_tokens": "1024",
    "temperature": "0.7",
    "top_p": "0.8",
    "top_k": "20",
    "min_p": "0",                       # 0: off; else drop the tokens below min_p times the best one's probability
    "repeat_penalty": "1.0",            # 1.0: off; llama.cpp's repeat penalty (the output is then not token-exact)
    "repeat_last_n": "64",              # the tokens that repeat_penalty looks at (-1: the whole context)
}

INT_KEYS = {"repeat_stop", "lookup", "port", "ctx", "fast_clusters", "checkpoint_every", "checkpoint_budget_mb", "prefix_max", "prefix_min", "park_max",
            "turn_save", "turn_keep", "turn_disk_mb", "turn_min",
            "max_tokens", "top_k", "repeat_last_n"}
FLOAT_KEYS = {"temperature", "top_p", "min_p", "repeat_penalty"}
PATH_KEYS = {"pack", "state_dir"}


class Config(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def path(self, k, *more):
        return os.path.join(self[k], *more)


def _candidates(explicit):
    if explicit:
        return [explicit]
    out = []
    if os.environ.get("KINDI_CONFIG"):
        out.append(os.environ["KINDI_CONFIG"])
    out += ["kindi.conf", os.path.expanduser("~/.kindi/kindi.conf")]
    return out


def load(explicit=None, overrides=None):
    vals = dict(DEFAULTS)
    source = None
    for p in _candidates(explicit):
        if p and os.path.exists(p):
            cp = configparser.ConfigParser()
            cp.read(p)
            if cp.has_section("kindi"):
                for k, v in cp.items("kindi"):
                    if k not in DEFAULTS:
                        raise SystemExit("kindi: unknown config key %r in %s" % (k, p))
                    vals[k] = v
            source = p
            break
        if explicit:
            raise SystemExit("kindi: config file not found: %s" % p)
    for k in DEFAULTS:
        e = os.environ.get("KINDI_" + k.upper())
        if e is not None:
            vals[k] = e
    for k, v in (overrides or {}).items():
        if v is not None:
            vals[k] = str(v)
    cfg = Config()
    for k, v in vals.items():
        v = v.strip()
        if k in INT_KEYS:
            cfg[k] = int(v)
        elif k in FLOAT_KEYS:
            cfg[k] = float(v)
        elif k in PATH_KEYS:
            cfg[k] = os.path.abspath(os.path.expanduser(v))
        else:
            cfg[k] = v
    if cfg["model_name"] == "auto":
        manifest = os.path.join(cfg["pack"], "kindi_pack.json")
        m = json.load(open(manifest)) if os.path.isfile(manifest) else {}
        if m.get("backend") in ("kindi_gemma4", "kindi_lightning"):
            cfg["model_name"] = m.get("model_id", "gemma4")
        else:
            # the GGUF's general.name, then its file name: some files name the model only "Hf"
            size = None
            for text in (m.get("model") or "", m.get("source_file") or ""):
                size = re.search(r"(\d+(?:\.\d+)?)b(?:\b|-|_)", text.lower())   # Ternary-Bonsai-2-9B -> 9b
                if size:
                    break
            cfg["model_name"] = "bonsai2-%sb" % (size.group(1) if size else "2")
    if cfg["head"] not in ("exact", "fast"):
        raise SystemExit("kindi: head must be exact or fast, not %r" % cfg["head"])
    cfg["source"] = source or "(built-in defaults)"
    return cfg
