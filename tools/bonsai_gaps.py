"""Add llama.cpp's top-2 log-probability gap at each expected token to a Ternary-Bonsai reference file (kindi verify
then counts a first difference at a gap under verify.TIE_GAP as a tie). llama.cpp is used as a black box: a running
llama-server on the same GGUF (temperature 0, top_k 1), asked for the two most likely tokens at every step.

python3 tools/bonsai_gaps.py kindi/reference-4b.json http://127.0.0.1:8099 [--check]
Where the server's greedy tokens differ from the reference (another llama.cpp run of the same file splits there), its
tokens are kept as "alt": kindi verify counts Kindi taking the alt token at that point as a tie, as nanbeige_cli does
with llama.cpp's -ub 1 run."""
import json
import math
import sys
import urllib.request

path, url = sys.argv[1], sys.argv[2].rstrip("/")
ref = json.load(open(path, encoding="utf-8"))
for r in ref["prompts"]:
    body = {"prompt": r["ids"], "n_predict": len(r["expected"]), "temperature": 0, "top_k": 1, "n_probs": 2,
            "cache_prompt": False, "post_sampling_probs": False}
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))
    probs = out.get("completion_probabilities") or []
    toks, gaps = [], []
    for p in probs:
        tops = p.get("top_logprobs") or p.get("probs") or []
        lp = [t["logprob"] if "logprob" in t else math.log(max(t["prob"], 1e-30)) for t in tops[:2]]
        toks.append(p.get("id", tops[0].get("id") if tops else None))
        gaps.append(round(lp[0] - lp[1], 6) if len(lp) == 2 else None)
    same = toks == r["expected"]
    print("%-40s %d tokens, llama-server agrees with the reference: %s, smallest gap %.4f at %d" % (
        repr(r["prompt"])[:40], len(gaps), same, min(g for g in gaps if g is not None),
        min(range(len(gaps)), key=lambda k: gaps[k] if gaps[k] is not None else 9e9)))
    if not same:
        print("   reference", r["expected"])
        print("   server   ", toks)
    # the gaps follow the server's own path: keep them up to its first difference from the reference (aligned), and
    # keep the server's tokens as "alt" when the two llama.cpp runs differ (a point where llama.cpp itself is split)
    d = next((k for k, (a, b) in enumerate(zip(toks, r["expected"])) if a != b), len(gaps) - 1)
    r["gap"] = gaps[:d + 1]
    if not same:
        r["alt"] = toks
    else:
        r.pop("alt", None)
ref.setdefault("gap_source", "llama-server top-2 log-probability gap at each expected token (tools/bonsai_gaps.py)")
if "--check" not in sys.argv:
    json.dump(ref, open(path, "w", encoding="utf-8"), indent=1)
    print("written", path)
