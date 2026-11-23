#!/usr/bin/env python3
"""Verify Kindi Inkling-2-Mini engine against reference-inkling2-mini.json (greedy match + top-2 tie rule) and benchmark pp512/tg128."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from kindi.inkling2 import Engine  # noqa: E402
from kindi.inkling2_tokenizer import Inkling2Tokenizer  # noqa: E402

MODEL = Path.home() / "models/inkling2/Inkling-2-Mini-UD-Q4_K_M.gguf"
REF_PATH = Path(__file__).resolve().parent / "kindi/reference-inkling2-mini.json"
TIE_GAP = 0.08  # logprob gap threshold for near-tie acceptance


def verify_all(eng, tok, only_names=None, reference=REF_PATH):
    with open(reference, encoding="utf-8") as f:
        ref = json.load(f)
    if only_names:
        unknown = only_names - {case["name"] for case in ref["cases"]}
        if unknown:
            raise ValueError("unknown reference cases: " + ", ".join(sorted(unknown)))
    total_toks = 0
    n_pass = 0
    n_tie = 0
    n_fail = 0
    for c in ref["cases"]:
        name = c["name"]
        if only_names and name not in only_names:
            continue
        ids = c["ids"]
        exp = c["expected"]
        gaps = c["gap"]
        alts = c["alt"]
        t0 = time.time()
        eng.reset()
        tok0, gap0 = eng.prefill(ids, pos=0, return_gap=True)
        c_pass = c_tie = c_fail = 0
        first_fail = None
        pred = tok0
        pgap = gap0
        for step, ref_tok in enumerate(exp):
            ref_gap = gaps[step]
            ref_alt = alts[step]
            if pred == ref_tok:
                c_pass += 1
            elif (pred == ref_alt and ref_gap < TIE_GAP) or (ref_gap < 0.02 and pgap < 0.05):
                c_tie += 1
            else:
                c_fail += 1
                if first_fail is None:
                    first_fail = (step, pred, tok.decode([pred]), ref_tok, tok.decode([ref_tok]), ref_alt, ref_gap, pgap)
            # Teacher-force the reference token so every step tests single-step numerical accuracy
            if step + 1 < len(exp):
                pred, pgap = eng.step(ref_tok, return_gap=True)
        dt = time.time() - t0
        total_toks += len(exp)
        n_pass += c_pass
        n_tie += c_tie
        n_fail += c_fail
        msg = "CASE %-14s in=%4d steps=%3d PASS=%3d TIE=%2d FAIL=%2d (%.2fs)" % (
            name, len(ids), len(exp), c_pass, c_tie, c_fail, dt)
        if first_fail:
            s, pr, prs, rt, rts, ra, rg, pg = first_fail
            msg += " FIRST_FAIL@%d: got %d(%r) vs ref %d(%r) alt=%d ref_gap=%.4f our_gap=%.4f" % (
                s, pr, prs, rt, rts, ra, rg, pg)
        print(msg)
        sys.stdout.flush()
    rate = 100.0 * (n_pass + n_tie) / max(1, total_toks)
    print("SUMMARY: total=%d PASS=%d TIE=%d FAIL=%d (PASS+TIE=%.2f%%)" % (
        total_toks, n_pass, n_tie, n_fail, rate))
    return n_fail == 0


def bench_engine(eng, n_prompt=512, n_gen=128, repeats=3):
    prompt_ids = [100 + (i * 37) % 5000 for i in range(n_prompt)]
    pp_rates = []
    tg_rates = []
    for r in range(repeats):
        eng.reset()
        eng.profile(reset=True)
        t0 = time.time()
        tok, _ = eng.prefill(prompt_ids, pos=0, return_gap=True)
        t_pp = time.time() - t0
        pp_rates.append(n_prompt / t_pp)
        _, prof_pp = eng.profile(reset=True)

        t1 = time.time()
        cur = tok
        for _ in range(n_gen - 1):
            cur, _ = eng.step(cur, return_gap=True)
        t_tg = time.time() - t1
        tg_rates.append((n_gen - 1) / t_tg)
        _, prof_tg = eng.profile(reset=True)
        print("REPEAT %d: pp%d = %.2f tok/s (%.2fs) pp_prof=%s | tg%d = %.2f tok/s (%.2fs) tg_prof=%s" % (
            r + 1, n_prompt, pp_rates[-1], t_pp,
            {k: round(v, 3) for k, v in prof_pp.items()},
            n_gen, tg_rates[-1], t_tg,
            {k: round(v, 3) for k, v in prof_tg.items()}))
        sys.stdout.flush()
    import numpy as np
    print("BENCH_RESULT mode=%s threads=%s pp%d=%.2f+-%.2f tg%d=%.2f+-%.2f" % (
        eng.mode, os.environ.get("KINDI_THREADS", "auto"),
        n_prompt, float(np.mean(pp_rates)), float(np.std(pp_rates)),
        n_gen, float(np.mean(tg_rates)), float(np.std(tg_rates))))


def verify_draft(eng, draft):
    from kindi.inkling2_chat import render

    prompts = [
        "Solve step by step: a train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed?",
        "Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring and tests.",
        "Explain why the sky appears blue during the day and red at sunset, in clear prose for a student.",
    ]
    tok = Inkling2Tokenizer(eng.gguf.kv)
    eng.attach_dspark(draft)
    matches = True
    for i, prompt in enumerate(prompts):
        ids = tok.encode(render(eng.model_path, [{"role": "user", "content": prompt}], enable_thinking=False))
        eng.generate(ids, 8)
        start = time.perf_counter()
        plain = eng.generate(ids, 96)
        draft_start = time.perf_counter()
        speculative, stats = eng.generate_dspark(ids, 96, return_stats=True)
        elapsed = time.perf_counter() - draft_start
        first_diff = next((j for j, pair in enumerate(zip(plain, speculative)) if pair[0] != pair[1]), None)
        matches &= plain == speculative
        print("P%d match=%s first_diff=%s plain=%.2f tok/s draft=%.2f tok/s accept=%.3f" % (
            i, plain == speculative, first_diff, len(plain) / (draft_start - start),
            len(speculative) / elapsed, stats["accept_rate"]), flush=True)
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="all",
                        choices=("verify", "bench", "all", "profile_experts", "verify_draft"))
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--reference", default=REF_PATH)
    parser.add_argument("--cases", help="comma-separated reference case names")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="inkling2-experts.json")
    parser.add_argument("--draft", help="DSpark GGUF for verify_draft")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.mode == "verify_draft" and not args.draft:
        parser.error("verify_draft requires --draft")
    eng = Engine(args.model, ctx=2048)
    try:
        if args.mode == "verify_draft":
            return 0 if verify_draft(eng, args.draft) else 1
        if args.mode == "profile_experts":
            with open(args.reference, encoding="utf-8") as f:
                for case in json.load(f)["cases"]:
                    eng.reset()
                    eng.prefill(case["ids"] + case["expected"][:32], pos=0)
            eng.save_usage(args.out)
        if args.mode in ("verify", "all"):
            tok = Inkling2Tokenizer(eng.gguf.kv)
            only = set(args.cases.split(",")) if args.cases else None
            if not verify_all(eng, tok, only_names=only, reference=args.reference):
                return 1
        if args.mode in ("bench", "all"):
            bench_engine(eng, repeats=args.repeats)
        return 0
    finally:
        eng.close()


if __name__ == "__main__":
    sys.exit(main())
