"""Kindi prompt cache examples: four workloads against one OpenAI-compatible server. Each request prints one JSON
line with its prompt tokens, the tokens read from the cache, the time and the reply.

    python3 examples/prompt_cache_bench.py http://127.0.0.1:8093/v1/chat/completions

Run it once against a server started with the cache off (KINDI_G4_PROMPT_CACHE=0 for Nemotron) and once with it on
(the default), then compare: the replies must be the same, and the cached requests much faster. See
docs/PROMPT-CACHE.md for the results on a GTX 1080.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import json, sys, time, urllib.request
url = sys.argv[1]
def chat(msgs, n=60):
    body = json.dumps({"model": "x", "messages": msgs, "max_tokens": n, "temperature": 0}).encode()
    t = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=1800))
    return r["choices"][0]["message"]["content"], r["usage"], time.time() - t
def out(case, step, u, dt, text):
    print(json.dumps({"case": case, "step": step, "prompt": u["prompt_tokens"], "cached": u["prompt_tokens_details"]["cached_tokens"], "s": round(dt, 2), "text": text}), flush=True)
rules = " ".join("Rule %d: answer briefly, cite the file and the line, never guess." % i for i in range(120))
# 1. an agent: one long system prompt, then turns that extend the conversation
msgs = [{"role": "system", "content": "You are a careful assistant for a software team. " + rules}]
for i, t in enumerate(["List three uses of a hash map.", "Now give one drawback of each.", "Which one matters most for a cache? One sentence."]):
    msgs.append({"role": "user", "content": t}); text, u, dt = chat(msgs); msgs.append({"role": "assistant", "content": text}); out("agent", i + 1, u, dt, text)
# 2. regenerate: the same prompt twice, with a long answer in between
m2 = [{"role": "system", "content": "You are a patient teacher. " + rules}, {"role": "user", "content": "Explain, step by step and at length, how TCP sets up and closes a connection."}]
for i in range(2):
    text, u, dt = chat(m2, 600); out("regenerate", i + 1, u, dt, text)
# 3. RAG: one long document, then different questions about it
doc = " ".join("Section %d. The service keeps its settings in config/%d.yaml; the key timeout_%d is %d seconds, and retries_%d is %d." % (i, i, i, 10 + i, i, i % 5) for i in range(220))
for i, q in enumerate(["What is timeout_17? Answer with the number.", "Which file holds the settings of section 42?", "What is retries_101? Answer with the number."]):
    text, u, dt = chat([{"role": "user", "content": doc + "\n\nQuestion: " + q}], 400); out("rag", i + 1, u, dt, text)
# 4. two agents that take turns, each with its own long system prompt
A = [{"role": "system", "content": "Agent A, a code reviewer. " + rules}]
B = [{"role": "system", "content": "Agent B, a release manager. " + rules.replace("Rule", "Policy")}]
for i, (who, conv, t) in enumerate([("A", A, "Review: def f(x): return x*2"), ("B", B, "Plan a 0.3.0 release in three steps."),
                                    ("A", A, "Suggest one test for it."), ("B", B, "What could block step two?")]):
    conv.append({"role": "user", "content": t}); text, u, dt = chat(conv); conv.append({"role": "assistant", "content": text}); out("two agents (%s)" % who, i + 1, u, dt, text)
