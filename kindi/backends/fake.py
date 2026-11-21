# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic fake backend for tests (T4.1).

FakeBackend implements the full backend contract with no GPU, no weights and no
tokenizer files, so the API surface (Feature 5) can be built and tested on a
CPU-only host. Token ids are a byte-level encoding: id 0 is EOS, ids 1..256 are
the UTF-8 bytes of the text plus one. Generation is a seeded pseudo-random stream
over the same byte vocabulary, or a scripted output when ``scripted_text`` is
given; scripted output is what the tool-call parser tests (T5.2) will drive.
"""
import json
import random
import time

from .base import Backend, BackendCapabilities, CancelToken, GenerationEvent
from . import registry

EOS = 0


class FakeBackend(Backend):
    capabilities = BackendCapabilities(
        name="fake",
        model_ids=("fake", "kindi-fake"),
        architectures=("fake",),
        devices=("cpu",),
        max_context=4096,
        supports_thinking=True,
        supports_tools=True,
        tool_format="fake",
        supports_speculation=False,
        supports_token_filter=True,
        cache_kind="none",
        max_concurrent_sequences=1,
        deterministic_greedy=True,
    )

    def __init__(self, seed=0, scripted_text=None, step_latency=0.0):
        self.seed = int(seed)
        self.scripted_text = scripted_text
        self.step_latency = float(step_latency or 0.0)
        self.loaded = False
        self.config = None
        self.parked = {}

    # lifecycle ------------------------------------------------------------
    def load(self, config):
        self.config = dict(config or {})
        self.loaded = True
        return self

    def close(self):
        self.loaded = False

    def info(self):
        return {
            "engine": self.capabilities.name,
            "loaded": self.loaded,
            "max_context": self.capabilities.max_context,
            "cache_kind": self.capabilities.cache_kind,
            "deterministic_greedy": True,
        }

    # text primitives ------------------------------------------------------
    def tokenize(self, text, add_special_tokens=True):
        data = (text or "").encode("utf-8")
        ids = [byte + 1 for byte in data]
        if add_special_tokens:
            ids = [EOS] + ids
        return ids

    def detokenize(self, token_ids):
        data = bytearray()
        for token_id in token_ids or []:
            token_id = int(token_id)
            if 1 <= token_id <= 256:
                data.append(token_id - 1)
        return bytes(data).decode("utf-8", "replace")

    def apply_chat_template(self, messages, tools=None, enable_thinking=False):
        lines = []
        if tools:
            names = []
            for tool in tools or []:
                fn = tool.get("function", tool) if isinstance(tool, dict) else {}
                if fn.get("name"):
                    names.append(fn["name"])
            if names:
                lines.append("tools: " + ",".join(names))
        for message in messages or []:
            role = message.get("role", "user") if isinstance(message, dict) else "user"
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and "text" in part)
            lines.append("%s: %s" % (role, content))
        lines.append("assistant:")
        return "\n".join(lines)

    # generation -----------------------------------------------------------
    def _wait(self, cancel):
        """Sleep for one step of latency in small slices so cancel stays responsive."""
        remaining = self.step_latency
        while remaining > 0:
            if cancel is not None and cancel.is_set():
                return False
            slice_seconds = min(0.005, remaining)
            time.sleep(slice_seconds)
            remaining -= slice_seconds
        return not (cancel is not None and cancel.is_set())

    def generate(self, prompt_ids, params, session=None, cancel=None):
        if not self.loaded:
            raise RuntimeError("FakeBackend is not loaded; call load() first")
        cancel_token = cancel if cancel is not None else CancelToken()
        max_tokens = max(0, int(params.max_tokens)) if params is not None else 0
        text = self.scripted_text
        if text is None:
            rng = random.Random(0xC0FFEE ^ self.seed ^ (len(prompt_ids or []) * 131))
            alphabet = "abcdefghijklmnopqrstuvwxyz .,!?0123456789"
            text = "".join(rng.choice(alphabet) for _ in range(max_tokens))
        else:
            text = text[:max_tokens] if max_tokens else text
        started = time.time()
        produced = 0
        for ch in text:
            if cancel_token.is_set():
                yield GenerationEvent("done", finish_reason="cancelled",
                                      timing={"gen_s": round(time.time() - started, 4)})
                return
            if not self._wait(cancel_token):
                yield GenerationEvent("done", finish_reason="cancelled",
                                      timing={"gen_s": round(time.time() - started, 4)})
                return
            token_id = ord(ch) + 1 if ord(ch) < 256 else 63
            produced += 1
            yield GenerationEvent("token", token_id=token_id, text=ch)
        finish = "length" if max_tokens and produced >= max_tokens and len(text) >= max_tokens else "stop"
        yield GenerationEvent("done", finish_reason=finish,
                              timing={"gen_s": round(time.time() - started, 4),
                                      "gen_tok_s": round(produced / max(0.001, time.time() - started), 2)})

    # cache and verification hooks -----------------------------------------
    def cache_stats(self):
        return {"cache_kind": "none", "parked_sessions": len(self.parked)}

    def park(self, session):
        if session is None:
            return False
        self.parked[session] = True
        return True

    def reference_gguf(self):
        return None

    def verify_prompts(self):
        return [{"prompt": "The capital of France is", "expected": []}]


registry.register_backend(FakeBackend)
