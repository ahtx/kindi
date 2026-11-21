# SPDX-License-Identifier: AGPL-3.0-only
"""T4.2 Bonsai adapter: the dense Ternary-Bonsai-2 engines behind the backend contract.

The adapter wraps the engines today built by ``kindi.engine.make_engine_legacy`` (the
plain GPU engine, the two-card plan engine and the CPU engine — and, through the same
factory, the tandem engine). ``load`` builds exactly the object ``make_engine_legacy``
built before the adapter existed, so serve/verify/bench keep their numbers; the adapter
adds the contract's text primitives (Qwen3.5 tokenizer, chat template) and a short
generation loop for API consumers. ``make_engine`` resolves dense Bonsai packs through
this adapter by default; ``KINDI_LEGACY_ENGINE_PATH=1`` opts out.

A pack is optional at ``load`` time: without one the backend stays unloaded (no GPU
touched), so the contract tests can exercise lifecycle/info without a model.
Tokenize/generate raise BackendUnavailable until a pack loads. The engine/tok/engine
imports stay inside ``load`` so importing this module (and the backends package) never
needs CuPy or NumPy on a CPU-only host.
"""
import json
import os
import time

from . import registry
from .base import Backend, BackendCapabilities, GenerationEvent, StopScanner

MODEL_ID_PREFIX = "Ternary-Bonsai-2-"
SIZES = ("2b", "4b", "9b", "27b")
ADD_SPECIAL = "<|endoftext|>"


def model_id_for_manifest(man):
    """Registry model id for a Bonsai pack manifest, or None for other packs.

    A pack resolves when its `model` is a ``Ternary-Bonsai-2-<size>`` name —
    or, like the 27B (whose GGUF ``general.name`` is the placeholder "Hf"),
    when the `source_file` carries the size (the same fallback
    `kindi/config.py` uses to name the served model).
    """
    import re as _re

    man = man or {}
    name = str(man.get("model") or "")
    if name.startswith(MODEL_ID_PREFIX):
        size = name.rsplit("-", 1)[-1].lower()
        if size in SIZES:
            return "bonsai2-" + size
        return None
    m = _re.search(r"(\d+(?:\.\d+)?)b(?:\b|-|_)", str(man.get("source_file") or "").lower())
    if m and (m.group(1) + "b") in SIZES and "bonsai" in str(man.get("source_file") or "").lower():
        return "bonsai2-" + m.group(1) + "b"
    return None


class BonsaiBackend(Backend):
    """Ternary-Bonsai-2 dense engine, behind the backend contract."""

    # The contract tests instantiate every registered backend with no arguments
    # and call the text primitives without a model. For this adapter that is
    # meaningless: skip the round-trip test, keep the lifecycle ones.
    REQUIRES_PACK = True

    capabilities = BackendCapabilities(
        name="bonsai",
        model_ids=("bonsai2-2b", "bonsai2-4b", "bonsai2-9b", "bonsai2-27b"),
        architectures=("qwen35",),
        devices=("cuda", "cpu", "rocm"),
        max_context=262144,
        supports_thinking=True,
        supports_tools=True,
        tool_format="json",
        supports_speculation=False,
        supports_token_filter=True,
        supports_json_grammar=True,
        cache_kind="none",
        max_concurrent_sequences=1,
        deterministic_greedy=True,
    )

    def __init__(self):
        self.loaded = False
        self.config = None
        self.engine = None
        self.tok = None
        self.man = None
        self.pack = None
        self.ctx = 0
        self.variant = "qwen35"
        self.log = print

    # lifecycle ----------------------------------------------------------

    def load(self, config):
        """Build the engine for ``config["pack"]``. An empty config loads nothing."""
        config = dict(config or {})
        self.config = config
        self.log = config.get("log") or print
        pack = config.get("pack")
        if not pack:
            self.loaded = False
            self.engine = None
            self.tok = None
            return self
        if not os.path.isdir(pack):
            raise registry.BackendUnavailable("no Bonsai pack directory at %r" % (pack,))

        from ..chat import template_variant
        from ..engine import make_engine_legacy
        from ..tokenizer import Tok

        self.pack = pack
        self.ctx = int(config.get("ctx") or 4096)
        with open(os.path.join(pack, "kindi_pack.json")) as f:
            self.man = json.load(f)
        self.engine = make_engine_legacy(
            pack,
            ctx=self.ctx,
            clusters=config.get("clusters"),
            log=self.log,
        )
        self.tok = Tok(self.engine.vocab, os.path.join(pack, "merges.txt"))
        self.variant = template_variant(self.man)
        self.loaded = True
        return self

    def close(self):
        self.engine = None
        self.tok = None
        self.man = None
        self.pack = None
        self.loaded = False

    def info(self):
        out = {
            "backend": "bonsai",
            "model_ids": list(self.capabilities.model_ids),
            "loaded": bool(self.loaded),
        }
        if self.engine is not None:
            out["engine"] = type(self.engine).__name__
            out["pack"] = self.pack
            out["model"] = str((self.man or {}).get("model") or "")
            vocab = getattr(self.engine, "vocab", None)
            try:
                out["vocab"] = len(vocab)
            except TypeError:
                out["vocab"] = 0
            out["ctx"] = self.ctx
            out["tandem"] = str((self.config or {}).get("tandem") or "off")
        return out

    def reference_gguf(self):
        if self.loaded and self.man:
            name = self.man.get("source_file")
            return name or None
        return None

    def cache_stats(self):
        return {}

    # text primitives ----------------------------------------------------

    def tokenize(self, text, add_special_tokens=True):
        self._require_loaded()
        return list(self.tok.encode(text or ""))

    def detokenize(self, token_ids):
        self._require_loaded()
        return self.tok.decode([int(t) for t in (token_ids or [])])

    def apply_chat_template(self, messages, tools=None, enable_thinking=False):
        from .. import chat

        rendered = chat.render(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            variant=self.variant,
        )
        if isinstance(rendered, tuple):
            return "".join(rendered)
        return rendered

    # generation ---------------------------------------------------------

    def _require_loaded(self):
        if not self.loaded or self.engine is None or self.tok is None:
            raise registry.BackendUnavailable(
                "the bonsai backend has no model loaded; "
                "load({'pack': <pack dir>}) first"
            )

    def generate(self, prompt_ids, params, session=None, cancel=None):
        """Greedy or tempered generation with the engine's own feed/step pair.

        Prefill feeds every prompt token but the last (which is stepped, exactly
        like ``Runner``), then decodes one token per step. ``params.stop`` is
        honored as decoded-text substrings; ``<|im_end|>`` stops; the done
        event carries ``finish_reason`` stop/length/cancelled and the measured
        decode time. No speculation: drafted/accepted are 0.
        """
        self._require_loaded()
        eng = self.engine
        ids = [int(t) for t in prompt_ids] if prompt_ids else []
        if not ids:
            ids = [int(self.tok.sid.get(ADD_SPECIAL, 0))]
        max_tokens = max(0, int(params.max_tokens)) if params is not None else 0
        rng = None
        if params is not None and (
            params.token_filter is not None or float(params.temperature or 0.0) > 1e-9
        ):
            import numpy as np

            rng = np.random.default_rng(int(params.seed) if params.seed is not None else 0)

        eng.reset()
        for pos in range(len(ids) - 1):
            eng.feed(ids[pos], pos)
        nxt = eng.step(ids[-1], len(ids) - 1)
        eos = self.tok.sid.get("<|im_end|>")
        stops = StopScanner((params.stop or ()) if params is not None else ())
        text = ""
        generated = []
        started = time.time()
        finish = "length"
        for _ in range(max_tokens):
            if cancel is not None and cancel.is_set():
                finish = "cancelled"
                break
            if rng is not None:
                nxt = self._sample_from_logits(
                    eng, params, generated=generated, prompt_ids=ids, rng=rng
                )
            generated.append(int(nxt))
            piece = self.tok.decode([int(nxt)])
            yield GenerationEvent(kind="token", token_id=int(nxt), text=piece)
            text += piece
            if (eos is not None and int(nxt) == eos) or stops.hit(text):
                finish = "stop"
                break
            if len(generated) >= max_tokens:
                finish = "length"
                break
            nxt = eng.step(int(nxt), len(ids) - 1 + len(generated))
        yield GenerationEvent(
            kind="done",
            finish_reason=finish,
            timing={"decode_s": round(time.time() - started, 4)},
            drafted=0,
            accepted=0,
        )

    def _sample_from_logits(self, eng, params, generated, prompt_ids, rng):
        """Sample the next token from the engine's ``ylog`` logits.

        ``params.temperature <= 0`` with a token filter is a masked argmax;
        otherwise temperature/top-k/top-p/min-p with the caller's seeded rng.
        """
        import numpy as np

        logits = None
        ylog = getattr(eng, "ylog", None)
        if ylog is not None:
            try:
                logits = ylog.get() if hasattr(ylog, "get") else np.asarray(ylog)
            except Exception:  # noqa: BLE001 - no logits means greedy-only for this build
                logits = None
        if logits is None:
            raise registry.BackendUnavailable(
                "sampling needs the engine's logits; this engine build does not "
                "expose them (temperature=0 only)"
            )
        logits = np.asarray(logits, dtype=np.float64).ravel().copy()
        filt = params.token_filter
        if filt is not None:
            allowed = filt({"prompt_ids": list(prompt_ids), "generated": list(generated)})
            if allowed is not None:
                allowed = set(int(t) for t in allowed)
                bad = np.fromiter(
                    (i not in allowed for i in range(logits.size)),
                    dtype=bool,
                    count=logits.size,
                )
                logits[bad] = -np.inf
        temp = float(params.temperature or 0.0)
        if temp <= 0:
            return int(np.argmax(logits))
        logits = logits / temp
        top_k = int(params.top_k or 0)
        if 0 < top_k < logits.size:
            kth = np.partition(logits, -top_k)[-top_k]
            logits[logits < kth] = -np.inf
        finite = np.isfinite(logits)
        if not finite.any():
            return int(np.argmax(logits))
        shifted = logits - np.max(logits[finite])
        exps = np.where(finite, np.exp(shifted), 0.0)
        total = exps.sum()
        if not np.isfinite(total) or total <= 0:
            return int(np.argmax(logits))
        probs = exps / total
        top_p = float(params.top_p or 1.0)
        if top_p < 1.0:
            order = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[order])
            cut = int(np.searchsorted(cum, top_p))
            keep = order[: max(1, cut)]
            masked = np.zeros_like(probs)
            masked[keep] = probs[keep]
            kept_total = masked.sum()
            if kept_total > 0:
                probs = masked / kept_total
        min_p = float(params.min_p or 0.0)
        if min_p > 0 and probs.size:
            threshold = min_p * float(probs.max())
            probs = np.where(probs >= threshold, probs, 0.0)
            kept_total = probs.sum()
            if kept_total > 0:
                probs = probs / kept_total
        norm = probs.sum()
        if not np.isfinite(norm) or norm <= 0:
            return int(np.argmax(logits))
        probs = probs / norm
        return int(rng.choice(logits.size, p=probs))


registry.register_backend(BonsaiBackend)
