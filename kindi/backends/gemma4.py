# SPDX-License-Identifier: AGPL-3.0-only
"""T4.2 Gemma 4 adapter: the native Gemma 4 decoder behind the contract.

The adapter wraps ``kindi.gemma4.Engine`` (``kindi.gemma4_tp.TPEngine`` when the two-card
tensor-parallel mode is on) — the same objects the server's ``kindi_gemma4`` branch builds —
and adds the contract's text primitives (``kindi.gemma4_tokenizer.Tok``, the GGUF chat
template through ``kindi.gemma4_chat``) plus a short generation loop for API consumers.
``kindi.engine.make_engine`` resolves ``kindi_gemma4`` packs through this adapter by
default; ``KINDI_LEGACY_ENGINE_PATH=1`` opts out of the gateway entirely.

The server keeps its own richer path (the prompt cache, the JSON-schema and tool-call
grammars, the thinking-channel sorter, logprobs, the fast head). Like the Bonsai adapter,
this one is the engine plus the contract surface: whole prompt per request, exact head,
greedy or sampled from the full logits.

A pack is optional at ``load`` time: without one the backend stays unloaded (no GPU
touched), so the contract tests can exercise lifecycle/info without a model.
Tokenize/generate raise BackendUnavailable until a pack loads. The engine/tokenizer imports
stay inside ``load`` so importing this module (and the backends package) never needs CuPy or
NumPy on a CPU-only host.
"""
import codecs
import json
import os
import time

from . import registry
from .base import Backend, BackendCapabilities, GenerationEvent, StopScanner

MODEL_ID = "gemma4-e4b"
PACK_BACKEND = "kindi_gemma4"
PREFILL_BATCH = 512                    # the local KV ring takes one prompt batch of at most 512


def model_id_for_manifest(man):
    """Registry model id for a Gemma 4 pack manifest, or None for other packs."""
    if not isinstance(man, dict) or man.get("backend") != PACK_BACKEND:
        return None
    return MODEL_ID


def two_cards():
    """Gemma 4 in tensor parallel: KINDI_G4_TP=1 always, 0 never; auto (the default) when exactly
    two GPUs of the same kind are visible. The server's Gemma branch (`_two_cards`) asks this too,
    so adapter and server split on the same rule."""
    mode = os.environ.get("KINDI_G4_TP", "auto")
    if mode in ("0", "1"):
        return mode == "1"
    try:
        import cupy as cp

        n = cp.cuda.runtime.getDeviceCount()
        if n != 2:
            return False
        names = [cp.cuda.runtime.getDeviceProperties(i)["name"] for i in range(2)]
        return names[0] == names[1]
    except Exception:  # noqa: BLE001 - no CUDA: one engine
        return False


class Gemma4Backend(Backend):
    """Native Gemma 4 (E4B) decoder, behind the backend contract."""

    # Like the Bonsai/Nanbeige/Lightning adapters: tokenization needs a pack, so the
    # round-trip contract test is skipped and the lifecycle ones are kept.
    REQUIRES_PACK = True

    capabilities = BackendCapabilities(
        name="gemma4",
        model_ids=(MODEL_ID,),
        architectures=("gemma4",),
        devices=("cuda",),
        max_context=131072,            # the model's; the engine clamps it to device memory
        supports_thinking=True,
        supports_tools=True,
        tool_format="gemma4",
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
        self.stop_ids = set()
        self.log = print

    # lifecycle ----------------------------------------------------------

    def load(self, config):
        """Build the engine for ``config["pack"]``. An empty config loads nothing."""
        config = dict(config or {})
        self.config = config
        self.log = config.get("log") or print
        pack = config.get("pack")
        if not pack:
            self.close()
            return self
        if not os.path.isdir(pack):
            raise registry.BackendUnavailable("no Gemma 4 pack directory at %r" % (pack,))

        from ..gemma4 import Engine
        from ..gemma4_tokenizer import Tok
        from ..gemma4_tp import TPEngine

        self.pack = pack
        self.ctx = int(config.get("ctx") or 8192)
        with open(os.path.join(pack, "kindi_pack.json")) as f:
            self.man = json.load(f)
        if not self.man.get("model_path"):
            raise registry.BackendUnavailable(
                "%s/kindi_pack.json has no model_path (run `kindi convert <model.gguf>`)" % (pack,))
        self.man["pack_path"] = pack                  # the engine reads head_clusters.npz from here
        # KINDI_G4_TP: auto when two matching cards are visible, which is when the server
        # splits every layer across the pair instead of building one engine
        engine = TPEngine if two_cards() else Engine
        self.engine = engine(self.man, ctx=self.ctx, log=self.log)
        self.ctx = self.engine.ctx                    # the loader clamps it to what fits
        self.tok = Tok(self.man["model_path"])
        self.stop_ids = set(self.tok.stop_ids)
        self.loaded = True
        return self

    def close(self):
        engine, self.engine = self.engine, None
        close = getattr(engine, "close", None)
        if close is not None:
            close()
        self.tok = None
        self.man = None
        self.pack = None
        self.loaded = False

    def info(self):
        out = {
            "backend": "gemma4",
            "model_ids": list(self.capabilities.model_ids),
            "loaded": bool(self.loaded),
        }
        if self.engine is not None:
            out["engine"] = type(self.engine).__name__
            out["pack"] = self.pack
            out["model"] = str((self.man or {}).get("model") or "")
            out["vocab"] = len(self.tok.tokens) if self.tok else 0
            out["ctx"] = self.ctx
        return out

    def reference_gguf(self):
        if self.loaded and self.man:
            return self.man.get("source_file") or None
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
        self._require_loaded()
        from ..gemma4_chat import render

        return render(
            self.man["model_path"], messages, tools=tools,
            template_kwargs={"enable_thinking": bool(enable_thinking)},
        )

    # generation ---------------------------------------------------------

    def _require_loaded(self):
        if not self.loaded or self.engine is None or self.tok is None:
            raise registry.BackendUnavailable(
                "the gemma4 backend has no model loaded; load({'pack': <pack dir>}) first")

    def _prefill(self, ids):
        """Feed every prompt token but the last (the last one returns the first logits)."""
        eng = self.engine
        end = len(ids) - 1
        pos = 0
        while pos < end:
            stop = min(end, pos + PREFILL_BATCH)
            if stop - pos > 1:
                eng.forward_batch(ids[pos:stop], pos)
            else:
                eng.forward(ids[pos], pos, logits=False)
            pos = stop
        return eng.forward(ids[-1], len(ids) - 1, logits=True)

    def _next_token(self, logits, params, generated, rng, prompt_ids):
        cp = self.engine.cp
        temp = float(params.temperature or 0.0) if params is not None else 0.0
        allowed = None
        if params is not None and params.token_filter is not None:
            allowed = params.token_filter({"prompt_ids": list(prompt_ids),
                                           "generated": list(generated)})
            if allowed is not None:
                allowed = set(int(t) for t in allowed)
        if temp <= 0 and allowed is None:
            return int(cp.argmax(logits).item())      # no host copy of the vocabulary
        from ..sampling import pick_from_logits

        token = pick_from_logits(
            cp.asnumpy(logits), temp, float(params.top_p or 1.0), int(params.top_k or 0), rng,
            float(params.min_p or 0.0), allowed,
        )
        return int(cp.argmax(logits).item()) if token is None else token

    def generate(self, prompt_ids, params, session=None, cancel=None):
        """Greedy or tempered generation with the engine's prefill/step pair.

        The whole prompt is prefilled from a reset engine (the server's prompt cache stays
        there), then one token per step. ``params.stop`` is honored as decoded-text
        substrings; the tokenizer's stop ids end the turn; the done event carries
        ``finish_reason`` stop/length/cancelled and the measured decode time. Exact head, no
        speculation: drafted/accepted are 0.
        """
        self._require_loaded()
        eng = self.engine
        ids = [int(t) for t in prompt_ids] if prompt_ids else []
        if not ids:
            ids = [int(self.tok.bos)]
        if len(ids) > self.ctx:
            raise registry.BackendUnavailable(
                "the prompt is %d tokens but this pack's context is %d" % (len(ids), self.ctx))
        max_tokens = max(0, int(params.max_tokens)) if params is not None else 0
        max_tokens = max(0, min(max_tokens, self.ctx - len(ids)))
        temp = float(params.temperature or 0.0) if params is not None else 0.0
        filt = params.token_filter if params is not None else None
        resample = filt is not None or temp > 1e-9
        rng = None
        if temp > 1e-9:
            import numpy as np

            rng = np.random.RandomState(int(params.seed) if params.seed is not None else 0)
        eng.reset()
        logits = self._prefill(ids)
        stops = StopScanner((params.stop or ()) if params is not None else ())
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text = ""
        generated = []
        started = time.time()
        finish = "length"
        for n in range(max_tokens):
            if cancel is not None and cancel.is_set():
                finish = "cancelled"
                break
            token = self._next_token(logits, params, generated, rng, ids) if resample else int(
                eng.cp.argmax(logits).item())
            if token in self.stop_ids:
                finish = "stop"
                break
            generated.append(token)
            piece = decoder.decode(self.tok.decode_bytes(token))
            text += piece
            yield GenerationEvent(kind="token", token_id=token, text=piece)
            if stops.hit(text):
                finish = "stop"
                break
            if len(generated) < max_tokens:
                logits = eng.forward(token, len(ids) + n, logits=True)
        yield GenerationEvent(
            kind="done",
            finish_reason=finish,
            timing={"decode_s": round(time.time() - started, 4)},
            drafted=0,
            accepted=0,
        )


registry.register_backend(Gemma4Backend)
