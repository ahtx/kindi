# SPDX-License-Identifier: AGPL-3.0-only
"""T4.2 DeepSeek V4.1 adapter: the DSpark MoE engine behind the contract.

The adapter wraps ``kindi.dsv41.Model`` — the CPU dense/expert engine the standalone
``python -m kindi.dsv41_server`` serves — and adds the contract's text primitives (the
tokenizer in the GGUF, DeepSeek's own ``encoding.py`` chat template) plus a short generation
loop for API consumers. ``kindi.engine.make_engine`` resolves ``kindi_dsv41`` packs through
this adapter by default; ``KINDI_LEGACY_ENGINE_PATH=1`` opts out of the gateway entirely.

Unlike the other families there is no pack converter for DeepSeek V4.1: a pack is a
directory holding ``kindi_pack.json`` with ``backend: kindi_dsv41``, the ``model_path`` the
engine maps (the first GGUF shard), and optionally ``encoding`` (the ``encoding.py`` that
ships with the model, used for the chat template). The engine's CPU worker pools, the GPU
forward kernels (``KINDI_DSV_GPU``), the DSpark drafter, the RAM slots and the turn saves
stay with the standalone server; the adapter is the engine plus the contract surface: whole
prompt per request, greedy or sampled, no speculation.

A pack is optional at ``load`` time: without one the backend stays unloaded (the C library
is not compiled and no shard is mapped), so the contract tests can exercise lifecycle/info
without a model. Tokenize/generate raise BackendUnavailable until a pack loads.
"""
import codecs
import json
import os
import time

from . import registry
from .base import Backend, BackendCapabilities, GenerationEvent, StopScanner

MODEL_ID = "deepseek-v41-flash"
PACK_BACKEND = "kindi_dsv41"
ARCHITECTURES = ("deepseek41",)


def model_id_for_manifest(man):
    """Registry model id for a DeepSeek V4.1 pack manifest, or None for other packs."""
    if not isinstance(man, dict) or man.get("backend") != PACK_BACKEND:
        return None
    return MODEL_ID


class DeepseekV41Backend(Backend):
    """DeepSeek V4.1 Flash (DSpark MoE, compiled CPU engine), behind the contract."""

    # Like the other adapters: tokenization needs a pack, so the round-trip contract test
    # is skipped and the lifecycle ones are kept.
    REQUIRES_PACK = True

    capabilities = BackendCapabilities(
        name="dsv41",
        model_ids=(MODEL_ID,),
        architectures=ARCHITECTURES,
        # the dense layers and the experts run in the compiled CPU library; the CUDA forward
        # kernels are opt-in (KINDI_DSV_GPU) and are not what the engine is built around
        devices=("cpu",),
        max_context=8192,              # the engine's own default; it takes whatever fits RAM
        supports_thinking=True,
        supports_tools=True,
        tool_format="deepseek",
        supports_speculation=False,    # the DSpark drafter stays in the standalone server
        supports_token_filter=True,
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
        self._enc = None

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
            raise registry.BackendUnavailable("no DeepSeek V4.1 pack directory at %r" % (pack,))

        with open(os.path.join(pack, "kindi_pack.json")) as f:
            man = json.load(f)
        path = man.get("model_path")
        if not path:
            raise registry.BackendUnavailable(
                "%s/kindi_pack.json has no model_path (the first GGUF shard)" % (pack,))

        from ..dsv41 import Model

        self.pack = pack
        self.man = man
        self.ctx = int(config.get("ctx") or 8192)
        self.engine = Model(path, ctx=self.ctx, tmax=int(config.get("tmax") or 64))
        self.ctx = int(self.engine.ctx)
        self.tok = self.engine.tok
        self.stop_ids = {int(self.engine.eos)}
        self.loaded = True
        return self

    def close(self):
        self.engine = None
        self.tok = None
        self.man = None
        self.pack = None
        self._enc = None
        self.loaded = False

    def info(self):
        out = {
            "backend": "dsv41",
            "model_ids": list(self.capabilities.model_ids),
            "loaded": bool(self.loaded),
        }
        if self.engine is not None:
            out["engine"] = type(self.engine).__name__
            out["pack"] = self.pack
            out["model"] = str((self.man or {}).get("model") or "")
            out["vocab"] = len(self.tok.vocab) if self.tok else 0
            out["ctx"] = self.ctx
        return out

    def reference_gguf(self):
        if self.loaded and self.man:
            return self.man.get("model_path") or None
        return None

    def cache_stats(self):
        return {}

    # text primitives ----------------------------------------------------

    def tokenize(self, text, add_special_tokens=True):
        self._require_loaded()
        return list(self.tok.encode(text or "", special=add_special_tokens))

    def detokenize(self, token_ids):
        self._require_loaded()
        return self.tok.decode([int(t) for t in (token_ids or [])])

    def apply_chat_template(self, messages, tools=None, enable_thinking=False):
        """DeepSeek's own renderer: ``encoding.py`` from the model repo (manifest ``encoding``)."""
        self._require_loaded()
        from ..dsv41_server import to_encoding

        enc = self._encoding()
        return enc.encode_messages(
            to_encoding(messages or [], tools or None),
            thinking_mode="thinking" if enable_thinking else "chat",
        )

    def _encoding(self):
        if self._enc is None:
            path = (self.man or {}).get("encoding") or os.path.join(self.pack or "", "encoding.py")
            path = os.path.expanduser(str(path))
            if not os.path.isfile(path):
                raise registry.BackendUnavailable(
                    "no DeepSeek encoding.py for the chat template: point the pack manifest's "
                    "\"encoding\" at the file that ships with the model (tried %s)" % (path,))
            from ..dsv41_server import load_encoding

            self._enc = load_encoding(path)
        return self._enc

    # generation ---------------------------------------------------------

    def _require_loaded(self):
        if not self.loaded or self.engine is None or self.tok is None:
            raise registry.BackendUnavailable(
                "the dsv41 backend has no model loaded; load({'pack': <pack dir>}) first")

    def generate(self, prompt_ids, params, session=None, cancel=None):
        """Greedy or tempered generation over the engine's ``forward``.

        The whole prompt goes through one ``forward`` (the engine chunks it by its own tmax),
        then one token per step. ``params.stop`` is honored as decoded-text substrings; the
        engine's EOS ends the turn; the done event carries ``finish_reason``
        stop/length/cancelled and the measured decode time. No speculation: drafted/accepted
        are 0 (the DSpark drafter stays in the standalone server).
        """
        self._require_loaded()
        import numpy as np

        eng = self.engine
        ids = [int(t) for t in prompt_ids] if prompt_ids else []
        if not ids:
            raise registry.BackendUnavailable("the dsv41 backend needs at least one prompt token")
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
            rng = np.random.RandomState(int(params.seed) if params.seed is not None else 0)
        eng.reset()
        logits = eng.forward(ids)
        stops = StopScanner((params.stop or ()) if params is not None else ())
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text = ""
        generated = []
        started = time.time()
        finish = "length"
        for _ in range(max_tokens):
            if cancel is not None and cancel.is_set():
                finish = "cancelled"
                break
            token = self._next_token(logits, params, generated, rng, ids) if resample else int(np.argmax(logits))
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
                logits = eng.forward([token])
        yield GenerationEvent(
            kind="done",
            finish_reason=finish,
            timing={"decode_s": round(time.time() - started, 4)},
            drafted=0,
            accepted=0,
        )

    def _next_token(self, logits, params, generated, rng, prompt_ids):
        """One token from the full logits vector; the sampler chain is ``sampling.pick_from_logits``."""
        import numpy as np

        temp = float(params.temperature or 0.0)
        allowed = None
        if params.token_filter is not None:
            allowed = params.token_filter({"prompt_ids": list(prompt_ids),
                                           "generated": list(generated)})
            if allowed is not None:
                allowed = set(int(t) for t in allowed)
        if temp <= 0 and allowed is None:
            return int(np.argmax(logits))
        from ..sampling import pick_from_logits

        token = pick_from_logits(logits, temp, float(params.top_p or 1.0), int(params.top_k or 0),
                                 rng, float(params.min_p or 0.0), allowed)
        return int(np.argmax(logits)) if token is None else token


registry.register_backend(DeepseekV41Backend)
