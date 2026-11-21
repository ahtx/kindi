# SPDX-License-Identifier: AGPL-3.0-only
"""T4.2 Nanbeige adapter: the llama-architecture K/I-quant engine behind the contract.

The adapter wraps ``kindi.nanbeige.Engine`` (the same class ``nanbeige_gen``'s
server backend and ``nanbeige_cli``'s verify/bench build from the GGUF the
pack manifest points at) and adds the contract's text primitives (SPM
tokenizer, Nanbeige chat template) plus a short generation loop for API
consumers. ``kindi.engine.make_engine`` resolves ``kindi_nanbeige`` packs
through this adapter by default; ``KINDI_LEGACY_ENGINE_PATH=1`` opts out
to the legacy builder (which has no Nanbeige branch, so Nanbeige serve still
needs the adapter to load anything at all). The engine object is the same
class the legacy path built, so serve/verify/bench keep their numbers.

This adapter is the non-drafting gateway path: ``generate()`` runs one token per
step and reports ``drafted=0``/``accepted=0``, and ``supports_speculation`` is
False. The server's draft sources (``lookup`` and ``mtp_head``) belong to
``nanbeige_gen.NanbeigeBackend``'s check-pass loop, so this adapter does not load
them; `kindi verify`/`bench`, which build the same base engine, do not either.

A pack is optional at ``load`` time: without one the backend stays unloaded
(no GPU touched), so the contract tests can exercise lifecycle/info without
a model. Tokenize/generate raise BackendUnavailable until a pack loads.
The engine/tokenizer imports stay inside ``load`` so importing this module
(and the backends package) never needs CuPy or NumPy on a CPU-only host.
"""
import json
import os
import time

from . import registry
from .base import Backend, BackendCapabilities, GenerationEvent, StopScanner

MODEL_ID = "nanbeige-33b"
PACK_BACKEND = "kindi_nanbeige"


def model_id_for_manifest(man):
    """Registry model id for a Nanbeige pack manifest, or None for other packs."""
    if not isinstance(man, dict) or man.get("backend") != PACK_BACKEND:
        return None
    return MODEL_ID


class NanbeigeBackend(Backend):
    """Nanbeige 4.3 33B (llama I-quant) engine, behind the backend contract."""

    # Like the Bonsai adapter: tokenization needs a pack, so the round-trip
    # contract test is skipped and the lifecycle ones are kept.
    REQUIRES_PACK = True

    capabilities = BackendCapabilities(
        name="nanbeige",
        model_ids=(MODEL_ID,),
        architectures=("llama",),
        devices=("cuda",),
        max_context=262144,
        supports_thinking=True,
        supports_tools=True,
        tool_format="xml",
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
            self.loaded = False
            self.engine = None
            self.tok = None
            return self
        if not os.path.isdir(pack):
            raise registry.BackendUnavailable("no Nanbeige pack directory at %r" % (pack,))

        from ..gguf import GGUF
        from ..nanbeige import Engine
        from ..spm import SPM

        self.pack = pack
        self.ctx = int(config.get("ctx") or 4096)
        with open(os.path.join(pack, "kindi_pack.json")) as f:
            self.man = json.load(f)
        self.engine = Engine(
            self.man["model_path"], ctx=self.ctx,
            kv=config.get("kv_cache"), log=self.log,
        )
        self.tok = SPM(GGUF(self.man["model_path"]).kv)
        # every control token ends the turn, as llama.cpp's end-of-generation
        # set does (same rule as nanbeige_gen.NanbeigeBackend)
        self.stop_ids = {i for i, ty in enumerate(self.tok.types) if ty == 3}
        if self.tok.eos is not None:
            self.stop_ids.add(self.tok.eos)
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
            "backend": "nanbeige",
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
        from .. import nanbeige_chat

        return nanbeige_chat.render(
            messages, tools=tools, enable_thinking=enable_thinking
        )

    # generation ---------------------------------------------------------

    def _require_loaded(self):
        if not self.loaded or self.engine is None or self.tok is None:
            raise registry.BackendUnavailable(
                "the nanbeige backend has no model loaded; "
                "load({'pack': <pack dir>}) first"
            )

    def generate(self, prompt_ids, params, session=None, cancel=None):
        """Greedy or tempered generation with the engine's prefill/step pair.

        The prompt is prefilled in batch (as ``nanbeige_cli.verify`` does),
        then one token per step. ``params.stop`` is honored as decoded-text
        substrings; control tokens stop; the done event carries
        ``finish_reason`` stop/length/cancelled and the measured decode time.
        No speculation: drafted/accepted are 0.
        """
        self._require_loaded()
        eng = self.engine
        ids = [int(t) for t in prompt_ids] if prompt_ids else []
        if not ids:
            ids = [int(self.tok.bos)] if self.tok.bos is not None else [0]
        max_tokens = max(0, int(params.max_tokens)) if params is not None else 0
        rng = None
        if params is not None and (
            params.token_filter is not None or float(params.temperature or 0.0) > 1e-9
        ):
            import numpy as np

            rng = np.random.RandomState(
                int(params.seed) if params.seed is not None else 0
            )
        nxt = eng.prefill_batch(ids)
        pos = len(ids)
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
                nxt = self._sample_next(params, greedy=int(nxt), generated=generated,
                                        rng=rng, prompt_ids=ids)
            generated.append(int(nxt))
            piece = self.tok.decode([int(nxt)])
            yield GenerationEvent(kind="token", token_id=int(nxt), text=piece)
            text += piece
            if int(nxt) in self.stop_ids or stops.hit(text):
                finish = "stop"
                break
            if len(generated) >= max_tokens:
                finish = "length"
                break
            # pos tracks the token just generated's own position: the prompt
            # occupies 0..len(ids)-1, so the first generated token is at
            # len(ids). Mirrors Engine.generate's pos bookkeeping exactly.
            nxt = eng.step(int(nxt), pos)
            pos += 1
        yield GenerationEvent(
            kind="done",
            finish_reason=finish,
            timing={"decode_s": round(time.time() - started, 4)},
            drafted=0,
            accepted=0,
        )

    def _sample_next(self, params, greedy, generated, rng, prompt_ids):
        """Sample the next token from the last step's logits.

        Mirrors ``nanbeige_gen.NanbeigeBackend._pick``: without a token
        filter, candidates are the tokens within 30 temperatures of the
        best, scored through ``sampling.Sampler._pick``. With a token
        filter the allowed ids are read in full, so a grammar-legal token
        outside the candidate window is not lost.
        """
        import numpy as np

        temp = float(params.temperature or 0.0)
        filt = params.token_filter
        allowed = None
        if filt is not None:
            allowed = filt({"prompt_ids": list(prompt_ids), "generated": list(generated)})
        eng = self.engine
        cp = eng.cp
        c = eng.cards[-1]
        with cp.cuda.Device(c.dev):
            z = c.logits.ravel()[:len(self.tok.tokens)]
            if allowed is not None:
                cand = np.asarray(sorted(int(t) for t in allowed), dtype=np.int64)
                if not cand.size:
                    raise ValueError("constrained sampling: no allowed token in the model's candidates")
                sc = cp.asnumpy(z[cand]).astype(np.float64)
            else:
                idx = cp.flatnonzero(z >= z.max() - 30.0 * max(temp, 1e-9))
                cand = cp.asnumpy(idx).astype(np.int64)
                sc = cp.asnumpy(z[idx]).astype(np.float64)
        if temp <= 0:
            return int(cand[int(np.argmax(sc))])
        if allowed is None and cand.size > 256:
            keep = np.argpartition(sc, -256)[-256:]
            cand, sc = cand[keep], sc[keep]
        from ..sampling import Sampler

        return int(Sampler._pick(
            cand, sc, temp, float(params.top_p or 1.0), int(params.top_k or 0), rng
        ))


registry.register_backend(NanbeigeBackend)
