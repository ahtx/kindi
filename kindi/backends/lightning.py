# SPDX-License-Identifier: AGPL-3.0-only
"""T4.2 Lightning adapter: the Nemotron-H MoE engine behind the contract.

The adapter wraps ``kindi.lightning.Engine`` (the same class
``lightning_gen``'s server backend and ``lightning_cli``'s verify/bench
build from the GGUF the pack manifest points at) and adds the contract's
text primitives (the Lightning tokenizer, the GGUF chat template) plus a
short generation loop for API consumers. ``kindi.engine.make_engine``
resolves ``kindi_lightning`` packs through this adapter by default;
``KINDI_LEGACY_ENGINE_PATH=1`` opts out. The adapter builds the same engine
object the legacy path built, so serve/verify/bench keep their numbers.
The one-request state reuse and the Mamba snapshots of
``lightning_gen.LightningBackend`` stay in the legacy server path; the
adapter reads the whole prompt per request.

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

MODEL_ID = "nemotron-lightning-30b"
PACK_BACKEND = "kindi_lightning"


def model_id_for_manifest(man):
    """Registry model id for a Lightning pack manifest, or None for other packs."""
    if not isinstance(man, dict) or man.get("backend") != PACK_BACKEND:
        return None
    return MODEL_ID


class LightningBackend(Backend):
    """Nemotron 3.5 Lightning 30B-A3B (Nemotron-H MoE) engine, behind the contract."""

    # Like the Bonsai/Nanbeige adapters: tokenization needs a pack, so the
    # round-trip contract test is skipped and the lifecycle ones are kept.
    REQUIRES_PACK = True

    capabilities = BackendCapabilities(
        name="lightning",
        model_ids=(MODEL_ID,),
        architectures=("nemotron_h_moe",),
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
        self.close()                    # a reload must not orphan the previous engine/reader
        self.config = config
        self.log = config.get("log") or print
        pack = config.get("pack")
        if not pack:
            return self
        if not os.path.isdir(pack):
            raise registry.BackendUnavailable("no Lightning pack directory at %r" % (pack,))

        from ..gguf import GGUF
        from ..lightning import Engine
        from ..lightning_tokenizer import LightningTokenizer

        self.pack = pack
        self.ctx = int(config.get("ctx") or 32768)
        with open(os.path.join(pack, "kindi_pack.json")) as f:
            self.man = json.load(f)
        self.engine = Engine(self.man["model_path"], ctx=self.ctx, log=self.log)
        self.tok = LightningTokenizer(GGUF(self.man["model_path"]).kv)
        # every control token ends the turn, as in lightning_gen.LightningBackend
        # (the tokenizer always has eos: it defaults to 2)
        self.stop_ids = {self.tok.eos} | {i for i, ty in enumerate(self.tok.types) if ty == 3}
        self.loaded = True
        return self

    def close(self):
        engine, self.engine = self.engine, None
        if engine is not None:
            engine.close()
        self.tok = None
        self.man = None
        self.pack = None
        self.loaded = False

    def info(self):
        out = {
            "backend": "lightning",
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
            name = self.man.get("source_file")
            return name or None
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
        self._require_loaded()
        from .. import lightning_chat

        return lightning_chat.render(
            self.man["model_path"], messages, tools=tools,
            enable_thinking=enable_thinking,
        )

    # generation ---------------------------------------------------------

    def _require_loaded(self):
        if not self.loaded or self.engine is None or self.tok is None:
            raise registry.BackendUnavailable(
                "the lightning backend has no model loaded; "
                "load({'pack': <pack dir>}) first"
            )

    def generate(self, prompt_ids, params, session=None, cancel=None):
        """Greedy or tempered generation with the engine's prefill/step pair.

        The whole prompt is prefilled from a reset engine (the legacy
        server backend's Mamba-state reuse stays there), then one token per
        step. ``params.stop`` is honored as decoded-text substrings; control
        tokens (EOS plus every type-3 token) stop; the done event carries
        ``finish_reason`` stop/length/cancelled and the measured decode time.
        No speculation: drafted/accepted are 0.
        """
        self._require_loaded()
        eng = self.engine
        ids = [int(t) for t in prompt_ids] if prompt_ids else [int(self.tok.bos)]
        if len(ids) > self.ctx:
            raise registry.BackendUnavailable(
                "the prompt is %d tokens but this pack's context is %d" % (len(ids), self.ctx))
        max_tokens = max(0, int(params.max_tokens)) if params is not None else 0
        max_tokens = max(0, min(max_tokens, self.ctx - len(ids)))
        temp = float(params.temperature or 0.0) if params is not None else 0.0
        # the engine's token is finalized at each step, so sampling (or a token mask, which
        # even at temperature 0 decides between the allowed candidates) needs one more pass
        resample = params is not None and (params.token_filter is not None or temp > 1e-9)
        rng = None
        if temp > 1e-9:
            import numpy as np

            rng = np.random.RandomState(
                int(params.seed) if params.seed is not None else 0
            )
        eng.reset()
        greedy = eng.prefill(ids, 0)
        stops = StopScanner((params.stop or ()) if params is not None else ())
        text = ""
        generated = []
        started = time.time()
        finish = "length"
        for _ in range(max_tokens):
            if cancel is not None and cancel.is_set():
                finish = "cancelled"
                break
            if resample:
                greedy = self._sample_next(params, greedy=int(greedy), generated=generated,
                                           rng=rng, prompt_ids=ids)
            generated.append(int(greedy))
            piece = self.tok.decode([int(greedy)])
            yield GenerationEvent(kind="token", token_id=int(greedy), text=piece)
            text += piece
            if int(greedy) in self.stop_ids or stops.hit(text):
                finish = "stop"
                break
            if len(generated) < max_tokens:
                greedy = eng.step(int(greedy))
        yield GenerationEvent(
            kind="done",
            finish_reason=finish,
            timing={"decode_s": round(time.time() - started, 4)},
            drafted=0,
            accepted=0,
        )

    def _sample_next(self, params, greedy, generated, rng, prompt_ids):
        """Sample the next token from the engine's candidate window.

        The token filter masks the full candidate set before the 256 cap (an
        argmax over the allowed ids at temperature 0), and the sampler chain is
        ``sampling.Sampler._pick``.
        """
        import numpy as np

        allowed = None
        if params.token_filter is not None:
            allowed = params.token_filter({"prompt_ids": list(prompt_ids),
                                           "generated": list(generated)})
            if allowed is not None:
                allowed = set(int(t) for t in allowed)
        temperature = float(params.temperature or 0.0)
        if temperature <= 0 and allowed is None:
            return greedy
        if allowed is not None:
            cand = np.asarray(sorted(allowed), dtype=np.int64)
            if not cand.size:
                return greedy
            sc = self.engine.cp.asnumpy(self.engine.logits[cand]).astype(np.float64)
        else:
            w = 16.0 * max(temperature, 0.1)
            got = self.engine.candidates(w)
            while got is None and w > 0.5:
                w /= 2
                got = self.engine.candidates(w)
            if got is not None:
                cand, sc = got
            else:
                z = self.engine.logits
                idx = self.engine.cp.flatnonzero(z >= z.max() - 30.0 * max(temperature, 0.1))
                cand = self.engine.cp.asnumpy(idx).astype(np.int64)
                sc = self.engine.cp.asnumpy(z[idx]).astype(np.float64)
        if cand.size > 256:
            keep = np.argpartition(sc, -256)[-256:]
            cand, sc = cand[keep], sc[keep]
        if temperature <= 0:
            return int(cand[int(np.argmax(sc))])
        from ..sampling import Sampler

        return Sampler._pick(cand, sc, temperature, float(params.top_p or 1.0),
                             int(params.top_k or 0), rng, float(params.min_p or 0.0))


registry.register_backend(LightningBackend)
