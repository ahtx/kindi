# SPDX-License-Identifier: AGPL-3.0-only
"""Contract tests for the backend plugin interface (T4.1).

The suite runs over the fake backend today and auto-extends to every registered
backend. A registered backend whose constructor needs arguments should not be
registered without a factory that supplies them; that is what register_backend
is for.
"""
import pytest

from kindi.backends import (
    BackendCapabilities,
    CancelToken,
    FakeBackend,
    GenerationEvent,
    SamplingParams,
    backend_classes,
)


def _instantiate(backend_cls):
    # Backends registered for production use may need constructor arguments; the
    # contract tests only cover backends constructible without arguments. The
    # fake backend is the reference implementation of that shape.
    return backend_cls()


@pytest.mark.parametrize("backend_cls", backend_classes() or [FakeBackend])
def test_capabilities_have_the_contract_fields(backend_cls):
    backend = _instantiate(backend_cls)
    caps = backend.capabilities
    assert isinstance(caps, BackendCapabilities)
    for field in ("name", "model_ids", "architectures", "devices", "max_context",
                  "supports_thinking", "supports_tools", "tool_format",
                  "supports_speculation", "supports_token_filter", "supports_json_grammar", "cache_kind",
                  "max_concurrent_sequences", "deterministic_greedy"):
        assert hasattr(caps, field), field
    assert caps.max_concurrent_sequences >= 1
    assert isinstance(caps.to_dict(), dict)


@pytest.mark.parametrize("backend_cls", backend_classes() or [FakeBackend])
def test_lifecycle_info_and_cache_stats(backend_cls):
    backend = _instantiate(backend_cls)
    backend.load({})
    try:
        info = backend.info()
        assert isinstance(info, dict)
        assert isinstance(backend.cache_stats(), dict)
        assert backend.verify_prompts() is not None
        assert backend.reference_gguf() is None or isinstance(backend.reference_gguf(), str)
    finally:
        backend.close()


@pytest.mark.parametrize("backend_cls", backend_classes() or [FakeBackend])
def test_text_primitives_round_trip(backend_cls):
    if getattr(backend_cls, "REQUIRES_PACK", False):
        pytest.skip(
            "%s needs a model pack to tokenize; its text primitives are verified "
            "through the kindi verify/bench gate, not this harness" % backend_cls.__name__
        )
    backend = _instantiate(backend_cls)
    backend.load({})
    try:
        prompt = backend.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        assert isinstance(prompt, str) and "hello" in prompt
        ids = backend.tokenize(prompt)
        assert isinstance(ids, list) and ids
        assert all(isinstance(token_id, int) for token_id in ids)
        assert isinstance(backend.detokenize(ids), str)
    finally:
        backend.close()


def test_fake_backend_generates_and_honours_cancel():
    backend = FakeBackend(seed=7, scripted_text="hello world")
    backend.load({})
    params = SamplingParams(max_tokens=32)
    events = list(backend.generate(backend.tokenize("hi"), params))
    assert events[-1].kind == "done"
    assert events[-1].finish_reason == "stop"
    text = "".join(event.text for event in events if event.kind == "token")
    assert text == "hello world"
    assert all(isinstance(event, GenerationEvent) for event in events)

    cancel = CancelToken()
    cancel.set()
    cancelled = list(backend.generate([1, 2], SamplingParams(max_tokens=5), cancel=cancel))
    assert cancelled[-1].finish_reason == "cancelled"

    # token_filter is part of the contract even though T5.4 implements it later.
    assert SamplingParams().token_filter is None
    assert FakeBackend.capabilities.supports_token_filter is True


def test_fake_backend_is_deterministic_for_a_seed():
    first = FakeBackend(seed=3)
    first.load({})
    second = FakeBackend(seed=3)
    second.load({})
    params = SamplingParams(max_tokens=24)
    ids = first.tokenize("prompt")
    out1 = [e.text for e in first.generate(ids, params) if e.kind == "token"]
    out2 = [e.text for e in second.generate(ids, params) if e.kind == "token"]
    assert out1 == out2 and out1
