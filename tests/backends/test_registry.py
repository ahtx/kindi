# SPDX-License-Identifier: AGPL-3.0-only
"""Registry tests for the backend plugin interface (T4.1)."""

import pytest

from kindi.backends import Backend, BackendCapabilities, FakeBackend
from kindi.backends import registry


class ToyBackend(Backend):
    capabilities = BackendCapabilities(
        name="toy",
        model_ids=("toy-1b",),
        architectures=("toyarch",),
        devices=("cpu",),
        max_context=1024,
        deterministic_greedy=True,
    )

    def load(self, config):
        self.config = dict(config or {})
        return self

    def close(self):
        pass

    def info(self):
        return {"engine": "toy", "loaded": True}

    def tokenize(self, text, add_special_tokens=True):
        return [ord(c) + 1 for c in (text or "")]

    def detokenize(self, token_ids):
        return "".join(chr(int(t) - 1) for t in token_ids if int(t) >= 1)

    def apply_chat_template(self, messages, tools=None, enable_thinking=False):
        return "\n".join("%s: %s" % (m.get("role", "user"), m.get("content", ""))
                         for m in messages or [])

    def generate(self, prompt_ids, params, session=None, cancel=None):
        yield "done"

    def cache_stats(self):
        return {"cache_kind": "none"}


@pytest.fixture(autouse=True)
def isolate_registry():
    saved = dict(registry._REGISTRY)
    order = list(registry._ORDER)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
    registry._ORDER[:] = order


def test_register_and_list():
    registry.register_backend(ToyBackend)
    rows = registry.list_backends()
    names = [row["name"] for row in rows]
    assert "toy" in names
    assert "fake" in names
    toy = [row for row in rows if row["name"] == "toy"][0]
    assert toy["model_ids"] == ["toy-1b"]
    assert toy["architectures"] == ["toyarch"]
    assert toy["devices"] == ["cpu"]
    assert toy["factory"] == "ToyBackend"


def test_duplicate_registration_is_refused():
    registry.register_backend(ToyBackend)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_backend(ToyBackend)


def test_resolve_by_model_id():
    registry.register_backend(ToyBackend)
    backend = registry.resolve("toy-1b")
    assert isinstance(backend, ToyBackend)


def test_resolve_by_architecture_via_metadata(monkeypatch):
    registry.register_backend(ToyBackend)
    monkeypatch.setattr(registry, "gguf_architecture", lambda path: "toyarch")
    import os
    real_file = os.path.join(os.path.dirname(__file__), "test_registry.py")
    backend = registry.resolve(real_file)
    assert isinstance(backend, ToyBackend)


def test_resolve_unknown_names_what_is_registered():
    with pytest.raises(registry.BackendNotFound) as excinfo:
        registry.resolve("no-such-model")
    message = str(excinfo.value)
    assert "no-such-model" in message
    assert "fake" in message


def test_resolve_respects_device_filter():
    registry.register_backend(ToyBackend)
    with pytest.raises(registry.BackendNotFound):
        registry.resolve("toy-1b", device="cuda")


def test_external_module_loading_registers_on_import(tmp_path, monkeypatch):
    module = tmp_path / "toy_ext_backend.py"
    module.write_text(
        "from kindi.backends.base import Backend, BackendCapabilities\n"
        "from kindi.backends import registry\n"
        "class ExtBackend(Backend):\n"
        "    capabilities = BackendCapabilities(name='ext', model_ids=('ext-1b',), devices=('cpu',))\n"
        "    def load(self, config):\n"
        "        return self\n"
        "    def close(self): pass\n"
        "    def info(self): return {'engine': 'ext'}\n"
        "    def tokenize(self, text, add_special_tokens=True): return []\n"
        "    def detokenize(self, token_ids): return ''\n"
        "    def apply_chat_template(self, messages, tools=None, enable_thinking=False): return ''\n"
        "    def generate(self, prompt_ids, params, session=None, cancel=None): yield 'done'\n"
        "    def cache_stats(self): return {}\n"
        "registry.register_backend(ExtBackend)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("KINDI_BACKEND_MODULES", "toy_ext_backend")
    loaded = registry.load_external_backends()
    assert loaded == ["toy_ext_backend"]
    assert "ext" in [row["name"] for row in registry.list_backends()]


def test_external_module_failure_is_named(tmp_path, monkeypatch):
    (tmp_path / "broken_ext_backend.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(registry.BackendUnavailable, match="broken_ext_backend"):
        registry.load_external_backends("broken_ext_backend")


def test_fake_backend_is_resolvable():
    backend = registry.resolve("fake")
    assert isinstance(backend, FakeBackend)


def test_a_shipped_backend_refuses_a_second_registration():
    with pytest.raises(ValueError, match="already registered"):
        registry.register_backend(FakeBackend)


def test_ambiguous_resolution_is_refused_not_first_wins():
    """Two backends match equally well: registration order must not silently pick one."""
    class _Twin(FakeBackend):
        capabilities = BackendCapabilities(name="twin", model_ids=("kindi-fake",),
                                           architectures=("fake",))

    registry.register_backend(_Twin, replace=True)
    try:
        with pytest.raises(registry.BackendNotFound, match="ambiguous: fake and twin"):
            registry.resolve("kindi-fake")
        # a model id only one of them declares still resolves
        assert isinstance(registry.resolve("fake"), FakeBackend)
    finally:
        registry.unregister_backend("twin")


def test_stop_scanner_finds_a_match_across_two_pieces():
    from kindi.backends.base import StopScanner

    scanner = StopScanner(("END",))
    assert scanner.hit("before EN") is False
    assert scanner.hit("before END") is True
    assert StopScanner(()).hit("anything") is False


def test_stop_scanner_keeps_only_the_overlap_instead_of_rescanning():
    from kindi.backends.base import StopScanner

    scanner = StopScanner(("<eos>",))
    text = "x" * 5000
    assert scanner.hit(text) is False
    assert scanner.hit(text + "x") is False
    assert scanner.hit(text + "x<eo") is False
    assert scanner.hit(text + "x<eos>") is True


class _CountingText(str):
    """A str that counts the characters slicing hands out, so a rescan is measurable."""

    read = 0

    def __getitem__(self, item):
        piece = str.__getitem__(self, item)
        _CountingText.read += len(piece)
        return piece


def test_stop_scanner_reads_only_the_tail_of_a_long_reply():
    from kindi.backends.base import StopScanner

    scanner = StopScanner(("<eos>",))
    _CountingText.read = 0
    text = _CountingText("x" * 4000)
    assert scanner.hit(text) is False
    first = _CountingText.read
    for _ in range(200):
        text = _CountingText(text + "xxxx")
        assert scanner.hit(text) is False
    tail = _CountingText.read - first
    assert tail <= 200 * 12, "the scanner re-read %d chars for 800 new ones" % tail
    assert scanner.hit(_CountingText(text + "<eos>")) is True
