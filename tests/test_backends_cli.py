# SPDX-License-Identifier: AGPL-3.0-only
"""CLI tests for `kindi backends` (REQ-0019 T4.3)."""

import argparse
import json

import pytest

from kindi import __main__ as cli
from kindi.backends import Backend, BackendCapabilities, registry


class CliToyBackend(Backend):
    capabilities = BackendCapabilities(name="cli-toy", devices=("cpu",))

    def load(self, config):
        return self

    def close(self):
        pass

    def info(self):
        return {"engine": "cli-toy"}

    def tokenize(self, text, add_special_tokens=True):
        return []

    def detokenize(self, token_ids):
        return ""

    def apply_chat_template(self, messages, tools=None, enable_thinking=False):
        return ""

    def generate(self, prompt_ids, params, session=None, cancel=None):
        yield "done"

    def cache_stats(self):
        return {}


@pytest.fixture(autouse=True)
def isolate_registry():
    saved = dict(registry._REGISTRY)
    order = list(registry._ORDER)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
    registry._ORDER[:] = order


def _args(json_output=False):
    return argparse.Namespace(json=json_output)


def test_backends_lists_registered_fake_backend(capsys):
    assert cli.cmd_backends({}, _args()) == 0
    out = capsys.readouterr().out
    assert "fake" in out
    assert "devices=cpu" in out
    assert "architectures=fake" in out
    assert "model_ids=fake,kindi-fake" in out


def test_backends_json_is_machine_readable(capsys):
    assert cli.cmd_backends({}, _args(json_output=True)) == 0
    rows = json.loads(capsys.readouterr().out)
    names = [row["name"] for row in rows]
    assert names[0] == "fake"
    assert rows[0]["devices"] == ["cpu"]
    assert rows[0]["factory"] == "FakeBackend"


def test_backends_loads_configured_external_modules(monkeypatch, capsys):
    import kindi.backends as backends_package

    def load_toy():
        registry.register_backend(CliToyBackend, replace=True)
        return ["toy_ext_backend"]

    monkeypatch.setattr(backends_package, "load_configured_backends", load_toy)
    assert cli.cmd_backends({}, _args()) == 0
    out = capsys.readouterr().out
    assert "fake" in out
    assert "cli-toy" in out
