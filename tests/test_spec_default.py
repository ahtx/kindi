# SPDX-License-Identifier: AGPL-3.0-only
"""The measured mode-A draft length default, without a GPU or model pack."""
from types import SimpleNamespace

from kindi import spec


def _engine(mtp=None, drafter=None):
    return SimpleNamespace(mtp=mtp, drafter=drafter, dspine=None, W=1)


def test_mode_a_defaults_to_k1(monkeypatch):
    monkeypatch.setenv("KINDI_SPEC", "mtp")
    monkeypatch.delenv("KINDI_SPEC_K", raising=False)
    eng = _engine(mtp=object())
    spec.spec_begin(eng, greedy=True)
    assert eng.spec_mode == "mtp"
    assert eng.spec_k == 1


def test_explicit_spec_k_still_wins(monkeypatch):
    monkeypatch.setenv("KINDI_SPEC", "mtp")
    monkeypatch.setenv("KINDI_SPEC_K", "4")
    eng = _engine(mtp=object())
    spec.spec_begin(eng, greedy=True)
    assert eng.spec_k == 4


def test_other_speculation_modes_keep_k4(monkeypatch):
    monkeypatch.setenv("KINDI_SPEC", "ngram")
    monkeypatch.delenv("KINDI_SPEC_K", raising=False)
    eng = _engine()
    spec.spec_begin(eng, greedy=True)
    assert eng.spec_mode == "ngram"
    assert eng.spec_k == 4
