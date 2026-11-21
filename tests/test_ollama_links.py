# SPDX-License-Identifier: AGPL-3.0-only
"""`tools/ollama_links.py`: give Ollama's content-addressed blobs readable .gguf names."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "ollama_links.py"
spec = importlib.util.spec_from_file_location("ollama_links", SCRIPT)
ollama_links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ollama_links)

MODEL = "application/vnd.ollama.image.model"
GGUF = b"GGUF" + b"\0" * 12


def _manifest(root, rel, layers):
    p = root / "manifests" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"layers": layers}), encoding="utf-8")
    return p


def _layer(digest, size=len(GGUF), media=MODEL):
    return {"mediaType": media, "digest": "sha256:" + digest, "size": size}


def _models(root, missing=False):
    """Two registries, a projector and a second model layer; optionally a manifest whose blob is gone."""
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    for digest, data in (("aa11", GGUF), ("bb22", GGUF + b"b"), ("cc33", GGUF + b"c"),
                         ("dd44", GGUF + b"d"), ("ee55", GGUF + b"e")):
        (blobs / ("sha256-" + digest)).write_bytes(data)
    _manifest(root, "registry.ollama.ai/library/gemma3/4b",
              [_layer("aa11"), _layer("nolicense", 10, "application/vnd.ollama.image.license"),
               _layer("noproj", 10, "application/vnd.ollama.image.projector")])
    _manifest(root, "registry.ollama.ai/example.com/team/bar/1b", [_layer("bb22")])
    _manifest(root, "registry.ollama.ai/library/big/1b",
              [_layer("ee55", 10), _layer("dd44", 500)])          # the large model layer wins
    if missing:
        (blobs / "sha256-cc33").unlink()
        _manifest(root, "library/gone/1b", [_layer("cc33")])
    return root


def test_discover_reads_manifests_and_names_by_model(tmp_path):
    models = _models(tmp_path / "models")
    found = {e["name"]: e for e in ollama_links.discover(models)}
    assert sorted(found) == ["big-1b", "example.com-team-bar-1b", "gemma3-4b"]
    # license and projector layers are not models; the largest model layer is the weights
    assert found["gemma3-4b"]["blob"].name == "sha256-aa11"
    assert found["big-1b"]["blob"].name == "sha256-dd44"
    assert found["big-1b"]["size"] == 500


def test_discover_keeps_a_manifest_whose_blob_is_missing(tmp_path):
    models = _models(tmp_path / "models", missing=True)
    found = {e["name"]: e for e in ollama_links.discover(models)}
    assert found["gone-1b"]["blob"].exists() is False


def test_main_links_each_model_and_is_idempotent(tmp_path, capsys):
    models = _models(tmp_path / "models")
    out = tmp_path / "links"
    assert ollama_links.main(["-m", str(models), "-o", str(out)]) == 0
    for name, blob in (("gemma3-4b.gguf", "sha256-aa11"), ("example.com-team-bar-1b.gguf", "sha256-bb22"),
                       ("big-1b.gguf", "sha256-dd44")):
        target = out / name
        assert target.is_file()
        assert target.read_bytes() == (models / "blobs" / blob).read_bytes()
        assert os.path.samefile(str(target), str(models / "blobs" / blob))   # linked, not copied
    assert ollama_links.main(["-m", str(models), "-o", str(out)]) == 0
    assert "already linked" in capsys.readouterr().out


def test_dry_run_changes_nothing(tmp_path):
    models = _models(tmp_path / "models")
    out = tmp_path / "links"
    assert ollama_links.main(["-m", str(models), "-o", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_ollama_models_environment_variable_finds_the_models(tmp_path, monkeypatch):
    models = _models(tmp_path / "models")
    out = tmp_path / "links"
    monkeypatch.setenv("OLLAMA_MODELS", str(models))
    assert ollama_links.main(["-o", str(out)]) == 0
    assert (out / "gemma3-4b.gguf").is_file()


def test_explicit_directory_wins_over_the_environment(tmp_path, monkeypatch):
    models = _models(tmp_path / "models")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "empty"))
    out = tmp_path / "links"
    assert ollama_links.main(["-m", str(models), "-o", str(out)]) == 0
    assert (out / "gemma3-4b.gguf").is_file()


def test_a_blob_directory_without_manifests_is_scanned_for_gguf(tmp_path):
    models = tmp_path / "models"
    (models / "blobs").mkdir(parents=True)
    (models / "blobs" / "sha256-abc").write_bytes(GGUF)
    (models / "blobs" / "sha256-notgguf").write_bytes(b"NOPE")
    found = ollama_links.discover(models)
    assert [e["name"] for e in found] == ["sha256-abc"]
    assert ollama_links.main(["-m", str(models), "-o", str(tmp_path / "links")]) == 0
    assert (tmp_path / "links" / "sha256-abc.gguf").read_bytes() == GGUF


def test_a_plain_folder_of_gguf_files_works(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "local-model.gguf").write_bytes(GGUF)
    assert [e["name"] for e in ollama_links.discover(models)] == ["local-model"]
    assert ollama_links.main(["-m", str(models), "-o", str(tmp_path / "links")]) == 0
    assert (tmp_path / "links" / "local-model.gguf").is_file()


def test_symlink_refusal_falls_back_to_a_hard_link(tmp_path, monkeypatch):
    blob = tmp_path / "blob"
    blob.write_bytes(GGUF)
    target = tmp_path / "link.gguf"

    def refuse(*a, **k):
        raise OSError(22, "symlinks need Developer Mode")

    monkeypatch.setattr(ollama_links.os, "symlink", refuse)
    assert ollama_links.make_link(blob, target) == "hardlink"
    assert target.read_bytes() == GGUF
    with pytest.raises(OSError):
        ollama_links.make_link(blob, tmp_path / "second.gguf", kind="symlink")


def test_missing_directory_is_an_error(tmp_path, capsys):
    assert ollama_links.main(["-m", str(tmp_path / "absent")]) == 2
    assert "no such directory" in capsys.readouterr().err


def test_foreign_file_is_kept_without_force_and_replaced_with_it(tmp_path):
    models = _models(tmp_path / "models")
    out = tmp_path / "links"
    out.mkdir()
    (out / "gemma3-4b.gguf").write_bytes(b"someone else's model")
    assert ollama_links.main(["-m", str(models), "-o", str(out)]) == 1
    assert (out / "gemma3-4b.gguf").read_bytes() == b"someone else's model"
    assert ollama_links.main(["-m", str(models), "-o", str(out), "--force"]) == 0
    assert (out / "gemma3-4b.gguf").read_bytes() == GGUF


def test_a_cross_volume_failure_explains_the_two_ways_out():
    assert "put -o on the models' drive" in ollama_links._link_hint(OSError(18, "cross-device"))
    assert ollama_links._link_hint(OSError(2, "no such file")) == ""
