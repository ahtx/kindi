# SPDX-License-Identifier: AGPL-3.0-only
"""`kindi verify` must route each release to its own reference file.

The release is part of the key: a Bonsai 2.5 pack compared against Bonsai 2's
reference would report a port failure that is really a wrong reference.
"""
import json

from kindi import verify


def _pack(tmp_path, man):
    (tmp_path / "kindi_pack.json").write_text(json.dumps(man), encoding="utf-8")
    return str(tmp_path)


def test_reference_for_routes_by_model_size(tmp_path):
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2-2B-PTQ1_0.gguf"})).endswith("reference.json")
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2-4B-PTQ1_0.gguf"})).endswith("reference-4b.json")
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2-9B-PTQ1_0.gguf"})).endswith("reference-9b.json")
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf"})).endswith("reference-27b.json")
    # the release is part of the key: 2.5 resolves to its own reference, never Bonsai 2 weights
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2.5-27B-PTQ1_0.gguf"})).endswith("reference-2.5-27b.json")
    assert verify.reference_for(_pack(tmp_path, {"model": "Ternary-Bonsai-2.5-27B"})).endswith("reference-2.5-27b.json")


def test_reference_for_defaults_to_2b_and_missing_is_none(tmp_path):
    assert verify.reference_for(_pack(tmp_path, {})).endswith("reference.json")
    assert verify.reference_for(_pack(tmp_path, {"source_file": "Ternary-Bonsai-2-99B-x.gguf"})) is None
