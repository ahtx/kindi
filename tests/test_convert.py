# SPDX-License-Identifier: AGPL-3.0-only
"""Trunk byte-identity between the base and -mtp Bonsai 2.5 27B releases.

The -mtp release is the base model plus blk.64, so every trunk tensor's bytes must
match. Runs wherever the catalog files are; skips cleanly where they are not.

    KINDI_BONSAI25_DIR=/path/to/models python -m pytest tests/test_convert.py
"""
import hashlib
import os

import pytest

from kindi.convert import PTQ1_0, mtp_head_block
from kindi.gguf import GGUF

    # the catalog-pinned release files; KINDI_BONSAI25_DIR overrides the models dir
def _bonsai25_gguf_paths():
    # the catalog-pinned release files; KINDI_BONSAI25_DIR overrides the models dir
    from kindi import catalog
    root = os.environ.get("KINDI_BONSAI25_DIR", os.path.expanduser("~/.kindi/models"))
    return tuple(os.path.join(root, catalog.MODELS[key]["file"])
                 for key in ("bonsai-2.5-27b", "bonsai-2.5-27b-mtp"))


def _gguf_profile(g):
    return (g.kv.get("qwen35.block_count"), g.kv.get("qwen35.embedding_length"),
            g.kv.get("qwen35.feed_forward_length"), g.kv.get("qwen35.attention.head_count"),
            g.kv.get("qwen35.attention.head_count_kv"), g.kv.get("qwen35.ssm.group_count"))


def _tensor_bytes(ne, ty):
    # exact data bytes for the trunk types the 2.5 27B files contain (F32 and PTQ1_0 only);
    # anything else fails loudly instead of hashing a wrong range
    rows = 1
    for x in ne[1:]:
        rows *= x
    if ty == PTQ1_0:
        assert ne[0] % 128 == 0
        return rows * (ne[0] // 128) * 28
    if ty == 0:
        return rows * ne[0] * 4
    raise AssertionError("unexpected 2.5 27B trunk tensor type %r" % (ty,))


def _sha256_range(f, start, size):
    digest = hashlib.sha256()
    f.seek(start)
    remaining = size
    while remaining:
        chunk = f.read(min(1 << 25, remaining))
        if not chunk:
            raise AssertionError("short read (%d of %d bytes left)" % (remaining, size))
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def test_the_mtp_file_carries_the_base_files_trunk_tensors_byte_identical():
    # F1: the -mtp release is the base model plus blk.64, so every trunk tensor's bytes must match.
    # Runs wherever the catalog files are; skips cleanly where they are not.
    from kindi.gguf import GGUF
    base_path, mtp_path = _bonsai25_gguf_paths()
    missing = [p for p in (base_path, mtp_path) if not os.path.isfile(p)]
    if missing:
        pytest.skip("bonsai 2.5 27B GGUFs absent: %s" % ", ".join(missing))
    base, mtp = GGUF(base_path), GGUF(mtp_path)
    assert mtp_head_block(_gguf_profile(mtp), [t[0] for t in mtp.tensors]) == 64
    assert mtp_head_block(_gguf_profile(base), [t[0] for t in base.tensors]) is None
    base_names = [t[0] for t in base.tensors]
    mtp_names = [t[0] for t in mtp.tensors]
    assert len(set(base_names)) == len(base_names)
    assert len(set(mtp_names)) == len(mtp_names)
    extra = set(mtp_names) - set(base_names)
    assert len(extra) == 15
    assert all(n.startswith("blk.64.") or ".nextn." in n or n.startswith("nextn.") for n in extra)
    base_info = {t[0]: (i, t[1], t[2], t[3]) for i, t in enumerate(base.tensors)}
    mtp_info = {t[0]: (i, t[1], t[2], t[3]) for i, t in enumerate(mtp.tensors)}
    base_end = os.path.getsize(base_path) - base.data_start
    mtp_end = os.path.getsize(mtp_path) - mtp.data_start
    for g, path, end in ((base, base_path, base_end), (mtp, mtp_path, mtp_end)):
        offs = [t[3] for t in g.tensors]
        assert offs == sorted(offs)
    with open(base_path, "rb") as fb, open(mtp_path, "rb") as fm:
        for name, ne, ty, off in base.tensors:
            size = _tensor_bytes(ne, ty)
            digests = []
            for g, f, info, end in ((base, fb, base_info, base_end),
                                    (mtp, fm, mtp_info, mtp_end)):
                i, me, mty, moff = info[name]
                assert me == ne and mty == ty
                nxt = g.tensors[i + 1][3] if i + 1 < len(g.tensors) else end
                assert moff + size <= nxt, name
                if moff + size < nxt:
                    f.seek(g.data_start + moff + size)
                    assert set(f.read(nxt - moff - size)) == {0}, "%s: nonzero padding" % name
                digests.append(_sha256_range(f, g.data_start + moff, size))
            assert digests[0] == digests[1], name
