# SPDX-License-Identifier: AGPL-3.0-only
"""The Bonsai 2.5 intake that a pack cannot test: the -mtp head block, and `low` rendering `xhigh`.

No weights, no GPU and no C: `convert.mtp_head_block` is a dimension tuple and tensor names, and the
template is text. `kindi convert` on the real 6.2 GB file is still the only end-to-end check, but the
two decisions that make that run work are pure functions, and they are the ones a wrong index breaks.

    python -m pytest tests/test_bonsai25_intake.py
"""
import pytest

from kindi import chat, convert

# (block_count, width, ffn, heads, kv_heads, groups) - the 27B profile in convert.PROFILES
MODEL_27B = (64, 5120, 17408, 24, 4, 16)
DIMS_27B = MODEL_27B[1:]


def _tensors(blocks, head_at=None):
    """The tensor names a 27B GGUF carries, optionally with the MTP head layer at ``head_at``."""
    names = []
    for il in range(blocks):
        names.append("blk.%d.attn_norm.weight" % il)
        names.append("blk.%d.ffn_gate.weight" % il)
        if il == head_at:
            names += ["blk.%d.nextn.eh_proj.weight" % il,
                      "blk.%d.nextn.enorm.weight" % il,
                      "blk.%d.nextn.hnorm.weight" % il,
                      "blk.%d.nextn.shared_head_norm.weight" % il]
    return names


def test_prisms_mtp_file_is_recognised_by_its_65th_block():
    """What Prism ships: block_count counts the head, so the model is 64 blocks and the head is 64."""
    profile = (65,) + DIMS_27B
    assert convert.mtp_head_block(profile, _tensors(65, head_at=64)) == 64


def test_a_block_count_that_excludes_the_head_still_finds_it():
    """The head is found by its own nextn.eh_proj marker, so the metadata's count cannot hide it."""
    profile = MODEL_27B                       # 64, as if the publisher did not count the head block
    assert convert.mtp_head_block(profile, _tensors(65, head_at=64)) == 64


def test_a_plain_27b_file_is_not_an_mtp_file():
    """The plain file has no 65th block: conversion must never look for one."""
    assert convert.mtp_head_block(MODEL_27B, _tensors(64)) is None


def test_a_layer_past_the_head_makes_it_a_plain_model():
    """A real layer at blk.65 means blk.64 is a model layer with nextn tensors, not the head."""
    names = _tensors(66, head_at=64)
    assert convert.mtp_head_block((65,) + DIMS_27B, names) is None
    assert convert.mtp_head_block(MODEL_27B, names) is None


def test_the_marker_alone_is_not_enough():
    """A head marker at a block index that is not a known model's layer count is not a head."""
    names = _tensors(65, head_at=64)
    assert convert.mtp_head_block((65,) + (2048, 6144, 8, 2, 16), names) is None   # 2B geometry
    assert convert.mtp_head_block(None, names) is None
    assert convert.mtp_head_block((None,) + DIMS_27B, names) is None


def test_low_renders_the_xhigh_instruction():
    """The 2.5 card does not support `low`; it must not select the template's own `low` text."""
    assert chat.REASONING_EFFORTS["low"] == chat.REASONING_EFFORTS["xhigh"]
    assert chat.REASONING_EFFORTS["low"] == chat._XHIGH_INSTR           # noqa: SLF001 - the property
    assert chat.REASONING_EFFORTS["medium"] == ""


def test_an_effort_is_rendered_into_the_system_turn():
    """`render` returns (prompt, generation prompt); the instruction lands in the system turn."""
    msgs = [{"role": "user", "content": "hi"}]
    low = chat.render(msgs, enable_thinking=True, variant="bonsai27", reasoning_effort="low")
    xhigh = chat.render(msgs, enable_thinking=True, variant="bonsai27", reasoning_effort="xhigh")
    medium = chat.render(msgs, enable_thinking=True, variant="bonsai27", reasoning_effort="medium")
    default = chat.render(msgs, enable_thinking=True, variant="bonsai27")
    assert low == xhigh == default
    assert chat._XHIGH_INSTR in low[0] and chat._XHIGH_INSTR not in medium[0]   # noqa: SLF001
    assert low[0].startswith("<|im_start|>system\n")


def test_an_unknown_effort_is_refused():
    with pytest.raises(ValueError):
        chat.render([{"role": "user", "content": "hi"}], enable_thinking=True,
                    variant="bonsai27", reasoning_effort="high")
