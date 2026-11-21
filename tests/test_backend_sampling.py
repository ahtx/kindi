# SPDX-License-Identifier: AGPL-3.0-only
"""The adapter sampler: a full-vocabulary pick with an optional allowed-id mask."""

import numpy as np

from kindi.sampling import pick_from_logits


def test_pick_from_logits_masks_before_argmax():
    z = np.array([0.0, 9.0, 8.0, 1.0])
    assert pick_from_logits(z, 0.0, 1.0, 0, None) == 1
    assert pick_from_logits(z, 0.0, 1.0, 0, None, allowed={2}) == 2


def test_pick_from_logits_returns_none_for_an_empty_mask():
    z = np.array([0.0, 9.0, 8.0, 1.0])
    assert pick_from_logits(z, 0.0, 1.0, 0, None, allowed=set()) is None
