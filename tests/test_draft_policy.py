# SPDX-License-Identifier: AGPL-3.0-only
"""Adaptive draft-length control: the policy must draft more when the drafter
is accepted often, stop when it is not, and never exceed kmax."""
from kindi.draft_policy import DraftPolicy

COSTS = {1: 1.18, 2: 1.47, 3: 1.67, 4: 1.94, 5: 2.16, 6: 2.6, 7: 3.02}


def test_high_acceptance_picks_the_longest_draft():
    p = DraftPolicy("2b", kmax=3, costs=COSTS)
    for _ in range(20):
        p.observe(3, 3)
    assert p.acceptance > 0.9
    assert p.propose() == 3


def test_low_acceptance_picks_no_draft():
    p = DraftPolicy("2b", kmax=4, costs=COSTS)
    for _ in range(40):
        p.observe(0, 3)
    assert p.acceptance < 0.15
    assert p.propose() == 0


def test_propose_never_exceeds_kmax_and_never_goes_negative():
    p = DraftPolicy("x", kmax=2, costs=COSTS)
    for _ in range(30):
        p.observe(2, 2)
    assert 0 <= p.propose() <= 2
    assert DraftPolicy("y", kmax=0).propose() == 0


def test_low_step_confidence_shortens_the_draft():
    p = DraftPolicy("2b", kmax=4, costs=COSTS)
    for _ in range(30):
        p.observe(4, 4)
    confident = p.propose(confidence=0.95)
    unconfident = p.propose(confidence=0.2)
    assert confident > unconfident


def test_observe_updates_stats_and_acceptance():
    p = DraftPolicy("2b", kmax=3, costs=COSTS)
    p.observe(2, 3, confidence=0.9)
    s = p.stats()
    assert s["steps"] == 1 and s["drafted"] == 3 and s["kept"] == 2
    assert 0.0 < s["acceptance"] < 1.0 and s["confidence"] == 0.9


def test_unknown_cost_falls_back_to_never_drafting_past_it():
    p = DraftPolicy("z", kmax=7, costs={1: 1.18})
    for _ in range(20):
        p.observe(1, 1)
    assert p.propose() == 1


def test_limit_k_only_shortens_the_base():
    p = DraftPolicy("x", kmax=3, costs=COSTS)
    for _ in range(5):
        p.observe(3, 3)                      # everything is kept: longest draft is worth it
    assert p.limit_k(7) == 3                 # clamped by kmax=3
    assert p.limit_k(2) == 2                 # the smaller base passes through
    for _ in range(60):
        p.observe(0, 3)                      # nothing is kept: drafting costs more than it earns
    assert p.limit_k(4) == 0                 # drafts stop entirely


def test_limit_k_passes_the_base_through_with_no_history():
    p = DraftPolicy("x", kmax=3, costs=COSTS)
    p.acc = p.tries = 0.0                    # no history at all: propose() would be 0
    assert p.limit_k(4) == 4                 # ... but the controller must not kill the first drafts
