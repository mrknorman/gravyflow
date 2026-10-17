"""
A partner time (``partners.partner_time``): where a detector a transient was not
seen in is read instead.
"""

import numpy as np
import pytest

from gravyflow.src.dataset.acquisition.partners import partner_time

RATE = 4096.0
BEFORE = AFTER = 48.2
SEPARATION = 192.8
BLOCK = (8192.0 * 150000, 8192.0 * 150001)
T = BLOCK[0] + 4000.0


def draw(allowed, t=T, key=("spec", "t", "H1"), block=BLOCK, separation=SEPARATION):
    return partner_time(np.asarray(allowed, dtype=np.float64), t, BEFORE, AFTER, block, separation, key, RATE)


def test_a_partner_lies_whole_in_the_allowed_time_of_its_block_and_away_from_the_transient():
    allowed = np.array([
        [BLOCK[0] - 500.0, BLOCK[0] + 300.0],   # crosses the block's start
        [T - 400.0, T + 100.0],                  # mostly too near the transient
        [T + 150.0, T + 260.0],                  # too short beyond the separation
        [BLOCK[1] - 200.0, BLOCK[1] + 900.0],   # crosses the block's end
    ])
    for k in range(400):
        centre = draw(allowed, key=("spec", repr(T), f"H1-{k}"))
        span = (centre - BEFORE, centre + AFTER)
        assert any(a <= span[0] and span[1] <= b for a, b in allowed), f"{span} leaves the allowed time"
        assert BLOCK[0] <= span[0] and span[1] <= BLOCK[1], f"{span} leaves the block"
        assert abs(centre - T) >= SEPARATION, f"centre {centre - T:+.3f} s from the transient"
        assert centre * RATE == np.round(centre * RATE), "centre off the stored grid"


def test_the_same_key_gives_the_same_partner_and_other_keys_spread():
    allowed = [[T - 2000.0, T + 2000.0]]
    assert draw(allowed) == draw(allowed)
    centres = {draw(allowed, key=("spec", repr(T), f"H1-{k}")) for k in range(200)}
    assert len(centres) == 200
    assert min(centres) < T - SEPARATION and max(centres) > T + SEPARATION


@pytest.mark.parametrize("allowed", [
    [],
    [[T - 150.0, T + 150.0]],                          # all within the separation
    [[BLOCK[1] - 50.0, BLOCK[1] + 5000.0]],            # no whole span inside the block
    [[T + 300.0, T + 300.0 + BEFORE + AFTER - 0.01]],  # shorter than a span
], ids=["no time", "too near", "outside the block", "too short"])
def test_no_room_gives_no_partner(allowed):
    assert draw(allowed) is None
