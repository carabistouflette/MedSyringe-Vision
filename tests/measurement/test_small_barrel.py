import cv2
import numpy as np
import pytest

from medsyringe.small_capacity import accept_markings, tick_tracks
from medsyringe.small_fill import inspect_barrel


@pytest.mark.parametrize("digits,unit,expected", [
    (["", "1", "2", "3", "4", "5"], "ml", 5),
    (["1", "2", "3"], "mL", 3),
    (["1", "2", "3"], "nl", None),
    (["1", "2", "3", ""], "ml", None),
    (["1", "2", "4"], "ml", None),
    (["1", "", "3"], "ml", None),
])
def test_requires_terminal_digit_coherent_scale_and_explicit_unit(digits, unit, expected):
    match = accept_markings([(digit, .92) for digit in digits], [(unit, .95)])
    assert (match[0] if match else None) == expected


def test_recovers_regular_major_ticks_despite_unaligned_strokes():
    regular = tuple((150, 120 + 80 * n, 225) for n in range(6))
    all_strokes = sorted((*regular, (80, 162, 155), (148, 203, 181), (170, 251, 245)), key=lambda p: p[1])
    assert tick_tracks(all_strokes, 800, 300)[0] == (regular, 80)


@pytest.mark.parametrize("seal,liquid,presence", [(0., None, "empty"), (2., None, "unknown"),
                                                  (2., (190, 220, 235), "present")])
def test_colored_stopper_is_separate_from_chamber_color(seal, liquid, presence):
    zero, pitch, capacity = 120, 90, 5
    track = [(150, zero + i * pitch, 225) for i in range(capacity + 1)]
    frame = np.full((850, 290, 3), 225, np.uint8)
    row = round(zero + pitch * seal)
    if liquid is not None:
        frame[zero:row, 69:146] = liquid
    frame[row:row + 25, 68:147] = (112, 58, 31)
    result = inspect_barrel(frame, track, pitch, 1, capacity)
    assert result["presence"] == presence
    assert result["relative_opacity"] is None
    if presence == "empty":
        assert result["fill_ml"] == 0.
    if presence == "unknown":
        assert result["fill_ml"] is None and result["interval_ml"] is None
    if presence == "present":
        assert result["interval_ml"][0] < seal < result["interval_ml"][1]
        assert result["appearance"]["apparent_color"]["family"] == "orange"


def test_uniform_colored_plastic_is_not_a_liquid_column():
    frame = np.full((800, 270, 3), 225, np.uint8)
    frame[110:650, 70:145] = (130, 215, 230)
    cv2.rectangle(frame, (70, 295), (144, 322), (39, 73, 31), -1)
    track = [(150, 120 + 90 * n, 225) for n in range(6)]
    result = inspect_barrel(frame, track, 90, 1, 5)
    assert result["presence"] == "unknown"
    assert result["appearance"] is None
