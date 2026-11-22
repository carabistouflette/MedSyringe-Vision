import numpy as np
import pytest

from medsyringe.hybrid_detection import candidates, rank, retain_baseline, rule, vectors
from medsyringe.liquid_color import describe_region
from medsyringe.marking_views import agreement, supplement
from medsyringe.markings_guard import confirm_untyped_capacity


def test_hybrid_agreement_fuses_and_keeps_distinct_instances():
    shared = [0.12, 0.10, 0.35, 0.55]
    alternative = [0.13, 0.10, 0.37, 0.55]
    separate = [0.55, 0.10, 0.71, 0.60]
    branches = {"primary": [dict(bbox=shared, score=.8)],
                "sam": [dict(bbox=alternative, score=.88)], "rf": [dict(bbox=separate, score=.75)]}
    proposals = candidates(branches)
    assert vectors(proposals).shape == (3, 13)
    result = rank(proposals, rule(proposals, "agreement"), mode="weighted")
    assert len(result) == 2
    assert result[0]["support_channels"] == ["primary", "sam"]
    assert shared[0] < result[0]["bbox"][0] < alternative[0]
    preserved = retain_baseline(branches["primary"], result, .5, .7)
    assert [p["bbox"] for p in preserved] == [shared, separate]


def test_ocr_fallback_requires_agreement_and_respects_existing_scale():
    capacity = dict(value=20., unit="mL", score=.93)
    choice = {"candidates": [], "capacity": capacity, "scale": None}
    wrong = {**choice, "capacity": dict(value=60., unit="mL", score=.92)}
    assert agreement(choice, wrong)["capacity"] is None
    joint = agreement(choice, choice)
    assert joint["capacity"]["value"] == 20.
    initial = dict(candidates=[], capacity=None, scale=dict(value="60-scale"))
    assert supplement(initial, joint)["capacity"] is None
    assert supplement({**initial, "scale": None}, joint)["capacity"]["unit"] == "mL"


def test_supported_color_keeps_opacity_unknown():
    rgb = np.full((20, 20, 3), 220, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[3:17, 8:12] = True
    rgb[mask] = [148, 85, 30]
    result = describe_region(rgb, mask, "sRGB assumed")
    assert result["opacity"] is None
    assert result["apparent_color"]["srgb"] == [148, 85, 30]
    assert result["region"] == pytest.approx([.4, .15, .6, .85])
    assert describe_region(rgb, np.zeros_like(mask), "sRGB assumed")["lightness"] is None


def test_untyped_capacity_needs_matching_evidence():
    capacity = dict(value=25., unit="mL")
    result = dict(capacity=capacity, scale=None)
    assert confirm_untyped_capacity(result, {})["capacity"] is None
    assert confirm_untyped_capacity(result, {"capacity": capacity})["capacity"] == capacity
