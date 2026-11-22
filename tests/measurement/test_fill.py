import numpy as np
from PIL import Image, ImageOps

from medsyringe.measurement.fill import estimate
from medsyringe.measurement.fill_orientation import upright_liquid_for_fill


def inputs(turn=False):
    pixels = np.full((345, 120, 3), 206, dtype=np.uint8)
    pixels[70:200, 22:76] = [205, 132, 67]
    pixels[200:222, 22:76] = [25, 22, 21]
    image = Image.fromarray(pixels)

    def reflect(rect):
        l, t, r, b = rect
        return [1-r, 1-b, 1-l, 1-t] if turn else rect

    candidates = []
    for value in (5, 10, 15, 20):
        row = 70+10*value
        candidates.append(dict(text=str(value), rotation=180 if turn else 0, family="raw", score=.92,
                               crop_bbox=reflect([.58, (row-3)/345, .75, (row+3)/345])))
    inscriptions = dict(capacity=dict(value=20., unit="mL"), candidates=candidates)
    liquid = dict(lightness={"lstar": 56.}, region=reflect([.22, 75/345, .60, 196/345]))
    return image.transpose(Image.Transpose.ROTATE_180) if turn else image, inscriptions, liquid


SETTINGS = dict(ocr_min=.6, min_drop=8., continuity=.8, tick_check=False)


def test_fill_uses_the_front_of_the_piston_in_both_directions():
    first = estimate(*inputs(), SETTINGS)
    turned = estimate(*inputs(turn=True), SETTINGS)
    for answer in (first, turned):
        assert answer["ml"] == 13
        assert answer["percent"] == 65
        assert answer["interval_ml"][0] <= 13 <= answer["interval_ml"][1]
        assert len(answer["graduations"]) == 4
    assert abs(first["piston_line"][0][1] + turned["piston_line"][0][1] - 1) < .02


def test_abstain_without_numbered_scale_or_visible_liquid():
    image, markings, liquid = inputs()
    no_labeled_scale = {**markings, "candidates": markings["candidates"][:2]}
    assert "ml" not in estimate(image, no_labeled_scale, liquid, SETTINGS)
    assert "ml" not in estimate(image, markings, None, SETTINGS)


def test_two_numbers_require_visible_periodic_ticks():
    image, markings, liquid = inputs()
    pair = {**markings, "candidates": markings["candidates"][1::2]}
    answer = estimate(image, pair, liquid, {**SETTINGS, "tick_check": True})
    assert answer["status"] == "insufficient evidence"
    assert "graduation pattern" in answer["reason"]


def test_abstain_when_the_colored_liquid_column_has_a_large_gap():
    image, markings, liquid = inputs()
    pixels = np.array(image)
    pixels[100:155] = 206
    assert "ml" not in estimate(Image.fromarray(pixels), markings, liquid, SETTINGS)


def test_orientation_retry_maps_liquid_evidence_back_to_the_input():
    original, _, liquid = inputs()
    image, markings, expected = inputs(turn=True)

    def measure(upright):
        assert np.array_equal(np.asarray(upright), np.asarray(ImageOps.contain(original, (420, 420))))
        return liquid

    assert upright_liquid_for_fill(image, markings, SETTINGS, measure)["region"] == expected["region"]
