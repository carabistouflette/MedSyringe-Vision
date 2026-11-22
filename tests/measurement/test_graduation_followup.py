import numpy as np

from medsyringe.measurement.graduation_ocr import supplement_graduations
from medsyringe.measurement.piston_landmarks import lumen_support, refine_rim


POLICY = dict(readers=["tiny","medium"], confidence=.93, stroke=4.)


def mark(text, y, score=.99, **metadata):
    return dict(text=text, score=score, family="stroke-20-0", rotation=0,
                view_bbox=[70,y-8,110,y+8], crop_bbox=[.35,(y-8)/400,.55,(y+8)/400], **metadata)


def test_actual_unit_and_number_are_required_and_preserve_existing_capacity():
    anchors = [mark(str(value), 65 + value*8) for value in (5,10,15)]
    prior = dict(capacity=None, scale=None, candidates=anchors, status="markings only")
    terminal = mark("20 mL", 225, reader="tiny", stroke_strength=12., window="pair")
    observation = dict(expected_template=20, anchors=anchors, reads=[terminal])
    accepted = supplement_graduations(prior, [observation], POLICY)
    assert accepted["capacity"]["value"] == 20
    assert accepted["scale"]["value"] == "20-scale"
    assert accepted["graduation_crop_evidence"] == [terminal]
    conflict = {**prior, "capacity": {"value": 60, "unit": "mL"}}
    assert supplement_graduations(conflict, [observation], POLICY) == conflict
    weak = {**observation, "reads": [{**terminal, "score": .5}]}
    assert supplement_graduations(prior, [weak], POLICY) == prior


def test_piston_rim_does_not_require_warm_color_and_lumen_ends_before_rim():
    lab = np.zeros((200,80,3), dtype=np.float32)
    lab[...,0] = 70.
    lab[98:111,:,0] = 27.
    mask = np.zeros(lab.shape[:2], dtype=np.uint8)
    mask[15:175,22:58] = 1
    frame = dict(lab=lab, mask=mask)
    local = dict(shoulder=np.array([39.,20.]), piston=np.array([39.,100.]),
                 flange=np.array([39.,166.]))
    actual = refine_rim(frame, local, 11.)
    assert actual is not None and abs(actual["piston"][1]-98) < 4
    support = lumen_support(frame, actual)
    assert support.any() and not support[95:].any()
    short = {**actual, "piston":np.array([39.,24.])}
    assert not lumen_support(frame, short).any()
