from PIL import Image
import pytest
import torch

from scripts.experiments.expanded.sam3_evaluate import combine
from scripts.experiments.expanded.sam3_study import crop_predictions, mask_geometry


def test_mask_bounds_come_from_foreground_pixels():
    mask = torch.zeros((8, 10), dtype=torch.bool)
    mask[2:5, 3:9] = True
    box, area = mask_geometry(mask)
    assert box == [.3, .25, .9, .625]
    assert area == pytest.approx(18/80)
    assert mask_geometry(torch.zeros_like(mask)) == (None, 0.)


def test_projection_preserves_mask_geometry_and_rejects_internal_border():
    result = [
        dict(bbox=[.1,.2,.8,.9], score=.8,mask_bbox=[.2,.3,.7,.8],mask_area_fraction=.24),
        dict(bbox=[0.,.2,.8,.9], score=.9,mask_bbox=None,mask_area_fraction=0.)
    ]
    image = Image.new("RGB",(100,100))
    found = list(crop_predictions(lambda _:result,image,[(20,20,80,80)]))
    assert len(found)==1
    assert found[0]["bbox"]==pytest.approx([.26,.32,.68,.74])
    assert found[0]["mask_bbox"]==pytest.approx([.32,.38,.62,.68])
    assert found[0]["mask_area_fraction"]==pytest.approx(.24*.36)


def test_fallback_keeps_prior_detections_by_identity():
    accepted = dict(bbox=[.1,.2,.3,.4],score=.8)
    alternative = dict(bbox=[.2,.3,.4,.5],score=.9)
    baseline = {"accepted":[accepted],"empty":[]}
    frames = {"empty":{"proposals":{"full":[alternative]}}}
    augmented=combine(baseline,frames,.7,"full",.6,None)
    assert augmented["accepted"][0] is accepted
    assert augmented["empty"][0]["bbox"] == alternative["bbox"]
