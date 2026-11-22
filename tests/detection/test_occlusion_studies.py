import pytest
import numpy as np

from scripts.occlusion_detector import crop_labels, fuse
from scripts.occlusion_proposals import oracle
from scripts.occlusion_sam_tiling import handoff
from scripts.occlusion_seal import convert_to_volume


def proposal(box, score=.7, **fields):
    return dict(bbox=box, score=score, **fields)


def test_oracle_cannot_double_count_a_shared_proposal():
    left = proposal([.1, .2, .5, .5])
    right = proposal([.3, .2, .7, .5])
    shared = proposal([.1, .2, .7, .5])
    assert oracle([left, right], [shared])["matched"] == 1
    assert oracle([left, right], [shared, left])["matched"] == 2


def test_crop_does_not_drop_an_intersected_target():
    window = [.25, .25, .75, .75]
    inside = proposal([.375, .375, .625, .625])
    overlap = proposal([.125, .125, .375, .375])
    outside = proposal([.01, .01, .24, .24])
    assert crop_labels([inside, outside], window) == [{"bbox": [.25, .25, .75, .75]}]
    assert crop_labels([inside, overlap], window) is None


def test_fusion_preserves_existing_and_rejects_duplicate_tiles():
    existing = proposal([.10, .25, .35, .4])
    distinct = proposal([.55, .45, .75, .65])
    combined = fuse([existing], [existing, distinct, distinct], .5)
    assert [p["bbox"] for p in combined] == [existing["bbox"], distinct["bbox"]]
    assert combined[0] == existing


def test_tile_agreement_requires_distinct_view_support():
    alone = proposal([.55, .45, .75, .65], window=[0, 0, 3, 3])
    different_tile = proposal(alone["bbox"], window=[0, 1, 3, 4])
    assert handoff({"full": [], "tiles": [alone, alone]}, [], "agreement") == []
    assert len(handoff({"full": [], "tiles": [alone, different_tile]}, [], "agreement")) == 1


def test_volume_requires_rim_and_scale_and_will_not_extrapolate():
    frame = dict(front=np.array([0, 0]), axis=np.array([1, 0]))
    row = {"marked_scale": [{"ml": n, "point": [20*n, 5]} for n in (1, 2, 3)]}
    known = {"rim_status": "estimated", "location": 40}
    volume = convert_to_volume(row, frame, known, 2)
    assert (volume["low_ml"], volume["high_ml"]) == pytest.approx((1.9, 2.1))
    assert convert_to_volume(row, frame, {**known, "location": 19}, 2)["status"] == "unknown"
    assert convert_to_volume({"marked_scale": None}, frame, known, 2)["status"] == "unknown"
    assert convert_to_volume(row, None, {"rim_status": "unknown"}, None)["status"] == "unknown"
