from itertools import pairwise

import numpy as np

from medsyringe.detection.additive import duplicates
from scripts.experiments.identity.syringe_assembly_model import fragment_views, pair_is_near, training_weights
from scripts.experiments.identity.syringe_assembly_study import blocked, fuse
from scripts.experiments.identity.thin_syringe_tiles import WINDOWS, windows


def test_strips_preserve_the_long_axis_and_cover_the_short_axis():
    size = (1001, 703)
    view = windows(size, WINDOWS["horizontal"])
    assert len(view) == 5
    assert all(l == 0 and r == size[0] for l, _, r, _ in view)
    assert view[0][1] == 0 and view[-1][3] == size[1]
    assert all(a[3] >= b[1] for a, b in pairwise(view))


def test_training_fragments_cover_only_parts_of_an_annotated_assembly():
    obj = {"bbox": [.3, .1, .4, .8], "parts": {"barrel": [.35, .31], "rod": [.35, .72]}}
    found = fragment_views(obj, (1200, 800))
    assert len(found) == 4
    assert all(.3 <= box[0] < box[2] <= .4 for box in found)
    assert any(p[1] >= .58 and p[3] >= .78 for p in found)


def test_weak_overlap_can_be_vetoed_if_the_model_sees_a_fragment():
    complete = {"bbox": [.4, .2, .5, .7], "score": .7}
    fragment = {"bbox": [.42, .65, .48, .83], "score": .78}
    a, b = tuple(complete["bbox"]), tuple(fragment["bbox"])
    assert not duplicates(a, b) and pair_is_near(a, b, (1000, 1000))
    rule = dict(method="pixel-veto", floor=.6, veto=.7)
    pair = dict(pixels=dict(same=.08, fragment=.82, distinct=.08, distractor=.02))
    assert not blocked(complete, fragment, {**rule, "method": "overlap"}, {})
    assert fuse([complete], [fragment], rule, {(a, b): pair}) == [complete]


def test_distinct_neighbor_can_be_added_without_relaxing_near_identical_boxes():
    original = {"bbox": [.4, .2, .52, .7], "score": .7}
    peer = {"bbox": [.45, .2, .57, .7], "score": .74}
    rule = dict(method="pixel-identity", floor=.6, veto=.7)
    pair = dict(pixels=dict(same=.02, fragment=.02, distinct=.95, distractor=.01))
    a, b = tuple(original["bbox"]), tuple(peer["bbox"])
    assert duplicates(a, b)
    assert not blocked(original, peer, rule, {(a, b): pair})
    assert blocked(original, original, rule, {(a, a): pair})


def test_pair_weights_balance_classes_and_repeated_acquisitions():
    weights = training_weights(["same", "same", "same", "distinct"], ["burst", "burst", "one", "two"])
    assert np.isclose(weights[:2].sum(), weights[2])
    assert np.isclose(weights[:3].sum(), weights[3])
