import numpy as np

from scripts.native_candidate_verifier import (
    acquisition_weights, compare, examples, fuse, identity_scores, reference_index, simple_metrics,
)


def candidate(bbox, score=.8):
    return dict(bbox=bbox, score=score, rf=score, decisions={"tight": score},
                tight=[1., 0.], context=[1., 0.])


def test_proposal_assignment_abstains_between_two_close_instances():
    proposal = [.15, .21, .55, .44]
    objects = [{"bbox": [.1, .2, .5, .45]}, {"bbox": [.2, .2, .6, .45]}]
    assert reference_index(proposal, objects) is None
    row = {"objects": objects, "lookalikes": [{"bbox": [.7, .2, .9, .45]}]}
    observed = examples(row, [candidate(proposal)])
    assert [state["origin"] for state in observed] == [
        "reviewed-syringe", "reviewed-syringe", "reviewed-lookalike",
    ]


def test_balanced_weights_do_not_amplify_a_burst():
    weights = acquisition_weights(["burst", "burst", "single", "negative"], [True, True, True, False])
    assert np.isclose(weights[:2].sum(), weights[2])
    assert np.isclose(weights[:3].sum(), weights[3])


def test_identity_veto_keeps_near_identical_boxes_and_can_allow_neighbors():
    existing = candidate([.1, .2, .5, .45])
    identical = candidate(existing["bbox"])
    nearby = candidate([.25, .2, .65, .45])
    frame = {"incumbent": [existing], "proposals": [identical, nearby]}
    state = {"weight": [0.] * 7, "bias": 10., "scale": None}
    scores = identity_scores({"sample": frame}, state)["sample"]
    assert (0, 1) not in scores
    rule = {"appearance": "tight", "floor": .5, "distinct": .9}
    result = fuse(frame, rule, scores)
    assert [p["bbox"] for p in result] == [existing["bbox"], nearby["bbox"]]
    assert len(fuse(frame, {**rule, "distinct": None}, scores)) == 1


def test_admission_guard_protects_counts_on_individual_images():
    true = candidate([.2, .2, .4, .4])
    wrong = candidate([.6, .6, .7, .7])
    rows = [{"id": "exact", "objects": [true]}, {"id": "miss", "objects": [true]}]
    base = simple_metrics(rows, {"exact": [wrong], "miss": []})
    outcome = compare(rows, base, {"exact": [wrong, true], "miss": [true]})
    assert not outcome["safe"]
    assert outcome["count_regression"]
    assert not outcome["per_image_false_positive_regression"]
