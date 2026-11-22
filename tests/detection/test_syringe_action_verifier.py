from scripts.syringe_action_diagnostic import classify_actions
from scripts.syringe_action_verifier import apply, audit, context, expected_action
from scripts.native_candidate_verifier import simple_metrics
from medsyringe.evaluate import match


A = [.10, .20, .30, .60]
B = [.33, .21, .48, .55]
WIDE_A = [.05, .15, .34, .65]
FRAGMENT = [.18, .31, .24, .45]
OBJECTS = [dict(bbox=A, parts={"hub": [.12, .22]}), dict(bbox=B)]
RULE = dict(features="visual", add=.8, replace=.8, overlap="strict")


def proposal(box, slot, action, probability=.96):
    scores = dict(add=.02, reject=.02, replace=.02)
    scores[action] = probability
    return dict(bbox=box, slot=slot, view="first", score=.45, decisions={"visual": scores})


def test_fit_supervision_separates_instances_from_fragments_and_duplicate_boxes():
    assert expected_action(A, 0, [WIDE_A], OBJECTS, [0]) == "replace"
    assert expected_action(B, None, [A], OBJECTS, [0]) == "add"
    assert expected_action(A, 0, [A], OBJECTS, [0]) == "reject"
    assert expected_action(FRAGMENT, 0, [A], OBJECTS, [0]) == "reject"
    assert expected_action(A, 0, [WIDE_A], [dict(bbox=A, parts={"hub": [.09, .10]})], [0]) == "reject"


def test_context_has_explicit_absence_and_prefers_the_closest_incumbent():
    assert context(A, [B]) == (None, [0.] * 8)
    slot, geometry = context(A, [B, WIDE_A])
    assert slot == 1
    assert geometry[0] == 1.
    assert geometry[1] > .5
    assert geometry[3] == 1.


def test_a_replacement_preserves_slot_an_addition_appends_once_and_reject_changes_nothing():
    before = {"frame": [dict(bbox=WIDE_A, score=.9, id="keep-id")]}
    ranked = {"frame": [proposal(B, None, "add"), proposal(B, None, "add", .90),
                        proposal(A, 0, "replace"), proposal(FRAGMENT, 0, "reject")]}
    after, logged = apply(before, ranked, RULE)
    assert before["frame"][0]["bbox"] == WIDE_A
    assert after["frame"] == [dict(bbox=A, score=.9, id="keep-id"), dict(bbox=B, score=.45)]
    assert [event["action"] for event in logged["frame"]] == ["replace", "add"]
    rows = [dict(id="frame", objects=OBJECTS)]
    info = classify_actions(rows, before, after, logged)["frame"]
    assert info[0]["after_iou_with_same_target"] == 1.
    assert info[1]["new_target"]


def test_a_replacement_cannot_claim_a_second_incumbents_box():
    before = {"frame": [dict(bbox=WIDE_A, score=.9), dict(bbox=B, score=.8)]}
    after, log = apply(before, {"frame": [proposal(B, 0, "replace")]}, RULE)
    assert after == before
    assert not log


def test_equal_counts_cannot_hide_lost_matches_or_duplicate_detections():
    rows = [dict(id="frame", objects=OBJECTS)]
    baseline = {"frame": [dict(bbox=A, score=.9), dict(bbox=B, score=.8)]}
    worse = {"frame": [dict(bbox=A, score=.9), dict(bbox=FRAGMENT, score=.8)]}
    delta = audit(rows, simple_metrics(rows, baseline), match(rows, baseline, .75)[1], worse, {})
    assert delta["count_regression"] is False
    assert delta["lost_reference"]
    assert delta["per_image_false_positive_regression"]
    assert not delta["safe_at_75"]


def test_strict_localization_gate_rejects_loss_above_counting_iou():
    rows = [dict(id="frame", objects=OBJECTS)]
    baseline = {"frame": [dict(bbox=A, score=.9), dict(bbox=B, score=.8)]}
    worse = {"frame": [dict(bbox=WIDE_A, score=.9), dict(bbox=B, score=.8)]}
    delta = audit(rows, simple_metrics(rows, baseline), match(rows, baseline, .75)[1], worse, {})
    assert delta["safe"]
    assert delta["losses_at_75"] == [["frame", 0]]
    assert not delta["safe_at_75"]
