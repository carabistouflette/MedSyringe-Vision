from scripts.experiments.relocalization.pair_relocalization import check, owner, refine_image, targets
from scripts.experiments.relocalization.relocalization_selector import snapshot


A = [.12, .21, .55, .40]
B = [.38, .39, .79, .57]
A_FRAGMENT = [.13, .23, .32, .38]
POLICY = dict(arm="factorized", floor=.85, link=.80)


def prediction(bbox, name):
    return dict(bbox=bbox, score=.90, id=name)


def offer(slot, proposal, bbox, same=.96, improvement=.92):
    return dict(slot=slot, proposal=proposal, bbox=bbox,
                scores=dict(same=same, improvement=improvement))


def test_fit_distinguishes_a_better_box_a_fragment_and_a_neighbor():
    objects = [dict(bbox=A), dict(bbox=B)]
    a, fragment, b = (owner(box, objects) for box in (A, A_FRAGMENT, B))
    assert targets(A, objects, fragment, a) == (True, True)
    assert targets(A_FRAGMENT, objects, a, fragment) == (True, False)
    assert targets(B, objects, a, b) == (False, None)


def test_replacement_preserves_existing_identity_score_and_number():
    current = [prediction(A_FRAGMENT, "syringe-a"), prediction(B, "syringe-b")]
    after = refine_image(current, [offer(0, 0, A)], POLICY)
    assert len(after) == len(current) and after[1] == current[1]
    assert after[0]["bbox"] == A and after[0]["prior_bbox"] == A_FRAGMENT
    assert after[0]["score"] == current[0]["score"] and after[0]["id"] == "syringe-a"
    assert current[0]["bbox"] == A_FRAGMENT


def test_conflicting_slot_claims_trigger_abstention():
    current = [prediction(A_FRAGMENT, "first"), prediction(B, "second")]
    tied = [offer(0, 0, A, same=.95), offer(1, 0, A, same=.93)]
    assert refine_image(current, tied, POLICY) == current


def test_a_small_regression_is_detected_even_if_iou_matches_and_count_survive():
    row = dict(id="sample", objects=[dict(bbox=A)])
    base = {"sample": [prediction(A, "syringe-a")]}
    moved = {"sample": [prediction([.12,.21,.50,.40], "syringe-a")]}
    result = check([row], base, snapshot([row], base), moved)
    assert result["count_preserved"] and not any(result["lost"].values())
    assert not result["passes"] and result["harmful_changes"][0]["reason"] == "reduced-localization"
