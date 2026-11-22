from scripts.experiments.relocalization.relocalization_selector import eligible, evidence, guard, refine, snapshot, visible_anchors_preserved


def test_neighbor_ownership_and_fragment_safeguards():
    current = [.1, .1, .7, .6]
    candidate = [.18, .2, .57, .42]
    assert eligible(current, [], candidate)
    assert not eligible(current, [candidate], candidate)
    assert not eligible(current, [], [.2, .2, .23, .24])


def test_agreement_uses_the_other_view():
    a = dict(bbox=[0, 0, .4, .3], view="first", score=.8)
    b = dict(bbox=[0, 0, .4, .3], view="crop", score=.3)
    c = dict(bbox=[.75, .75, .9, .9], view="crop", score=.99)
    assert evidence(a, [a, b, c]) == (1., .3)


def test_fit_only_visible_anchors_veto_truncation():
    obj = {"parts": {"flange": [.31, .4], "rod": None, "barrel": [.71, .41]}}
    assert visible_anchors_preserved([.3, .37, .73, .45], obj)
    assert not visible_anchors_preserved([.46, .37, .73, .45], obj)


def test_replacements_are_one_to_one_and_guard_target_identity():
    first = {"bbox": [.08, .1, .4, .34], "score": .92, "id": "first"}
    second = {"bbox": [.60, .5, .8, .7], "score": .89, "id": "second"}
    correct = [.17, .13, .33, .29]
    baseline = {"x": [first, second]}
    rows = [{"id": "x", "objects": [{"bbox": correct}, {"bbox": second["bbox"]}]}]
    bank = {"x": {3: [dict(bbox=correct, instance_slot=i, score=.8, view="crop", scores={"consensus": .9})
                     for i in (0, 1)]}}
    policy = {"arm": "consensus", "budget": 3, "floor": .6, "multiple": True}
    updated = refine(baseline, bank, policy)
    assert [p["id"] for p in updated["x"]] == ["first", "second"]
    assert updated["x"][0]["prior_bbox"] == first["bbox"]
    assert updated["x"][1] == second
    assert guard(rows, baseline, snapshot(rows, baseline), updated)["safe"]
    assert not guard(rows, baseline, snapshot(rows, baseline), {"x": [first, first]})["safe"]
