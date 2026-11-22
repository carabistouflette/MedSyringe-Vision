from scripts.syringe_box_quality import order_boxes, supervise


RULE = dict(conflicts="learned", distinct=.70)
FULL = [.20, .20, .60, .40]
FRAGMENT = [.20, .20, .39, .39]
NEIGHBOR = [.50, .60, .83, .80]


def proposal(box, rank=.90, iou=.75, identity=.90):
    return {"bbox": box, "score": .6, "rank": rank, "pair-iou": iou, "identity": identity}


def test_quality_training_pairs_a_fragment_with_its_own_syringe():
    row = dict(objects=[dict(bbox=FULL), dict(bbox=NEIGHBOR)])
    crops = [proposal(b) for b in (FULL, FRAGMENT, NEIGHBOR)]
    labels, pairs = supervise(row, crops)
    assert [name for _, name, _, _ in labels] == ["complete", "fragment", "complete"]
    assert [(good, bad) for good, bad, _ in pairs] == [(0, 1)]


def test_box_quality_reorders_competing_candidates_at_the_same_presence_cutoff():
    crops = [proposal(FRAGMENT, rank=.97, iou=.40), proposal(FULL, rank=.85, iou=.88)]
    relation = {(0, 1): .02}
    assert order_boxes([], crops, relation, "coupled", RULE, .80)[0]["bbox"] == FRAGMENT
    assert order_boxes([], crops, relation, "pair-iou", RULE, .80)[0]["bbox"] == FULL


def test_high_localization_alone_does_not_add_an_unsupported_syringe():
    crops = [proposal(FULL, identity=.4, iou=.99)]
    assert order_boxes([], crops, {}, "identity-pair-iou", RULE, .75) == []
    assert order_boxes([], crops, {}, "shared-identity-pair-iou", RULE, .75) == []


def test_sharing_requires_a_direct_match_and_cannot_duplicate_an_incumbent():
    inner = [.21, .21, .43, .39]
    crops = [proposal(inner, identity=.90, iou=.52), proposal(FULL, identity=.4, iou=.91)]
    relation = {(0, 1): .03}
    assert order_boxes([], crops, relation, "shared-identity-pair-iou", RULE, .75)[0]["bbox"] == FULL
    assert order_boxes([], crops, {(0, 1): .8}, "shared-identity-pair-iou", RULE, .75)[0]["bbox"] == inner
    assert order_boxes([dict(bbox=FULL)], crops, {(0, 1): .03, (1, 2): .03},
                       "shared-identity-pair-iou", RULE, .75) == []
