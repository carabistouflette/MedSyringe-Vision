from scripts.occlusion_relocalization import additions, replacements

POLICY = dict(floor=.40, maximum_area_ratio=.80, containment=.75, peer_floor=None)
ORIGINAL = dict(id="object-1", bbox=[.1, .1, .8, .8], score=.70)
TIGHTER = dict(bbox=[.30, .2, .50, .65], score=.80)
RIVAL = dict(bbox=[.57, .2, .75, .64], score=.72)
OUTSIDE = dict(bbox=[.82, .2, .95, .45], score=.65)


def test_refinement_keeps_instance_identity_and_count():
    revised, = replacements([ORIGINAL], [TIGHTER], [], POLICY)
    assert revised["bbox"] == TIGHTER["bbox"]
    assert revised["prior_bbox"] == ORIGINAL["bbox"]
    assert revised["id"] == ORIGINAL["id"]
    assert revised["score"] == ORIGINAL["score"]


def test_refinement_abstains_when_two_objects_could_explain_one_box():
    assert replacements([ORIGINAL], [TIGHTER, RIVAL], [], POLICY) == [ORIGINAL]
    assert replacements([ORIGINAL, OUTSIDE], [TIGHTER], [], POLICY) == [ORIGINAL, OUTSIDE]


def test_addition_respects_the_old_box_after_replacement():
    refined = replacements([ORIGINAL], [TIGHTER], [], POLICY)
    merged = additions(refined, [ORIGINAL], [RIVAL, OUTSIDE], .4)
    assert [p["bbox"] for p in merged] == [TIGHTER["bbox"], OUTSIDE["bbox"]]
