from scripts.part_association import apply, conflict, evidence, part_labels


def test_part_owner_uses_visible_unambiguous_landmarks():
    objects = [
        dict(bbox=[.1, .2, .5, .4], parts={"rod": [.2, .25]}),
        dict(bbox=[.1, .25, .5, .45], parts={"rod": [.2, .30]}),
        dict(bbox=[.1, .25, .5, .45], parts={"rod": None}),
    ]
    assert part_labels(objects, {"bbox": [.185, .23, .215, .27]}, "rod", (200, 200)) == 0
    assert part_labels(objects, {"bbox": [.185, .26, .215, .29]}, "rod", (200, 200)) is None
    assert part_labels(objects, {"bbox": [.185, .23, .215, .27]}, "seal", (200, 200)) is None


def test_associated_evidence_downweights_high_scoring_part_on_the_wrong_axis():
    a = dict(bbox=[.2, .2, .5, .3], name="barrel", id=0, score=.8)
    b = dict(bbox=[.4, .22, .6, .30], name="rod", id=1, score=.6)
    c = dict(bbox=[.4, .31, .6, .39], name="rod", id=2, score=.9)
    result = evidence([.1, .1, .7, .45], [a, b, c], {(0, 1): .9, (0, 2): .05})
    assert result == dict(unlinked=.8, linked=.54, assigned=[0, 1])


def test_shared_reliable_landmark_blocks_part_reuse_and_distinct_parts_allow_overlap():
    a = dict(bbox=[.1, .1, .6, .7], linked=.6, assigned=[1, 2])
    b = dict(bbox=[.22, .19, .72, .79], linked=.55, assigned=[1, 3])
    c = dict(bbox=b["bbox"], linked=.55, assigned=[4, 5])
    rule = dict(anchor=.4, exclusive=True)
    assert conflict(a, b, rule)
    assert not conflict(a, c, rule)
    c["linked"] = .15
    assert conflict(a, c, rule)


def test_additive_policy_keeps_incumbent_and_blocks_ineligible_duplicates():
    original = {"frame": [dict(bbox=[.1, .1, .5, .3], score=.8, id="incumbent")]}
    original_parts = {"frame": [dict(linked=.4, assigned=[1, 2])]}
    candidates = {"frame": [
        dict(bbox=[.11, .12, .49, .32], score=.9, box=.9, linked=.4, assigned=[1, 2]),
        dict(bbox=[.55, .55, .85, .73], score=.88, box=.94, linked=.4, assigned=[3, 4]),
    ]}
    rule = dict(arm="linked", weight=.25, floor=.75, exclusive=True, anchor=.25)
    result, audit = apply(original, candidates, original_parts, rule)
    assert result["frame"][0] == original["frame"][0]
    assert len(result["frame"]) == 2
    assert audit["frame"][0]["assigned"] == [3, 4]
