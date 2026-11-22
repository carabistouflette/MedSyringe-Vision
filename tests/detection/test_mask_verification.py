from scripts.experiments.crowds.detection_mask_verification import candidates, detect


def test_crowded_proposals_preserve_existing_detections_and_require_support():
    first = {"bbox": [.1, .1, .2, .3], "score": .82}
    second = {"bbox": [.3, .3, .4, .5], "score": .79}
    independent = {"bbox": [.6, .2, .8, .25], "score": .7, "stability": .95,
                   "margin": .02, "elongation": 3., "primary": False, "open_votes": 1}
    unsupported = {**independent, "bbox": [.7, .6, .9, .7], "margin": .1, "open_votes": 0}
    duplicate = {**independent, "bbox": first["bbox"], "margin": .3}
    policy = [.55, -.005, 1., "external", "multiple", .9]
    ranked = candidates({"grids": {"16": [duplicate, unsupported, independent]}}, [first, second])
    assert ranked == [unsupported, independent]
    assert detect([first], ranked, policy) == [first]
    revised = detect([first, second], ranked, policy)
    assert revised[:2] == [first, second] and len(revised) == 3
    assert revised[-1]["bbox"] == independent["bbox"]
