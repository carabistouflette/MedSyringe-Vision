import pytest

from scripts.experiments.ranking.complete_syringe_selection import predict
from scripts.experiments.ranking.completeness_calibration import scored


def test_pure_ranking_changes_the_preference_without_changing_the_detector_prefilter():
    whole = {"bbox":[.1,.1,.8,.5], "score":.352,"complete":.351}
    fragment = {"bbox":[.2,.15,.75,.48], "score":.402,"complete":.342}
    too_weak = {"bbox":[.1,.65,.8,.85], "score":.11,"complete":.99}
    raw = {"x":[whole,fragment,too_weak]}
    variants = scored(raw)
    base, pairs = {"x":[]}, {"x":{}}
    rule = dict(arm="body-greedy",method="overlap",minimum=0,detector_floor=.12,floor=.25,distinct=.65)
    pure = predict(base,variants["pure"],pairs,rule)["x"]
    mix = predict(base,variants["blended"],pairs,rule)["x"]
    assert [p["bbox"] for p in pure] == [whole["bbox"]]
    assert [p["bbox"] for p in mix] == [fragment["bbox"]]
    assert pure[0]["detector_score"] == pytest.approx(.352)
    assert pure[0]["original_complete"] == pytest.approx(.351)
    assert whole["complete"] == .351
