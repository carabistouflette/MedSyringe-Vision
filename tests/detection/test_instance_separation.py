import numpy as np
from PIL import Image, ImageDraw

from scripts.experiments.ranking.instance_separation import axis_profile, blocked, fuse


def test_paired_edges_distinguish_a_body_line_from_flat_background():
    blank = Image.new("RGB", (100, 70), "gray")
    body = blank.copy()
    ImageDraw.Draw(body).rectangle((5, 28, 95, 41), fill="white")
    state = {"axis": np.array([1., 0.])}
    assert axis_profile(body, [0., 0., 1., 1.], state)[2] > axis_profile(blank, [0., 0., 1., 1.], state)[2]


def test_fragment_veto_overrides_a_high_distinct_score():
    a, b = (.1, .1, .6, .5), (.23, .13, .7, .53)
    rule = {"method": "axis-identity", "distinct": .65}
    result = {"distinct": .67, "same": .02, "fragment": .29}
    record = {(a, b): {"axis": result}}
    assert blocked(a, b, record, rule)
    assert blocked(a, b, {}, rule)
    result["fragment"] = .04
    assert not blocked(a, b, record, rule)


def test_distinct_override_preserves_incumbent_and_is_scoped_to_overlap():
    a, b, c = (.1, .1, .6, .5), (.23, .13, .7, .53), (.75, .7, .95, .95)
    base = {"frame": [{"bbox": a, "score": .9}]}
    raw = {"frame": [{"bbox": b, "score": .8}, {"bbox": c, "score": .8}]}
    evidence = {"frame": {(a, b): {"axis": {"distinct": .9, "same": 0., "fragment": .03}}}}
    rule = {"method": "axis-identity", "distinct": .75, "floor": .2, "minimum": 1}
    result = fuse(base, raw, evidence, rule)
    assert result["frame"][0] == base["frame"][0]
    assert [p["bbox"] for p in result["frame"]] == [a, b]
