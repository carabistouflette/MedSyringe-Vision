import numpy as np

from medsyringe.part_geometry import PARTS, geometry


def test_geometry_requires_barrel_flange_rod_order():
    whole = [.10, .30, .90, .70]
    points = np.array([[10, 45], [20, 44], [35, 45], [57, 44], [80, 46]])
    parts = dict(barrel=[dict(bbox=[.15, .38, .48, .54], score=.9)],
                 rod=[dict(bbox=[.68, .43, .87, .50], score=.9)],
                 flange=[dict(bbox=[.52, .37, .61, .56], score=.8)], seal=[])
    features, sources = geometry(whole, parts, points, 100, 100)
    assert features[-1] == .8
    assert sources["seal"] == []
    parts["flange"][0]["bbox"] = [.11, .37, .18, .56]
    features, _ = geometry(whole, parts, points, 100, 100)
    assert features[4] > 0 and features[5] == 0


def test_outside_components_do_not_support_candidate():
    body = [.20, .30, .80, .70]
    parts = {key: [dict(bbox=[.00, .10, .1, .30], score=.99)] for key in PARTS}
    measured, _ = geometry(body, parts, np.zeros((0, 2)), 100, 100)
    assert measured == [0.] * 6
