from scripts.complete_syringe_selection import best_compatible, select_boxes, supervise


def test_joint_selection_can_skip_a_high_ranked_fragment_blocking_two_bodies():
    conflicts = [{1, 2}, {0}, {0}]
    assert best_compatible([.8, .55, .55], conflicts) == (1, 2)
    assert best_compatible([1.2, .55, .55], conflicts) == (0,)
    assert best_compatible([], []) == ()


def test_visible_extent_supervision_distinguishes_truncation():
    full, fragment, far = [.1,.1,.9,.25], [.2,.1,.4,.25], [.1,.7,.9,.8]
    row = {"objects":[{"bbox":full}]}
    bank = [{"bbox":b} for b in (full, fragment, far)]
    labels = {(tuple(b), kind):value for b,value,kind in supervise(row,bank,(100,100),[])}
    assert labels[tuple(full),"detector-complete"]
    assert not labels[tuple(fragment),"detector-fragment"]
    assert not labels[tuple(far),"detector-background"]


def test_joint_keeps_incumbent_and_uses_the_same_candidate_floor():
    old = {"bbox":[.1,.1,.4,.3],"score":.9}
    raw = [{"bbox":[.11,.1,.41,.3],"score":.8,"complete":.98},
           {"bbox":[.6,.6,.8,.9],"score":.5,"complete":.85},
           {"bbox":[.1,.6,.3,.9],"score":.1,"complete":.99}]
    rule = dict(arm="body-joint",minimum=1,floor=.5,detector_floor=.2,method="overlap",distinct=.65)
    actual = select_boxes([old],raw,{},rule)
    assert [p["bbox"] for p in actual] == [old["bbox"],raw[1]["bbox"]]
    assert actual[0]==old
