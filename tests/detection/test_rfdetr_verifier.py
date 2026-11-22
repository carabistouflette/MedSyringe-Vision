import numpy as np

from scripts.experiments.transfer.rfdetr_verifier import apply, expanded, extras


def test_verification_only_filters_additions():
    initial = dict(bbox=[.0, .1, .2, .3], score=.9)
    review = dict(bbox=[.0, .1, .2, .3], score=.3)
    weak = dict(bbox=[.2, .3, .4, .5], score=.7, source="auxiliary")
    strong = dict(bbox=[.5, .5, .7, .7], score=.7, source="auxiliary")
    before = {"frame": [initial, weak, strong, review], "other": [initial]}
    found = extras(before, .7)
    assert found == {"frame": [weak, strong], "other": []}
    kept = apply(before, {"frame": np.array([[-1.0], [1.0]])},
                 {"weights": [1.0], "bias": 0.0, "cutoff": .5})
    assert kept == {"frame": [initial, strong, review], "other": [initial]}
    assert kept["frame"][0] is initial


def test_context_padding_clips_to_image():
    assert np.allclose(expanded([.1, .2, .9, 1.], .2), [0, .04, 1, 1])


def test_two_heads_must_both_accept_an_addition():
    from scripts.experiments.transfer.rfdetr_verifier_refine import kept_by

    base = dict(bbox=[.0, .1, .2, .3], score=.9)
    first = dict(bbox=[.2, .3, .4, .5], score=.7, source="auxiliary")
    second = dict(bbox=[.5, .6, .7, .8], score=.7, source="auxiliary")
    sample = {"hybrid": {"x": [base, first, second]}}
    heads = [dict(encoder=kind, padding=0, weights=[1.], bias=0., cutoff=0.)
             for kind in ("mobile", "dino")]
    features = {("mobile", 0): {"x": np.array([[1.], [1.]])},
                ("dino", 0): {"x": np.array([[-1.], [1.]])}}
    assert kept_by(heads, sample, features) == {"x": [base, second]}


def test_runtime_adds_only_verified_boxes_and_preserves_base():
    from PIL import Image
    from medsyringe.detection.auxiliary_detector import VerifiedRecovery
    from types import SimpleNamespace

    runner = object.__new__(VerifiedRecovery)
    runner.options = dict(support=.4, aux_cutoff=.25, iou=.3)
    runner.head = dict(weights=[1.], bias=0., cutoff=.5)
    runner.encoder = SimpleNamespace(encode=lambda _: np.array([[1.], [-1.]]))
    first = dict(bbox=[.2, .2, .4, .4], score=.75)
    second = dict(bbox=[.5, .5, .75, .75], score=.65)
    runner.propose = lambda _: [first, second]
    initial = dict(bbox=[.05, .05, .15, .15], score=.9)
    weaker = [dict(pred, score=.5) for pred in (first,second)]
    image = Image.new("RGB", (80, 80))
    raw = [initial, *weaker]
    kept = runner.apply(image, raw, .7)
    assert kept[0] is initial and kept[2:] == weaker
    assert kept[1]["bbox"] == first["bbox"]
    assert kept[1]["auxiliary_score"] == first["score"]
    assert kept[1]["verification_score"] == 1.
    assert runner.apply(image, [initial], .7) == [initial]
