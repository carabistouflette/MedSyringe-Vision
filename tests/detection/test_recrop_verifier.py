import pytest
from PIL import Image, ImageOps

from scripts import recrop_verifier as study
from scripts.recrop_proposal_bank import candidates, windows


def test_crop_windows_are_bounded_nested_and_skip_exact_repeats():
    source = [{"bbox": [.02 * i, .06, .02 * i + .14, .14], "score": 1 - .05 * i} for i in range(12)]
    source[1]["bbox"] = source[0]["bbox"]
    source[3]["bbox"] = [.05, .05, .95, .95]
    views = windows(source, (500, 400))
    assert [x["rank"] for x in views if x["slot"] <= 6] == [1, 3, 5, 6]
    assert [x["slot"] for x in views if x["rank"] > 6] == [7, 8]
    assert len({x["window"] for x in views}) == len(views)
    assert all(0 <= x["window"][0] < x["window"][2] <= 500 for x in views)


def test_repeat_evidence_counts_independent_crop_slots():
    box = [.2, .2, .4, .4]
    proposals = [dict(bbox=box, score=.12, slot=1), dict(bbox=box, score=.10, slot=1),
                 dict(bbox=box, score=.08, slot=4), dict(bbox=box, score=.06, slot=8)]
    frames = dict(seeds=[dict(bbox=box, score=.18)], proposals=proposals)
    short, medium, fallback = (candidates(frames, n)[0] for n in (3, 6, 8))
    assert [p["repeats"] for p in (short, medium, fallback)] == [0, 1, 2]
    assert fallback["first_pass"] == pytest.approx(.18)


def test_fitting_proposals_keep_visual_and_weak_labels_separate(monkeypatch):
    gold = [0., .10, .50, .20]
    part = [.02, .10, .15, .20]
    uncertain = [0., .10, .24, .20]
    lookalike = [.65, .35, .95, .40]
    background = [.66, .65, .95, .90]
    monkeypatch.setattr(study, "candidates", lambda *_: [dict(bbox=b) for b in
                        (gold, part, uncertain, lookalike, background)])
    row = dict(objects=[dict(bbox=gold)], lookalikes=[dict(bbox=lookalike)])
    examples = study.fitting_examples(row, {}, [dict(bbox=part, label="fragment")], (100, 100))
    assert {p["label"] for p in examples} == study.CLASSES
    assert all(p["bbox"] != uncertain for p in examples if p["origin"] == "second-pass-proposal")
    assert any(p["bbox"] == part and p["origin"] == "visual-fragment" for p in examples)
    assert any(p["bbox"] == background and p["origin"] == "second-pass-proposal" for p in examples)


def test_additive_fusion_preserves_existing_detection_and_rejects_duplicate():
    old, new = [.1, .1, .3, .3], [.6, .6, .8, .8]
    incumbent = [dict(bbox=old, score=.9)]
    pool = [dict(bbox=b, score=.08, seed_rank=1, slot=1, repeats=0,
                 scores={"hybrid": .9}, classes={}) for b in (old, new)]
    data = {"image": {"incumbent": incumbent, "pools": {3: pool}}}
    predicted = study.fuse(data, dict(budget=3, model="hybrid", identity="overlap", floor=.75), None)
    assert predicted["image"][:1] == incumbent
    assert [p["bbox"] for p in predicted["image"]] == [old, new]


def test_extremely_thin_crop_stays_resizable_by_appearance_encoder():
    class Encoder:
        def encode(self, images):
            return [ImageOps.pad(patch, (168, 168)).size for patch in images]

    photo = Image.new("RGB", (1600, 300))
    box = (0., .2, 1., .201)
    assert study.embeddings(Encoder(), photo, [box]) == {box: (168, 168)}
