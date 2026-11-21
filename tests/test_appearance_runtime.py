from PIL import Image

from medsyringe.appearance_runtime import RELEASE, AppearanceRuntime
from medsyringe.local_models import WorkerError
from scripts.appearance_worker import explain


def test_verified_sources_are_visible_and_demoted_boxes_stay_reviewable():
    rejected = dict(bbox=[.10,.12,.33,.41],score=.88)
    confirmed = dict(bbox=[.35,.39,.80,.51],score=.93)
    low = dict(bbox=[.80,.15,.95,.26],score=.14)
    original = dict(ranked=[rejected,confirmed,low],floor=.67)
    new = dict(bbox=[.40,.65,.55,.91],score=.67,semantic_margin=.72)
    revised = explain(original,[confirmed,new,low])
    assert revised[0] is confirmed
    assert revised[1]["source"] == "crop appearance"
    assert revised[1]["verification_score"] == .72
    assert revised[2]["score"] < .67
    assert revised[2]["original_score"] == .88
    assert revised[2]["bbox"] == rejected["bbox"]
    assert revised[2]["review_reason"]
    assert revised[3] is low
    assert original["ranked"] == [rejected,confirmed,low]


def test_crop_worker_failure_preserves_the_previous_detector(tmp_path, monkeypatch):
    config = tmp_path / RELEASE
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    ranked = [dict(bbox=[.10,.12,.33,.41],score=.88)]
    verifier = AppearanceRuntime(tmp_path, lambda _: (ranked,.67,"original fallback evidence"))

    def fail(_):
        raise WorkerError("Unavailable worker")

    monkeypatch.setattr(verifier.worker,"run",fail)
    result = verifier.detect(Image.new("RGB",(10,10)))
    assert result == (ranked,.67,"original fallback evidence")
    assert verifier.status == "baseline retained after an unavailable crop verifier"
    assert verifier.worker.options["ranked"] is ranked
