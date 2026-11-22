from threading import Barrier
from types import SimpleNamespace

from PIL import Image

from medsyringe.analysis.inference import Engine
from medsyringe.runtime.local_models import WorkerError


def test_arc_checks_overlap_and_preserve_separate_evidence(monkeypatch):
    engine = Engine.__new__(Engine)
    engine.identity = {"primary_backend": "xpu"}
    engine.overlap_crop_checks = True
    engine.markings = engine.liquid = engine.sam3 = True
    engine.fill_settings = None
    engine.attributes = {}
    gate = Barrier(2, timeout=3)

    def markings(_region):
        gate.wait()
        raise WorkerError("OCR unavailable")

    def liquid(_region):
        gate.wait()
        return {"status": "measured", "lightness": 38.0}

    engine.expanded = SimpleNamespace(read_markings=markings, measure_liquid=liquid)
    monkeypatch.setattr("medsyringe.analysis.inference.describe", lambda *_: {
        "scale": {"value": None}, "color": {"value": None},
    })
    boxes = [{"bbox": [0, 0, 1, 1], "score": .9}]
    obj, = engine.describe_objects(Image.new("RGB", (50, 30)), boxes, ocr=True)

    assert obj["bbox"] == boxes[0]["bbox"]
    assert obj["inscriptions"]["status"] == "unavailable"
    assert obj["liquid_appearance"]["lightness"] == 38.0
    assert obj["scale"] is None


def test_count_only_keeps_the_detector_and_skips_every_crop_worker():
    engine = Engine.__new__(Engine)
    engine.identity = {"primary_backend": "xpu"}
    engine.markings = engine.liquid = engine.sam3 = True
    engine.verifier = None
    engine.fill_settings = True
    box = [0, 0, 1, 1]
    engine.expanded = SimpleNamespace(detect=lambda _image: (
        [{"bbox": box, "score": .9}, {"bbox": box, "score": .4}], .6, None
    ))
    report = engine.analyze(Image.new("RGB", (50, 30)), ocr=True, count_only=True)
    assert report["accepted_count"] == 1
    assert report["analysis_mode"] == "count only"
    assert len(report["below_threshold_candidates"]) == 1
    obj, = report["objects"]
    assert obj["bbox"] == box and obj["score"] == .9
    assert obj["scale"] is None and obj["liquid_appearance"] is None
    assert obj["inscriptions"]["status"] == "not requested"
    assert obj["fill_level"]["reason"] == "Only detection and counting were requested."
