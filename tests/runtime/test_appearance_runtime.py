import base64
from io import BytesIO

from PIL import Image

from medsyringe.runtime.appearance_runtime import RELEASE, AppearanceRuntime
from medsyringe.measurement.attributes import crop
from medsyringe.runtime.local_models import WorkerError, decode_picture
from scripts.workers.appearance_text_encoder import pack_patches, prepare_patch, unpack_patches
from scripts.workers.appearance_worker import explain
from scripts.experiments.detector_audits.detection_crop_semantics import PADDINGS, expanded


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


def test_packed_crops_preserve_encoder_pixels_and_order():
    image = Image.effect_noise((800, 600), 30).convert("RGB")
    candidates = [{"bbox": [.02, .28, .6, .39]}, {"bbox": [.8, .03, .94, .54]}]
    packed = pack_patches(image, candidates)
    wire = BytesIO()
    packed.save(wire, format="PNG", compress_level=1)
    decoded = decode_picture(base64.b64encode(wire.getvalue()))
    expected = [prepare_patch(crop(image, expanded(p["bbox"], padding)))
                for padding in PADDINGS for p in candidates]
    actual = unpack_patches(decoded, candidates)
    assert [tile.tobytes() for tile in actual] == [tile.tobytes() for tile in expected]
    assert pack_patches(image.resize((224, 224)), candidates) is None
