from pathlib import Path
from types import SimpleNamespace

import pytest

from medsyringe.data import read_json
from scripts.experiments.transfer import rfdetr_transfer
from scripts.experiments.transfer.rfdetr_compare import oracle, ranked_region


def test_zenodo_polygon_area_and_class():
    contents = "2 0 0 .5 0 0 .5\n3 0 0 1 0 0 1\n"
    assert list(rfdetr_transfer.polygons(contents, 3)) == [
        {"bbox": [0, 0, 1, 1], "polygon": [0, 0, 1, 0, 0, 1], "area": .5}]
    for malformed in ("3 0 0 1 0 0", "3 0 0 0 0 0 1", "3 0 0 2 0 0 1"):
        with pytest.raises(ValueError):
            list(rfdetr_transfer.polygons(malformed, 3))


def test_coco_export_keeps_masks_only_when_supplied(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(rfdetr_transfer, "source", lambda _: Image.new("RGB", (100, 50)))
    box = [0, 0, 1, 1]
    row = dict(id="one", group="one", objects=[
        dict(bbox=box),
        dict(bbox=box, polygon=[0, 0, 1, 0, 0, 1], area=.5),
    ])
    rfdetr_transfer.export([row], tmp_path)
    first, second = read_json(tmp_path / rfdetr_transfer.ANNOTATIONS)["annotations"]
    assert "segmentation" not in first and first["area"] == 5000
    assert second["segmentation"] == [[0, 0, 100, 0, 0, 50]] and second["area"] == 2500


def test_videos_group_frames_by_clip():
    assert rfdetr_transfer.public_clip(Path("train/images/frame_0005_3_jpg.rf.hash.jpg")) == "0005"
    assert rfdetr_transfer.public_clip("test/images/frame_0005_9_jpg.rf.other.jpg") == "0005"
    with pytest.raises(ValueError):
        rfdetr_transfer.public_clip("test/images/untitled.jpg")


def test_oracle_distinguishes_proposal_from_duplicate_rejection():
    box = [.1, .1, .4, .4]
    misses = [{"id": "x", "box": box}]
    base = {"x": [{"bbox": box, "score": .9}]}
    auxiliary = {"x": [{"bbox": box, "score": .7}]}
    assert oracle(misses, base, auxiliary, .8)[0] == dict(
        id="x", box=box, max_iou=1, max_nonduplicate_iou=0, max_matching_score=None)


def test_rfdetr_predictions_exclude_background_and_normalize():
    from PIL import Image
    result = SimpleNamespace(xyxy=[[0, 0, 12, 12], [-2, 3, 12, 6]],
                             confidence=[.99, .7],
                             data={"class_name": ["__background__", "syringe"]})
    model = SimpleNamespace(predict=lambda *args, **kwargs: result)
    assert list(ranked_region(model, Image.new("RGB", (10, 20)), 12)) == [
        dict(bbox=[0, .25, 1, .5], score=.7)]
