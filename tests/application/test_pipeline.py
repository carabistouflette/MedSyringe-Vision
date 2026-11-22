import io
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image
from pytest import approx
from pillow_heif import from_pillow

from medsyringe.app import create_app
from medsyringe.data import coco, dhash, digest, leakage, read_image, valid_box, write_json
from medsyringe.detectors import box_iou, suppress
from medsyringe.evaluate import best_threshold, evaluate
from medsyringe.scene import compare, inventory
from medsyringe.refinement import fuse, restore_box

BOX = [0.2, 0.2, 0.4, 0.8]


def test_lossless_orientation_normalization(tmp_path):
    source = tmp_path / "source.jpg"
    original = Image.new("RGB", (25, 40), "#ce7925")
    exif = original.getexif()
    exif[274] = 6
    original.save(source, exif=exif)
    data = source.read_bytes()
    image = read_image(source)
    assert image.size == (40, 25)
    assert source.read_bytes() == data
    heif = tmp_path / "portrait.heic"
    from_pillow(image).save(heif, quality=100)
    decoded = read_image(heif)
    assert decoded.size == (40, 25)
    converted = tmp_path / "portrait.png"
    decoded.save(converted)
    assert read_image(converted).tobytes() == decoded.tobytes()


def test_leakage_contains_all_evidence():
    origin = dict(id="a", group="day-one", split="train", sha256="aaa", dhash="0000")
    related = dict(id="b", group="day-one", split="test", sha256="aaa", dhash="0003")
    assert leakage([origin, related]) == [dict(images=["a", "b"], evidence=["group", "sha256", "dhash"])]
    assert not leakage([origin, {**related, "split": "train"}])
    assert not leakage([origin, dict(id="c", group="day-two", split="test", sha256="ccc", dhash="ffff")])


def test_coco_coordinates_and_box_validation():
    rows = [dict(id="a", image="a.png", width=100, height=200,
                 objects=[dict(bbox=BOX, color="orange")])]
    assert coco(rows)["annotations"][0]["bbox"] == [20, 40, 20, 120.00000000000001]
    assert not valid_box([0.3, 0.2, 0.1, 0.4])
    assert not valid_box([float("nan"), 0.2, 0.8, 0.9])
    assert box_iou(BOX, BOX) == 1
    assert dhash(Image.new("RGB", (20, 20))) == "0000000000000000"


def test_match_counts_duplicate_and_empty():
    rows = [dict(id="positive", objects=[dict(bbox=BOX)]), dict(id="negative", objects=[])]
    predicted = {
        "positive": [dict(bbox=BOX, score=0.9), dict(bbox=BOX, score=0.7)],
        "negative": [dict(bbox=BOX, score=0.4)],
    }
    assert best_threshold(rows, predicted) == 0.9
    result = evaluate(rows, predicted, 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert (result["ap50"], result["ap50_95"], result["correct_empty"]) == (1, 1, 1)
    assert suppress(predicted["positive"]) == [predicted["positive"][0]]


def test_inventory_unknown_and_observation_identity():
    objects = [dict(id="a", scale="20-scale"), dict(id="b", scale=None)]
    result = inventory(objects, {"20-scale": 0, "60-scale": 2}, "scale")
    assert result["extra"]["20-scale"] == 1
    assert result["minimum_missing"] == 1
    assert result["unresolved"] == ["b"]
    before = dict(id="before", revision=0, objects=[])
    after = dict(id="after", revision=1, objects=objects)
    assert compare(before, after)["count_change"] == 2
    assert compare(before, after)["identity_status"] == "unresolved"


def test_annotator_corrections_conflict_export(tmp_path):
    images = tmp_path / "data/original"
    images.mkdir(parents=True)
    source = images / "sample.png"
    Image.new("RGB", (8, 12)).save(source)
    inventory = [dict(id="sample", image="original/sample.png", width=8, height=12,
                      split="train", source="sample.heic", sha256="external",
                      derived_sha256=digest(source))]
    draft = {"sample": dict(status="draft", flags=[], provenance="test fixture",
                           objects=[dict(bbox=[0.1, 0.1, 0.5, 0.9], color="orange")])}
    write_json(tmp_path / "artifacts/source_inventory.json", inventory)
    write_json(tmp_path / "artifacts/annotations.json", draft)
    client = TestClient(create_app(tmp_path))
    assert client.get("/").status_code == 200
    assert len(client.get("/api/library").json()) == 1
    assert client.post("/api/observations", files={"file": ("file.txt", b"txt")}).status_code == 422
    assert client.get("/api/library-coco").json()["images"] == []
    item = client.post("/api/library/sample").json()
    endpoint = f"/api/observations/{item['id']}"
    assert client.get(endpoint + "/image").headers["content-type"] == "image/jpeg"
    review = dict(reviewer="Reviewer", note="Revised the box", revision=0,
                  status="approved", objects=[dict(id="object-1", bbox=BOX, color="amber-brown")])
    result = client.post(endpoint + "/review", json=review).json()
    assert result["revision"] == 1
    assert result["original"]["objects"] == item["objects"]
    assert result["objects"][0]["bbox"] == BOX
    assert result["objects"][0]["origin"] == "human"
    assert client.post(endpoint + "/review", json=review).status_code == 409
    assert len(client.get("/api/library-coco").json()["images"]) == 1
    expected = {"field": "count", "counts": {"syringe": 2}}
    assert client.post(endpoint + "/inventory", json=expected).json()["missing"] == {"syringe": 1}

    archive = client.post(endpoint + "/export", json={"revision": 1, "expected": expected})
    assert archive.headers["content-type"] == "application/zip"
    with ZipFile(io.BytesIO(archive.content)) as zip_file:
        assert set(zip_file.namelist()) == {"report.json", "observation.jpg"}
        exported = json.loads(zip_file.read("report.json"))
        assert exported["inventory_check"]["missing"] == {"syringe": 1}
        assert exported["original"]["objects"] == item["objects"]
    assert client.post(endpoint + "/export", json={"revision": 0}).status_code == 409


def test_detection_view_boxes_return_to_original_coordinates():
    assert box_iou(restore_box([0.2, 0.6, 0.8, 0.8], "quarter_turn"), BOX) == approx(1)
    assert box_iou(restore_box([0.2, 0.2, 0.8, 0.4], "opposite_turn"), BOX) == approx(1)
    assert box_iou(restore_box([0.6, 0.2, 0.8, 0.8], "mirror"), BOX) == approx(1)


def test_consensus_strengthens_matching_boxes_and_retains_separate_instances():
    second = [0.65, 0.2, 0.75, 0.8]
    results = fuse({"original": [{"bbox": BOX, "score": 0.46}, {"bbox": second, "score": 0.4}],
                    "mirror": [{"bbox": BOX, "score": 0.45}]},
                   strategy="consensus", overlap=0.6)
    assert len(results) == 2
    assert results[0]["score"] > 0.46 and results[0]["support"] == 2
    assert results[1]["score"] == 0.4 and results[1]["support"] == 1


def test_learned_geometry_head_scores_proposals_without_losing_box_evidence():
    from medsyringe.proposal_learning import adjust_scores, describe_proposals, fit_variant, labels_for

    image = Image.new("RGB", (48, 32), "white")
    candidates = [{"bbox": BOX, "score": .8, "support": 2},
                  {"bbox": [.6, .2, .9, .7], "score": .2}]
    geometry, texture = describe_proposals(image, candidates)
    labels = labels_for(candidates, [BOX])
    assert (geometry.shape, texture.shape, labels.tolist()) == ((2, 3), (2, 37), [True, False])
    original = dict(geometry=geometry, texture=texture, labels=labels, burst="one", augmented=False)
    trained = fit_variant("logistic-geometry", [original, {**original, "augmented": True}])
    assert trained["metrics"]["training_proposals"] == 2
    rescored = adjust_scores(candidates, (geometry, texture), [trained], .5)
    assert rescored[0]["score"] > rescored[1]["score"]
    assert (rescored[0]["bbox"], rescored[0]["support"]) == (BOX, 2)


def test_crop_verification_routes_rejected_proposals_to_review():
    from medsyringe.verification import filter_proposals

    candidates = [{"bbox": BOX, "score": 0.9}, {"bbox": BOX, "score": 0.7}]
    kept, review = filter_proposals(candidates, [-0.4, 0.5], 0.0)
    assert kept == [{**candidates[1], "verification_score": 0.5}]
    assert review == [{**candidates[0], "verification_score": -0.4}]
    assert filter_proposals([], [], 0.0) == ([], [])


def test_ssd_hard_negatives_use_only_empty_scenes():
    import torch
    from medsyringe.supervised import empty_scene_loss

    logits = torch.tensor([[[0.0, 4.0], [4.0, 0.0]], [[0.0, 5.0], [0.0, 2.0]]],
                          requires_grad=True)
    loss = empty_scene_loss(logits, [{"labels": [1]}, {"labels": []}], top=1)
    assert loss.item() == approx(torch.nn.functional.softplus(torch.tensor(5.)).item())
    loss.backward()
    assert logits.grad[0].abs().sum().item() == 0
    assert logits.grad[1, 0, 1].item() > 0
    assert logits.grad[1, 1].abs().sum().item() == 0


def test_localization_offsets_round_trip():
    import numpy as np
    from medsyringe.localization import correct_boxes, relative_offsets

    boxes = np.array([[.20, .20, .50, .80]])
    targets = np.array([[.16, .18, .54, .82]])
    assert correct_boxes(boxes, relative_offsets(boxes, targets), 1) == approx(targets)


def test_cross_model_fusion_only_supports_overlapping_proposals():
    from scripts.compare_supervised import combine

    others = [.60, .20, .90, .70]
    current = [{"bbox": BOX, "score": .78}, {"bbox": others, "score": .77}]
    direct = [{"bbox": BOX, "score": .95}]
    fused = combine(current, direct, weight=.2, box_weight=0, min_overlap=.35)
    assert fused[0]["bbox"] == BOX
    assert fused[0]["score"] == approx(.814)
    assert fused[1]["score"] == approx(.616)


def test_letterbox_mapping_ignores_padding_and_preserves_original_box():
    from scripts.scale_study import inverse

    mapping = {"frame": (0, 80, 320, 160)}
    predictions = {"frame": [{"bbox": [.20, .35, .40, .65], "score": .9},
                             {"bbox": [.1, .02, .2, .08], "score": .8}]}
    output = inverse(mapping, predictions, 320)["frame"]
    assert len(output) == 1
    assert output[0]["bbox"] == approx(BOX)


def test_recall_guard_rejects_exchanging_one_detected_instance_for_another():
    from medsyringe.recall_guard import compare, release_allowed

    def result(missed):
        return dict(images=2, objects=2, tp=2-len(missed), fn=len(missed), fp=0,
                    false_negatives=[dict(id=identifier, box=BOX) for identifier in missed])

    baseline = result(["first"])
    swapped = compare(baseline, result(["second"]))
    assert swapped["recovered_matches"] == 1
    assert swapped["lost_matches"] == [{"id": "second", "box": tuple(BOX)}]
    assert not swapped["passed"]
    assert compare(baseline, result([]))["passed"]
    assert not release_allowed({"fresh": {"baseline": baseline, "candidate": result([])}})


def test_additive_detector_does_not_modify_or_reorder_accepted_proposals():
    from medsyringe.additive import additions, project

    duplicate = {"bbox": [0.21, 0.21, 0.38, 0.70], "score": .98}
    retained = {"bbox": BOX, "score": .68}
    missed = {"bbox": [.65, .20, .80, .75], "score": .42}
    candidate = {**missed, "score": .9}
    output = additions([retained, missed], [duplicate, candidate], .68,
                       support=.4, aux_cutoff=.8, iou=.4)
    assert output[0] is retained
    assert output[1]["bbox"] == missed["bbox"]
    assert output[1]["score"] == .68
    assert output[2] is missed
    assert project([0.05, .05, .5, .5], (200, 100, 800, 700), (1000, 1000)) == approx(
        [.23, .13, .5, .4])
    assert project([0.0, .05, .5, .5], (200, 100, 800, 700), (1000, 1000)) is None


def test_mosaic_box_coordinates_follow_the_source_quadrant():
    import torch
    from scripts.mosaic_study import mosaic

    blank = torch.zeros((3, 320, 320))
    output, target = mosaic([blank] * 4, [torch.empty((0, 4))] * 3 + [
        torch.tensor([[.0, .0, 1., 1.]])], (128, 180))
    assert output.shape == blank.shape
    assert target["boxes"].tolist() == [[128., 180., 320., 320.]]


def test_public_polygon_conversion_keeps_only_the_syringe_label():
    from scripts.public_transfer_study import syringe_boxes

    labels = "0 0.1 0.2 0.4 0.2 0.4 0.8 0.1 0.8\n1 0.0 0.0 1.0 0.0 1.0 1.0"
    assert syringe_boxes(labels, 0) == [{"bbox": [0.1, 0.2, 0.4, 0.8]}]


def test_zoom_pads_thin_images_and_restores_coordinates():
    from scripts.zoom_study import zoom

    class Detector:
        def propose(self, image):
            assert image.size == (20, 10)
            return [{"bbox": [0.20, 0.30, 0.40, 0.60], "score": .7}]

    image = Image.new("RGB", (20, 4))
    results = list(zoom(Detector(), image, (0, 0, 20, 4)))
    assert len(results) == 1
    assert results[0]["bbox"] == approx([.20, 0, .40, .75])
