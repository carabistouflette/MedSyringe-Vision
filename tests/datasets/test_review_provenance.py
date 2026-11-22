from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient

from medsyringe.app import create_app
from medsyringe.scene import Store


def test_redraw_clears_stale_fields_and_reconfirmation_is_explicit(tmp_path):
    store = Store(tmp_path / "data/workspace")
    old = dict(id="one", bbox=[.1, .1, .3, .3], origin="model", score=.91, scale="20-scale",
               color="orange", transcription="20 mL", nominal_capacity_override_ml=20,
               liquid_presence_override="present", evidence={"scale": {"reason": "Old crop"}},
               inscriptions={"capacity": {"value": 20, "unit": "mL"}},
               fill_level={"status": "visual estimate", "ml": 4})
    for identifier in ("redraw", "reconfirm"):
        store.create(dict(name="photo.png", kind="inference", objects=[old]), b"preview", identifier)
    edit = dict(revision=0, reviewer="Reviewer", note="Checked the new box", status="approved")
    moved = dict(old, bbox=[.6, .6, .9, .9])
    with TestClient(create_app(tmp_path)) as client:
        route = "/api/observations/redraw"
        reviewed = client.post(route + "/review", json=dict(edit, objects=[moved]))
        assert reviewed.status_code == 200
        report = reviewed.json()
        revised, = report["objects"]
        assert revised["bbox"] == moved["bbox"] and revised["score"] is None
        assert revised["scale"] is None and revised["color"] is None and revised["transcription"] == ""
        assert revised["nominal_capacity_override_ml"] is None and revised["liquid_presence_override"] is None
        assert all(field not in revised for field in ("evidence", "inscriptions", "fill_level"))
        assert report["original"]["objects"] == [old]
        with ZipFile(BytesIO(client.post(route + "/export", json={"revision": 1}).content)) as archive:
            exported = json.loads(archive.read("report.json"))
        assert exported["field_status"]["one"]["nominal_capacity"]["ml"] is None
        assert exported["field_status"]["one"]["liquid_volume"]["ml"] is None
        assert exported["original"]["objects"] == [old]
        restored = client.post("/api/observations/reconfirm/review", json=dict(
            edit, objects=[dict(moved, confirmed_on_box=["scale", "transcription"])]
        )).json()
        assert restored["objects"][0]["scale"] == "20-scale"
        assert restored["objects"][0]["transcription"] == "20 mL"
        assert "confirmed_on_box" not in restored["objects"][0]
        assert restored["field_status"]["one"]["nominal_capacity"]["ml"] is None


def test_explicit_capacity_replaces_an_incompatible_appearance_type(tmp_path):
    store = Store(tmp_path / "data/workspace")
    obj = dict(id="one", bbox=[.1, .1, .7, .7], origin="model", scale="20-scale")
    store.create(dict(name="scene.png", kind="inference", objects=[obj]), b"preview", "scene")
    route = "/api/observations/scene"
    with TestClient(create_app(tmp_path)) as client:
        initial = client.get(route).json()["field_status"]["one"]
        assert initial["syringe_type"]["value"] == "20-scale"
        assert initial["nominal_capacity"]["ml"] is None
        revised = client.post(route + "/review", json=dict(
            revision=0, reviewer="Reviewer", note="Read the actual capacity", status="approved",
            objects=[dict(obj, nominal_capacity_override_ml=60)]
        )).json()["field_status"]["one"]
        assert revised["nominal_capacity"]["ml"] == 60
        assert revised["syringe_type"]["value"] is None


def test_joint_inventory_and_reference_export_preserve_object_associations(tmp_path):
    store = Store(tmp_path / "data/workspace")
    for identifier, colors in (("reference", ("orange", "amber-brown")),
                               ("observation", ("amber-brown", "orange"))):
        boxes = [dict(id=f"object-{i}", bbox=[.1, .1, .2, .2], scale=f"{ml}-scale", color=color)
                 for i, (ml, color) in enumerate(zip((20, 60), colors, strict=True))]
        raw = f"Original bytes of {identifier}".encode()
        store.create(dict(name=identifier, kind="inference", objects=boxes, source_extension="png",
                          source_sha256=sha256(raw).hexdigest()), b"preview", identifier, source=raw)
    route = "/api/observations/observation"
    with TestClient(create_app(tmp_path)) as client:
        comparison = client.get(route + "/compare/reference").json()
        assert comparison["count_change"] == 0
        assert all(value == 0 for value in comparison["attribute_count_changes"]["scale"].values())
        assert all(value == 0 for value in comparison["attribute_count_changes"]["color"].values())
        assert comparison["attribute_count_changes"]["scale_and_color"]["20-scale / orange"] == -1
        assert comparison["attribute_count_changes"]["scale_and_color"]["20-scale / amber-brown"] == 1
        expected = dict(field="scale_and_color", counts={
            "20-scale / orange": 1, "20-scale / amber-brown": 0,
            "60-scale / orange": 0, "60-scale / amber-brown": 1,
        })
        check = client.post(route + "/inventory", json=expected).json()
        assert check["minimum_missing"] == 2
        assert check["evidence"]["20-scale / amber-brown"] == ["object-0"]
        response = client.post(route + "/export", json=dict(
            revision=0, reference_id="reference", reference_revision=0, expected=expected,
        ))
        with ZipFile(BytesIO(response.content)) as archive:
            report = json.loads(archive.read("report.json"))
            assert archive.read(report["reference"]["original_image"]) == b"Original bytes of reference"
            assert archive.read(report["original_image"]) == b"Original bytes of observation"
            assert report["observation_comparison"] == comparison


def test_partial_joint_inventory_respects_known_fields_and_guaranteed_extras(tmp_path):
    store = Store(tmp_path / "data/workspace")
    partial = dict(id="uncertain", bbox=[.1, .1, .2, .2], scale="60-scale", color=None)
    matched = dict(id="matched", bbox=[.3, .3, .4, .4], scale="20-scale", color="orange")
    for name, objects in (("missing", [partial]), ("excess", [matched, partial])):
        store.create(dict(name=name, kind="inference", objects=objects), b"preview", name)
    expected = dict(field="scale_and_color", counts={
        "20-scale / orange": 1, "20-scale / amber-brown": 0,
        "60-scale / orange": 0, "60-scale / amber-brown": 0,
    })
    with TestClient(create_app(tmp_path)) as client:
        missing = client.post("/api/observations/missing/inventory", json=expected).json()
        assert missing["missing"]["20-scale / orange"] == missing["minimum_missing"] == 1
        assert missing["minimum_extra"] == 1
        excess = client.post("/api/observations/excess/inventory", json=expected).json()
        assert excess["minimum_missing"] == 0 and excess["minimum_extra"] == 1
        expected["counts"]["60-scale / amber-brown"] = 1
        compatible = client.post("/api/observations/excess/inventory", json=expected).json()
        assert compatible["minimum_missing"] == compatible["minimum_extra"] == 0


def test_reviewed_transcription_withdraws_prior_ocr_derivatives(tmp_path):
    store = Store(tmp_path / "data/workspace")
    original = dict(
        id="one", bbox=[.1, .1, .7, .7], origin="model", scale="20-scale",
        transcription="", inscriptions={"capacity": {"value": 20, "unit": "mL"},
                                        "scale": {"value": "20-scale"}},
        fill_level={"status": "visual estimate", "ml": 6, "interval_ml": [5, 7]},
    )
    store.create(dict(name="scene.png", kind="inference", objects=[original]), b"preview", "scene")
    route = "/api/observations/scene"
    with TestClient(create_app(tmp_path)) as client:
        review = client.post(route + "/review", json=dict(
            revision=0, reviewer="Reviewer", note="Corrected the reading", status="approved",
            objects=[dict(original, transcription="Capacity is illegible")],
        ))
        assert review.status_code == 200
        report = review.json()
        assert report["objects"][0]["transcription"] == "Capacity is illegible"
        assert report["objects"][0]["scale"] is None
        assert report["field_status"]["one"]["nominal_capacity"]["ml"] is None
        assert report["field_status"]["one"]["liquid_volume"]["ml"] is None
        assert report["original"]["objects"] == [original]
