from io import BytesIO
from hashlib import sha256
from zipfile import ZipFile
import json

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from medsyringe.app import create_app
from medsyringe.liquid_color import lab_pixels
from medsyringe.liquid_fields import color_from_region
from medsyringe.scene import Store


def png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_colour_requires_existing_localization_and_lightness_agreement():
    pixels = np.empty((80, 100, 3), dtype=np.uint8)
    pixels[:] = [228, 226, 225]
    pixels[20:60, 20:80] = [190, 93, 29]
    lightness = float(lab_pixels(pixels)[35, 35, 0])
    evidence = dict(status="appearance measured", region=[.2, .25, .8, .75],
                    lightness=dict(lstar=lightness, iqr=[lightness - 2, lightness + 2]))
    image = Image.fromarray(pixels)
    assert color_from_region(image, evidence)["srgb"] == [190, 93, 29]
    assert color_from_region(image, {**evidence, "region": [.2, 0., .8, .2]}) is None
    assert color_from_region(image, {**evidence, "lightness": {**evidence["lightness"], "lstar": 75}}) is None


def test_presence_and_capacity_review_are_exported_without_inventing_volume(tmp_path):
    store = Store(tmp_path / "data/workspace")
    store.create(dict(name="photo.png", kind="inference", dimensions=[100, 60], accepted_count=1,
                      objects=[dict(id="one", bbox=[.1, .1, .8, .8], origin="model",
                                    inscriptions={"capacity": {"value": 5, "unit": "mL"}},
                                    fill_level={"status": "visual estimate", "ml": 3., "capacity_ml": 5.})]),
                 png(np.full((60, 100, 3), 170, dtype=np.uint8)), "public-case")
    with TestClient(create_app(tmp_path)) as client:
        root = "/api/observations/public-case"
        original = client.get(root).json()["field_status"]["one"]
        assert original["nominal_capacity"]["ml"] == 5
        reviewed = client.post(root + "/review", json=dict(
            revision=0, reviewer="Reader", note="Piston appears seated", status="approved",
            objects=[dict(id="one", bbox=[.1, .1, .8, .8], nominal_capacity_override_ml=5,
                          liquid_presence_override="empty")]))
        assert reviewed.status_code == 200
        revised = reviewed.json()["field_status"]["one"]
        assert revised["liquid_presence"]["value"] is False
        assert revised["liquid_volume"]["ml"] is None
        assert reviewed.json()["original_field_status"]["one"]["liquid_volume"]["ml"] == 3
        exported = client.post(root + "/export", json={"revision": 1})
        assert exported.status_code == 200
        with ZipFile(BytesIO(exported.content)) as archive:
            evidence = json.loads(archive.read("report.json"))
        assert evidence["field_status"]["one"] == revised


def test_live_opacity_endpoint_measures_only_an_explicit_matched_pair(tmp_path):
    clear = np.broadcast_to(np.linspace(55, 228, 160, dtype=np.uint8)[None, :, None], (55, 160, 3)).copy()
    linear = ((clear.astype(float) / 255 + .055) / 1.055) ** 2.4
    fluid = np.round((1.055 * (.45 * linear + .04) ** (1 / 2.4) - .055) * 255).astype(np.uint8)
    files = dict(observed=("fluid.png", png(fluid), "image/png"),
                 clear_reference=("clear.png", png(clear), "image/png"))
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/liquid-opacity", files=files).json()["status"] == "indeterminate"
        observed = client.post("/api/liquid-opacity", files=files, data={"matched_setup": "true"}).json()
    assert observed["status"] == "estimated"
    assert abs(observed["value"] - .55) < .01
    assert observed["fit_r2"] >= .99


def test_reviewed_liquid_sample_reads_native_color_and_rejects_outside_box(tmp_path):
    pixels = np.full((100, 130, 3), 228, dtype=np.uint8)
    pixels[30:60, 55:75] = [36, 114, 215]
    payload = png(pixels)
    store = Store(tmp_path / "data/workspace")
    store.create(dict(name="frame.png", kind="inference", dimensions=[130, 100],
                      source_sha256=sha256(payload).hexdigest(), source_extension="png", accepted_count=1,
                      objects=[dict(id="syringe", bbox=[.2, .2, .8, .75], origin="model")]),
                 payload, "color-example", source=payload)
    with TestClient(create_app(tmp_path)) as client:
        location = "/api/observations/color-example/review"
        item = dict(id="syringe", bbox=[.2, .2, .8, .75], liquid_sample=[.0, .0, .1, .1])
        review = dict(revision=0, reviewer="Reader", note="Localized the visible blue column",
                      status="approved", objects=[item])
        assert client.post(location, json=review).status_code == 422
        item["liquid_sample"] = [.43, .32, .56, .56]
        response = client.post(location, json=review)
    assert response.status_code == 200
    fields = response.json()["field_status"]["syringe"]
    assert fields["liquid_presence"]["source"] == "reviewed sample"
    assert fields["liquid_colour"]["family"] == "blue"
    assert fields["liquid_colour"]["srgb"] == [36, 114, 215]
