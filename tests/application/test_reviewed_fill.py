import io
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient

from medsyringe.app import create_app
from medsyringe.review.scene import Store


def test_fill_review_validates_points_and_exports_append_only_evidence(tmp_path):
    store = Store(tmp_path / "data/workspace")
    box = [.2, .1, .8, .9]
    original = dict(id="object-1", bbox=box, scale="20-scale", score=.9, origin="model",
                    fill_level={"status": "insufficient evidence"})
    record = store.create(dict(name="test.jpg", kind="inference", dimensions=[1200, 1800],
                               objects=[original]), b"test image")
    path = f"/api/observations/{record['id']}"
    client = TestClient(create_app(tmp_path))
    manual = dict(capacity_ml=20, low={"point": [.4, .7], "ml": 5},
                  high={"point": [.5, .3], "ml": 15}, piston=[.45, .5])
    item = dict(id="object-1", bbox=box, scale="20-scale", manual_fill=manual)
    body = dict(revision=0, reviewer="Reviewer", note="Marked the piston", status="approved", objects=[item])
    for invalid in ({**manual, "piston": [.45, .15]}, {**manual, "capacity_ml": 60},
                    {**manual, "piston": [.92, .5]}, {**manual, "high": manual["low"]}):
        item["manual_fill"] = invalid
        assert client.post(path + "/review", json=body).status_code == 422
    item["manual_fill"] = manual
    reviewed = client.post(path + "/review", json=body)
    assert reviewed.status_code == 200, reviewed.text
    corrected = reviewed.json()["objects"][0]
    assert corrected["origin"] == "human"
    assert corrected["fill_level"]["ml"] == 10
    assert corrected["fill_level"]["percent"] == 50
    assert corrected["fill_level"]["landmarks"] == manual
    assert "interval_ml" not in corrected["fill_level"]
    body["revision"] = 1
    del item["manual_fill"]
    assert client.post(path + "/review", json=body).json()["objects"] == [corrected]
    response = client.post(path + "/export", json={"revision": 2})
    with ZipFile(io.BytesIO(response.content)) as archive:
        report = json.loads(archive.read("report.json"))
    assert report["original"]["objects"] == [original]
    assert report["objects"][0]["fill_level"] == corrected["fill_level"]
    body["revision"] = 2
    item["manual_fill"] = None
    assert "fill_level" not in client.post(path + "/review", json=body).json()["objects"][0]
