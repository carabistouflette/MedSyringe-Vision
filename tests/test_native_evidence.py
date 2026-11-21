from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image

from medsyringe.app import create_app
from medsyringe.scene import Store


def test_original_pixels_and_source_archive_follow_the_reviewed_box(tmp_path):
    image = Image.new("RGB", (240,120), "black")
    image.paste((200,10,40), (60,30,180,90))
    stream = BytesIO()
    image.save(stream,format="PNG")
    original = stream.getvalue()
    store = Store(tmp_path/"data/workspace")
    observation = store.create(dict(name="input.png",kind="inference",objects=[dict(id="object-1",bbox=[.25,.25,.75,.75])],
                                    source_extension="png",source_sha256=sha256(original).hexdigest()),
                               b"preview bytes",source=original)
    identifier = observation["id"]
    client = TestClient(create_app(tmp_path))
    route = f"/api/observations/{identifier}"
    native = client.get(route+"/objects/object-1/crop")
    assert native.status_code == 200 and native.headers["content-type"] == "image/png"
    patch = Image.open(BytesIO(native.content))
    assert patch.size == (120,60) and patch.getpixel((0,0)) == (200,10,40)
    assert client.get(route+"/objects/unknown/crop").status_code == 404
    draft = client.get(route+"/crop", params={"bbox": [.25, .25, .5, .5]})
    assert draft.status_code == 200 and draft.headers["cache-control"] == "private, no-store"
    with Image.open(BytesIO(draft.content)) as cropped:
        assert cropped.size == (60, 30) and cropped.getpixel((0, 0)) == (200, 10, 40)
    saved = client.get(route).json()
    assert saved["revision"] == 0 and saved["objects"][0]["bbox"] == [.25, .25, .75, .75]
    for invalid in ([.6, .25, .5, .5], [.25, .5], [-.1, .2, .5, .6], ["nan", .25, .5, .6]):
        assert client.get(route+"/crop", params={"bbox": invalid}).status_code == 422
    payload = client.post(route+"/export",json={"revision":0})
    with ZipFile(BytesIO(payload.content)) as exported:
        assert exported.read("source.png") == original
        details = json.loads(exported.read("report.json"))
        assert details["original"]["source_sha256"] == sha256(original).hexdigest()
        assert details["original_image"] == "source.png"


def test_legacy_observation_has_its_preview_and_export(tmp_path):
    store = Store(tmp_path/"data/workspace")
    record = store.create(dict(name="legacy.jpg",kind="inference",objects=[dict(id="known",bbox=[.1,.1,.5,.6])]),
                          b"legacy preview")
    client = TestClient(create_app(tmp_path))
    route = f"/api/observations/{record['id']}"
    assert client.get(route+"/objects/known/crop").status_code == 404
    assert client.get(route+"/image").content == b"legacy preview"
    with ZipFile(BytesIO(client.post(route+"/export",json={"revision":0}).content)) as exported:
        assert set(exported.namelist()) == {"observation.jpg","report.json"}
