from fastapi.testclient import TestClient

from medsyringe.app import create_app
from medsyringe.scene import Store


def test_review_links_only_one_object_to_the_original_candidate(tmp_path):
    store = Store(tmp_path / "data/workspace")
    bbox = [.2,.3,.65,.78]
    suggestion = dict(bbox=bbox,score=.15,original_score=.81,review_reason="Weak crop evidence")
    identifier = store.create(dict(name="review.jpg",kind="inference",objects=[],
                                  below_threshold_candidates=[suggestion]),b"test image")["id"]
    client = TestClient(create_app(tmp_path))
    endpoint = f"/api/observations/{identifier}/review"
    form = dict(reviewer="Reviewer",note="Verified the visible syringe",revision=0,status="approved")
    item = dict(id="manual-1",bbox=bbox,candidate_index=0)
    for invalid in ([{**item,"candidate_index":1}], [{**item,"bbox":[.1,.2,.4,.6]}],
                    [item,{**item,"id":"manual-2"}]):
        assert client.post(endpoint,json={**form,"objects":invalid}).status_code == 422
    revised = client.post(endpoint,json={**form,"objects":[item]}).json()
    assert revised["original"]["objects"] == []
    assert revised["original"]["below_threshold_candidates"] == [suggestion]
    accepted = revised["objects"][0]
    assert accepted["score"] is None and accepted["origin"] == "human"
    assert accepted["reviewed_candidate"] == dict(index=0,score=.81,reason="Weak crop evidence")
    updated = client.post(endpoint,json={**form,"revision":1,"objects":[accepted]}).json()
    assert updated["objects"] == [accepted]


def test_replacement_keeps_the_instance_and_removes_stale_crop_evidence(tmp_path):
    store = Store(tmp_path / "data/workspace")
    original = dict(
        id="object-1", bbox=[.2, .3, .6, .7], score=.92, origin="model", scale="20-scale",
        inscriptions={"capacity": {"value": 20}}, liquid_appearance={"lightness": {"lstar": 40}},
        fill_level={"ml": 14}, evidence={"scale": {"value": "20-scale"}},
    )
    suggestion = dict(bbox=[.15, .25, .65, .76], score=.24)
    observation = store.create(
        dict(name="reference.jpg", kind="inference", objects=[original],
             accepted_count=1, below_threshold_candidates=[suggestion]), b"test image",
    )
    client = TestClient(create_app(tmp_path))
    review = client.post(f"/api/observations/{observation['id']}/review", json=dict(
        revision=0, reviewer="Reviewer", note="Aligned the visible extent", status="approved",
        objects=[dict(id=original["id"], bbox=suggestion["bbox"], candidate_index=0)],
    ))
    assert review.status_code == 200
    output = review.json()
    assert output["original"]["objects"] == [original]
    assert output["original"]["accepted_count"] == 1
    revised, = output["objects"]
    assert revised["id"] == original["id"]
    assert revised["bbox"] == suggestion["bbox"]
    assert revised["origin"] == "human" and revised["score"] is None
    assert revised["reviewed_candidate"]["index"] == 0
    assert not {"inscriptions", "liquid_appearance", "fill_level", "evidence"}.intersection(revised)
