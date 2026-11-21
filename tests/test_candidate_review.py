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
