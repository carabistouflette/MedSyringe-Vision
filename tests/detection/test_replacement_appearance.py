from medsyringe.data import read_json
from scripts.experiments.relocalization import replacement_appearance as study
from scripts.experiments.relocalization import replacement_appearance_reserve as reserve
from scripts.experiments.relocalization.replacement_appearance import apply, gate_choices, screen_examples


ANCHOR = [.18, .20, .40, .41]
GOOD = [.14, .21, .46, .42]
BAD = [.19, .22, .33, .39]


def offer(box, proposal, joint, slot=0, same=.95):
    return dict(slot=slot, proposal=proposal, bbox=box, scores=dict(joint=joint, same=same))


def packet():
    return dict(base={"photo": [dict(bbox=ANCHOR, id="syringe-1", score=.9)]},
                choices={"photo": [offer(BAD, 0, .999), offer(GOOD, 1, .995)]},
                appearance={"photo": [
                    dict(bbox=ANCHOR, **{"targeted-text": -.01}),
                    dict(bbox=BAD, **{"targeted-text": -.02}),
                    dict(bbox=GOOD, **{"targeted-text": .01}),
                ]})


def test_proposal_gate_can_select_a_better_supported_alternative():
    item = packet()
    result = apply(item, dict(arm="targeted-proposal", joint=.99, cutoff=0.))
    assert len(result["photo"]) == 1
    assert result["photo"][0]["bbox"] == GOOD
    assert result["photo"][0]["id"] == item["base"]["photo"][0]["id"]
    assert item["choices"]["photo"][0]["scores"]["joint"] == .999


def test_pair_requirement_retains_the_incumbent_when_its_crop_is_uncertain():
    item = packet()
    result = apply(item, dict(arm="targeted-pair", joint=.99, cutoff=0.))
    assert result == item["base"]


def test_veto_does_not_remove_a_competing_ownership_claim():
    before = {"photo": [dict(bbox=ANCHOR), dict(bbox=BAD)]}
    choices = {"photo": [offer(GOOD, 2, .999, slot=0, same=.94), offer(GOOD, 2, .999, slot=1, same=.90)]}
    evidence = {"photo": {
        tuple(ANCHOR): {"targeted-text": .02},
        tuple(BAD): {"targeted-text": -.02},
        tuple(GOOD): {"targeted-text": .03},
    }}
    kept = gate_choices(before, choices, evidence, "targeted-pair", 0.)
    assert len(kept["photo"]) == 2
    assert kept["photo"][1]["scores"] == {"joint": -1., "same": .90}
    data = dict(base=before, choices=choices, appearance={"photo": [
        dict(bbox=list(box), **scores) for box,scores in evidence["photo"].items()]})
    assert apply(data, dict(arm="targeted-pair", joint=.99, cutoff=0.)) == before


def test_screen_uses_at_most_one_localized_proposal_per_scene_and_category():
    row = dict(objects=[], lookalikes=[dict(bbox=GOOD, kind="pipette"), dict(bbox=BAD, kind="pipette")])
    assert screen_examples(row, [dict(bbox=GOOD), dict(bbox=BAD)]) == [dict(bbox=GOOD, kind="pipette", iou=1.)]
    assert screen_examples(row, [dict(bbox=[.8, .8, .9, .9])]) == []


def test_scoring_accepts_the_cohort_iterator(monkeypatch, tmp_path):
    item = dict(packet(), examples={"photo": []})
    rows = [dict(id="photo", objects=[dict(bbox=GOOD)])]
    policy = tmp_path / "policy.json"
    result = tmp_path / "result.json"
    policy.write_text("{}")
    monkeypatch.setattr(study, "POLICY", policy)
    monkeypatch.setattr(study, "RESULT", result)
    monkeypatch.setattr(study, "frozen", lambda: dict(
        selection=dict(winner=None, active={}),
        controls={"ungated-joint": dict(arm="off", joint=.99, cutoff=None)},
    ))
    monkeypatch.setattr(reserve, "cohorts", lambda: iter([("reserve", (rows, item))]))
    study.score()
    recorded = read_json(result)
    assert recorded["cohorts"]["reserve"]["variants"]["ungated-joint"]["guard"]["count_preserved"]
