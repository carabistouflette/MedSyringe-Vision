from scripts.experiments.detector_training.detection_hillclimb_experiment import proposals
from scripts.experiments.detector_training.detection_hillclimb_rank import agree, augment


def test_additions_preserve_order_and_respect_the_published_veto():
    accepted = {"bbox": [.1,.15,.27,.7], "score": .82}
    denied = {"bbox": [.60,.2,.72,.75], "score": .006, "crop_score": .1}
    second = {"bbox": [.35,.3,.47,.80], "score": .51}
    baseline = dict(floor=.67,ranked=[accepted,denied])
    candidates = [dict(denied,score=.98),dict(accepted,score=.94),second]
    policy = dict(scope="all",duplicate="strict",threshold=.4,require_grounding=False,model="detector-a")
    ranked = augment(baseline,candidates,policy)
    assert ranked[0] is accepted and ranked[-1] is denied
    assert ranked[1]["bbox"] == second["bbox"] and ranked[1]["score"] == .67
    assert augment(baseline,candidates,{**policy,"scope":"empty"}) == baseline["ranked"]
    assert augment(baseline,candidates,{**policy,"require_grounding":True}) == baseline["ranked"]


def test_same_tile_does_not_count_twice_and_model_agreement_needs_overlap():
    main = dict(bbox=[.18,.26,.56,.67],score=.55,view=4)
    copy = {**main,"bbox":[.19,.25,.55,.67],"score":.37}
    frame = dict(full=[], windows=[main,copy])
    assert proposals(frame,"stable") == []
    frame["windows"][1]["view"] = 7
    assert proposals(frame,"stable") == [main]
    invalid = dict(bbox=[.70,.10,.95,.25],score=.79)
    assert agree([main],[invalid]) == []
    assert len(agree([main],[copy])) == 1
