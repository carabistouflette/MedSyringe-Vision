import pytest

from medsyringe.evaluate import match
from scripts.experiments.crowds.native_candidate_verifier import simple_metrics
from scripts.experiments.identity.syringe_object_features import ROUTES, TARGET, class_for_proposal
from scripts.experiments.identity.syringe_object_verifier import ARMS, guard, partition_audit, predictions, thresholds


SYRINGE = [.08, .20, .50, .48]
SMALLER = [.12, .24, .48, .45]
FRAGMENT = [.10, .24, .27, .42]
PIPETTE = [.55, .48, .95, .73]


def box(coords, score=.5):
    return dict(bbox=coords, score=score)


def scene():
    return dict(id="frame", objects=[box(SYRINGE)], lookalikes=[dict(box(PIPETTE), kind="adjustable pipettor")])


def test_fit_labels_skip_partial_syringes_and_use_reviewed_lookalike_identity():
    assert class_for_proposal(box(SYRINGE), scene()) == TARGET
    assert class_for_proposal(box(FRAGMENT), scene()) is None
    assert class_for_proposal(box(PIPETTE), scene()) == "pipette"
    assert class_for_proposal(box([.8, .8, .9, .9]), scene()) == "other"


def test_verifier_only_applies_to_candidate_additions():
    current = dict(box(SYRINGE, .9), id="retained")
    before = {"frame": [current]}
    offered = {"frame": [box(PIPETTE)]}
    evidence = {"frame": {tuple(PIPETTE): {"typed": -.8}}}
    after, skipped = predictions(before, offered, evidence, "typed", 0.)
    assert before == after == {"frame": [current]}
    assert after["frame"] is not before["frame"]
    assert skipped["frame"][0]["evidence"] == {"typed": -.8}


def test_selection_preserves_recovered_syringe_while_removing_new_false_positive():
    rows = [scene()]
    base = {"frame": []}
    proposals = {"frame": [box(SYRINGE), box(PIPETTE)]}
    rank = {"frame": {tuple(SYRINGE): dict.fromkeys(ARMS, .2), tuple(PIPETTE): dict.fromkeys(ARMS, -.7)}}
    selected = thresholds(rows, base, dict.fromkeys(ROUTES, proposals), rank)
    best = selected["winner"]
    assert best["rejected"] == 1 and best["tp"] == 1 and best["fp"] == 0
    assert best["safe"] and best["preserves_recovery"]
    assert all(a["preserves_recovery"] for a in selected["finalists"]["active"].values())


def test_preservation_includes_strict_localization():
    rows = [scene()]
    base = simple_metrics(rows, {"frame": []})
    both = {"frame": [box(SYRINGE, .9), box(SMALLER)]}
    refs = {iou: match(rows, both, iou)[1] for iou in (.50, .75)}
    delta = guard(rows, base, refs, {"frame": [box(SMALLER)]})
    assert delta["safe"]
    assert delta["lost_recovered"] == {"0.5": {}, "0.75": {"frame": [0]}}
    assert not delta["preserves_recovery"]


def test_partition_audit_rejects_shared_bytes_even_if_ids_and_groups_changed():
    fit = [dict(id="train", group="group A", sha256="same")]
    reserve = [dict(id="reserve", group="group B", sha256="same")]
    with pytest.raises(ValueError, match="Shared provenance between fit and reserve"):
        partition_audit(fit, [], reserve)
