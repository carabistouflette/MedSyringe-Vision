"""Replay the previous frozen box replacement on the exposed recrop reserve."""
import argparse

from medsyringe.additive import overlap_fraction
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from scripts.occlusion_detector import ART, FREEZE, INITIAL, TRAIN, inference
from scripts.occlusion_relocalization import POLICY, TILED, area, diagnostics, replacements
from scripts.recrop_proposal_bank import WORK, candidates, read_bank
from scripts.recrop_reserve import REFERENCE, RELEASE, references

CACHE = WORK / "legacy-localization-preflight.json"
OUTPUT = ART / "recrop_relocalization_preflight.json"
AUDIT = ART / "recrop_verifier_error_audit.json"


def binding():
    return {str(p): digest(p) for p in (REFERENCE, POLICY, FREEZE, INITIAL, TRAIN)}


def collect():
    predictions, _ = inference(references())
    write_json(CACHE, dict(binding=binding(), predictions=predictions))


def test_arm(rows, baseline, predictions, peers, settings):
    corrected = {key: replacements(items, predictions[key], peers[key], settings)
                 for key, items in baseline.items()}
    return dict(**diagnostics(rows, baseline, corrected, detailed=True),
                boxes={key: [p["bbox"] for p in values] for key, values in corrected.items()})


def score():
    cache = read_json(CACHE)
    if cache["binding"] != binding():
        raise ValueError("The legacy proposals refer to different sources or checkpoints.")
    settings = read_json(POLICY)["selected"]["policy"]
    peer = next(m for m in TILED if m != settings["model"])
    original, companion = (cache["predictions"][m] for m in (settings["model"], peer))
    rows, bank = read_bank("reserve")
    baseline = read_json(RELEASE)["outputs"]
    cropped = {r["id"]: candidates(bank[r["id"]], 3) for r in rows}
    protocols = {"released": baseline, "legacy-tiles": original,
                 "three-recrops": cropped, "tiles-and-recrops": {key: [*items, *cropped[key]]
                                                                  for key, items in original.items()}}
    arms = {name: test_arm(rows, baseline, pool, companion, settings) for name, pool in protocols.items()
            if name != "released"}
    arms["released"] = diagnostics(rows, baseline, baseline, detailed=True)
    case_details = {}
    for row in read_json(AUDIT)["reserve_missed_references"]:
        key = row["id"]
        incumbent = row["baseline"]["candidate"]["bbox"]
        first = row["first_pass"]["candidate"]
        recrop = row["second_pass"]["3"]["candidate"]
        measured = {"released_iou": row["baseline"]["best_iou"], "first_pass_iou": row["first_pass"]["best_iou"],
                    "recrop_iou": row["second_pass"]["3"]["best_iou"],
                    "candidate_agreement_iou": box_iou(first["bbox"], recrop["bbox"])}
        for name, ranked in (("legacy-tiles", original[key]), ("three-recrops", cropped[key]),
                             ("tiles-and-recrops", protocols["tiles-and-recrops"][key])):
            chosen = arms[name]["boxes"][key][0]
            ref = next(r for r in rows if r["id"] == key)["objects"][row["target_index"]]["bbox"]
            measured[name] = dict(iou=box_iou(ref, chosen), changed=chosen != incumbent,
                                  best_candidate_in_pool=max((box_iou(ref, p["bbox"]) for p in ranked), default=0.))
        isolated_output, = replacements(baseline[key], [recrop], companion[key], settings)
        measured["isolated_recrop"] = dict(score=recrop["score"],
            area_ratio=area(recrop["bbox"])/area(incumbent), containment=overlap_fraction(incumbent, recrop["bbox"]),
            eligible_in_isolation="prior_bbox" in isolated_output)
        case_details[key] = measured
    write_json(OUTPUT, dict(code=digest(__file__), inputs={**binding(), **{str(p): digest(p) for p in (
              CACHE, RELEASE, AUDIT, WORK / "reserve.json")}}, frozen_legacy_settings=settings, arms=arms, cases=case_details,
              scope="Retrospective feasibility check on exposed scenes. No model or policy was fitted or selected."))
    for name, item in arms.items():
        print(name, item["iou50"], item["iou75"], "moved", item["changed_boxes"], flush=True)
    print(case_details, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "score"))
    {"collect": collect, "score": score}[parser.parse_args().stage]()
