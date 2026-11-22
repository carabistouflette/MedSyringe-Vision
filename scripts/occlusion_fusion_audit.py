"""Identify localized RF-DETR hypotheses blocked by overlap with released boxes."""
from medsyringe.additive import duplicates
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from scripts.occlusion_detector import ART, DATA, FREEZE, RESULTS, frozen_frames
from scripts.occlusion_proposals import oracle


def run():
    cohorts, released = frozen_frames()
    floors = read_json(FREEZE)["floors"]
    predictions = read_json(DATA / "scored-predictions.json")
    findings = {}
    for row in [*cohorts["new-public-reserve"], *cohorts["prior-small-occlusion"]]:
        key = row["id"]
        baseline = released[key]
        targets = row["objects"]
        prior = oracle(targets, baseline)["targets"]
        findings[key] = {}
        for arm, floor in floors.items():
            accepted = [p for p in predictions[arm][key] if p["score"] >= floor]
            detected = oracle(targets, accepted)["targets"]
            blocked = []
            for index, (obj, candidate) in enumerate(zip(targets, detected, strict=True)):
                if candidate["best_iou"] < .50:
                    continue
                target = obj["bbox"]
                box = candidate["best_candidate"]["bbox"]
                collisions = [dict(bbox=p["bbox"], mutual_iou=box_iou(p["bbox"], box),
                                   iou_to_target=box_iou(p["bbox"], target))
                              for p in baseline if duplicates(p["bbox"], box)]
                blocked.append(dict(target=index, candidate=candidate["best_candidate"],
                                    candidate_iou=candidate["best_iou"],
                                    existing_iou_to_target=prior[index]["best_iou"], conflicts=collisions))
            findings[key][arm] = blocked
    write_json(ART / "occlusion_fusion_diagnostic.json",
               dict(observations=findings, replay_sha256=digest(RESULTS),
                    scope="Post-evaluation audit of high-scoring RF proposals blocked by released-region overlap."))


if __name__ == "__main__":
    run()
