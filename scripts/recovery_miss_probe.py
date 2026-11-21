"""Inspect text and box-prompt masks on already known public misses."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from scripts.expanded_proposals import ART, image_for, rows_for
from scripts.recovery_mask_study import LOCK as MASK_LOCK, local_masks
from scripts.recovery_study import LOCK, dataset, probability
from scripts.sam3_study import Segmenter, configure_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if not MASK_LOCK.exists():
        raise ValueError("Freeze the mask-based selection before error inspection.")
    configure_cpu()
    selection = read_json(LOCK)
    policy = selection["shadow"]["policy"]
    model = selection["models"].get(policy["method"])
    frames = dataset("reserve")["frames"]
    rows = {r["id"]: r for r in rows_for("reserve")}
    baseline = read_json(ART / "detector_reserve.json")
    misses = baseline["split_metrics"]["all"]["baseline"]["false_negatives"]
    segmenter = Segmenter(504, "xpu")
    probe = []
    for miss in misses:
        identifier, truth = miss["id"], miss["box"]
        weak = frames[identifier]["candidates"]
        shortlist = max(weak, key=lambda item: item["proposal"]["score"])
        p = shortlist["proposal"]
        image = image_for(rows[identifier], args.source)
        observations = local_masks(segmenter, image, p["bbox"])
        for value in observations.values():
            if value:
                value["reference_iou"] = box_iou(value["mask_bbox"], truth)
        entry = dict(id=identifier, candidate=p, candidate_iou=box_iou(p["bbox"],truth), masks=observations,
                     recovery_confidence=probability(shortlist["values"],model,policy["method"]))
        probe.append(entry)
        print(identifier, round(entry["recovery_confidence"],3), observations, flush=True)
    write_json(ART / "public_miss_mask_probe.json", dict(cases=probe, calibrated_rule=policy,
               mask_selection_sha256=digest(MASK_LOCK),
               scope="Post-selection error inspection of known misses. No threshold changes or deployment decisions."))


if __name__ == "__main__":
    main()
