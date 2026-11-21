"""Explore tiled transfer-learning proposals without removing accepted detections."""
import argparse
from pathlib import Path

import torch

from medsyringe.additive import additions, duplicates, overlapping_tiles, project
from medsyringe.data import digest, leakage, read_json, train_rows, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from medsyringe.supervised import detector, predict, tensor
from scripts.optimize_detector import FINAL, counters
from scripts.precision_study import baseline, source
from scripts.refine_detector import ART, OLD
from scripts.supervised_study import sets

ROOT = Path(".")
REPORT = ART / "additive_calibration.json"
FRESH = ART / "additive_external_manifest.json"


def proposals(rows, split):
    release, frozen = read_json(FINAL), read_json(OLD)
    ranked = baseline(frozen, release, rows, split)
    training = read_json(ART / "supervised_study.json")
    file = Path(training["checkpoint"])
    if digest(file) != training["checkpoint_sha256"]:
        raise ValueError("The auxiliary model checkpoint changed.")
    model = detector(ROOT)
    model.load_state_dict(torch.load(file, weights_only=True, map_location="cpu"))
    views = {}
    for row in rows:
        image = source(row)
        w, h = image.size
        windows = [(0, 0, w, h), *overlapping_tiles(image.size)]
        inputs = [(str(i), tensor(image.crop(region))) for i, region in enumerate(windows)]
        outputs = predict(model, inputs)
        full, tiled = [], []
        for i, region in enumerate(windows):
            for p in outputs[str(i)]:
                box = project(p["bbox"], region, image.size)
                if box:
                    (tiled if i else full).append({**p, "bbox": box})
        views[row["id"]] = {"full": full, "tiles": tiled, "both": full+tiled}
    return ranked, views, release["cutoff"]


def apply(rows, ranked, auxiliary, cutoff, setting):
    params = {k: setting[k] for k in ("aux_cutoff", "support", "iou")}
    return {r["id"]: additions(ranked[r["id"]], auxiliary[r["id"]][setting["views"]],
                               cutoff, **params) for r in rows}


def select(rows):
    ranked, auxiliary, cutoff = proposals(rows, "precision-fresh")
    base = evaluate(rows, ranked, cutoff)
    trials = []
    for views in ("full", "tiles", "both"):
        for aux_cutoff in (.45, .65, .80, .90):
            for support in (.25, .45, .60):
                for iou in (.35, .50):
                    setting = dict(views=views, aux_cutoff=aux_cutoff, support=support, iou=iou)
                    pred = apply(rows, ranked, auxiliary, cutoff, setting)
                    result = evaluate(rows, pred, cutoff)
                    trials.append(dict(**setting, **counters(result), recall_guard=compare(base, result)))
    eligible = [t for t in trials if t["recall_guard"]["passed"] and t["tp"] > base["tp"]
                and t["precision"] >= base["precision"]]
    choice = max(eligible, key=lambda t: (t["f1"], t["precision"], t["recall"])) if eligible else None
    missed = {}
    for item in base["false_negatives"]:
        identifier, target = item["id"], item["box"]
        pool = auxiliary[identifier]
        accepted = [p for p in ranked[identifier] if p["score"] >= cutoff]
        survivors = [p for p in pool["both"] if
                     not any(duplicates(p["bbox"], q["bbox"]) for q in accepted)]
        missed[identifier] = dict(
            reference_box=target,
            full_max_iou=max((box_iou(target, p["bbox"]) for p in pool["full"]), default=0.0),
            tile_max_iou=max((box_iou(target, p["bbox"]) for p in pool["tiles"]), default=0.0),
            after_duplicate_check_max_iou=max(
                (box_iou(target, p["bbox"]) for p in survivors), default=0.0),
        )
    write_json(REPORT, dict(baseline=counters(base), selected=choice, trials=trials, missed=missed,
                            unseen_sha256=digest(FRESH) if FRESH.exists() else None,
                            detector_sha256=digest(FINAL),
                            ssd_study_sha256=digest(ART / "supervised_study.json")))
    print("Development", counters(base), "selected", choice, flush=True)


def score(rows, split):
    configured = read_json(REPORT)
    if (configured["unseen_sha256"] != digest(FRESH) or
        configured["detector_sha256"] != digest(FINAL) or
        configured["ssd_study_sha256"] != digest(ART / "supervised_study.json")):
        raise ValueError("References or models changed after calibration.")
    if not configured["selected"]:
        print("No configuration met the calibration criteria.", flush=True)
        return
    ranked, auxiliary, cutoff = proposals(rows, split)
    base = evaluate(rows, ranked, cutoff)
    result = evaluate(rows, apply(rows, ranked, auxiliary, cutoff, configured["selected"]), cutoff)
    check = compare(base, result)
    print(split, counters(base), counters(result), check, flush=True)
    write_json(ART / f"additive_{split}.json", dict(baseline=base, candidate=result,
                 recall_guard=check, calibration_sha256=digest(REPORT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "audit", "historical", "fil-rouge"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    training, validation, historical = sets()
    if args.stage == "select":
        select(validation)
        return
    fresh = read_json(FRESH)["rows"]
    if leakage([*[{**r, "split": "training"} for r in training], *validation, *historical, *fresh]):
        raise ValueError("Grouped input partitions overlap.")
    if args.stage == "audit":
        score(fresh, "additive-audit")
    elif args.stage == "historical":
        score(historical, "test")
    else:
        score(train_rows(ROOT), "train")


if __name__ == "__main__":
    main()
