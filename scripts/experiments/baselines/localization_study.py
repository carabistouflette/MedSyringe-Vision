"""Train a box refiner and continuous-IoU head from frozen proposals."""
from pathlib import Path

from joblib import dump
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from medsyringe.data import digest, read_json, train_rows, write_json
from medsyringe.detection.detectors import box_iou, model_hashes
from medsyringe.evaluate import best_threshold, evaluate
from medsyringe.detection.localization import relative_offsets, rerank
from medsyringe.detection.proposal_learning import describe_proposals
from scripts.experiments.detector_training.optimize_detector import FINAL, WEIGHTS, counters
from scripts.experiments.detector_audits.precision_study import DATA, FRESH, baseline, source
from scripts.experiments.detector_training.refine_detector import ART, HOLDOUT, OLD
from scripts.experiments.transfer.supervised_study import sets

ROOT = Path(".")


def dataset(frozen, stage):
    training, validation, historical = sets()
    old_sets = [
        ("train", train_rows(ROOT)),
        ("precision-training", read_json(DATA)["train"]),
        ("precision-test", read_json(DATA)["test"]),
        ("validation", read_json(ART / "validation_manifest.json")["rows"]),
        ("holdout", read_json(HOLDOUT)["rows"]),
        ("precision-fresh", read_json(FRESH)["rows"]),
        ("test", historical),
    ]
    rows = {}
    for split, group in old_sets:
        ranked = baseline(frozen, stage, group, split)
        for row in group:
            proposals = ranked[row["id"]][:24]
            rows[row["id"]] = (proposals, describe_proposals(source(row), proposals))
        print("Prepared", split, len(group), flush=True)
    return training, validation, historical, rows


def train(rows, data):
    features, labels, boxes, targets, weights = [], [], [], [], []
    for row in rows:
        proposals, descriptors = data[row["id"]]
        if not proposals:
            continue
        gold = [obj["bbox"] for obj in row["objects"]]
        features.extend(np.column_stack(descriptors))
        for p in proposals:
            box = p["bbox"]
            overlap, best = max(((box_iou(box, target), target) for target in gold),
                                default=(0, box))
            labels.append(overlap)
            boxes.append(box)
            targets.append(best)
            # Prevent repeat source exposures from dominating box regression.
            weights.append(1/len(proposals))
    features, labels, weights = map(np.asarray, (features, labels, weights))
    accepted = labels >= .25
    box = make_pipeline(StandardScaler(), Ridge(alpha=20))
    box.fit(features[accepted], relative_offsets(np.array(boxes)[accepted],
             np.array(targets)[accepted]), ridge__sample_weight=weights[accepted])
    quality = HistGradientBoostingRegressor(max_iter=70, max_leaf_nodes=9,
                                            l2_regularization=1, random_state=607)
    # Rare overlaps receive extra weight so the model learns a useful ranking
    # even when most low-ranked proposals are background.
    quality.fit(features, labels, sample_weight=weights*(1+3*labels))
    return dict(box=box, quality=quality), dict(samples=len(labels),
        box_regression_samples=int(accepted.sum()), total_images=len(rows))


def predictions(rows, data, models, stage, strength, weight):
    return {r["id"]: rerank(*data[r["id"]], models, box_strength=strength,
                           quality_weight=weight, overlap=stage["overlap"]) for r in rows}


def main():
    frozen, stage = read_json(OLD), read_json(FINAL)
    if (model_hashes(ROOT, frozen["backbone"]) != frozen["bundle_sha256"] or
        digest(WEIGHTS) != stage["proposal_heads_sha256"]):
        raise ValueError("The upstream model changed.")
    training, val, historical, data = dataset(frozen, stage)
    rep = {}
    for row in training:
        if row["id"].startswith("IMG_"):
            rep.setdefault(row["objects"][0]["burst"], row["id"])
    original = set(rep.values())
    fit_rows = [r for r in training if not r["id"].startswith("IMG_") or r["id"] in original]
    with threadpool_limits(limits=3):
        models, training_stats = train(fit_rows, data)
    weightfile = ROOT / "data/localization-heads.joblib"
    dump(models, weightfile, compress=3)
    base = {r["id"]: data[r["id"]][0] for r in val}
    trials = []
    for strength in (0, .5, 1):
        for weight in (0, .25, .50):
            if not strength and not weight:
                continue
            pred = predictions(val, data, models, stage, strength, weight)
            cutoff = best_threshold(val, pred) if weight else stage["cutoff"]
            metrics = evaluate(val, pred, cutoff)
            trials.append(dict(box_strength=strength, quality_weight=weight, cutoff=cutoff,
                               **counters(metrics)))
    selected = max(trials, key=lambda t: (t["f1"], t["recall"], t["ap50_95"]))
    revised = predictions(historical, data, models, stage,
                          selected["box_strength"], selected["quality_weight"])
    write_json(ART / "localization_study.json", dict(
        released_detector_sha256=digest(FINAL), samples=training_stats,
        training_ids=[r["id"] for r in fit_rows], validation_ids=[r["id"] for r in val],
        checkpoint=str(weightfile), checkpoint_sha256=digest(weightfile),
        baseline=counters(evaluate(val, base, stage["cutoff"])),
        trials=trials, selected=selected,
        historical=counters(evaluate(historical, revised, selected["cutoff"])),
        historical_baseline=counters(evaluate(historical, {r["id"]: data[r["id"]][0]
                            for r in historical}, stage["cutoff"])),
        note="Previously consulted sets only. Fresh assessment required before adoption."))
    print("Localization candidate", selected, flush=True)


if __name__ == "__main__":
    main()
