"""Train proposal heads, select on validation, and score the frozen detector."""
import argparse
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import digest, leakage, read_json, train_rows, write_json
from medsyringe.detection.detectors import (MODEL_SOURCES, NMS_IOU, PROMPT, RAW_MINIMUM, THREADS,
                        box_iou, cached_proposals, features, model_hashes, rerank, suppress)

SEED = 607
IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]
CODE = {"data.py": "data.py", "detectors.py": "detection/detectors.py", "evaluate.py": "evaluate.py"}
MODELS = {"dino": "dino", "owl": "owl", "dino-trained": "dino", "dino-augmented": "dino"}


def code_hashes():
    package = Path(__file__).parent
    return {name: digest(package / path) for name, path in CODE.items()}


def match(rows, predictions, min_iou):
    ranked = sorted(
        ((r["id"], p) for r in rows for p in predictions[r["id"]]),
        key=lambda item: -item[1]["score"],
    )
    gold = {r["id"]: [obj["bbox"] for obj in r["objects"]] for r in rows}
    used = {identifier: set() for identifier in gold}
    assignments = []
    for identifier, prediction in ranked:
        overlaps = [(box_iou(prediction["bbox"], bbox), i)
                    for i, bbox in enumerate(gold[identifier]) if i not in used[identifier]]
        iou, target = max(overlaps, default=(0, None))
        if iou >= min_iou:
            used[identifier].add(target)
        else:
            target = None
        assignments.append((identifier, prediction, target, iou))
    return assignments, used


def scores(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r,
                f1=2*p*r/(p+r) if p+r else 0.0)


def average_precision(assignments, total):
    if not total:
        return None
    true = np.cumsum([m[2] is not None for m in assignments])
    if not len(true):
        return 0.0
    recall, precision = true / total, true / np.arange(1, len(true)+1)
    return float(np.mean([max(precision[recall >= threshold], default=0.0)
                          for threshold in np.linspace(0, 1, 101)]))


def best_threshold(rows, predictions):
    assignments, _ = match(rows, predictions, 0.5)
    total = sum(len(r["objects"]) for r in rows)
    tp, fp = 0, 0
    best = (-1.0, -1.0, 1.0)
    for i, (_, prediction, target, _) in enumerate(assignments):
        tp += target is not None
        fp += target is None
        threshold = prediction["score"]
        if i+1 < len(assignments) and assignments[i+1][1]["score"] == threshold:
            continue
        result = scores(tp, fp, total-tp)
        best = max(best, (result["f1"], result["precision"], threshold))
    return best[2]


def evaluate(rows, predictions, threshold):
    total = sum(len(r["objects"]) for r in rows)
    aps = {f"{iou:.2f}": average_precision(match(rows, predictions, iou)[0], total)
           for iou in IOU_THRESHOLDS}
    accepted = {identifier: [p for p in preds if p["score"] >= threshold]
                for identifier, preds in predictions.items()}
    assignments, used = match(rows, accepted, 0.50)
    false_positives = [{"id": identifier, "prediction": p, "best_unmatched_iou": iou}
                       for identifier, p, target, iou in assignments if target is None]
    misses = [{"id": r["id"], "box": obj["bbox"]}
              for r in rows for i, obj in enumerate(r["objects"]) if i not in used[r["id"]]]
    true = sum(map(len, used.values()))
    negatives = [r for r in rows if not r["objects"]]
    abstentions = sum(not accepted[r["id"]] and bool(predictions[r["id"]]) for r in rows)
    return {
        "images": len(rows), "objects": total, "threshold": threshold,
        **scores(true, len(false_positives), len(misses)), "ap50": aps["0.50"],
        "ap50_95": float(np.mean(list(aps.values()))) if total else None, "ap_by_iou": aps,
        "exact_count": sum(len(accepted[r["id"]]) == len(r["objects"]) for r in rows),
        "negative_images": len(negatives),
        "correct_empty": sum(not accepted[r["id"]] for r in negatives),
        "images_with_only_below_threshold_proposals": abstentions,
        "false_positives": false_positives, "false_negatives": misses,
    }


def predict(raw, checkpoint=None):
    return {identifier: suppress(rerank(row["proposals"], checkpoint)
                                 if checkpoint else row["proposals"])
            for identifier, row in raw.items()}


def fit(rows, raw, augmented=None):
    started = perf_counter()
    items, labels = [], []
    for variant, proposals in ((False, raw), (True, augmented)):
        if proposals is None:
            continue
        for row in rows:
            truth = [obj["bbox"] for obj in row["objects"]]
            if variant:
                truth = [[1-r, y, 1-x, b] for x, y, r, b in truth]
            for pred in proposals[row["id"]]["proposals"]:
                iou = max((box_iou(pred["bbox"], target) for target in truth), default=0)
                items.append(features(pred))
                labels.append(iou >= 0.5)
    scaler = StandardScaler().fit(items)
    model = LogisticRegression(class_weight="balanced", random_state=SEED)
    model.fit(scaler.transform(items), labels)
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                weights=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                fit_seconds=perf_counter()-started, samples=len(labels), positives=sum(labels),
                fitting_images=len(rows), augmented=augmented is not None, seed=SEED)


def latency(raw):
    seconds = [r["seconds"] * 1000 for r in list(raw.values())[1:]]
    if not seconds:
        return None
    return dict(n=len(seconds), median_ms=float(np.median(seconds)),
                p95_ms=float(np.percentile(seconds, 95)))


def validate(root):
    artifacts = root / "artifacts"
    val = artifacts / "validation_manifest.json"
    if (artifacts / "freeze.json").exists():
        raise ValueError("This experiment is already frozen. Use a separate workspace to reproduce it.")
    rows = read_json(val)["rows"]
    training = train_rows(root)
    if not any(r["objects"] for r in rows) or not any(not r["objects"] for r in rows):
        raise ValueError("Validation requires both positive and negative scenes.")
    if leakage([*training, *rows]):
        raise ValueError("Training/validation leakage check failed.")
    raw_dino = cached_proposals(root, "dino", "validation", "original", rows)
    raw_owl = cached_proposals(root, "owl", "validation", "original", rows)
    train_dino = cached_proposals(root, "dino", "train", "original", training)
    aug_dino = cached_proposals(root, "dino", "train", "augmented", training)
    results, checkpoints = {}, {}
    for name, backbone in MODELS.items():
        raw = raw_dino if backbone == "dino" else raw_owl
        checkpoint = None
        if name.endswith(("trained", "augmented")):
            checkpoint = fit(training, train_dino, aug_dino if name.endswith("augmented") else None)
            checkpoints[name] = checkpoint
            write_json(artifacts / f"{name}.json", checkpoint)
        preds = predict(raw, checkpoint)
        threshold = best_threshold(rows, preds)
        results[name] = {**evaluate(rows, preds, threshold), "backbone_latency": latency(raw),
                         "weights_bytes": (root / "data/models" / backbone /
                                           "model.safetensors").stat().st_size}
        print(name, *(f"{results[name][key]:.3f}" for key in ("ap50", "ap50_95", "f1")), flush=True)
    winner = max(results, key=lambda name: (results[name]["ap50_95"], results[name]["ap50"]))
    result_file = artifacts / "validation_results.json"
    write_json(result_file, results)
    freeze = {
        "winner": winner, "backbone": MODELS[winner], "threshold": results[winner]["threshold"],
        "model_id": MODEL_SOURCES[MODELS[winner]][0], "revision": MODEL_SOURCES[MODELS[winner]][1],
        "bundle_sha256": model_hashes(root, MODELS[winner]),
        "proposal_checkpoint": checkpoints.get(winner),
        "validation_sha256": digest(val), "validation_results_sha256": digest(result_file),
        "training_inventory_sha256": digest(artifacts / "source_inventory.json"),
        "draft_annotations_sha256": digest(artifacts / "annotations.json"),
        "code_sha256": code_hashes(), "environment_lock_sha256": digest(root / "uv.lock"),
        "prompt": PROMPT, "raw_minimum": RAW_MINIMUM, "nms_iou": NMS_IOU, "threads": THREADS,
        "platform": platform.platform(), "python": platform.python_version(),
        "seed": SEED,
    }
    write_json(artifacts / "freeze.json", freeze)
    print("Frozen", winner, freeze["threshold"], flush=True)


def test(root):
    artifacts = root / "artifacts"
    freeze = read_json(artifacts / "freeze.json")
    current = dict(code_sha256=code_hashes(), bundle_sha256=model_hashes(root, freeze["backbone"]),
                   validation_sha256=digest(artifacts / "validation_manifest.json"),
                   validation_results_sha256=digest(artifacts / "validation_results.json"),
                   environment_lock_sha256=digest(root / "uv.lock"))
    if any(freeze[k] != value for k, value in current.items()):
        raise ValueError("The evaluated pipeline differs from the frozen selection.")
    test_manifest = artifacts / "test_manifest.json"
    rows = read_json(test_manifest)["rows"]
    val = read_json(artifacts / "validation_manifest.json")["rows"]
    if leakage([*train_rows(root), *val, *rows]):
        raise ValueError("Cross-partition leakage check failed.")
    if not any(r["objects"] for r in rows) or not any(not r["objects"] for r in rows):
        raise ValueError("Independent test requires both positive and negative scenes.")
    raw = cached_proposals(root, freeze["backbone"], "test", "original", rows)
    predictions = predict(raw, freeze["proposal_checkpoint"])
    threshold = freeze["threshold"]
    subsets = {
        "single": [r for r in rows if len(r["objects"]) == 1],
        "multiple": [r for r in rows if len(r["objects"]) > 1],
        "verified-negative": [r for r in rows if not r["objects"]],
        "annotated-occlusion": [r for r in rows if any(o.get("occluded") for o in r["objects"])],
        "no-annotated-occlusion": [r for r in rows if r["objects"] and
                                    not any(o.get("occluded") for o in r["objects"])],
    }
    result = evaluate(rows, predictions, threshold)
    result.update(model=freeze["winner"], freeze_sha256=digest(artifacts / "freeze.json"),
                  test_manifest_sha256=digest(test_manifest), latency=latency(raw),
                  slices={name: evaluate(group, predictions, threshold)
                          for name, group in subsets.items() if group})
    write_json(artifacts / "test_predictions.json", predictions)
    write_json(artifacts / "test_results.json", result)
    print(f"Test: {result['images']} scenes, precision {result['precision']:.3f}, "
          f"recall {result['recall']:.3f}, AP50 {result['ap50']:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validation", "test"])
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    (validate if args.stage == "validation" else test)(args.root)
