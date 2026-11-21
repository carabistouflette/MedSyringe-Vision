"""Train a crop verifier on new distractors and check a locked external set."""
import argparse
from pathlib import Path

from joblib import load
import numpy as np
from PIL import ImageOps
from sklearn.linear_model import LogisticRegression

from medsyringe.attributes import crop
from medsyringe.data import digest, download, leakage, read_image, read_json, train_rows, write_json
from medsyringe.detectors import box_iou, model_hashes
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare, release_allowed
from medsyringe.verification import (BACKBONE, CropEncoder, classifier_scores, eligible,
                                      filter_proposals)
from scripts.optimize_detector import FINAL, WEIGHTS, counters, describe, rescore
from scripts.refine_detector import ART, HOLDOUT, OLD, ROOT, cache, predict

DATA = ART / "precision_study_manifest.json"
RELEASE = ART / "precision_candidate.json"
HEAD = ART / "precision_head.json"
FRESH = ART / "semantic_holdout_manifest.json"


def source(row):
    path = ROOT / "data" / row["image"]
    if not path.exists() and row.get("image_url"):
        download(row["image_url"], path)
    if digest(path) != row.get("derived_sha256", row["sha256"]):
        raise ValueError(f"Image checksum mismatch: {row['id']}")
    return read_image(path)


def baseline(frozen, stage, rows, partition):
    for row in rows:
        if row.get("image_url"):
            source(row)
    entries = cache(frozen, partition, rows, stage["views"])
    ranked = predict(entries, stage["views"], stage["strategy"], stage["overlap"])
    return rescore(ranked, describe(rows, ranked), load(WEIGHTS), stage["weight"], stage["overlap"])


def vectors(encoder, rows, predictions, cutoff):
    result = {}
    for row in rows:
        candidates = eligible(predictions[row["id"]], cutoff)
        image = source(row) if candidates else None
        result[row["id"]] = encoder.encode([crop(image, p["bbox"]) for p in candidates])
    return result


def apply(predictions, embeddings, head, cutoff):
    out = {}
    for key, proposals in predictions.items():
        accepted, _ = filter_proposals(eligible(proposals, cutoff), classifier_scores(
            embeddings[key], head["weights"], head["bias"]), head["cutoff"])
        out[key] = accepted
    return out


def fit(frozen, stage, data, encoder, checksum):
    if RELEASE.exists():
        raise ValueError("The candidate is already frozen.")
    original = train_rows(ROOT)
    # Deterministic subset across the acquisition bursts keeps one long session
    # from dominating the shape of the crop classifier.
    selected = {}
    for r in original:
        selected.setdefault(r["objects"][0]["burst"], r)
    older_validation = read_json(ART / "validation_manifest.json")["rows"]
    older_holdout = read_json(HOLDOUT)["rows"]
    training = [*selected.values(), *data["train"], *older_validation, *older_holdout]
    mined = baseline(frozen, stage, data["train"], "precision-training")
    mined.update(baseline(frozen, stage, older_validation, "validation"))
    mined.update(baseline(frozen, stage, older_holdout, "holdout"))
    X, y, weights = [], [], []
    for r in training:
        image = source(r)
        boxes = [o["bbox"] for o in r["objects"]]
        patches = [(crop(image, box), 1) for box in boxes]
        for p in eligible(mined.get(r["id"], []), stage["cutoff"]):
            overlap = max((box_iou(p["bbox"], box) for box in boxes), default=0)
            if overlap >= .5:
                patches.append((crop(image, p["bbox"]), 1))
            elif overlap < .20:
                patches.append((crop(image, p["bbox"]), 0))
        if not boxes:
            patches.append((image, 0))
        images = [view for patch, _ in patches for view in (patch, ImageOps.mirror(patch))]
        if images:
            features = encoder.encode(images)
            X.extend(features)
            y.extend([label for _, label in patches for _ in range(2)])
            weights.extend([1/len(features)] * len(features))
    rows = data["test"]
    predictions = baseline(frozen, stage, rows, "precision-test")
    embeddings = vectors(encoder, rows, predictions, stage["cutoff"])
    before = evaluate(rows, predictions, stage["cutoff"])
    trials = []
    for strength in (1, 10, 100):
        model = LogisticRegression(C=strength, class_weight="balanced", random_state=607)
        model.fit(np.asarray(X), y, sample_weight=weights)
        prototype = dict(weights=model.coef_[0].tolist(), bias=float(model.intercept_[0]))
        values = sorted({float(val) for batch in embeddings.values() for val in
                         classifier_scores(batch, prototype["weights"], prototype["bias"])})
        # Midpoints avoid a threshold exactly on a validation score.
        for cutoff in [values[0]-1, *[(l+r)/2 for l,r in zip(values, values[1:])]]:
            head = dict(**prototype, cutoff=cutoff)
            kept = apply(predictions, embeddings, head, stage["cutoff"])
            result = evaluate(rows, kept, stage["cutoff"])
            if compare(before, result)["passed"]:
                trials.append((result["fp"], -cutoff, strength, head, counters(result)))
    chosen = min(trials, key=lambda item: item[:3])
    write_json(HEAD, chosen[3])
    write_json(RELEASE, dict(id="syringe-crop-verifier-v1", base_sha256=digest(FINAL),
                 manifest_sha256=digest(DATA), fresh_sha256=digest(FRESH), head_sha256=digest(HEAD), backbone_sha256=checksum,
                 code_sha256=digest("medsyringe/verification.py")))
    write_json(ART / "precision_validation.json", dict(
        baseline=counters(before), candidate=chosen[4], fitted_C=chosen[2],
        train_scenes=len(training), calibration_scenes=len(rows), positive_crops=sum(y)//2,
        negative_crops=(len(y)-sum(y))//2, eligible_settings=len(trials)))
    print("Validation", counters(before), chosen[4], flush=True)


def score(frozen, stage, data, encoder):
    release, head = read_json(RELEASE), read_json(HEAD)
    if (release["manifest_sha256"] != digest(DATA) or release["base_sha256"] != digest(FINAL)
        or release["fresh_sha256"] != digest(FRESH)):
        raise ValueError("The reference or base detector changed after selection.")
    if release["head_sha256"] != digest(HEAD):
        raise ValueError("The classifier weights changed after selection.")
    comparisons = {}
    for name, rows, partition in (
        ("fresh", read_json(FRESH)["rows"], "precision-fresh"),
        ("calibration", data["test"], "precision-test"),
        ("fil_rouge", train_rows(ROOT), "train"),
        ("historical", read_json(ART / "test_manifest.json")["rows"], "test"),
        ("previous_external", read_json(HOLDOUT)["rows"], "holdout"),
    ):
        predictions = baseline(frozen, stage, rows, partition)
        embeddings = vectors(encoder, rows, predictions, stage["cutoff"])
        kept = apply(predictions, embeddings, head, stage["cutoff"])
        before = evaluate(rows, predictions, stage["cutoff"])
        after = evaluate(rows, kept, stage["cutoff"])
        comparisons[name] = dict(baseline=before, candidate=after)
        print(name, counters(before), counters(after), flush=True)
    guard = {name: compare(result["baseline"], result["candidate"])
             for name, result in comparisons.items()}
    evidence = ART / "precision_holdout.json"
    allowed = release_allowed(comparisons)
    write_json(evidence, dict(**comparisons, recall_guard=guard,
                             release_sha256=digest(RELEASE), deployed=allowed))
    if allowed:
        write_json(ART / "precision_release.json", {**release, "assessment_sha256": digest(evidence)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "test"])
    args = parser.parse_args()
    data = read_json(DATA)
    frozen, stage = read_json(OLD), read_json(FINAL)
    blocked = [*train_rows(ROOT), *read_json(ART / "validation_manifest.json")["rows"],
               *read_json(ART / "test_manifest.json")["rows"], *read_json(HOLDOUT)["rows"]]
    if leakage([*blocked, *data["train"], *data["test"], *read_json(FRESH)["rows"]]):
        raise ValueError("Dataset partitions are not disjoint.")
    if model_hashes(ROOT, frozen["backbone"]) != frozen["bundle_sha256"]:
        raise ValueError("Detector weights changed.")
    if digest(WEIGHTS) != stage["proposal_heads_sha256"]:
        raise ValueError("Base proposal head weights changed.")
    import torch
    torch.set_num_threads(4)
    checksum = digest(BACKBONE)
    encoder = CropEncoder(Path("."), checksum)
    if args.stage == "fit":
        fit(frozen, stage, data, encoder, checksum)
    else:
        if checksum != read_json(RELEASE)["backbone_sha256"]:
            raise ValueError("The encoder checkpoint changed after selection.")
        score(frozen, stage, data, encoder)


if __name__ == "__main__":
    main()
