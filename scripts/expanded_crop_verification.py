"""Train a hard-negative crop verifier and enforce instance-level recall."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageOps
from sklearn.linear_model import LogisticRegression

from medsyringe.attributes import crop
from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from medsyringe.verification import eligible
from scripts.expanded_detection_study import load, short
from scripts.expanded_proposals import ART, cache_file, image_for, rows_for
from scripts.rfdetr_verifier import DINOV2_INFO, DinoEncoder, expanded
from scripts.sam3_study import configure_cpu

LOCK = ART / "crop_verification_selection.json"
AUDIT = ART / "crop_verification_regression.json"
FLOOR_PATH = Path("artifacts/detector_release.json")
SEED = 617


def locked_inputs():
    caches = [cache_file(cohort, branch) for cohort, branches in
              (("fit", ("cpu", "sam", "fallback")), ("calibration", ("xpu", "sam", "fallback-xpu")))
              for branch in branches]
    files = (__file__, FLOOR_PATH, DINOV2_INFO, ART / "annotations.json", ART / "inventory.json",
             ART / "release.json", "artifacts/precision_study_manifest.json", "artifacts/sam3_reserve.json",
             "artifacts/rfdetr_pen_manifest.json", "scripts/rfdetr_verifier.py", *caches)
    return {str(p): digest(p) for p in files}


def decontaminate(rows, against):
    """Drop whole groups with duplicate or shared-provenance matches."""
    labeled = [{**r, "split": "active"} for r in rows]
    reference = [{**r, "split": "protected"} for r in against]
    by_id = {r["id"]: r["group"] for r in rows}
    collision = leakage([*labeled, *reference])
    groups = {by_id[pair["images"][0]] for pair in collision}
    return [r for r in rows if r["group"] not in groups], dict(groups=sorted(groups), collisions=collision)


def examples(rows, frames, floor, training):
    result = []
    for row in rows:
        frame = frames[row["id"]]
        accepted = eligible(frame["prior"], floor)
        local = list(accepted)
        if training:
            extras = sorted((p for branch in frame["branches"].values() for p in branch),
                            key=lambda p: -p["score"])
            for candidate in extras:
                if len(local) >= len(accepted)+5:
                    break
                if all(box_iou(candidate["bbox"], p["bbox"]) < .85 for p in local):
                    local.append(candidate)
        for proposal in local:
            box = proposal["bbox"]
            impact = max((box_iou(box, t["bbox"]) for t in row["objects"]), default=0.)
            if training and .15 < impact < .5:
                continue
            votes = sum(any(p["score"] >= .35 and box_iou(box, p["bbox"]) >= .45 for p in proposals)
                        for proposals in frame["branches"].values())
            result.append((row, proposal, impact >= .5, votes))
    return result


def features(items, source, encoder):
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (16, 16), (8, 8), 9)
    appearances, histograms, padded = [], [], []
    prev, image = None, None
    for row, proposal, _, _ in items:
        if row["id"] != prev:
            image = image_for(row, source)
            prev = row["id"]
        patch = crop(image, expanded(proposal["bbox"], .08))
        pixels = np.asarray(ImageOps.pad(patch, (64, 64), color=(124, 116, 104)))
        gradient = hog.compute(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)).ravel()
        histograms.append(gradient/max(np.linalg.norm(gradient), 1e-8))
        padded.append(patch)
        if len(padded) == 16:
            appearances.extend(encoder.encode(padded))
            padded.clear()
    if padded:
        appearances.extend(encoder.encode(padded))
    embeddings, gradients = np.array(appearances), np.array(histograms)
    return dict(dino=embeddings, hog=gradients, fused=np.column_stack((embeddings, gradients)))


def fit_head(matrix, items):
    labels = np.array([value for _, _, value, _ in items])
    strata = Counter((row["group"], value) for row, _, value, _ in items)
    weights = np.array([1/strata[row["group"], value] for row, _, value, _ in items])
    for mask in (labels, ~labels):
        weights[mask] *= len(items)/(2*weights[mask].sum())
    model = LogisticRegression(C=2, random_state=SEED).fit(matrix, labels, sample_weight=weights)
    return dict(weights=model.coef_[0].tolist(), bias=float(model.intercept_[0]))


def predictions(rows, items, matrix, head, cutoff, protected_votes):
    scores = matrix@np.asarray(head["weights"]) + head["bias"]
    probabilities = 1/(1+np.exp(-np.clip(scores, -30., 30.)))
    answer = {row["id"]: [] for row in rows}
    filtered = []
    for probability, (row, proposal, _, votes) in zip(probabilities, items, strict=True):
        if probability >= cutoff or votes >= protected_votes:
            answer[row["id"]].append(proposal)
        else:
            filtered.append(dict(id=row["id"], bbox=proposal["bbox"], verification_score=float(probability),
                                 branch_votes=votes))
    return answer, filtered


def select(source):
    if LOCK.exists():
        raise ValueError("The crop verifier is already frozen.")
    if "detection" in read_json(ART / "release.json")["features"]:
        raise ValueError("This study expects the currently deployed baseline detector.")
    floor = read_json(FLOOR_PATH)["cutoff"]
    train, training = load("fit")
    development, calibration = load("calibration")
    protected = rows_for("test") + rows_for("reserve")
    development, dev_exclusions = decontaminate(development, protected)
    train, fit_exclusions = decontaminate(train, [*development, *protected])
    fit_items = examples(train, training, floor, True)
    dev_items = examples(development, calibration, floor, False)
    print("Fit", len(train), "images,", len(fit_items), "crops.", flush=True)
    encoder = DinoEncoder()
    inputs = features(fit_items, source, encoder)
    validation = features(dev_items, source, encoder)
    heads = {key: fit_head(matrix, fit_items) for key, matrix in inputs.items()}
    baseline = evaluate(development, {row["id"]: calibration[row["id"]]["prior"] for row in development}, floor)
    trials = []
    for key, cutoff, votes in product(heads, (.05, .10, .20, .35, .50), (2, 4)):
        results, _ = predictions(development, dev_items, validation[key], heads[key], cutoff, votes)
        result = evaluate(development, results, floor)
        trials.append(dict(method=key, cutoff=cutoff, protected_votes=votes, counts=short(result),
                           guard=compare(baseline, result)))
    valid = [t for t in trials if t["guard"]["passed"] and t["counts"]["fp"] < baseline["fp"]]
    choice = max(valid, key=lambda t: (-t["counts"]["fp"], t["counts"]["ap50_95"], -t["cutoff"]), default=None)
    shadow = max(trials, key=lambda t: (t["guard"]["passed"], -t["counts"]["fp"], -t["cutoff"]))
    write_json(LOCK, dict(fit=dict(images=len(train), groups=len({r["group"] for r in train}),
                                  crops=len(fit_items), positives=sum(i[2] for i in fit_items)),
                         exclusion=dict(fit=fit_exclusions, development=dev_exclusions), inputs=locked_inputs(),
                         baseline=short(baseline), trials=trials, heads=heads, selected=choice,
                         shadow=choice or shadow))
    print("Calibration", short(baseline), choice or shadow, flush=True)


def audit(source):
    freeze = read_json(LOCK)
    if freeze["inputs"] != locked_inputs():
        raise ValueError("Verifier inputs changed since calibration.")
    encoder = DinoEncoder()
    floor = read_json(FLOOR_PATH)["cutoff"]
    choice = freeze["shadow"]
    key = choice["method"]
    scopes = {}
    for split in ("test", "reserve"):
        rows, frames = load(split)
        baseline = evaluate(rows, {k: f["prior"] for k, f in frames.items()}, floor)
        items = examples(rows, frames, floor, False)
        matrix = features(items, source, encoder)[key]
        candidate, filtered = predictions(rows, items, matrix, freeze["heads"][key], choice["cutoff"],
                                          choice["protected_votes"])
        after = evaluate(rows, candidate, floor)
        scopes[split] = dict(baseline=short(baseline), candidate=short(after), guard=compare(baseline, after),
                             removed=filtered)
        print(split, short(baseline), short(after), scopes[split]["guard"], flush=True)
    safe = all(o["guard"]["passed"] and o["candidate"]["fp"] <= o["baseline"]["fp"] for o in scopes.values())
    improved = any(o["candidate"]["fp"] < o["baseline"]["fp"] for o in scopes.values())
    write_json(AUDIT, dict(passed=bool(freeze["selected"] and safe and improved),
                           selection_sha256=digest(LOCK), cases=scopes,
                           scope="Historically consulted regression cohorts; no new independent test."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "audit"))
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    configure_cpu()
    (select if args.stage == "select" else audit)(args.source)


if __name__ == "__main__":
    main()
