"""Evaluate an additive below-cutoff recovery on existing grouped proposals."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou, suppress
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.expanded_crop_verification import FLOOR_PATH, decontaminate, examples, features
from scripts.expanded_detection_study import load, short
from scripts.expanded_proposals import ART, cache_file, rows_for
from scripts.rfdetr_verifier import DinoEncoder
from scripts.sam3_study import configure_cpu

RELEASE = ART / "crop_verifier_release.json"
WORK = Path("data/expanded/additive-recovery")
LOCK = ART / "additive_recovery_selection.json"
AUDIT = ART / "additive_recovery_regression.json"
SEED = 264


def fingerprints(split):
    branches = ("xpu", "sam", "fallback-xpu") if split == "calibration" else ("cpu", "sam", "fallback")
    paths = [Path(__file__).relative_to(Path.cwd()), RELEASE, FLOOR_PATH, ART / "annotations.json",
             *[cache_file(split, branch) for branch in branches]]
    return {str(path): digest(path) for path in paths}


def match_score(box, branch, overlap):
    return max((p["score"] for p in branch if box_iou(p["bbox"], box) >= overlap), default=0.)


def candidate_features(proposal, channels, crop_probability):
    box = proposal["bbox"]
    sam = channels["sam"]
    rf = channels["rf"]
    left, top, right, bottom = box
    area = (right-left)*(bottom-top)
    mask_support = max((p["score"] for p in sam if p["mask_bbox"] and box_iou(p["mask_bbox"], box) >= .5 and
                        p["mask_area_fraction"] >= area*.10), default=0.)
    return [crop_probability, proposal["score"], match_score(box, sam, .40), match_score(box, sam, .60),
            match_score(box, rf, .40), match_score(box, rf, .60), mask_support, np.log(max(area, 1e-6))]


def collect(source, split):
    key = fingerprints(split)
    target = WORK / f"{split}.json"
    if target.exists():
        if read_json(target)["inputs"] != key:
            raise ValueError(f"The cached inputs for {split} changed.")
        return
    if split in ("test", "reserve") and not LOCK.exists():
        raise ValueError("Freeze recovery before inspecting the regression groups.")
    floor = read_json(FLOOR_PATH)["cutoff"]
    model = read_json(RELEASE)
    rows, frames = load(split)
    exclusions = {}
    if split in ("fit", "calibration"):
        protected = [*rows_for("test"), *rows_for("reserve")]
        if split == "fit":
            dev, _ = decontaminate(rows_for("calibration"), protected)
            protected.extend(dev)
        rows, exclusions = decontaminate(rows, protected)
    encoder = DinoEncoder()
    accepted = examples(rows, frames, floor, False)
    lows = []
    for row in rows:
        frame = frames[row["id"]]
        pool = suppress([p for p in frame["prior"] if .24 <= p["score"] < floor], .55)[:3]
        lows.extend((row, p, False, 0) for p in pool)
    items = [*accepted, *lows]
    embeddings = features(items, source, encoder)["dino"]
    scores = embeddings @ np.asarray(model["head"]["weights"]) + model["head"]["bias"]
    probabilities = 1/(1+np.exp(-np.clip(scores, -30., 30.)))
    prepared = {row["id"]: {"accepted": [], "rejected": [], "candidates": []} for row in rows}
    for (row, proposal, _, votes), probability in zip(accepted, probabilities[:len(accepted)], strict=True):
        frame = prepared[row["id"]]
        name = "accepted" if probability >= model["cutoff"] or votes >= 2 else "rejected"
        frame[name].append(proposal)
    for (row, proposal, _, _), probability in zip(lows, probabilities[len(accepted):], strict=True):
        key_id = row["id"]
        values = candidate_features(proposal, frames[key_id]["branches"], float(probability))
        iou = max((box_iou(proposal["bbox"], obj["bbox"]) for obj in row["objects"]), default=0.)
        prepared[key_id]["candidates"].append(dict(proposal=proposal, values=values, iou=iou))
    references = [{k: row[k] for k in ("id", "group", "objects")} for row in rows]
    write_json(target, dict(inputs=key, exclusions=exclusions, frames=prepared, references=references))
    print(split, len(accepted), "published candidates and", len(lows), "below-cutoff proposals", flush=True)


def feature_model(dataset, trial):
    items = [(row["group"], p) for row in dataset["references"] for p in
             dataset["frames"][row["id"]]["candidates"] if p["iou"] >= .50 or p["iou"] <= .15]
    strata = Counter((g, p["iou"] >= .50) for g, p in items)
    features = np.array([p["values"][:trial] for _, p in items])
    target = np.array([p["iou"] >= .50 for _, p in items])
    weights = np.array([1/strata[g, p["iou"] >= .50] for g, p in items])
    for label in (True, False):
        weights[target == label] *= len(weights)/(2*weights[target == label].sum())
    scaler = StandardScaler().fit(features, sample_weight=weights)
    fitted = LogisticRegression(C=.5, random_state=SEED).fit(scaler.transform(features), target, sample_weight=weights)
    return dict(weights=fitted.coef_[0].tolist(), bias=float(fitted.intercept_[0]), mean=scaler.mean_.tolist(),
                scale=scaler.scale_.tolist(), features=trial)


def probability(values, model, method):
    if method in ("crop", "crop+sam"):
        return values[0]
    centered = (np.asarray(values[:model["features"]])-model["mean"])/model["scale"]
    logit = np.clip(centered @ model["weights"] + model["bias"], -30., 30.)
    return float(1/(1+np.exp(-logit)))


def decide(frame, policy, models, floor, shadow=False):
    if frame["accepted"] and not shadow:
        return frame["accepted"], None
    supported = []
    method = policy["method"]
    for item in frame["candidates"]:
        p = item["proposal"]
        if any(box_iou(p["bbox"], rejected["bbox"]) >= .50 for rejected in frame["rejected"]):
            continue
        values = item["values"]
        if method == "crop+sam" and max(values[2], values[6]) < policy["sam"]:
            continue
        confidence = probability(values, models.get(method), method)
        if confidence >= policy["confidence"]:
            supported.append((confidence, p["score"], item))
    if not supported:
        return [], None
    confidence, _, winner = max(supported, key=lambda item: item[:2])
    prediction = {**winner["proposal"], "score": floor, "verification_probability": confidence,
                  "original_score": winner["proposal"]["score"], "source": "additive crop recovery"}
    return [prediction], winner


def dataset(split):
    payload = read_json(WORK / f"{split}.json")
    if payload["inputs"] != fingerprints(split):
        raise ValueError(f"Recovery inputs changed for {split}.")
    return payload


def run(dataset, policy, heads, shadow=False):
    floor = read_json(FLOOR_PATH)["cutoff"]
    rows, frames = dataset["references"], dataset["frames"]
    before, after, rescues = {}, {}, {}
    for row in rows:
        identifier = row["id"]
        frame = frames[identifier]
        before[identifier] = frame["accepted"]
        after[identifier], rescue = decide(frame, policy, heads, floor, shadow)
        if rescue:
            rescues[identifier] = dict(iou=rescue["iou"], proposal=after[identifier][0], features=rescue["values"])
    result = evaluate(rows, after, floor)
    baseline = evaluate(rows, before, floor)
    return dict(baseline=short(baseline), candidate=short(result), guard=compare(baseline, result), rescues=rescues)


def select():
    if LOCK.exists():
        raise ValueError("The additive policy was already frozen.")
    training, development = dataset("fit"), dataset("calibration")
    heads = {method: feature_model(training, num) for method, num in (("stacked-basic", 2), ("stacked-hybrid", 8))}
    trials = []
    for method, cutoff, sam in product(("crop", "crop+sam", *heads), (.55, .68, .78, .88, .95), (.15, .35)):
        if method != "crop+sam" and sam == .35:
            continue
        policy = dict(method=method, confidence=cutoff, sam=sam)
        outcome = run(development, policy, heads)
        proxy = run(development, policy, heads, shadow=True)
        # Proxy removes accepted positive controls so that the policy can be
        # assessed on existing low-score evidence without consulting the reserve.
        # Every false addition on a naturally empty image is still counted.
        safe_proxy = all(r["iou"] >= .50 for r in proxy["rescues"].values())
        safe_live = (outcome["guard"]["passed"] and
                     outcome["candidate"]["fp"] == outcome["baseline"]["fp"])
        trials.append(dict(policy=policy, shadow_tp=proxy["candidate"]["tp"], shadow_fp=proxy["candidate"]["fp"],
                           natural_additions=len(outcome["rescues"]), safe_proxy=safe_proxy, safe_live=safe_live))
    order = lambda t: (t["shadow_tp"], t["policy"]["confidence"], t["policy"]["method"] == "crop+sam")
    selected = max((t for t in trials if t["safe_live"] and t["safe_proxy"] and t["shadow_tp"]), key=order, default=None)
    shadow = selected or max(trials, key=lambda t: (-t["shadow_fp"], *order(t)))
    write_json(LOCK, dict(selected=selected, shadow=shadow, trials=trials, models=heads,
                          fit_sha256=digest(WORK / "fit.json"), calibration_sha256=digest(WORK / "calibration.json"),
                          scope="Development rescue simulation by masking published predictions. No reserve data used."))
    print("Recovery calibration", shadow, flush=True)


def audit():
    lock = read_json(LOCK)
    scores = {split: run(dataset(split), lock["shadow"]["policy"], lock["models"])
              for split in ("test", "reserve")}
    safe = all(s["guard"]["passed"] and s["candidate"]["fp"] <= s["baseline"]["fp"] for s in scores.values())
    useful = any(s["guard"]["recovered_matches"] for s in scores.values())
    write_json(AUDIT, dict(passed=bool(lock["selected"] and safe and useful), scores=scores,
                           selection_sha256=digest(LOCK), scope="Previously consulted regression cohorts."))
    for k, v in scores.items():
        print(k, v["candidate"], v["guard"], {i: r["iou"] for i, r in v["rescues"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "select", "audit"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--split", choices=("fit", "calibration", "test", "reserve"))
    args = parser.parse_args()
    configure_cpu()
    if args.stage == "collect":
        if args.source is None or args.split is None:
            parser.error("Collection requires --source and --split.")
        collect(args.source, args.split)
    else:
        (select if args.stage == "select" else audit)()
