"""Fit image landmarks and test their effect on fill and apparent lightness."""
import argparse
import csv
from collections import Counter
from itertools import product
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import ImageOps
from sklearn.linear_model import LogisticRegression

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.measurement.fill import SCALES, calibration, estimate, liquid_box
from medsyringe.measurement.graduation_ocr import supplement_graduations
from medsyringe.measurement.liquid_appearance import appearance
from medsyringe.measurement.liquid_color import describe_region, display_rgb, lab_pixels
from medsyringe.measurement.liquid_region import HEAD, RegionFeatures
from medsyringe.measurement.piston_landmarks import LANDMARKS, features, geometry, lumen_support, refine_rim
from scripts.experiments.expanded.expanded_color_context import controls
from scripts.experiments.expanded.expanded_liquid import ART as EXPANDED, CACHE, mask_syringe, records
from scripts.experiments.expanded.expanded_proposals import image_for
from scripts.experiments.measurement.fill_study import ART, LEGACY_ROWS, prepared, references
from scripts.experiments.measurement.graduation_ocr_study import CACHE as OCR_CACHE, LOCK as OCR_LOCK
from scripts.experiments.expanded.sam3_study import Segmenter, configure_cpu

MARKS = ART / "piston_landmarks.tsv"
LOCK = ART / "piston_selection.json"
AUDIT = ART / "piston_regression.json"
LIGHT = EXPANDED / "lumen_appearance_regression.json"


def inputs():
    files = (MARKS, Path("medsyringe/measurement/piston_landmarks.py"), Path("scripts/experiments/measurement/piston_study.py"),
             EXPANDED / "annotations.json", EXPANDED / "liquid_traces.json",
             Path("artifacts/auxiliary_crop_backbone.json"), Path("artifacts/sam3_checkpoint.json"))
    return {str(path): digest(path) for path in files}


def landmarks():
    with MARKS.open() as file:
        data = list(csv.DictReader((line for line in file if not line.startswith("#")), delimiter="\t"))
    rows, _ = references()
    output = {}
    for row in data:
        identifier = row["id"]
        points = np.array([[float(row[f"{name}_{axis}"]) for axis in "xy"] for name in LANDMARKS])
        if identifier in output or not all(0 <= value <= 1 for value in points.ravel()):
            raise ValueError(f"Invalid landmarks for {identifier}.")
        output[identifier] = (row["role"], rows[identifier], points)
    seen = {}
    for role, row, _ in output.values():
        prior = seen.setdefault(row["group"], role)
        if prior != role:
            raise ValueError(f"Landmark group crosses roles: {row['group']}")
    return output


def frame_for(row):
    path = CACHE / "liquid" / row["split"] / f"{row['id']}.npz"
    with np.load(path) as archive:
        frame = {key: archive[key] for key in archive.files}
    frame["lab"] = lab_pixels(frame["rgb"])
    return frame


def fit(reference, method):
    heads = {}
    locations = (np.moveaxis(np.indices((16, 16)), 0, -1)[..., ::-1]+.5)/16
    counts = Counter(row["group"] for row, _, _ in reference)
    for index, name in enumerate(LANDMARKS):
        matrices, labels, weights = [], [], []
        for row, frame, known in reference:
            h, w = frame["mask"].shape
            distances = np.linalg.norm((locations-known[index])*[w,h], axis=-1)/max(w,h)
            selected = (distances <= .067) | (distances >= .13)
            target = distances[selected] <= .067
            balance = np.bincount(target.astype(int), minlength=2)
            group_weight = 1/counts[row["group"]]
            matrices.extend(features(frame, method)[selected])
            labels.extend(target)
            weights.extend(group_weight/balance[target.astype(int)])
        predictor = LogisticRegression(C=12., max_iter=300, random_state=284)
        weights = np.asarray(weights)
        predictor.fit(matrices, labels, sample_weight=weights/weights.mean())
        heads[name] = dict(weights=predictor.coef_[0].tolist(), bias=float(predictor.intercept_[0]))
    return dict(method=method, heads=heads)


def estimate_landmarks(frame, model, policy):
    item = geometry(frame, model)
    if item is None or item["score"] < policy["min_score"]:
        return None
    return refine_rim(frame, item, policy["drop"]) if policy["drop"] else item


def errors(reference, model, policy):
    cases = {}
    for row, frame, points in reference:
        prediction = estimate_landmarks(frame, model, policy)
        if not prediction:
            cases[row["id"]] = None
            continue
        h, w = frame["mask"].shape
        targets = points*[w,h]
        barrel = np.linalg.norm(targets[2]-targets[0])
        observed = np.array([prediction[k] for k in LANDMARKS])
        cases[row["id"]] = dict(points=(observed/[w,h]).tolist(), reviewed=points.tolist(),
                                normalized_errors=(np.linalg.norm(observed-targets, axis=1)/barrel).tolist(),
                                contrast=prediction.get("verified_contrast"))
    values = [case["normalized_errors"][1] for case in cases.values() if case]
    result = dict(images=len(cases), answered=len(values), within_06=sum(v <= .06 for v in values),
                  within_10=sum(v <= .10 for v in values), median_error=median(values) if values else None,
                  worst_error=max(values, default=None))
    return dict(metrics=result, cases=cases)


def select():
    if LOCK.exists():
        raise ValueError("The landmark component was already frozen.")
    frames = {role: [(row, frame_for(row), points) for _, (purpose, row, points) in landmarks().items()
                     if role == purpose] for role in ("fit", "development")}
    models = {method: fit(frames["fit"], method) for method in ("dense", "dense+color")}
    trials = []
    for method, confidence, drop in product(models, (-1.5, -.5, .5), (0., 7., 11., 15.)):
        policy = dict(method=method, min_score=confidence, drop=drop)
        result = errors(frames["development"], models[method], policy)["metrics"]
        trials.append(dict(policy=policy, metrics=result))
    safe = [trial for trial in trials if trial["metrics"]["answered"] >= 3 and
            trial["metrics"]["answered"] == trial["metrics"]["within_10"]]
    order = lambda t: (t["metrics"]["within_06"], t["metrics"]["within_10"], t["policy"]["drop"] > 0,
                      -(t["metrics"]["median_error"] if t["metrics"]["answered"] else 1.))
    chosen = max(safe, key=order, default=None)
    shadow = chosen or max(trials, key=order)
    write_json(LOCK, dict(inputs=inputs(), selected=chosen, shadow=shadow, trials=trials, models=models,
                          reviewed={role: len(items) for role, items in frames.items()}))
    print("Piston development", shadow, flush=True)


def frozen():
    lock = read_json(LOCK)
    if lock["inputs"] != inputs():
        raise ValueError("The landmark model inputs changed.")
    strategy = lock["shadow"]["policy"]
    return lock, lock["models"][strategy["method"]], strategy


def project_fill(image, texts, liquid, updated_piston, settings):
    cap = (texts.get("capacity") or {}).get("value")
    if cap not in SCALES or not liquid_box(liquid):
        return None
    ref = calibration(texts["candidates"], int(cap), image.size, settings["ocr_min"])
    if not ref:
        return None
    shoulder, piston, flange = updated_piston*np.asarray(image.size)
    vector = flange-shoulder
    alignment = vector@ref["axis"]/np.linalg.norm(vector)
    if alignment < .96:
        return None
    ml = (piston-ref["zero"])@ref["axis"]/ref["pixels_per_ml"]
    low, high = ref["limits"]
    if not max(0., low) <= ml <= min(cap, high):
        return None
    step = SCALES[int(cap)][1]
    return int(round(ml/step)*step)


def cached_fill(source, model, strategy, split):
    rows, _ = references()
    original = {r["id"]: r["bbox"] for r in read_json(LEGACY_ROWS)["files"]}
    settings = read_json(ART / "release.json")["settings"]
    lock = read_json(OCR_LOCK)
    patches = read_json(OCR_CACHE / f"{split}.json")["frames"]
    counts = {key: dict(added=0, in_bracket=0, false_clear=0) for key in ("stroke_ocr", "landmarks", "combined")}
    output = {}
    for name, image, texts, liquid, bracket in prepared(source, split):
        current = estimate(image, texts, liquid, settings)
        if "ml" in current:
            continue
        row = rows[name]
        landmarks_prediction = estimate_landmarks(frame_for(row), model, strategy)
        ocr = supplement_graduations(texts, patches[name], lock["shadow"]["policy"])
        alternative = estimate(image, ocr, liquid, settings)
        estimates = dict(stroke_ocr=alternative.get("ml"), landmarks=None, combined=None)
        if landmarks_prediction:
            h,w = frame_for(row)["mask"].shape
            coords = np.array([landmarks_prediction[k] for k in LANDMARKS])/[w,h]
            if split == "holdout-60" or row["group"] == "amber-20-low-fill-dark-wall":
                x, y, r, b = row["objects"][0]["bbox"]
                ox, oy, ore, ob = original[name]
                coords = (coords*[r-x,b-y] + [x-ox,y-oy])/[ore-ox, ob-oy]
            estimates["landmarks"] = project_fill(image, texts, liquid, coords, settings)
            estimates["combined"] = estimates["stroke_ocr"]
            if estimates["combined"] is None:
                estimates["combined"] = project_fill(image, ocr, liquid, coords, settings)
        elif estimates["stroke_ocr"] is not None:
            estimates["combined"] = estimates["stroke_ocr"]
        for key, value in estimates.items():
            if value is None:
                continue
            counts[key]["added"] += 1
            counts[key]["in_bracket"] += bool(bracket and bracket[0] <= value <= bracket[1])
            counts[key]["false_clear"] += bracket is None
        output[name] = dict(estimates=estimates, reference=bracket, baseline_reason=current["reason"])
    return dict(counts=counts, cases=output)


def audit(source):
    lock, model, strategy = frozen()
    sample = [(row, frame_for(row), points) for _, (role, row, points) in landmarks().items() if role == "regression"]
    replay = errors(sample, model, strategy)
    results = {split: cached_fill(source, model, strategy, split) for split in
               ("development", "holdout", "holdout-60", "clear")}
    stats = replay["metrics"]
    passed = bool(lock["selected"] and stats["answered"] == stats["within_10"] and stats["within_06"] >= 3)
    write_json(AUDIT, dict(passed=passed, landmark_replay=replay, fill=results, selection_sha256=digest(LOCK),
                           scope="Landmark references are approximate visual clicks. Secondary fill projection "
                                 "uses previously evaluated brackets."))
    print("Piston regression", stats, flush=True)
    for name, record in results.items():
        print(name, record["counts"], flush=True)


def appearance_record(frame, model, strategy, legacy):
    logits = cv2.resize(frame["grid"] @ legacy["weights"] + legacy["bias"], frame["mask"].shape[::-1])
    before = appearance(frame["raw"], frame["mask"], logits >= legacy["cutoff"])
    after = before
    point = estimate_landmarks(frame, model, strategy)
    if point:
        lumen = lumen_support(frame, point)
        lab = frame["lab"]
        chroma = np.linalg.norm(lab[..., 1:], axis=-1)
        warm = (chroma >= 14.) & (lab[..., 2] >= 11.) & (lab[..., 0] <= 94.)
        region = lumen & warm & (logits >= legacy["cutoff"])
        if region.sum() < max(45, .22*lumen.sum()):
            region.fill(False)
        after = describe_region(frame["rgb"], region, str(frame["profile"]))
    result = dict(geometry=bool(point), legacy=before, candidate=after, valid_region=frame.get("distance") is not None)
    if result["valid_region"]:
        core = frame["distance"] <= .023
        reference = float(np.median(frame["lab"][core][:,0]))
        for name, obs in (("legacy", before), ("candidate", after)):
            if not obs["lightness"]:
                continue
            h,w = core.shape
            x,y,r,b = obs["region"]
            coverage = core[int(y*h):int(np.ceil(b*h)),int(x*w):int(np.ceil(r*w))].sum()/core.sum()
            if coverage >= .35:
                result[name+"_error"] = abs(obs["lightness"]["lstar"]-reference)
    return result


def appearance_audit():
    _, model, strategy = frozen()
    legacy = read_json(HEAD)
    outputs = {split: {row["id"]: appearance_record(frame, model, strategy, legacy)
                       for row, frame in records(split)} for split in ("calibration", "test")}
    segmenter, encoder = Segmenter(504, "xpu"), RegionFeatures("xpu")
    outside = {}
    for row in controls():
        image = ImageOps.contain(crop(image_for(row, Path(".")), row["objects"][0]["bbox"]), (420,420))
        display, profile = display_rgb(image)
        rgb = np.asarray(display)
        frame = dict(raw=np.asarray(image.convert("RGB")), rgb=rgb, lab=lab_pixels(rgb), profile=profile,
                     grid=encoder.extract(image), mask=mask_syringe(segmenter, image))
        outside[row["id"]] = appearance_record(frame, model, strategy, legacy)
    outputs["public"] = outside
    totals = {}
    for split, cases in outputs.items():
        scores = {}
        for name in ("legacy", "candidate"):
            errors = [r[name+"_error"] for r in cases.values() if name+"_error" in r]
            scores[name] = dict(correct_region=len(errors), median_lstar_error=median(errors) if errors else None,
                                false_appearance=sum(not r["valid_region"] and bool(r[name]["lightness"])
                                                     for r in cases.values()))
        totals[split] = dict(images=len(cases), geometry=sum(r["geometry"] for r in cases.values()), **scores)
    improved = (totals["test"]["candidate"]["false_appearance"] < totals["test"]["legacy"]["false_appearance"]
                and totals["public"]["candidate"]["false_appearance"] == 0)
    preserved = totals["calibration"]["candidate"]["correct_region"] >= totals["calibration"]["legacy"]["correct_region"]
    write_json(LIGHT, dict(passed=improved and preserved, selection_sha256=digest(LOCK), summary=totals,
                           outputs=outputs, scope="Appearance only. Previously consulted clear and public controls."))
    print("Lumen appearance", totals, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "audit", "appearance"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.stage == "select":
        select()
    elif args.stage == "audit" and args.source:
        audit(args.source)
    elif args.stage == "appearance":
        configure_cpu()
        appearance_audit()
    else:
        parser.error("The fill audit requires --source.")
