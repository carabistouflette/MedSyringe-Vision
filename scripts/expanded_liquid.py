"""Fit color-aware dense localizers and compare them to the legacy liquid measurement."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np
from PIL import ImageOps
from sklearn.linear_model import LogisticRegression

from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.liquid_appearance import appearance
from medsyringe.liquid_color import describe_region, display_rgb, lab_pixels
from medsyringe.liquid_region import HEAD, RegionFeatures
from scripts.expanded_annotations import ANNOTATIONS, ART
from scripts.expanded_proposals import CACHE, image_for
from scripts.liquid_fit import distance

TRACE = ART / "liquid_traces.json"
LOCK = ART / "liquid_freeze.json"


def mask_syringe(segmenter, image):
    import torch

    with torch.inference_mode():
        inputs = segmenter.processor(images=image, text="syringe",
                  size={"height": segmenter.edge, "width": segmenter.edge}, return_tensors="pt").to(segmenter.device)
        result = segmenter.processor.post_process_instance_segmentation(segmenter.model(**inputs), threshold=.48)[0]
    options = [(float(score)*float(mask.mean()), mask) for score, mask in
               zip(result["scores"].cpu().numpy(), result["masks"].cpu().numpy(), strict=True)
               if .15 <= mask.mean() <= .95]
    if not options:
        return np.zeros((image.height, image.width), dtype=np.uint8)
    return cv2.resize(max(options, key=lambda x: x[0])[1].astype(np.uint8), image.size,
                      interpolation=cv2.INTER_NEAREST)


def reference(row, point_list, shape):
    width, height = shape[1], shape[0]
    l, t, r, b = row["objects"][0]["bbox"]
    points = [[(x-l)/(r-l), (y-t)/(b-t)] for x, y in point_list]
    return distance(points, shape, width, height)


def features(grid, lab, method):
    if method == "dense":
        return grid
    scaled = cv2.resize(lab/np.array([100., 80., 80.], dtype=np.float32), (16, 16), interpolation=cv2.INTER_AREA)
    return np.concatenate((grid, scaled), axis=-1)


def fingerprint():
    paths = (ANNOTATIONS, TRACE, HEAD, Path(__file__), Path("scripts/sam3_study.py"),
             Path("artifacts/sam3_checkpoint.json"), Path("artifacts/auxiliary_crop_backbone.json"),
             Path("medsyringe/liquid_color.py"))
    return {str(p): digest(p) for p in paths}


def collect(source, split):
    from scripts.sam3_study import Segmenter, configure_cpu

    configure_cpu()
    if split == "test" and not LOCK.is_file():
        raise ValueError("Freeze the appearance estimator before running held-out cases.")
    rows = [r for r in read_json(ANNOTATIONS)["rows"] if r["split"] == split]
    key = fingerprint()
    folder = CACHE / "liquid" / split
    marker = folder / "inputs.json"
    settings = read_json(marker) if marker.exists() else {}
    folder.mkdir(parents=True, exist_ok=True)
    if settings != key:
        for file in folder.glob("*.npz"):
            file.unlink()
        write_json(marker, key)
    segmenter, encoder = None, None
    for index, row in enumerate(rows, 1):
        output = folder / f"{row['id']}.npz"
        if output.exists():
            continue
        if segmenter is None:
            segmenter, encoder = Segmenter(504, "xpu"), RegionFeatures("xpu")
        tick = perf_counter()
        image = ImageOps.contain(crop(image_for(row, source), row["objects"][0]["bbox"]), (420, 420))
        screen, profile = display_rgb(image)
        mask = mask_syringe(segmenter, image)
        grid = encoder.extract(image)
        np.savez_compressed(output, raw=np.asarray(image), rgb=np.asarray(screen), grid=grid, mask=mask, profile=profile)
        if index % 8 == 0 or index == 1:
            print(split, index, row["id"], round(perf_counter()-tick, 2), flush=True)


def records(split):
    rows = [r for r in read_json(ANNOTATIONS)["rows"] if r["split"] == split]
    marks = read_json(TRACE)["regions"]
    location = CACHE / "liquid" / split
    if read_json(location / "inputs.json") != fingerprint():
        raise ValueError("Appearance cache inputs changed.")
    for row in rows:
        with np.load(location / f"{row['id']}.npz") as payload:
            frame = {k: payload[k] for k in payload.files}
        frame["lab"] = lab_pixels(frame["rgb"])
        points = marks.get(row["id"], {}).get("points")
        frame["distance"] = reference(row, points, frame["mask"].shape) if points else None
        yield row, frame


def train(rows, method, regularization):
    samples, labels, weights = [], [], []
    for row, frame in rows:
        grid = features(frame["grid"], frame["lab"], method)
        if frame["distance"] is None:
            selection = cv2.resize(frame["mask"], (16, 16), interpolation=cv2.INTER_AREA) > 0
            target = np.zeros((16, 16), dtype=bool)
        else:
            geometry = cv2.resize(frame["distance"], (16, 16), interpolation=cv2.INTER_AREA)
            target = geometry <= .045
            selection = target | (geometry >= .105)
        if not selection.any():
            continue
        group_weight = 1./sum(other["group"] == row["group"] for other, _ in rows)
        certain = target[selection]
        balance = np.bincount(certain.astype(int), minlength=2).clip(1)
        samples.extend(grid[selection])
        labels.extend(certain)
        weights.extend(group_weight/balance[certain.astype(int)])
    X, y, w = np.asarray(samples), np.asarray(labels), np.asarray(weights)
    for label in (True, False):
        w[y == label] *= len(w)/(2*w[y == label].sum())
    fit = LogisticRegression(C=regularization, random_state=617, max_iter=400).fit(X, y, sample_weight=w)
    return dict(weights=fit.coef_[0].tolist(), bias=float(fit.intercept_[0]), method=method, c=regularization)


def support(frame, model, cutoff):
    grid = features(frame["grid"], frame["lab"], model["method"])
    logits = cv2.resize(grid @ model["weights"] + model["bias"], frame["mask"].shape[::-1])
    chroma = np.linalg.norm(frame["lab"][..., 1:], axis=-1)
    interior = frame["mask"].astype(bool) & (logits >= cutoff) & (chroma >= 13.)
    linked = cv2.morphologyEx(interior.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, membership, stats, _ = cv2.connectedComponentsWithStats(linked)
    if count < 2:
        return np.zeros_like(interior)
    chosen = interior & (membership == 1+np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if chosen.sum() < max(90, .045*frame["mask"].sum()):
        return np.zeros_like(interior)
    return chosen


def assess(rows, method, model=None, cutoff=None):
    output = {}
    legacy = read_json(HEAD) if method == "legacy" else None
    for row, frame in rows:
        distances = frame["distance"]
        raw = frame["raw"]
        if method == "legacy":
            logits = cv2.resize(frame["grid"] @ legacy["weights"] + legacy["bias"], frame["mask"].shape[::-1])
            measurement = appearance(raw, frame["mask"], logits >= legacy["cutoff"])
        elif method == "color_only":
            measurement = appearance(raw, frame["mask"])
        else:
            interior = support(frame, model, cutoff)
            measurement = describe_region(frame["rgb"], interior, str(frame["profile"]))
        reference_lab = None
        if distances is not None:
            reference_lab = np.median(frame["lab"][distances < .023], axis=0).tolist()
        record = dict(measurement=measurement, reference_lab=reference_lab, supported=False,
                      false_measurement=not bool(reference_lab) and bool(measurement["lightness"]))
        if reference_lab and measurement["lightness"]:
            height, width = distances.shape
            l, t, r, b = measurement["region"]
            window = distances[int(t*height):int(np.ceil(b*height)), int(l*width):int(np.ceil(r*width))]
            coverage = float((window < .023).sum()/max(1, (distances < .023).sum()))
            record.update(supported=coverage >= .35, centerline_box_coverage=round(coverage, 4),
                          lstar_error=abs(measurement["lightness"]["lstar"]-reference_lab[0]))
            if measurement.get("apparent_color"):
                record["lab_delta_e76"] = float(np.linalg.norm(np.asarray(measurement["apparent_color"]["lab"])-reference_lab))
        output[row["id"]] = record
    errors = [o["lstar_error"] for o in output.values() if o["supported"]]
    color_errors = [o["lab_delta_e76"] for o in output.values() if o["supported"] and "lab_delta_e76" in o]
    return dict(images=len(rows), answered=sum(bool(o["measurement"]["lightness"]) for o in output.values()),
                supported=sum(o["supported"] for o in output.values()),
                uncolored_answered=sum(o["false_measurement"] for o in output.values()),
                median_lstar_error=median(errors) if errors else None,
                median_delta_e76=median(color_errors) if color_errors else None), output


def select():
    if LOCK.exists():
        raise ValueError("The appearance candidate is already frozen.")
    fit, valid = list(records("fit")), list(records("calibration"))
    baseline, legacy_frames = assess(valid, "legacy")
    color_only, _ = assess(valid, "color_only")
    trials = []
    for family in ("dense", "dense_color"):
        for c in (.25, 2.):
            model = train(fit, family, c)
            for cutoff in (-.45, .0, .45, .9, 1.35, 1.8):
                metrics, _ = assess(valid, family, model, cutoff)
                trials.append(dict(model=model, cutoff=cutoff, metrics=metrics))
    trials.sort(key=lambda trial: (trial["metrics"]["supported"],
                 -(trial["metrics"]["median_lstar_error"] or 100.),
                 -(trial["metrics"]["median_delta_e76"] or 100.)), reverse=True)
    winner = trials[0]
    metrics, details = assess(valid, winner["model"]["method"], winner["model"], winner["cutoff"])
    fit_uncolored = [(r, f) for r, f in fit if r["observed_condition"] == "uncolored"]
    fit_control, _ = assess(fit_uncolored, winner["model"]["method"], winner["model"], winner["cutoff"])
    write_json(ART / "liquid_calibration.json", dict(legacy=baseline, color_only=color_only, selected=metrics,
               trials=[dict(method=t["model"]["method"], c=t["model"]["c"], cutoff=t["cutoff"], metrics=t["metrics"])
                       for t in trials], legacy_frames=legacy_frames, selected_frames=details, fit_clear_control=fit_control))
    write_json(LOCK, dict(fingerprints=fingerprint(), winner=winner, fit_clear_control=fit_control,
                          calibration_sha256=digest(ART / "liquid_calibration.json")))
    print("Liquid calibration", "legacy", baseline, "color only", color_only, "candidate", metrics,
          "fit clear", fit_control, flush=True)


def audit():
    frozen = read_json(LOCK)
    if frozen["fingerprints"] != fingerprint():
        raise ValueError("Appearance inputs changed after selection.")
    held = list(records("test"))
    selected = frozen["winner"]
    old, old_cases = assess(held, "legacy")
    new, new_cases = assess(held, selected["model"]["method"], selected["model"], selected["cutoff"])
    write_json(ART / "liquid_test.json", dict(legacy=old, candidate=new, legacy_frames=old_cases,
               candidate_frames=new_cases, freeze_sha256=digest(LOCK)))
    print("Liquid clear holdout", old, new, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("stage", choices=("fit", "calibration", "test", "select", "audit"))
    args = parser.parse_args()
    if args.stage == "select":
        select()
    elif args.stage == "audit":
        audit()
    else:
        collect(args.source, args.stage)


if __name__ == "__main__":
    main()
