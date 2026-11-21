"""Learn a dense DINOv2 foreground proposal map from weak box supervision."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np
from PIL.Image import Resampling
from sklearn.linear_model import LogisticRegression

from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.detectors import box_iou, suppress
from medsyringe.evaluate import evaluate
from scripts.detection_context_study import inputs as contextual
from scripts.detection_guided_geometry import references
from scripts.detection_hillclimb_experiment import ART, DATA, inputs
from scripts.detection_real_box_transfer import benchmark_sources
from scripts.expanded_detection_study import short
from scripts.precision_study import source

MODEL = Path("data/models/dinov2-small")
LOCK = Path("artifacts/auxiliary_crop_backbone.json")
OUTPUT = ART / "dense_appearance_development.json"
HEAD = ART / "dense_appearance_head.json"
CACHE = DATA / "proposals/dense-appearance-development.json"
PATCH = 14
EDGE = 392
SEED = 607
THRESHOLDS = (.40, .55, .70, .85)
PADDING = (0., .15)


def data():
    real, _, old = benchmark_sources()
    splits = {
        "public": inputs("public-development"),
        "dental": [r for r in old if r["domain"] == "dental-development"],
        "transparent": contextual(),
    }
    all_rows = [*({**r, "split": "fit"} for r in real),
                *({**r, "split": "development"} for rows in splits.values() for r in rows)]
    if leakage(all_rows):
        raise ValueError("Weak patch supervision shares a protected source.")
    return real, splits


def labels(rows):
    return {row["id"]: [row["sha256"], row["group"],
                        [o["bbox"] for o in row["objects"]],
                        [o["bbox"] for o in row.get("lookalikes", [])]]
            for row in rows}


class Encoder:
    def __init__(self):
        import torch
        from transformers import Dinov2Model

        if any(digest(MODEL / name) != value for name, value in read_json(LOCK)["files"].items()):
            raise ValueError("Dense appearance requires the pinned DINOv2 weights.")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        self.torch = torch
        self.model = Dinov2Model.from_pretrained(MODEL, local_files_only=True).eval()

    def __call__(self, image):
        target = tuple(max(4, round(value * EDGE / max(image.size) / PATCH)) for value in image.size)
        width, height = target
        pixels = np.asarray(image.resize((width * PATCH, height * PATCH), Resampling.BICUBIC),
                            dtype=np.float32) / 255
        pixels = ((pixels - np.float32([.485, .456, .406])) / np.float32([.229, .224, .225]))
        with self.torch.inference_mode():
            tensor = self.torch.tensor(pixels.transpose(2, 0, 1))[None]
            vectors = self.model(tensor).last_hidden_state[0, 1:]
            vectors = self.torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy()
        return vectors.reshape(height, width, -1)


def locations(shape):
    height, width = shape[:2]
    y, x = np.mgrid[:height, :width]
    return (x + .5) / width, (y + .5) / height


def covers(x, y, box, scale):
    left, top, right, bottom = box
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    return ((np.abs(x - center_x) <= (right - left) * scale / 2) &
            (np.abs(y - center_y) <= (bottom - top) * scale / 2))


def fitting(frame, row):
    x, y = locations(frame.shape)
    positive = np.zeros(x.shape, dtype=bool)
    uncertain = positive.copy()
    for annotation in row["objects"]:
        box = annotation["bbox"]
        center = covers(x, y, box, .45)
        positive |= center
        uncertain |= covers(x, y, box, 1.30)
        if not center.any():
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            positive.flat[np.argmin((x - cx)**2 + (y - cy)**2)] = True
    distractor = np.zeros(x.shape, dtype=bool)
    for annotation in row.get("lookalikes", []):
        distractor |= covers(x, y, annotation["bbox"], .90)
    return frame[positive], frame[~uncertain & ~positive], frame[distractor & ~uncertain & ~positive]


def training(encode, rows):
    rng = np.random.default_rng(SEED)
    groups = Counter(row["group"] for row in rows)
    samples, targets, weights = [], [], []
    for number, row in enumerate(rows, start=1):
        entry = fitting(encode(source(row)), row)
        for features, target, maximum, importance in zip(entry, (1, 0, 0), (30, 18, 24), (1., 1., 2.), strict=True):
            count = min(maximum, len(features))
            if not count:
                continue
            indices = rng.choice(len(features), count, replace=False)
            samples.append(features[indices])
            targets.extend([target] * count)
            weights.extend([importance / (count * groups[row["group"]])] * count)
        if number % 20 == 0:
            print("Dense patch fit", number, "/", len(rows), flush=True)
    weight = np.asarray(weights)
    weight *= len(weight) / weight.sum()
    model = LogisticRegression(C=1., max_iter=300, class_weight="balanced", random_state=SEED)
    model.fit(np.concatenate(samples), targets, sample_weight=weight)
    result = dict(coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                  fitting_groups=len(groups), fitting_images=len(rows), samples=len(targets),
                  positive_patches=sum(targets), inputs=labels(rows),
                  weights_sha256=digest(LOCK), implementation_sha256=digest(__file__))
    write_json(HEAD, result)
    return result


def candidates(probabilities, cutoff, padding):
    height, width = probabilities.shape
    bitmap = (probabilities >= cutoff).astype(np.uint8)
    # Close single-patch gaps in elongated foreground responses.
    closed = cv2.morphologyEx(bitmap, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, components, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    proposals = []
    for index, (x, y, w, h, size) in enumerate(stats[1:], start=1):
        if size < 2 or w * h > .80 * width * height:
            continue
        eligible = probabilities[(components == index) & (bitmap != 0)]
        if len(eligible) < 2:
            continue
        score = float(np.mean(np.sort(eligible)[-6:]))
        box = [max(0., (x-padding*w)/width), max(0., (y-padding*h)/height),
               min(1., (x+w*(1+padding))/width), min(1., (y+h*(1+padding))/height)]
        proposals.append(dict(bbox=box, score=score))
    return suppress(proposals, .45)[:8]


def run():
    fit, dev = data()
    encode = Encoder()
    head = read_json(HEAD) if HEAD.exists() else training(encode, fit)
    if (head["inputs"] != labels(fit) or
            head["implementation_sha256"] != digest(__file__) or head["weights_sha256"] != digest(LOCK)):
        raise ValueError("The patch head training dependencies have changed.")
    binding = dict(head=digest(HEAD), refs={scope: labels(rows) for scope, rows in dev.items()})
    saved = read_json(CACHE) if CACHE.exists() else dict(binding=binding, frames={})
    if saved["binding"] != binding:
        raise ValueError("The candidate cache is tied to another model or image cohort.")
    for scope, rows in dev.items():
        frames = saved["frames"].setdefault(scope, {})
        for row in rows:
            name = row["id"]
            if name in frames:
                continue
            start = perf_counter()
            logits = encode(source(row)) @ head["coefficients"] + head["intercept"]
            probabilities = 1. / (1. + np.exp(-logits.clip(-30, 30)))
            frames[name] = dict(heat=probabilities.tolist(), seconds=perf_counter() - start)
            write_json(CACHE, saved)
            print("Dense patch development", scope, len(frames), "/", len(rows), flush=True)


def measure():
    if OUTPUT.exists():
        raise ValueError("This development comparison already exists.")
    _, groups = data()
    saved = read_json(CACHE)
    if saved["binding"] != dict(head=digest(HEAD), refs={key: labels(rows) for key, rows in groups.items()}):
        raise ValueError("The dense appearance benchmark references changed.")
    previous = references("public-development")
    results = []
    for threshold, padding in product(THRESHOLDS, PADDING):
        outcomes = {}
        for scope, rows in groups.items():
            frames = saved["frames"][scope]
            if set(frames) != {r["id"] for r in rows}:
                raise ValueError("A complete development cohort is required.")
            ranks = {k: candidates(np.asarray(v["heat"]), threshold, padding) for k, v in frames.items()}
            potential = 0
            if scope == "public":
                misses = evaluate(rows, previous, 0.)["false_negatives"]
                potential = sum(any(box_iou(p["bbox"], m["box"]) >= .5 for p in ranks[m["id"]])
                                for m in misses)
            outcomes[scope] = dict(metrics=short(evaluate(rows, ranks, 0.)),
                                   recoverable_public_misses=potential if scope == "public" else None,
                                   median_seconds=median(v["seconds"] for v in frames.values()))
        results.append(dict(patch_cutoff=threshold, padding=padding, outcomes=outcomes))
    write_json(OUTPUT, dict(comparisons=results, weights_sha256=digest(HEAD), cache_sha256=digest(CACHE),
                            notes="Weak patch supervision from independent real fit groups only. "
                                  "Candidates are measured on three development sets. "
                                  "Oracle proposal reach permits false proposals and cannot qualify a release."))
    print("Evaluated", len(results), "dense appearance controls", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "measure"))
    (run if parser.parse_args().action == "run" else measure)()
