"""Fit a pair classifier for duplicate, attached fragment, distinct syringe and distractor."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.additive import overlap_fraction
from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detectors import box_iou
from scripts.native_candidate_bank import archive, cohorts
from scripts.native_candidate_verifier import reference_index
from scripts.occlusion_detector import ART, raster
from scripts.rfdetr_verifier import expanded
from scripts.thin_syringe_tiles import WORK

REVIEW = ART / "syringe_assembly_fit_review.json"
MODEL = ART / "syringe_assembly_pair_model.json"
SEED = 631
# Reviewed labels on the fixed crop sheets generated from the fitting-only proposal bank.
VISUAL = {
    "fragment": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 24, 25, 26, 27, 30, 31, 32, 33],
    "same": [16, 28, 29],
    "distractor": [22, 23],
}
MODEL_ARMS = ("boxes", "pixels")


def freeze_review():
    archived = read_json(WORK / "fragment-review-map.json")
    marks = {i: key for key, indices in VISUAL.items() for i in indices}
    records = [{**r, "label": marks[r["index"]]} for r in archived if r["index"] in marks]
    if {r["index"] for r in records} != set(marks):
        raise ValueError("The crop sheet and visual review list differ.")
    write_json(REVIEW, dict(rows=records, map_sha256=digest(WORK / "fragment-review-map.json"),
                            unreviewed=[15, 17],
                            policy="A linked safety cap, needle, barrel fragment or plunger remains part of one assembly. "
                                   "Detached accessories and container regions are distractors."))


def center(box):
    return (box[:2]+box[2:])/2


def corners(box):
    l, t, r, b = box
    return np.asarray(((l, t), (l, b), (r, t), (r, b)))


def around(parent, shift, zoom):
    box = np.asarray(parent)
    middle = center(box)+(box[2:]-box[:2])*np.asarray(shift)
    radii = (box[2:]-box[:2])*np.asarray(zoom)/2
    return np.r_[middle-radii, middle+radii].clip(0, 1).tolist()


def unit_axis(obj, size):
    """Use manually typed anchors where available and elongated bounds elsewhere."""
    dims = np.asarray(size)
    parts = obj.get("parts", {})
    for first, second in (("barrel", "rod"), ("barrel", "flange"), ("barrel", "seal")):
        if parts.get(first) is not None and parts.get(second) is not None:
            delta = (np.asarray(parts[first])-parts[second])*dims
            if np.linalg.norm(delta) > 4:
                return delta/np.linalg.norm(delta)
    box = np.asarray(obj["bbox"])
    spans = (box[2:]-box[:2])*dims
    if spans.max() < 1.65*spans.min():
        return None
    return np.eye(2)[int(spans.argmax())]


def fragment_views(obj, size):
    axis = unit_axis(obj, size)
    if axis is None:
        return []
    dims = np.asarray(size)
    box = np.asarray(obj["bbox"])*np.tile(dims, 2)
    origin = center(box)
    cross = np.array([-axis[1], axis[0]])
    shape = corners(box)-origin
    axial = np.max(np.abs(shape @ axis))
    width = min(np.max(np.abs(shape @ cross)), .35*axial)
    candidates = []
    for along, half, transverse in ((-.70, .29, .0), (.70, .29, .0), (.88, .30, .08), (0., .32, .0)):
        middle = origin+along*axial*axis+transverse*width*cross
        points = np.array([middle+sign*half*axial*axis+wind*.60*width*cross
                           for sign in (-1, 1) for wind in (-1, 1)])
        proposal = np.r_[points.min(axis=0)/dims, points.max(axis=0)/dims].clip(0, 1).tolist()
        if valid_box(proposal) and .015 <= box_iou(obj["bbox"], proposal) < .48:
            candidates.append(proposal)
    return candidates


def descriptor(image, bbox):
    """Measure visible long edges and pixel statistics in an unwarped source crop."""
    region = crop(image, expanded(bbox, .04))
    spans = np.asarray(region.size)
    region.thumbnail((160, 160), Image.Resampling.BILINEAR)
    rgb = np.asarray(region).astype(np.float32)/255.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if min(gray.shape) < 3:
        axis = np.eye(2)[int(spans.argmax())]
        reliability = 0.
        energy = 0.
    else:
        # Trimming crop-edge pixels reduces padding and exposure-boundary effects.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
        tensor = [[float((gx*gx).mean()), float((gx*gy).mean())],
                  [float((gx*gy).mean()), float((gy*gy).mean())]]
        eigenvalues, vectors = np.linalg.eigh(tensor)
        energy = float(eigenvalues.sum())
        reliability = float((eigenvalues[1]-eigenvalues[0])/max(energy, .001))
        axis = vectors[:, 0] if reliability >= .15 else np.eye(2)[int(spans.argmax())]
    return dict(axis=axis, reliability=reliability, energy=energy,
                color=rgb.reshape(-1, 3).mean(axis=0), lightness=float(gray.std()))


def features(first, second, size, left=None, right=None):
    a, b = (np.asarray(box, dtype=float) for box in (first, second))
    scale = np.asarray(size, dtype=float)
    widths = [(box[2:]-box[:2])*scale for box in (a, b)]
    d = (center(b)-center(a))*scale
    ratio = float(np.prod(widths[1])/np.prod(widths[0]))
    overlap = box_iou(a, b)
    smaller = overlap_fraction(a, b)
    result = [overlap, smaller, float(np.log(ratio)),
              *list(np.abs(d)/np.maximum(widths[0], 1.)),
              *[float(np.log(w[0]/w[1])) for w in widths]]
    if left is not None and right is not None:
        axis = left["axis"]
        across = np.array([-axis[1], axis[0]])
        extent = max(float(np.linalg.norm(widths[0])), 1.)
        result.extend((abs(float(d @ axis))/extent, abs(float(d @ across))/extent,
                       abs(float(axis @ right["axis"])), left["reliability"], right["reliability"],
                       np.linalg.norm(left["color"]-right["color"]), left["lightness"], right["lightness"],
                       abs(float(np.log((left["energy"]+.03)/(right["energy"]+.03))))))
    return result


def candidates(row, detections, reviewed, size):
    objects = row["objects"]
    result = [(r["reference"], r["bbox"], r["label"], "visual") for r in reviewed]
    for i, target in enumerate(objects):
        box = target["bbox"]
        for j, other in enumerate(objects):
            if i != j:
                result.append((box, other["bbox"], "distinct", "reference-pair"))
        real = [item["bbox"] for item in detections if reference_index(item["bbox"], objects) == i]
        result.extend((box, ref, "same", "detector-reference") for ref in real[:4])
        if len(objects) == 1:
            result.extend((box, around(box, shift, zoom), "same", "jitter") for shift, zoom in (
                ((.05, -.04), (1.05, 1.10)), ((-.07, .07), (.86, .91))))
            result.extend((box, f, "fragment", "oriented-split") for f in fragment_views(target, size))
        negatives = [item["bbox"] for item in detections if box_iou(item["bbox"], box) < .035
                     and all(box_iou(item["bbox"], o["bbox"]) < .035 for o in objects)
                     and pair_is_near(box, item["bbox"], size)]
        result.extend((box, n, "distractor", "mined-background") for n in negatives[:3])
    return result


def pair_is_near(a, b, size):
    first, second = np.asarray(a), np.asarray(b)
    dims = np.asarray(size)
    distance = np.linalg.norm((center(first)-center(second))*dims)
    longest = max(np.linalg.norm((box[2:]-box[:2])*dims) for box in (first, second))
    return max(np.prod(first[2:]-first[:2]), 1e-9) >= .12*np.prod(second[2:]-second[:2]) and distance <= .95*longest


def training_weights(labels, groups):
    histogram = Counter(zip(labels, groups))
    counts = Counter(name for name, _ in histogram)
    return np.asarray([len(labels)/(len(counts)*counts[k]*histogram[k, g])
                       for k, g in zip(labels, groups, strict=True)])


def train():
    cv2.setNumThreads(1)
    if MODEL.exists():
        raise FileExistsError(MODEL)
    freeze_review()
    review = read_json(REVIEW)
    fit = cohorts("fit")
    proposals = archive(fit, "fit")
    evidence, labels, groups, sources = [], [], [], []
    for row in fit:
        if not row["objects"]:
            continue
        picture = raster(row)
        selected = [r for r in review["rows"] if r["id"] == row["id"]]
        pairs = candidates(row, proposals[row["id"]], selected, picture.size)
        unique = {}
        for a, b, label, origin in pairs:
            if not (valid_box(a) and valid_box(b)):
                continue
            for key in (tuple(a), tuple(b)):
                if key not in unique:
                    unique[key] = descriptor(picture, key)
            evidence.append(features(a, b, picture.size, unique[tuple(a)], unique[tuple(b)]))
            labels.append(label)
            groups.append(row["group"])
            sources.append(origin)
    heads = {}
    matrix = np.asarray(evidence)
    weights = training_weights(labels, groups)
    for arm in MODEL_ARMS:
        selected = matrix[:, :7] if arm == "boxes" else matrix
        transform = StandardScaler().fit(selected)
        head = LogisticRegression(C=.35, max_iter=600, random_state=SEED)
        head.fit(transform.transform(selected), labels, sample_weight=weights)
        heads[arm] = dict(center=transform.mean_.tolist(), scale=transform.scale_.tolist(),
                          weights=head.coef_.tolist(), bias=head.intercept_.tolist(),
                          labels=head.classes_.tolist())
    counts = dict(classes=Counter(labels), origins=Counter(sources),
                  groups={label: len({g for y, g in zip(labels, groups, strict=True) if y == label})
                          for label in set(labels)})
    write_json(MODEL, dict(code=digest(__file__), review=digest(REVIEW), heads=heads, counts=counts,
                           seed=SEED, rf_fit=digest(Path("data/native-candidate-verifier/rf-fit.json"))))
    print(counts, flush=True)


def model():
    stored = read_json(MODEL)
    if (stored["code"] != digest(__file__) or stored["review"] != digest(REVIEW) or
            stored["rf_fit"] != digest(Path("data/native-candidate-verifier/rf-fit.json"))):
        raise ValueError("The assembly pair model changed.")
    return stored["heads"]


def probabilities(head, row):
    fields = (np.array(row[:len(head["center"])])-head["center"])/head["scale"]
    return {name: float(score) for name, score in zip(head["labels"],
        softmax(np.array(head["weights"]) @ fields + head["bias"]), strict=True)}


if __name__ == "__main__":
    train()
