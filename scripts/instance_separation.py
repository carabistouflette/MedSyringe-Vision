"""Qualify an axis-aware exception to overlap deduplication."""
import argparse
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import cv2
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.additive import duplicates, overlap_fraction
from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.native_candidate_bank import archive, cohorts
from scripts.native_candidate_verifier import compare, reference_index, simple_metrics
from scripts.occlusion_detector import ART, raster
from scripts.occlusion_relocalization import prepare_reference
from scripts.relocalization_reserve import evidence as previous
from scripts.relocalization_selector import ASSEMBLY, EXPOSED, NATIVE, RELOCAL, read_bank
from scripts.syringe_assembly_model import (
    MODEL, REVIEW, candidates as pair_examples, descriptor, features, model, probabilities, training_weights,
)

WORK = Path("data/instance-separation")
POLICY = ART / "instance_separation_policy.json"
RESULT = ART / "instance_separation_results.json"
METHODS = ("overlap", "prior-identity", "axis-identity")


def axis_profile(image, box, original):
    """Find a centerline from paired longitudinal edges at several barrel widths."""
    patch = crop(image, box)
    width, height = patch.size
    patch.thumbnail((140, 140))
    gray = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    axis = original["axis"]
    normal = np.array([-axis[1], axis[0]])
    span = np.array([gray.shape[1], gray.shape[0]], dtype=float)
    extent = 1 / max(np.max(np.abs(axis) / span), 1e-6)
    thickness = min(span.min(), .38 * extent)
    offset = np.linspace(-.54 * thickness, .54 * thickness, 35)
    longitudinal = np.linspace(-.46 * extent, .46 * extent, 41)
    point = (span - 1) / 2 + longitudinal[:, None, None] * axis + offset[None, :, None] * normal
    tile = cv2.remap(gray, point[:, :, 0].astype("float32"), point[:, :, 1].astype("float32"),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    edge = np.diff(tile, axis=1)
    center_options = []
    for gap in (3, 5, 8, 12):
        left, right = edge[:, :-gap], edge[:, gap:]
        paired = np.where(left * right < 0, np.minimum(abs(left), abs(right)), 0.)
        center_options.extend((float(v), j + gap / 2, gap, paired[:, j])
                              for j, v in enumerate(paired.mean(axis=0)) if 6 <= j + gap / 2 <= 28)
    strength, best, gap, votes = max(center_options, key=lambda x: x[0])
    displacement = (best - 16.5) * thickness * 1.08 / 34
    confidence = float(np.mean(votes > .018))
    l, t, r, b = np.array(box) * np.tile(image.size, 2)
    origin = np.array([(l + r) / 2, (t + b) / 2]) + normal * displacement * max(width, height) / max(span)
    return origin, axis, strength, confidence, gap / 34, float(np.quantile(votes, .85))


def pair_vector(a, b, photo, cache):
    for box in (tuple(a), tuple(b)):
        if box not in cache:
            data = descriptor(photo, box)
            cache[box] = data, axis_profile(photo, box, data)
    left, (x, ax, s, c, w, end) = cache[tuple(a)]
    right, (y, ay, S, C, W, END) = cache[tuple(b)]
    dis = y - x
    cross = [abs(float(np.cross(np.r_[axis, 0], np.r_[dis, 0])[2])) for axis in (ax, ay)]
    width = max(1., min((a[2]-a[0])*photo.width, (a[3]-a[1])*photo.height,
                        (b[2]-b[0])*photo.width, (b[3]-b[1])*photo.height))
    return [*features(a, b, photo.size, left, right),
            *[v / width for v in cross], s, S, c, C, w, W, end, END]


def collect_examples(rows, bank):
    review = read_json(REVIEW)["rows"]
    vectors, labels, groups, origins = [], [], [], Counter()
    for row in rows:
        if not row["objects"]:
            continue
        key, objects = row["id"], row["objects"]
        photo, cache = raster(row), {}
        raw = bank[key]
        examples = pair_examples(row, raw, [r for r in review if r["id"] == key], photo.size)
        precise = [(p["bbox"], i) for p in raw if (i := reference_index(p["bbox"], objects)) is not None]
        examples.extend((a, b, "same" if i == j else "distinct", "detector-pair")
                        for (a, i), (b, j) in combinations(precise, 2) if duplicates(a, b))
        for a, b, label, origin in examples:
            vectors.append(pair_vector(a, b, photo, cache))
            labels.append(label)
            groups.append(row["group"])
            origins[origin] += 1
    return np.array(vectors), labels, groups, origins


def train_head():
    rows = cohorts("fit")
    x, y, groups, sources = collect_examples(rows, archive(rows, "fit"))
    weights = training_weights(y, groups)
    norm = StandardScaler().fit(x, sample_weight=weights)
    fitted = LogisticRegression(C=.35, max_iter=650, random_state=641)
    fitted.fit(norm.transform(x), y, sample_weight=weights)
    return dict(center=norm.mean_.tolist(), scale=norm.scale_.tolist(),
                weight=fitted.coef_.tolist(), bias=fitted.intercept_.tolist(), labels=fitted.classes_.tolist()), dict(
                    totals=Counter(y), origins=sources, distinct_groups=len({g for g, y in zip(groups, y) if y == "distinct"}))


def bindings():
    paths = (MODEL, REVIEW, "data/native-candidate-verifier/rf-fit.json",
             "data/native-candidate-verifier/rf-development.json", "scripts/syringe_assembly_model.py",
             "scripts/native_candidate_bank.py")
    return {str(p): digest(p) for p in paths}


def pair_scores(rows, base, banks, head):
    cv2.setNumThreads(1)
    old = model()["pixels"]
    output = {}
    for row in rows:
        key = row["id"]
        if not base[key]:
            output[key] = {}
            continue
        photo, cache, collected = raster(row), {}, {}
        proposed = [p["bbox"] for p in banks[key] if p["score"] >= .12]
        boxes = list(dict.fromkeys(tuple(p["bbox"]) for p in base[key])) + [tuple(b) for b in proposed]
        for i, a in enumerate(boxes):
            for b in boxes[max(i+1, len(boxes)-len(proposed)):]:
                if not duplicates(a, b) or box_iou(a, b) >= .84:
                    continue
                vector = pair_vector(a, b, photo, cache)
                values = (np.asarray(vector) - head["center"]) / head["scale"]
                axis = dict(zip(head["labels"], map(float, softmax(np.asarray(head["weight"]) @ values + head["bias"]))))
                collected[a,b] = dict(prior=probabilities(old, vector), axis=axis)
        output[key] = collected
    return output


def fuse(base, banks, pairs, rule):
    outcomes = {}
    for key, incumbent in base.items():
        accepted = list(incumbent)
        # This experiment isolates crowded-scene additions.
        if len(incumbent) >= rule["minimum"]:
            for p in sorted(banks[key], key=lambda p: -p["score"]):
                if p["score"] < rule["floor"]:
                    break
                overlaps = [q for q in accepted if duplicates(q["bbox"], p["bbox"])]
                if rule["method"] != "overlap" and not overlaps:
                    continue
                if any(blocked(q["bbox"], p["bbox"], pairs[key], rule) for q in overlaps):
                    continue
                accepted.append(dict(p, score=.002 + .001*p["score"], detector_score=p["score"],
                                     source="instance separation", overlapping=len(overlaps)))
        outcomes[key] = accepted
    return outcomes


def blocked(a, b, evidence, rule):
    if rule["method"] == "overlap" or box_iou(a,b) >= .84:
        return True
    record = evidence.get((tuple(a), tuple(b))) or evidence.get((tuple(b), tuple(a)))
    if not record:
        return True
    distribution = record["axis" if rule["method"] == "axis-identity" else "prior"]
    return distribution["distinct"] < rule["distinct"] or distribution["same"] + distribution["fragment"] > .24


def choose(rows, base, bank, pairs):
    before = simple_metrics(rows, base)
    trials = []
    for method, minimum, floor in product(METHODS, (1, 2), (.12,.18,.25,.30,.35,.40,.45,.55,.65,.75,.85,.95,1.01)):
        for threshold in ((1.,) if method == "overlap" else (.65,.75,.85,.93,.98)):
            rule = dict(method=method, minimum=minimum, floor=floor, distinct=threshold)
            trials.append(dict(rule=rule, **compare(rows, before, fuse(base, bank, pairs, rule))))
    finalists = {m: max((t for t in trials if t["safe"] and t["rule"]["method"] == m),
                        key=lambda t: (t["gain"], t["exact_count"], t["rule"]["floor"])) for m in METHODS}
    best = max((t for t in finalists.values() if t["gain"] > 0),
               key=lambda t: (t["gain"], t["exact_count"]), default=None)
    probes = {m: dict(method=m, minimum=2, floor=.25, distinct=.65) for m in METHODS}
    return dict(finalists=finalists, selected=best, trials=trials, probes=probes)


def train():
    if POLICY.exists():
        raise FileExistsError(POLICY)
    cv2.setNumThreads(1)
    head, training = train_head()
    rows, base, provenance, _ = prepare_reference()
    bank = archive(rows, "development")
    decision = choose(rows, base, bank, pair_scores(rows, base, bank, head))
    write_json(POLICY, dict(code=digest(__file__), inputs=bindings(), archives=provenance,
                            model=head, training=training, decision=decision))
    print(training, decision["finalists"], decision["selected"], flush=True)


def sealed():
    lock = read_json(POLICY)
    if (lock["code"] != digest(__file__) or lock["inputs"] != bindings()
            or any(digest(p) != h for p, h in lock["archives"].items())):
        raise ValueError("The separation policy changed.")
    return lock


def report(rows, base, bank, lock):
    pairs = pair_scores(rows, base, bank, lock["model"])
    before = simple_metrics(rows, base)
    out = {}
    rules = {name: trial["rule"] for name, trial in lock["decision"]["finalists"].items()}
    rules.update({f"probe-{name}": rule for name, rule in lock["decision"]["probes"].items()})
    for name, rule in rules.items():
        pred = fuse(base, bank, pairs, rule)
        out[name] = dict(delta=compare(rows, before, pred), metrics=evaluate(rows, pred, 0.),
                         additions={key: v[len(base[key]):] for key, v in pred.items() if len(v) > len(base[key])})
    return dict(baseline=evaluate(rows, base, 0.), arms=out)


def target_audit(rows, base, bank, lock):
    row = next(r for r in rows if r["id"] == "refine-8670507")
    key = row["id"]
    index = set(simple_metrics([row], base)["targets"][key])
    scores = pair_scores([row], base, bank, lock["model"])[key]
    trace = []
    for i, target in enumerate(row["objects"]):
        if i in index:
            continue
        p = next(p for p in bank[key] if box_iou(p["bbox"], target["bbox"]) >= .50)
        trace.append(dict(target=i, candidate=p, evidence=[
            dict(iou=box_iou(q["bbox"], p["bbox"]), result=scores.get((tuple(q["bbox"]),tuple(p["bbox"]))))
            for q in base[key] if duplicates(q["bbox"], p["bbox"])]))
    return trace


def score():
    from scripts.instance_separation_reserve import evidence

    lock = sealed()
    old, frames = read_bank("archive")
    base = {**read_json(RELOCAL / "released.json")["predictions"], **read_json(NATIVE / "released.json")["outputs"],
            **read_json(ASSEMBLY)["outputs"]}
    mins = {key: v["seeds"] for key,v in frames.items()}
    familiar, before, bank, prior_inputs = previous()
    known = {key:v["seeds"] for key,v in bank.items()}
    reserve, control, raw, inputs = evidence()
    archive_result = report(old, base, mins, lock)
    previous_result = report(familiar, before, known, lock)
    fresh = report(reserve, control, raw, lock)
    chosen = lock["decision"]["selected"]
    winner = chosen["rule"]["method"] if chosen else None
    gate = bool(winner and fresh["arms"][winner]["delta"]["gain"] > 0
                and all(r["arms"][winner]["delta"]["safe"] for r in (archive_result, previous_result, fresh)))
    write_json(RESULT, dict(policy=digest(POLICY), inputs=inputs, previous_inputs=prior_inputs,
                            archive=archive_result, exposed=previous_result, independent=fresh,
                            six_syringes=target_audit(familiar, before, known, lock), winner=winner, qualifies=gate))
    print(winner, gate, {name:a["delta"] for name,a in fresh["arms"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "score"))
    {"train": train, "score": score}[parser.parse_args().stage]()
