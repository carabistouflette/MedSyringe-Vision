"""Fit and qualify a four-class verifier for automatic syringe recrops."""
import argparse
from collections import Counter
from itertools import product
from math import ceil
from statistics import median

import cv2
import numpy as np
from PIL import ImageOps
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.additive import duplicates
from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.native_candidate_reserve import FOLDER as NATIVE
from scripts.native_candidate_verifier import compare, reference_index, simple_metrics
from scripts.occlusion_detector import ART, raster
from scripts.occlusion_proposals import oracle
from scripts.occlusion_relocalization import prepare_reference
from scripts.occlusion_relocalization_reserve import DATA as RELOCAL
from scripts.recrop_proposal_bank import BUDGETS, WORK, area, candidates, read_bank
from scripts.rfdetr_verifier import DINOV2_INFO, DinoEncoder, expanded
from scripts.sam3_study import configure_cpu
from scripts.syringe_assembly_model import (
    MODEL as IDENTITY, REVIEW, descriptor, features, fragment_views, model as pair_model,
    pair_is_near, probabilities, training_weights,
)
from scripts.syringe_assembly_reserve import RELEASE as ASSEMBLY

POLICY = ART / "recrop_verifier_policy.json"
RESULT = ART / "recrop_verifier_results.json"
SEED = 657
MODELS = ("detector", "agreement", "geometry", "appearance", "hybrid")
IDENTITIES = ("overlap", "fragment-veto", "distinct-aware")
CLASSES = {"complete", "fragment", "lookalike", "background"}


def proposal_features(p, size):
    l, t, r, b = p["bbox"]
    width, height = (r - l) * size[0], (b - t) * size[1]
    return [np.log1p(max(width, height)), np.log1p(min(width, height)),
            np.log(width / height), p.get("window_area", 0.), p.get("seed_iou", 0.),
            p.get("containing_seed", 0.), p.get("repeats", 0.) / max(1, p.get("views", 1) - 1),
            *[np.log1p(50 * p.get(key, 0.)) for key in ("score", "seed_score", "first_pass")]]


def body_features(image, box, visual):
    """Measure axis agreement and edge support across thirds of the visible crop."""
    patch = crop(image, box)
    patch.thumbnail((112, 112))
    gray = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    if min(gray.shape) < 3:
        return [visual["reliability"], *(0.,) * 8]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0) / 8
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1) / 8
    axis_x, axis_y = visual["axis"]
    along, across = np.abs(axis_x * gx + axis_y * gy), np.abs(axis_x * gy - axis_y * gx)
    y, x = np.indices(gray.shape, dtype=np.float32)
    position = axis_x * (x - x.mean()) + axis_y * (y - y.mean())
    middle = np.abs(position) <= max(1., np.max(np.abs(position)) / 3)
    tips = np.abs(position) > max(1., .60 * np.max(np.abs(position)))
    tip_energy = sorted(float(across[section].mean()) if section.any() else 0.
                        for section in (tips & (position > 0), tips & (position < 0)))
    return [visual["reliability"], float(across.mean()), float(along.mean()),
            float(across[middle].mean()), float(along[middle].mean()), *tip_energy,
            float(np.percentile(gray, 90) - np.percentile(gray, 10)), visual["lightness"]]


def fraction_inside(box, enclosure):
    x, y, r, b = box
    X, Y, R, B = enclosure
    return area([max(x, X), max(y, Y), min(r, R), min(b, B)]) / area(box)


def fitting_examples(row, frame, reviews, size):
    objects, lookalikes = row["objects"], row.get("lookalikes", [])
    known = [dict(bbox=o["bbox"], label="complete", origin="reviewed-box") for o in objects]
    known.extend(dict(bbox=o["bbox"], label="lookalike", origin="reviewed-lookalike") for o in lookalikes)
    fragments = [r["bbox"] for r in reviews if r["label"] == "fragment"]
    known.extend(dict(bbox=b, label="fragment", origin="visual-fragment") for b in fragments)
    known.extend(dict(bbox=r["bbox"], label="lookalike", origin="visual-distractor")
                 for r in reviews if r["label"] == "distractor")
    for i, obj in enumerate(objects):
        for part in fragment_views(obj, size):
            if all(box_iou(v["bbox"], part) < .15 for j, v in enumerate(objects) if i != j):
                known.append(dict(bbox=part, label="fragment", origin="constructed-fragment"))
    mined, counts = [], Counter()
    for p in candidates(frame, max(BUDGETS)):
        box = p["bbox"]
        overlap = max((box_iou(box, o["bbox"]) for o in objects), default=0.)
        if reference_index(box, objects) is not None and overlap >= .58:
            label = "complete"
        elif overlap < .25 and any(box_iou(box, f) >= .45 for f in fragments):
            label = "fragment"
        elif overlap < .12 and any(box_iou(box, o["bbox"]) >= .30 or
                                   fraction_inside(box, o["bbox"]) >= .85 for o in lookalikes):
            label = "lookalike"
        elif overlap < .035 and all(fraction_inside(o["bbox"], box) < .15 for o in objects):
            label = "background"
        else:
            continue
        if counts[label] < 6:
            counts[label] += 1
            mined.append(dict(p, label=label, origin="second-pass-proposal"))
    return known + mined


def descriptors(image, boxes):
    return {box: descriptor(image, box) for box in sorted({tuple(b) for b in boxes})}


def embeddings(encoder, image, boxes):
    order = sorted({tuple(b) for b in boxes})
    patches = []
    for box in order:
        patch = crop(image, expanded(box, .10))
        # Extremely thin detector boxes need at least one pixel after DINO padding.
        minimum = ceil(max(patch.size) / 168)
        dx, dy = (max(0, minimum - pixels) for pixels in patch.size)
        if dx or dy:
            patch = ImageOps.expand(patch, (dx // 2, dy // 2, dx - dx // 2, dy - dy // 2),
                                    fill=(124, 116, 104))
        patches.append(patch)
    return dict(zip(order, encoder.encode(patches), strict=True))


def vectors(item, size, visual, appearance):
    metadata = proposal_features(item, size)
    return {"geometry": [*metadata, *visual], "appearance": appearance,
            "hybrid": [*metadata, *visual, *appearance]}


def fit_heads(rows, bank, encoder):
    reviewed = read_json(REVIEW)["rows"]
    labels, groups, sources = [], [], []
    matrix = {arm: [] for arm in MODELS[2:]}
    manifest = {}
    for index, row in enumerate(rows, 1):
        key, image = row["id"], raster(row)
        items = fitting_examples(row, bank[key], [r for r in reviewed if r["id"] == key], image.size)
        boxes = [p["bbox"] for p in items]
        visual, embedded = descriptors(image, boxes), embeddings(encoder, image, boxes)
        manifest[key] = dict(group=row["group"], examples=items)
        for p in items:
            box = tuple(p["bbox"])
            all_values = vectors(p, image.size, body_features(image, box, visual[box]), embedded[box])
            for arm, vector in all_values.items():
                matrix[arm].append(vector)
            labels.append(p["label"])
            groups.append(row["group"])
            sources.append(f'{p["origin"]}/{p["label"]}')
        if index % 20 == 0:
            print("fit descriptors", index, "/", len(rows), flush=True)
    if set(labels) != CLASSES:
        raise ValueError("All four verifier classes require supervision.")
    write_json(WORK / "fit-supervision.json", manifest)
    heads = {}
    weights = training_weights(labels, groups)
    for arm, values in matrix.items():
        values = np.asarray(values, dtype=np.float64)
        center = np.zeros(values.shape[1])
        scale = np.full(values.shape[1], 1 / 6.)
        length = 0 if arm == "appearance" else 19
        if length:
            norm = StandardScaler().fit(values[:, :length], sample_weight=weights)
            center[:length], scale[:length] = norm.mean_, norm.scale_
        head = LogisticRegression(C=.40, random_state=SEED, max_iter=600)
        head.fit((values-center)/scale, labels, sample_weight=weights)
        heads[arm] = dict(labels=head.classes_.tolist(), weights=head.coef_.tolist(),
                          bias=head.intercept_.tolist(), center=center.tolist(), scale=scale.tolist())
    return heads, dict(classes=Counter(labels), sources=Counter(sources),
                       groups={name: len({g for g, label in zip(groups, labels, strict=True) if label == name})
                               for name in sorted(CLASSES)})


def rank_frames(rows, base, bank, heads, encoder):
    ranked = {}
    for index, row in enumerate(rows, 1):
        key, image = row["id"], raster(row)
        pools = {k: candidates(bank[key], k) for k in BUDGETS}
        boxes = [p["bbox"] for p in base[key]] + [p["bbox"] for v in pools.values() for p in v]
        visual = descriptors(image, boxes)
        bodies = {box: body_features(image, box, data) for box, data in visual.items()}
        embedded = embeddings(encoder, image, boxes)
        for pool in pools.values():
            if not pool:
                continue
            values = [vectors(p, image.size, bodies[tuple(p["bbox"])], embedded[tuple(p["bbox"])]) for p in pool]
            for p in pool:
                p["scores"] = {"detector": p["score"], "agreement": min(p["rank_score"], 1.)}
            for arm, head in heads.items():
                inputs = (np.asarray([v[arm] for v in values])-head["center"])/head["scale"]
                outcome = softmax(inputs @ np.asarray(head["weights"]).T + head["bias"], axis=1)
                for p, vector in zip(pool, outcome, strict=True):
                    p["scores"][arm] = float(vector[head["labels"].index("complete")])
                    p.setdefault("classes", {})[arm] = dict(zip(head["labels"], map(float, vector), strict=True))
        ranked[key] = dict(incumbent=base[key], pools=pools, visual=visual, size=image.size, cache={})
        if index % 10 == 0:
            print("scored recrops", index, "/", len(rows), flush=True)
    return ranked


def blocked(a, b, mode, frame, head):
    conventional = duplicates(a, b)
    if mode == "overlap" or box_iou(a, b) >= .85 or not pair_is_near(a, b, frame["size"]):
        return conventional
    key = tuple(a), tuple(b)
    cache = frame["cache"]
    if key not in cache:
        first, second = (frame["visual"][k] for k in key)
        cache[key] = probabilities(head, features(a, b, frame["size"], first, second))
    p = cache[key]
    return p["same"] + p["fragment"] >= .65 or (
        conventional and (mode != "distinct-aware" or p["distinct"] < .90))


def fuse(ranked, rule, head):
    predictions = {}
    for key, frame in ranked.items():
        ordered = sorted(frame["pools"][rule["budget"]], key=lambda p: -p["scores"][rule["model"]])
        found = list(frame["incumbent"])
        for p in ordered:
            score = p["scores"][rule["model"]]
            if score < rule["floor"]:
                break
            if not any(blocked(q["bbox"], p["bbox"], rule["identity"], frame, head) for q in found):
                found.append(dict(bbox=p["bbox"], score=.002 + .001 * score, source="automatic recrop verifier",
                                  crop_score=p["score"], verifier_score=score, seed_rank=p["seed_rank"],
                                  slot=p["slot"], repeat_views=p["repeats"], classes=p["classes"]))
        predictions[key] = found
    return predictions


def quality(trial):
    rule = trial["rule"]
    return trial["gain"], trial["exact_count"], -rule["budget"], rule["identity"] == "overlap", rule["floor"]


def select(rows, baseline, ranked, head):
    control = simple_metrics(rows, baseline)
    floors = (.01, .02, .035, .05, .075, .10, .15, .20, .30, .40, .50, .60, .70, .80,
              .85, .90, .95, .975, .99, .999, 1.001)
    trials, arms = [], {}
    for model, budget, identity, floor in product(MODELS, BUDGETS, IDENTITIES, floors):
        rule = dict(model=model, budget=budget, identity=identity, floor=floor)
        predictions = fuse(ranked, rule, head)
        trials.append(dict(rule=rule, **compare(rows, control, predictions)))
    for model, budget in product(MODELS, BUDGETS):
        viable = [t for t in trials if t["safe"] and (t["rule"]["model"], t["rule"]["budget"]) == (model, budget)]
        arms[f"{model}/{budget}"] = max(viable, key=quality)
    winner = max((v for v in arms.values() if v["gain"] > 0), key=quality, default=None)
    return dict(selected=winner, arms=arms, trials=trials)


def inputs():
    return {str(path): digest(path) for path in (POLICY.with_name("recrop_verifier_reserve.json"),
            IDENTITY, REVIEW, DINOV2_INFO, WORK / "fit.json", WORK / "development.json",
            WORK / "fit-supervision.json", "scripts/recrop_proposal_bank.py", "scripts/recrop_reserve.py")}


def learn():
    if POLICY.exists():
        raise FileExistsError(POLICY)
    configure_cpu()
    cv2.setNumThreads(1)
    head = pair_model()["pixels"]
    encoder = DinoEncoder()
    rows, bank = read_bank("fit")
    heads, train = fit_heads(rows, bank, encoder)
    dev, bank = read_bank("development")
    _, base, archives, _ = prepare_reference()
    ranked = rank_frames(dev, base, bank, heads, encoder)
    selected = select(dev, base, ranked, head)
    development = report(dev, base, bank, ranked, selected["arms"], head)
    write_json(POLICY, dict(code=digest(__file__), inputs=inputs(), archives=archives, seed=SEED,
                            heads=heads, training=train, selection=selected, development=development,
                            release_criterion="Fresh matched-target improvement with no lost targets, added per-image "
                                              "false positives or greater per-image absolute count error in either audit."))
    print("training", train, "selected", selected["selected"], flush=True)


def sealed():
    data = read_json(POLICY)
    if data["code"] != digest(__file__) or data["inputs"] != inputs() or any(
        digest(path) != sha for path, sha in data["archives"].items()
    ):
        raise ValueError("The recrop policy or its sources changed.")
    return data


def report(rows, base, bank, ranked, rules, head):
    before = simple_metrics(rows, base)
    runs = {}
    for name, trial in rules.items():
        found = fuse(ranked, trial["rule"], head)
        runs[name] = dict(delta=compare(rows, before, found), metrics=evaluate(rows, found, 0.),
                          additions={key: values[len(base[key]):] for key, values in found.items()
                                     if len(values) > len(base[key])})
    coverage = {r["id"]: dict(first_pass=oracle(r["objects"], bank[r["id"]]["seeds"])["matched"],
                              **{str(k): oracle(r["objects"], [*bank[r["id"]]["seeds"],
                                             *ranked[r["id"]]["pools"][k]])["matched"] for k in BUDGETS}) for r in rows}
    return dict(baseline=evaluate(rows, base, 0.), arms=runs, oracle_by_image=coverage,
                oracle_total={k: sum(v[k] for v in coverage.values()) for k in ("first_pass", *map(str, BUDGETS))})


def evaluate_frozen():
    from scripts.recrop_reserve import RELEASE, assert_frozen_policy, references

    assert_frozen_policy()
    lock = sealed()
    configure_cpu()
    cv2.setNumThreads(1)
    encoder = DinoEncoder()
    head = pair_model()["pixels"]
    blocks = {}
    release = read_json(RELEASE)
    archive_sources = (RELOCAL / "released.json", NATIVE / "released.json", ASSEMBLY)
    old = {**read_json(archive_sources[0])["predictions"], **read_json(archive_sources[1])["outputs"],
           **read_json(archive_sources[2])["outputs"]}
    for role, base in (("archive", old), ("reserve", release["outputs"])):
        rows, bank = read_bank(role)
        if set(base) != {r["id"] for r in rows}:
            raise ValueError(f"Baseline belongs to a different {role}.")
        if role == "reserve" and set(base) != {r["id"] for r in references()}:
            raise ValueError("Unexpected acquisition in reserve.")
        scored = rank_frames(rows, base, bank, lock["heads"], encoder)
        block = report(rows, base, bank, scored, lock["selection"]["arms"], head)
        block["median_extra_cpu_s_all_windows"] = median(read_json(WORK / f"{role}.json")["times"].values())
        blocks[role] = block
    selected = lock["selection"]["selected"]
    winner = f'{selected["rule"]["model"]}/{selected["rule"]["budget"]}' if selected else None
    delta = {key: value["arms"][winner]["delta"] for key, value in blocks.items()} if winner else {}
    passed = bool(winner and all(v["safe"] for v in delta.values()) and delta["reserve"]["gain"] > 0)
    evidence = (RELEASE, *archive_sources, *(WORK / f"{r}.json" for r in blocks))
    write_json(RESULT, dict(policy=digest(POLICY), evidence={str(p): digest(p) for p in evidence},
                            selected=winner, passes_release_gate=passed, cohorts=blocks))
    print(winner, delta, "passes release", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("learn", "evaluate"))
    {"learn": learn, "evaluate": evaluate_frozen}[parser.parse_args().stage]()
