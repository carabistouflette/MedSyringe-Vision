"""Learn to replace a released syringe box while preserving its instance slot."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.detection.additive import overlap_fraction
from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate, match
from scripts.experiments.crowds.native_candidate_reserve import FOLDER as NATIVE
from scripts.experiments.occlusion.occlusion_detector import ART, raster
from scripts.experiments.occlusion.occlusion_relocalization import POLICY as LEGACY, diagnostics, prepare_reference, replacements
from scripts.experiments.occlusion.occlusion_relocalization_reserve import DATA as RELOCAL
from scripts.experiments.relocalization.recrop_proposal_bank import WORK as CACHED, area, candidates, read_bank
from scripts.experiments.relocalization.recrop_reserve import RELEASE as EXPOSED
from scripts.experiments.relocalization.recrop_verifier import body_features
from scripts.experiments.identity.syringe_assembly_model import around, descriptor, features, fragment_views, training_weights
from scripts.experiments.identity.syringe_assembly_reserve import RELEASE as ASSEMBLY

WORK = Path("data/relocalization-selector")
POLICY = ART / "relocalization_selector_policy.json"
RESULT = ART / "relocalization_selector_results.json"
ARMS = ("consensus", "learned-views", "learned-visual")
BUDGETS = (0, 3, 6)
SEED = 943


def pool(frame, budget):
    return [dict(p, view="first") for p in frame["seeds"]] + [
        dict(p, view="crop") for p in candidates(frame, budget)]


def eligible(current, other, candidate):
    """Avoid fragment-sized boxes and ambiguous swaps between released neighbors."""
    overlap = box_iou(current, candidate)
    if not (.14 <= area(candidate) / area(current) <= 1.10 and .12 <= overlap < .96
            and overlap_fraction(current, candidate) >= .72):
        return False
    return all(box_iou(b, candidate) < max(.35, overlap - .08) for b in other)


def evidence(candidate, options):
    cross = [p for p in options if p["view"] != candidate["view"]]
    aligned = [p for p in cross if box_iou(candidate["bbox"], p["bbox"]) >= .45]
    return (max((box_iou(candidate["bbox"], p["bbox"]) for p in cross), default=0.),
            max((p["score"] for p in aligned), default=0.))


def vector(anchor, candidate, other, options, image, cache):
    current, proposed = tuple(anchor), tuple(candidate["bbox"])
    agree, supporting = evidence(candidate, options)
    views = [box_iou(anchor, proposed), overlap_fraction(anchor, proposed),
             np.log(area(proposed) / area(anchor)),
             max((box_iou(p, proposed) for p in other), default=0.),
             candidate["score"], agree, supporting, candidate.get("repeats", 0),
             float(candidate["view"] == "crop")]
    for box in (current, proposed):
        if box not in cache:
            visual = descriptor(image, box)
            cache[box] = visual, body_features(image, box, visual)
    left, before = cache[current]
    right, after = cache[proposed]
    pair = features(current, proposed, image.size, left, right)
    return views, [*views, *pair[3:], *after, *np.subtract(after, before)]


def anchors(objects, index, first):
    box = objects[index]["bbox"]
    plausible = [p["bbox"] for p in first if box_iou(p["bbox"], box) >= .24 and all(
        box_iou(p["bbox"], obj["bbox"]) + .07 < box_iou(p["bbox"], box)
        for j, obj in enumerate(objects) if j != index)][:4]
    distorted = [around(box, shift, zoom) for shift, zoom in (
        ((0., 0.), (1.62, 1.50)), ((.05, -.12), (1.45, 1.85)), ((-.12, .08), (1.86, 1.38)),
        ((0., 0.), (1.10, 1.04)), ((-.09, -.06), (.78, .79)))]
    return [(a, "detector") for a in plausible] + [(a, "simulated") for a in distorted if valid_box(a)]


def visible_anchors_preserved(box, obj):
    l, t, r, b = box
    dx, dy = .08 * (r - l), .08 * (b - t)
    return all(l - dx <= p[0] <= r + dx and t - dy <= p[1] <= b + dy
               for p in obj.get("parts", {}).values() if p is not None)


def fit(rows, bank):
    matrices = [[], []]
    labels, groups, counts, ledger = [], [], Counter(), []
    for row in rows:
        if not row["objects"]:
            continue
        key, objects, cache = row["id"], row["objects"], {}
        image = raster(row)
        source = bank[key]
        detected = pool(source, 6)
        for i, obj in enumerate(objects):
            gold = obj["bbox"]
            other = [p["bbox"] for j, p in enumerate(objects) if i != j]
            synthetic = [dict(bbox=b, view="synthetic", score=0.) for b in fragment_views(obj, image.size)]
            for anchor, origin in anchors(objects, i, source["seeds"]):
                before = box_iou(anchor, gold)
                for candidate in [*detected, *synthetic]:
                    box = candidate["bbox"]
                    if not eligible(anchor, other, box):
                        continue
                    after = box_iou(box, gold)
                    same = all(after > box_iou(box, ref) + .05 for ref in other)
                    parts = visible_anchors_preserved(box, obj)
                    improving = same and parts and after >= .50 and after - before >= .08
                    stable = same and parts and after >= .50 and -.04 < after - before < .08
                    if stable:
                        continue
                    for matrix, values in zip(matrices, vector(anchor, candidate, other, detected, image, cache), strict=True):
                        matrix.append(values)
                    labels.append(improving)
                    groups.append(row["group"])
                    counts[f"{origin}/{candidate['view']}/{'improving' if improving else 'reject'}"] += 1
                    ledger.append(dict(id=key, group=row["group"], slot=i, anchor=anchor, candidate=box,
                                       origin=origin, view=candidate["view"], before_iou=before,
                                       after_iou=after, preserves_visible_parts=parts, improves=improving))
    weights = training_weights(labels, groups)
    heads = {}
    for name, matrix in zip(ARMS[1:], matrices, strict=True):
        fitted = StandardScaler().fit(matrix, sample_weight=weights)
        model = LogisticRegression(C=.3, max_iter=600, random_state=SEED)
        model.fit(fitted.transform(matrix), labels, sample_weight=weights)
        heads[name] = dict(center=fitted.mean_.tolist(), scale=fitted.scale_.tolist(),
                           coef=model.coef_[0].tolist(), bias=float(model.intercept_[0]))
    write_json(WORK / "fitting-pairs.json", ledger)
    return heads, dict(pairs=len(labels), improves=sum(labels), group_count=len(set(groups)), counts=counts)


def rank(rows, baseline, frames, heads):
    ranked = {}
    for row in rows:
        key = row["id"]
        current = baseline[key]
        if not current:
            ranked[key] = {budget: [] for budget in BUDGETS}
            continue
        frame = frames[key]
        image, cache = raster(row), {}
        ranked[key] = {}
        for budget in BUDGETS:
            options = pool(frame, budget)
            choices = []
            for i, item in enumerate(current):
                surrounding = [p["bbox"] for j, p in enumerate(current) if j != i]
                for candidate in options:
                    if not eligible(item["bbox"], surrounding, candidate["bbox"]):
                        continue
                    inputs = vector(item["bbox"], candidate, surrounding, options, image, cache)
                    agree = inputs[0][5]
                    score = {"consensus": .60 * candidate["score"] + .40 * agree}
                    for name, x in zip(ARMS[1:], inputs, strict=True):
                        head = heads[name]
                        normalized = (np.array(x) - head["center"]) / head["scale"]
                        score[name] = float(expit(normalized @ head["coef"] + head["bias"]))
                    choices.append(dict(candidate, instance_slot=i, scores=score, cross_view_iou=agree))
            ranked[key][budget] = choices
    return ranked


def refine(baseline, ranked, rule):
    arm, budget, floor, allow_multiple = (rule[k] for k in ("arm", "budget", "floor", "multiple"))
    output = {}
    for key, original in baseline.items():
        current, changed = list(original), set()
        if allow_multiple or len(current) == 1:
            choices = sorted(ranked[key][budget], key=lambda p: (-p["scores"][arm], -p["score"]))
            for p in choices:
                slot, score = p["instance_slot"], p["scores"][arm]
                if score < floor:
                    break
                if slot in changed or any(box_iou(p["bbox"], current[j]["bbox"]) >= .45 for j in changed):
                    continue
                current[slot] = {**current[slot], "bbox": p["bbox"], "prior_bbox": current[slot]["bbox"],
                                 "refinement_score": score, "refinement_view": p["view"],
                                 "refinement_detector_score": p["score"]}
                changed.add(slot)
        output[key] = current
    return output


def snapshot(rows, predictions):
    return {iou: match(rows, predictions, iou)[1] for iou in (.50, .75)}


def guard(rows, baseline, before, prediction):
    hits = snapshot(rows, prediction)
    gained = {str(iou): sorted((key, idx) for key, members in cohort.items()
                               for idx in members - before[iou][key]) for iou, cohort in hits.items()}
    lost = {str(iou): sorted((key, idx) for key, members in cohort.items()
                             for idx in members - hits[iou][key]) for iou, cohort in before.items()}
    same_counts = all(len(p) == len(baseline[key]) for key, p in prediction.items())
    return dict(gained=gained, lost=lost, count_preserved=same_counts,
                safe=same_counts and not any(lost.values()),
                changed=sum(p["bbox"] != q["bbox"] for key, p in prediction.items()
                            for p, q in zip(p, baseline[key])))


def select(rows, baseline, ranked):
    before = snapshot(rows, baseline)
    floors = (.30, .40, .50, .55, .60, .65, .70, .75, .80, .85, .90, .925, .95, .975, .99, 1.01)
    trials = []
    for arm, budget, multiple, floor in product(ARMS, BUDGETS, (False, True), floors):
        rule = dict(arm=arm, budget=budget, multiple=multiple, floor=floor)
        trials.append(dict(rule=rule, **guard(rows, baseline, before, refine(baseline, ranked, rule))))

    def quality(t):
        return len(t["gained"]["0.5"]), len(t["gained"]["0.75"]), -t["changed"], -t["rule"]["budget"], t["rule"]["floor"]

    winners = {arm: max((t for t in trials if t["safe"] and t["rule"]["arm"] == arm), key=quality) for arm in ARMS}
    winner = max((t for t in winners.values() if any(t["gained"].values())), key=quality, default=None)
    return dict(arms=winners, selected=winner, trials=trials)


def pinned():
    paths = (LEGACY, CACHED / "fit.json", CACHED / "development.json", WORK / "fitting-pairs.json",
             Path("scripts/experiments/relocalization/recrop_proposal_bank.py"), Path("scripts/experiments/relocalization/recrop_verifier.py"),
             Path("scripts/experiments/identity/syringe_assembly_model.py"))
    return {str(p): digest(p) for p in paths}


def train():
    if POLICY.exists():
        raise FileExistsError(POLICY)
    cv2.setNumThreads(1)
    fit_rows, fit_frames = read_bank("fit")
    dev, frames = read_bank("development")
    _, baseline, archives, _ = prepare_reference()
    if {r["group"] for r in fit_rows} & {r["group"] for r in dev}:
        raise ValueError("Fitting and development share an acquisition group.")
    heads, fitting = fit(fit_rows, fit_frames)
    selection = select(dev, baseline, rank(dev, baseline, frames, heads))
    write_json(POLICY, dict(code=digest(__file__), pinned=pinned(), archives=archives,
                            fitting=fitting, heads=heads, selection=selection, seed=SEED))
    print(fitting, selection["arms"], selection["selected"], flush=True)


def sealed():
    lock = read_json(POLICY)
    if (lock["code"] != digest(__file__) or lock["pinned"] != pinned()
            or any(digest(path) != sha for path, sha in lock["archives"].items())):
        raise ValueError("Frozen selection or input has changed.")
    return lock


def report(rows, baseline, frames, lock):
    cv2.setNumThreads(1)
    ranked = rank(rows, baseline, frames, lock["heads"])
    variants = {arm: refine(baseline, ranked, trial["rule"]) for arm, trial in lock["selection"]["arms"].items()}
    limits = read_json(LEGACY)["selected"]["policy"]
    variants["legacy-recrops"] = {key: replacements(original, candidates(frames[key], 3), [], limits)
                                 for key, original in baseline.items()}
    before = snapshot(rows, baseline)
    return dict(baseline=evaluate(rows, baseline, 0.), arms={
        arm: dict(guard=guard(rows, baseline, before, prediction),
                  evaluation=diagnostics(rows, baseline, prediction, detailed=True),
                  changes={key: [p for p in values if "prior_bbox" in p] for key, values in prediction.items()
                           if any("prior_bbox" in p for p in values)}) for arm, prediction in variants.items()})


def score():
    from scripts.experiments.relocalization.relocalization_reserve import evidence as independent

    lock = sealed()
    archive_sources = (RELOCAL / "released.json", NATIVE / "released.json", ASSEMBLY)
    old = {**read_json(archive_sources[0])["predictions"], **read_json(archive_sources[1])["outputs"],
           **read_json(archive_sources[2])["outputs"]}
    dev, baseline, _, _ = prepare_reference()
    summaries = {}
    for role, base in (("development", baseline), ("archive", old), ("exposed-recrop", read_json(EXPOSED)["outputs"])):
        rows, bank = read_bank("reserve" if role == "exposed-recrop" else role)
        summaries[role] = report(rows, base, bank, lock)
    fresh, base, bank, inputs = independent()
    summaries["reserve"] = report(fresh, base, bank, lock)
    chosen = lock["selection"]["selected"]
    arm = chosen["rule"]["arm"] if chosen else None
    gate = bool(arm and all(r["arms"][arm]["guard"]["safe"] for r in summaries.values())
                and any(summaries["reserve"]["arms"][arm]["guard"]["gained"].values()))
    write_json(RESULT, dict(policy=digest(POLICY), evaluated_inputs={**inputs, **{str(p): digest(p) for p in (
        *archive_sources, EXPOSED, CACHED / "archive.json", CACHED / "reserve.json")}},
        selected=arm, qualifies=gate, cohorts=summaries))
    print(arm, gate, {key: {name: r["guard"] for name, r in value["arms"].items()} for key, value in summaries.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "score"))
    {"train": train, "score": score}[parser.parse_args().stage]()
