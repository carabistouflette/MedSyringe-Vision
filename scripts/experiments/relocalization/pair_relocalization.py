"""Compare count-preserving syringe box replacements with explicit pair supervision."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from medsyringe.detection.additive import overlap_fraction
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.experiments.occlusion.occlusion_detector import ART
from scripts.experiments.ranking.overlap_candidate_features import WORK as ARCHIVE, inputs, load
from scripts.experiments.ranking.overlap_ranking import fit_logistic, probability
from scripts.experiments.ranking.overlap_ranking_review import nearby
from scripts.experiments.relocalization.relocalization_selector import (
    POLICY as PRIOR_REFERENCE, eligible, guard, rank as previous_rank, refine,
    sealed as previous_policy, snapshot, visible_anchors_preserved,
)
from scripts.experiments.identity.syringe_assembly_model import training_weights
from scripts.experiments.ranking.syringe_box_quality import POLICY as BOX_QUALITY, candidate_vector, frozen as box_quality
from scripts.experiments.identity.syringe_object_verifier import partition_audit

WORK = Path("data/pair-relocalization")
REFERENCE = ART / "pair_relocalization_reference.json"
POLICY = ART / "pair_relocalization_policy.json"
RESULT = ART / "pair_relocalization_results.json"
HEADS = ("geometry", "joint", "same", "improvement")
ARMS = ("visual-difference", "geometry", "joint", "factorized")
FLOORS = (.45, .55, .65, .75, .82, .90, .94, .97, .99, 1.01)
LINK_FLOORS = (.65, .80, .92)
OWNER_MARGIN = .10
UNKNOWN = -2
BACKGROUND = -1


def owner(box, objects):
    overlaps = sorted(((box_iou(box, p["bbox"]), i) for i, p in enumerate(objects)), reverse=True)
    first, i = overlaps[0] if overlaps else (0., BACKGROUND)
    second = overlaps[1][0] if len(overlaps) > 1 else 0.
    if first <= .08:
        return BACKGROUND, first
    if first >= .22 and first - second >= .08:
        return i, first
    return UNKNOWN, first


def targets(candidate, objects, anchor, alternative):
    current, before = anchor
    other, after = alternative
    if UNKNOWN in (current, other):
        return None, None
    same = current >= 0 and current == other
    if not same:
        return False, None
    if after < .50 or after - before <= -.04 or not visible_anchors_preserved(candidate, objects[current]):
        return True, False
    if after - before >= .10:
        return True, True
    return True, None


def encoded_box(p):
    return np.asarray(p["bbox"]), np.asarray(p["descriptor"]), np.asarray(p["features"])


def pair_vector(anchor, candidate, observed):
    a, va, fa = anchor
    b, vb, fb = candidate
    a_wh, b_wh = np.maximum(a[2:] - a[:2], 1e-6), np.maximum(b[2:] - b[:2], 1e-6)
    xy = ((b[:2] + b[2:]) - (a[:2] + a[2:])) / (a_wh + b_wh)
    geometry = np.r_[box_iou(a, b), overlap_fraction(a, b), np.abs(xy), np.log(b_wh/a_wh),
                     np.log(np.prod(a_wh)), np.log(np.prod(b_wh)),
                     observed["score"], observed["corroboration"], observed["view"] == "crop",
                     observed.get("repeats", 0), va, vb, vb - va]
    appearance = np.r_[np.sum(fa * fb, axis=1), np.linalg.norm(fa - fb, axis=1), 4 * (fb-fa).ravel()]
    return geometry, appearance


def fit_models(rows, bank):
    samples = {arm: [] for arm in HEADS}
    identity_strata = []
    stats = Counter()
    for r in rows:
        objects, key, group = r["objects"], r["id"], r["group"]
        boxes = bank[key]["proposed"]
        owners = [owner(p["bbox"], objects) for p in boxes]
        vectors = [encoded_box(p) for p in boxes]
        for i, (anchor, annotated) in enumerate(zip(boxes, owners, strict=True)):
            if annotated[0] == UNKNOWN:
                continue
            for j, (candidate, assigned) in enumerate(zip(boxes, owners, strict=True)):
                if i == j or not nearby(anchor["bbox"], candidate["bbox"]):
                    continue
                same, improves = targets(candidate["bbox"], objects, annotated, assigned)
                if same is None:
                    continue
                label = "background-pair" if annotated[0] == BACKGROUND or assigned[0] == BACKGROUND else "different-instance"
                stats["same-instance" if same else label] += 1
                geometry, visual = pair_vector(vectors[i], vectors[j], candidate)
                full = np.r_[geometry, visual]
                samples["same"].append((full, same, group))
                identity_strata.append("same-instance" if same else label)
                if same and improves is None:
                    continue
                useful = same and improves
                samples["geometry"].append((geometry, useful, group))
                samples["joint"].append((full, useful, group))
                if same:
                    samples["improvement"].append((full, improves, group))
    heads = {}
    for name, examples in samples.items():
        features, labels, groups = zip(*examples, strict=True)
        x = np.stack(features)
        weights = training_weights(identity_strata if name == "same" else labels, groups)
        scaler = StandardScaler().fit(x, sample_weight=weights)
        fitted = fit_logistic(scaler.transform(x), list(labels), weights, .25)
        heads[name] = dict(**fitted, center=scaler.mean_.tolist(), scale=scaler.scale_.tolist())
        stats[f"{name}/examples"] = len(labels)
        stats[f"{name}/positive"] = sum(labels)
        stats[f"{name}/groups"] = len(set(groups))
    return heads, stats


def quality(p, model):
    v = candidate_vector(p)
    width = model["width"]
    v[:width] = (v[:width] - model["center"]) / model["scale"]
    head = model["quality"]["visual-iou"]
    return float(np.clip(v @ head["weight"] + head["bias"], 0., 1.))


def rank_pairs(base, bank, heads, quality_model):
    choices = {}
    for key, anchors in base.items():
        items = bank[key]
        if [p["bbox"] for p in anchors] != [p["bbox"] for p in items["base"]]:
            raise ValueError(f"Archived incumbents do not match {key}.")
        before = [(encoded_box(p), quality(p, quality_model)) for p in items["base"]]
        after = [(encoded_box(p), quality(p, quality_model)) for p in items["proposed"]]
        choices[key] = []
        for j, candidate in enumerate(items["proposed"]):
            box = candidate["bbox"]
            for i, anchor in enumerate(anchors):
                incumbent = anchor["bbox"]
                if not nearby(incumbent, box) or box_iou(incumbent, box) >= .98:
                    continue
                vectors, expected = after[j]
                anchor_vector, current = before[i]
                geometry, visual = pair_vector(anchor_vector, vectors, candidate)
                full = np.r_[geometry, visual]
                scores = {}
                for name, head in heads.items():
                    x = geometry if name == "geometry" else full
                    scores[name] = probability(head, (x - head["center"]) / head["scale"])
                surrounding = [p["bbox"] for k,p in enumerate(anchors) if k != i]
                choices[key].append(dict(
                    slot=i, proposal=j, bbox=box, score=candidate["score"], scores=scores,
                    visual_difference=expected-current, visual_after=expected,
                    legacy_geometry=eligible(incumbent, surrounding, box),
                ))
    return choices


def refine_image(base, choices, rule):
    arm, floor, link = rule["arm"], rule["floor"], rule["link"]
    per_box = {}
    for p in choices:
        per_box.setdefault(p["proposal"], []).append(p)
    accepted = []
    for p in choices:
        scores = p["scores"]
        other = [q for q in per_box[p["proposal"]] if q["slot"] != p["slot"]]
        if arm == "visual-difference":
            safe = p["legacy_geometry"] and p["visual_after"] >= .50
            value = p["visual_difference"]
        else:
            best_competitor = max((q["scores"]["same"] for q in other), default=0.)
            safe = not other or scores["same"] >= best_competitor + OWNER_MARGIN
            if arm == "factorized":
                safe = safe and scores["same"] >= link
                value = scores["improvement"]
            else:
                value = scores[arm]
        if safe and value >= floor:
            accepted.append((value, p))
    out = list(base)
    changed, used = set(), set()
    for value, p in sorted(accepted, key=lambda x: (-x[0], x[1]["slot"], x[1]["proposal"])):
        slot, proposal, box = p["slot"], p["proposal"], p["bbox"]
        if slot in changed or proposal in used:
            continue
        if any(box_iou(box, q["bbox"]) >= .65 for i,q in enumerate(out) if i != slot):
            continue
        out[slot] = dict(base[slot], bbox=box, prior_bbox=base[slot]["bbox"],
                         refinement_score=value, replacement_evidence=p)
        changed.add(slot)
        used.add(proposal)
    return out


def predict(base, choices, rule):
    return {key: refine_image(items, choices[key], rule) for key, items in base.items()}


def check(rows, base, before, prediction):
    result = guard(rows, base, before, prediction)
    harms = []
    for row in rows:
        key, objects = row["id"], row["objects"]
        for slot, (a, p) in enumerate(zip(base[key], prediction[key], strict=True)):
            if a["bbox"] == p["bbox"]:
                continue
            i, start = owner(a["bbox"], objects)
            j, _ = owner(p["bbox"], objects)
            if not objects:
                harms.append(dict(id=key, slot=slot, reason="edited-negative-image"))
            if i >= 0:
                end = box_iou(p["bbox"], objects[i]["bbox"])
                if j >= 0 and j != i:
                    harms.append(dict(id=key, slot=slot, reason="switched-instance", reference=i, new_reference=j))
                if start - end > .05:
                    harms.append(dict(id=key, slot=slot, reason="reduced-localization", before=start, after=end))
    return dict(**result, harmful_changes=harms, passes=result["safe"] and not harms)


def choose(rows, base, choices):
    before = snapshot(rows, base)
    trials = []
    all_rules = product(ARMS, FLOORS, LINK_FLOORS)
    rules = [dict(arm=a, floor=f, link=p if a == "factorized" else None)
             for a,f,p in all_rules if a == "factorized" or p == LINK_FLOORS[0]]
    rules.extend(dict(arm="visual-difference", floor=f, link=None) for f in (.04,.08,.12,.18,.24,.30))
    for rule in rules:
        trials.append(dict(rule=rule, **check(rows, base, before, predict(base, choices, rule))))
    def preference(t):
        return len(t["gained"]["0.5"]), len(t["gained"]["0.75"]), -t["changed"], t["rule"]["floor"]
    finalists = {arm: max((t for t in trials if t["rule"]["arm"] == arm and t["passes"]), key=preference)
                 for arm in ARMS}
    plausible = [t for t in finalists.values() if any(t["gained"].values())]
    return dict(trials=trials, finalists=finalists, winner=max(plausible, key=preference, default=None))


def bindings():
    paths = (Path(__file__), REFERENCE, BOX_QUALITY, PRIOR_REFERENCE, ARCHIVE / "features-fit.json",
             ARCHIVE / "features-development.json", Path("scripts/experiments/relocalization/pair_relocalization_reserve.py"))
    return {str(p): digest(p) for p in paths}


def measure(rows, base, raw, choices, selected, earlier):
    earlier_rule = earlier["selection"]["selected"]["rule"]
    outputs = {"previous": refine(base, previous_rank(rows, base, raw, earlier["heads"]), earlier_rule)}
    outputs.update({a: predict(base, choices, t["rule"]) for a,t in selected["finalists"].items()})
    before = snapshot(rows, base)
    return dict(baseline=evaluate(rows, base, 0.), variants={
        a: dict(metrics=evaluate(rows, p, 0.), guard=check(rows, base, before, p),
                changes={key: [item for item in items if "prior_bbox" in item]
                         for key,items in p.items() if any("prior_bbox" in x for x in items)})
        for a,p in outputs.items()})


def freeze():
    if POLICY.exists():
        raise FileExistsError(POLICY)
    cv2.setNumThreads(1)
    fit, original, fit_raw = inputs("fit")
    dev, base, dev_raw = inputs("development")
    sizes, overlap = partition_audit(fit, dev, read_json(REFERENCE)["rows"])
    heads, counts = fit_models(fit, load("fit", fit, original, fit_raw))
    choices = rank_pairs(base, load("development", dev, base, dev_raw), heads, box_quality()["models"])
    selected = choose(dev, base, choices)
    data = measure(dev, base, dev_raw, choices, selected, previous_policy())
    write_json(POLICY, dict(binding=bindings(), fitting=counts, sizes=sizes, overlap=overlap,
                            heads=heads, selection=selected, development=data))
    print("Fitting", counts, flush=True)
    print("Selected", selected["winner"], flush=True)
    print("Development", {a:(len(v["guard"]["gained"]["0.5"]), len(v["guard"]["gained"]["0.75"]),
                              v["guard"]["changed"]) for a,v in data["variants"].items()}, flush=True)


def frozen():
    lock = read_json(POLICY)
    if lock["binding"] != bindings():
        raise ValueError("The frozen replacement policy inputs changed.")
    return lock


def score():
    from scripts.experiments.relocalization.pair_relocalization_reserve import packets

    cv2.setNumThreads(1)
    lock = frozen()
    previous, quality_model = previous_policy(), box_quality()["models"]
    outputs = {}
    for role, (rows, base, raw, bank) in packets().items():
        choices = rank_pairs(base, bank, lock["heads"], quality_model)
        outputs[role] = measure(rows, base, raw, choices, lock["selection"], previous)
    winner = lock["selection"]["winner"]
    qualified = False
    if winner:
        arm = winner["rule"]["arm"]
        qualified = all(a["variants"][arm]["guard"]["passes"] for a in outputs.values()) and any(
            outputs["reserve"]["variants"][arm]["guard"]["gained"].values())
    write_json(RESULT, dict(policy=digest(POLICY), selected=winner, qualified=qualified, cohorts=outputs))
    for role, outcome in outputs.items():
        print(role, "baseline", outcome["baseline"]["tp"],
              {a:(v["metrics"]["tp"], v["guard"]["changed"], v["guard"]["passes"])
               for a,v in outcome["variants"].items()}, flush=True)
    print("Qualified", qualified, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "score"))
    {"freeze": freeze, "score": score}[cli.parse_args().stage]()
