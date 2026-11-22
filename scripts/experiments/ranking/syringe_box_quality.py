"""Learn whether a crop contains a syringe and how well its box localizes it."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate, match
from scripts.experiments.crowds.native_candidate_verifier import simple_metrics
from scripts.experiments.occlusion.occlusion_detector import ART
from scripts.experiments.ranking.overlap_candidate_features import WORK as FEATURES, inputs, load
from scripts.experiments.ranking.overlap_ranking import (
    LOCK as UPSTREAM, audit, fit_logistic, predict_evidence, probability, sealed, select, unary,
)
from scripts.experiments.ranking.overlap_ranking_review import nearby, tags
from scripts.experiments.identity.syringe_assembly_model import training_weights
from scripts.experiments.identity.syringe_object_features import rules as upstream_rules
from scripts.experiments.identity.syringe_object_verifier import partition_audit

WORK = Path("data/syringe-box-quality")
REFERENCE = ART / "syringe_box_quality_reference.json"
POLICY = ART / "syringe_box_quality_policy.json"
RESULT = ART / "syringe_box_quality_results.json"
FLOORS = (.40, .50, .60, .70, .75, .80, .85, .90, .94, .97, .99, 1.01)
ARMS = {
    "coupled": ("rank", "rank", False),
    "geometry-iou": ("rank", "geometry-iou", False),
    "visual-iou": ("rank", "visual-iou", False),
    "pair-iou": ("rank", "pair-iou", False),
    "identity-only": ("identity", "identity", False),
    "identity-pair-iou": ("identity", "pair-iou", False),
    "shared-identity-pair-iou": ("identity", "pair-iou", True),
}


def supervise(row, proposals):
    """Keep clear background and observed syringe crops. Skip overlapping identities."""
    labeled, instances = [], defaultdict(list)
    for i, p in enumerate(proposals):
        label, target, iou = tags(p["bbox"], row["objects"])
        if label == "ambiguous":
            continue
        labeled.append((i, label, iou, target))
        if target is not None:
            instances[target].append((i, iou))
    pairs = []
    for options in instances.values():
        ordered = sorted(options, key=lambda entry: -entry[1])
        for (best, hi), (worse, lo) in combinations(ordered, 2):
            if hi >= .50 and hi - lo >= .12 and nearby(proposals[best]["bbox"], proposals[worse]["bbox"]):
                pairs.append((best, worse, hi - lo))
    return labeled, pairs


def candidate_vector(p):
    """Use the archived texture and embedding features with scale and aspect information."""
    x, y, r, b = p["bbox"]
    width, height = r - x, b - y
    return np.r_[np.log(max(width * height, 1e-9)), np.log(max(width, height) / max(min(width, height), 1e-9)),
                 unary(p)]


def weights(keys):
    count = Counter(keys)
    return np.asarray([len(keys) / (len(count) * count[k]) for k in keys])


def estimate_iou(features, target, sample_weight, differences=None):
    model = Ridge(alpha=18., solver="cholesky")
    if differences is None:
        model.fit(features, target, sample_weight=sample_weight)
        return dict(weight=model.coef_.tolist(), bias=float(model.intercept_))
    x, delta, weight = differences
    center = np.average(features, axis=0, weights=sample_weight)
    mean = np.average(target, weights=sample_weight)
    combined = np.vstack((features - center, x))
    model.set_params(fit_intercept=False).fit(combined, np.r_[target - mean, delta],
                                              sample_weight=np.r_[sample_weight, weight])
    return dict(weight=model.coef_.tolist(), bias=float(mean - center @ model.coef_))


def fit_models(rows, bank):
    features, classes, groups, targets, owner, local, deltas, delta_groups = [], [], [], [], [], [], [], []
    stats = Counter()
    for row in rows:
        proposals = bank[row["id"]]["proposed"]
        vectors = [candidate_vector(p) for p in proposals]
        labeled, pairs = supervise(row, proposals)
        index = {i: len(features) + j for j, (i, _, _, _) in enumerate(labeled)}
        for i, label, iou, instance in labeled:
            features.append(vectors[i])
            classes.append(instance is not None)
            targets.append(iou)
            groups.append(row["group"])
            stats[label] += 1
            if instance is not None:
                local.append(index[i])
                owner.append((row["group"], row["id"], instance))
        for first, second, margin in pairs:
            deltas.append((index[first], index[second], margin))
            delta_groups.append(row["group"])
    matrix = np.stack(features)
    width = matrix.shape[1] - 768
    normalizer = StandardScaler().fit(matrix[:, :width], sample_weight=weights(groups))
    matrix[:, :width] = normalizer.transform(matrix[:, :width])
    identity = fit_logistic(matrix, classes, training_weights(classes, groups), .25)
    full = matrix[local]
    target = np.asarray(targets)[local]
    location_weight = weights(owner)
    first, second, delta = map(np.asarray, zip(*deltas, strict=True))
    difference = matrix[first] - matrix[second]
    quality = {
        "geometry-iou": estimate_iou(full[:, :width], target, location_weight),
        "visual-iou": estimate_iou(full, target, location_weight),
        "pair-iou": estimate_iou(full, target, location_weight,
                                 (difference, delta, .65 * len(local) / len(delta) * weights(delta_groups))),
    }
    stats.update(images=len(rows), groups=len(set(groups)), quality_crops=len(local),
                 same_instance_pairs=len(deltas), pair_groups=len(set(delta_groups)),
                 pairs_with_truncation=int(sum(targets[j] < .5 for j in second)))
    return dict(identity=identity, quality=quality, width=width,
                center=normalizer.mean_.tolist(), scale=normalizer.scale_.tolist()), stats


def score_candidates(rows, bank, model, heads):
    ranked, affinity = predict_evidence(rows, bank, heads)
    for key, proposals in ranked.items():
        for p, source in zip(proposals, bank[key]["proposed"], strict=True):
            x = candidate_vector(source)
            width = model["width"]
            x[:width] = (x[:width] - model["center"]) / model["scale"]
            p["identity"] = probability(model["identity"], x)
            for name, head in model["quality"].items():
                dims = len(head["weight"])
                p[name] = float(np.clip(x[:dims] @ head["weight"] + head["bias"], 0., 1.))
    return ranked, affinity


def order_boxes(base, ranked, relation, arm, rule, floor):
    presence, order, share = ARMS[arm]
    count = len(base)
    effective = [p[presence] for p in ranked]
    if share:
        for i, j in combinations(range(len(ranked)), 2):
            affinity = relation.get((i + count, j + count), 1.)
            a, b = ranked[i], ranked[j]
            if affinity <= .05 and box_iou(a["bbox"], b["bbox"]) >= .35:
                if a[order] > b[order] + .04:
                    effective[i] = max(effective[i], b[presence])
                elif b[order] > a[order] + .04:
                    effective[j] = max(effective[j], a[presence])
    prepared = [dict(p, preference=p[order] if score >= floor else -1., effective_identity=score)
                for p, score in zip(ranked, effective, strict=True)]
    return select(base, prepared, relation, dict(rule, ranker="preference", floor=0., joint=False))


def predict(base, ranked, relations, arm, rule, floor):
    return {key: [*boxes, *order_boxes(boxes, ranked[key], relations[key], arm, rule, floor)]
            for key, boxes in base.items()}


def localization(rows, bank, ranked):
    pairs, examples = [], []
    for row in rows:
        key = row["id"]
        labels, local_pairs = supervise(row, bank[key]["proposed"])
        values = ranked[key]
        pairs.extend((values[i], values[j]) for i, j, delta in local_pairs)
        examples.extend((values[i], iou) for i, label, iou, instance in labels if instance is not None)
    results = {}
    for arm in ("rank", "identity", "geometry-iou", "visual-iou", "pair-iou"):
        correct = sum(1. if p[arm] > q[arm] else .5 if p[arm] == q[arm] else 0. for p, q in pairs)
        error = np.mean([abs(p[arm] - iou) for p, iou in examples]) if arm.endswith("-iou") and examples else None
        results[arm] = dict(ordered_correctly=correct / len(pairs) if pairs else None,
                            mean_absolute_iou_error=error)
    return dict(pairs=len(pairs), associated_boxes=len(examples), arms=results)


def controls(base, ranked, relations):
    return {name: {k: [*boxes, *select(boxes, ranked[k], relations[k], rule)] for k, boxes in base.items()}
            for name, rule in upstream_rules().items()}


def choose(rows, base, ranked, relations, conflict_rule):
    before, strict = simple_metrics(rows, base), match(rows, base, .75)[1]
    trials = []
    for arm, floor in product(ARMS, FLOORS):
        output = predict(base, ranked, relations, arm, conflict_rule, floor)
        trials.append(dict(arm=arm, floor=floor, **audit(rows, before, strict, output)))
    finalists = {arm: max((t for t in trials if t["arm"] == arm and t["passes"]), key=priority)
                 for arm in ARMS}
    candidates = [t for arm, t in finalists.items() if arm != "coupled" and t["gain"] > 0]
    winner = max(candidates, key=priority, default=None)
    return dict(trials=trials, finalists=finalists, winner=winner)


def priority(t):
    return t["gain"], t["gains_at_75"], t["exact_count"], -t["fp"], t["floor"]


def bindings():
    paths = (Path(__file__), REFERENCE, UPSTREAM, FEATURES / "features-fit.json", FEATURES / "features-development.json")
    return {str(p): digest(p) for p in paths}


def freeze():
    if POLICY.exists():
        raise FileExistsError(POLICY)
    cv2.setNumThreads(1)
    fit, start, archived = inputs("fit")
    dev, base, raw = inputs("development")
    sizes, overlaps = partition_audit(fit, dev, read_json(REFERENCE)["rows"])
    model, counts = fit_models(fit, load("fit", fit, start, archived))
    bank = load("development", dev, base, raw)
    ranked, affinity = score_candidates(dev, bank, model, sealed()["heads"])
    rule = upstream_rules()["calibrated"]
    selection = choose(dev, base, ranked, affinity, rule)
    metrics = {k: evaluate(dev, p, 0.) for k, p in controls(base, ranked, affinity).items()}
    write_json(POLICY, dict(bindings=bindings(), models=model, fit=counts, sizes=sizes, overlaps=overlaps,
                            conflict_rule=rule, selection=selection, development=dict(
                                baseline=evaluate(dev, base, 0.), controls=metrics, localization=localization(dev, bank, ranked))))
    print("Fit", dict(counts), flush=True)
    print("Development", {k:(t["floor"], t["tp"], t["fp"], t["gains_at_75"])
                          for k,t in selection["finalists"].items()}, flush=True)
    print("Selected", selection["winner"], flush=True)


def frozen():
    state = read_json(POLICY)
    if state["bindings"] != bindings():
        raise ValueError("The fitted or selected inputs changed.")
    return state


def measure(rows, base, bank, state, upstream):
    ranked, relations = score_candidates(rows, bank, state["models"], upstream)
    initial, strict = simple_metrics(rows, base), match(rows, base, .75)[1]
    outputs = controls(base, ranked, relations)
    variants = {f"previous-{name}": boxes for name, boxes in outputs.items()}
    variants.update({name: predict(base, ranked, relations, name, state["conflict_rule"], t["floor"])
                     for name, t in state["selection"]["finalists"].items()})
    return dict(baseline=evaluate(rows, base, 0.), localization=localization(rows, bank, ranked),
                variants={name: dict(metrics=evaluate(rows, p, 0.), delta=audit(rows, initial, strict, p),
                                     additions={k: boxes[len(base[k]):] for k, boxes in p.items() if len(boxes) > len(base[k])})
                          for name, p in variants.items()})


def score():
    from scripts.experiments.ranking.syringe_box_quality_reserve import packets

    state = frozen()
    upstream = sealed()["heads"]
    results = {role: measure(rows, base, bank, state, upstream) for role, (rows, base, bank) in packets().items()}
    winner = state["selection"]["winner"]
    qualified = False
    if winner:
        tests = [r["variants"][winner["arm"]]["delta"] for r in results.values()]
        qualified = all(t["passes"] for t in tests) and tests[0]["gain"] > 0
    write_json(RESULT, dict(policy=digest(POLICY), winner=winner, qualified=qualified, cohorts=results))
    for role, data in results.items():
        print(role, {k:(v["metrics"]["tp"],v["metrics"]["fp"],v["delta"]["passes"])
                     for k,v in data["variants"].items()}, flush=True)
    print("Qualified", qualified, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "score"))
    {"freeze": freeze, "score": score}[cli.parse_args().stage]()
