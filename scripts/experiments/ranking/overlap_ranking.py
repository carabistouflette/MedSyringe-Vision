"""Compare same-scene ranking and visually supervised duplicate conflicts."""
import argparse
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.detection.additive import duplicates, overlap_fraction
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate, match, scores
from scripts.experiments.ranking.complete_syringe_selection import best_compatible
from scripts.experiments.ranking.instance_separation import pair_vector
from scripts.experiments.crowds.native_candidate_verifier import compare, simple_metrics
from scripts.experiments.occlusion.occlusion_detector import raster
from scripts.experiments.occlusion.occlusion_proposals import oracle
from scripts.experiments.occlusion.occlusion_relocalization import prepare_reference
from scripts.experiments.ranking.overlap_candidate_features import ART, WORK, inputs, load
from scripts.experiments.ranking.overlap_ranking_review import REVIEW, load as reviews, nearby, pair_label, tags
from scripts.experiments.identity.syringe_assembly_model import training_weights

LOCK = ART / "overlap_ranking_policy.json"
RESULT = ART / "overlap_ranking_results.json"
ARMS = ("point-overlap", "rank-overlap", "point-learned", "rank-learned")
SEED = 917
CAP = 16
FLOORS = (.40, .50, .60, .65, .70, .75, .80, .85, .90, .94, .97, 1.01)


def unary(p):
    return np.r_[
        p["score"], p["corroboration"], p["view"] == "crop", p.get("repeats", 0),
        p["descriptor"], 5*np.asarray(p["features"]).ravel(),
    ]


def pair_features(photo, a, b, cache):
    va = np.asarray(a["features"])
    vb = np.asarray(b["features"])
    shape = np.asarray(pair_vector(a["bbox"], b["bbox"], photo, cache))
    reverse = np.asarray(pair_vector(b["bbox"], a["bbox"], photo, cache))
    return np.r_[(shape + reverse) / 2, np.abs(shape - reverse), np.sum(va * vb, axis=1),
                 min(a["score"], b["score"]), max(a["score"], b["score"])]


def fit_logistic(x, y, weights, c):
    matrix = np.asarray(x)
    model = LogisticRegression(C=c, random_state=SEED, max_iter=750)
    model.fit(matrix, y, sample_weight=weights)
    if max(model.n_iter_) == model.max_iter:
        raise ValueError("Classifier did not converge.")
    return dict(coef=model.coef_[0].tolist(), bias=float(model.intercept_[0]))


def fit_ranker(points, labels, groups, differences, ordered_groups):
    geometry = len(points[0]) - 2*384
    scaler = StandardScaler().fit(np.asarray(points)[:, :geometry])

    def scaled(matrix, delta=False):
        x = np.asarray(matrix).copy()
        x[:, :geometry] = (x[:, :geometry] - (0 if delta else scaler.mean_)) / scaler.scale_
        return x

    values, weights = scaled(points), training_weights(labels, groups)
    point = fit_logistic(values, labels, weights, .25)
    contrasts = scaled([*differences, *(-p for p in differences)], True)
    order_labels = [True] * len(differences) + [False] * len(differences)
    order_weights = training_weights(order_labels, ordered_groups * 2)
    rank_weight = .40 * len(labels) / len(order_labels)
    ranker = fit_logistic(np.vstack([values, contrasts]), [*labels, *order_labels],
                          np.r_[.60*weights, rank_weight*order_weights], .25)
    return dict(point=point, rank=ranker, geometry=geometry,
                center=scaler.mean_.tolist(), scale=scaler.scale_.tolist())


def fit_pairs(x, y, groups, weights):
    observed = training_weights(y, groups) * np.asarray(weights)
    scaler = StandardScaler().fit(x, sample_weight=observed)
    model = fit_logistic(scaler.transform(x), y, observed, .4)
    return dict(**model, center=scaler.mean_.tolist(), scale=scaler.scale_.tolist())


def learn(rows, frames, review):
    points, labels, groups, differences, ordered_groups = [], [], [], [], []
    matrix, outcomes, owners, pair_weights = [], [], [], []
    entries = Counter()
    reviewed = {r["id"]: {(s["i"], s["j"]): s for s in review if s["image"] == r["id"]} for r in rows}
    for row in rows:
        key, group, objects = row["id"], row["group"], row["objects"]
        image = raster(row)
        ps = frames[key]["proposed"]
        vectors = [unary(p) for p in ps]
        targets = [tags(p["bbox"], objects) for p in ps]
        positive = []
        negative = []
        for i, (label, target, overlap) in enumerate(targets):
            entries[f"boxes/{label}"] += 1
            if label == "ambiguous":
                continue
            is_positive = label == "complete"
            points.append(vectors[i])
            labels.append(is_positive)
            groups.append(group)
            (positive if is_positive else negative).append(i)
        for i, j in product(positive[:6], negative[:8]):
            differences.append(vectors[i] - vectors[j])
            ordered_groups.append(group)
            entries["contrasts/valid-vs-fragment-or-background"] += 1
        for i, j in combinations(positive, 2):
            a, b = targets[i], targets[j]
            if a[1] == b[1] and abs(a[2]-b[2]) >= .15:
                differences.append(np.sign(a[2]-b[2])*(vectors[i]-vectors[j]))
                ordered_groups.append(group)
                entries["contrasts/better-localization"] += 1
        cache = {}
        for i, j in combinations(range(len(ps)), 2):
            a, b = ps[i], ps[j]
            if not nearby(a["bbox"], b["bbox"]):
                continue
            human = reviewed[key].get((i, j))
            if human and (a["bbox"] != human["first"] or b["bbox"] != human["second"]):
                raise ValueError("Visual review does not match frozen proposal pair.")
            label = human["reviewed"] if human else pair_label(targets[i], targets[j])
            if label not in ("same", "different"):
                continue
            matrix.append(pair_features(image, a, b, cache))
            outcomes.append(label == "different")
            owners.append(group)
            pair_weights.append(4. if human else 1.)
            entries[f"pairs/{'reviewed' if human else 'geometric'}/{label}"] += 1
    heads = fit_ranker(points, labels, groups, differences, ordered_groups)
    heads["dedup"] = fit_pairs(matrix, outcomes, owners, pair_weights)
    return heads, dict(counts=entries, fitting_groups=len(set(groups)), distinct_groups=len({g for g, y in zip(owners, outcomes) if y}))


def probability(head, x):
    return float(expit(np.asarray(x) @ head["coef"] + head["bias"]))


def predict_evidence(rows, frames, heads):
    scored, pair_scores = {}, {}
    for row in rows:
        k, image = row["id"], raster(row)
        originals = frames[k]["base"]
        proposed = frames[k]["proposed"]
        scored[k] = []
        for p in proposed:
            x = unary(p)
            end = heads["geometry"]
            x[:end] = (x[:end] - heads["center"]) / heads["scale"]
            scored[k].append(dict(bbox=p["bbox"], score=p["score"], point=probability(heads["point"], x),
                                  rank=probability(heads["rank"], x)))
        cache = {}
        relation = {}
        head = heads["dedup"]
        boxes = [*originals, *proposed]
        for i, j in combinations(range(len(boxes)), 2):
            a, b = boxes[i], boxes[j]
            if j < len(originals) or not nearby(a["bbox"], b["bbox"]):
                continue
            vector = pair_features(image, a, b, cache)
            relation[i, j] = probability(head, (vector - head["center"])/head["scale"])
        pair_scores[k] = relation
    return scored, pair_scores


def conflict(a, b, affinity, rule):
    if box_iou(a, b) >= .84:
        return True
    overlaps = duplicates(a, b)
    if rule["conflicts"] == "overlap" or affinity is None:
        return overlaps
    # Apply a learned duplicate veto only to physically close pairs.
    if affinity <= .10 and nearby(a, b):
        return True
    return overlaps and affinity < rule["distinct"]


def select(original, candidates, affinity, rule):
    base_boxes = [p["bbox"] for p in original]
    blocked_by = {}
    order = []
    count = len(base_boxes)
    for i, p in enumerate(candidates, count):
        value = p[rule["ranker"]]
        if value < rule["floor"]:
            continue
        if any(conflict(b, p["bbox"], affinity.get((j,i)), rule) for j,b in enumerate(base_boxes)):
            continue
        blocked_by[i] = p
        order.append((i, value))
    order.sort(key=lambda q: (-q[1], q[0]))
    order = order[:CAP]
    edges = [set() for _ in order]
    for a, b in combinations(range(len(order)),2):
        i, j = order[a][0], order[b][0]
        if conflict(blocked_by[i]["bbox"], blocked_by[j]["bbox"], affinity.get(tuple(sorted((i,j)))), rule):
            edges[a].add(b)
            edges[b].add(a)
    if rule["joint"]:
        chosen = best_compatible([v-rule["floor"]+.03 for _, v in order], edges)
    else:
        chosen = []
        for i, blocked in enumerate(edges):
            if blocked.isdisjoint(chosen):
                chosen.append(i)
    return [dict(blocked_by[order[i][0]], score=.002+.001*order[i][1], rank_score=order[i][1])
            for i in chosen]


def predictions(baseline, candidates, pair_scores, rule):
    return {k: [*base, *select(base, candidates[k], pair_scores[k], rule)] for k, base in baseline.items()}


def audit(rows, before, strict, output):
    delta = compare(rows, before, output)
    matched = match(rows, output, .75)[1]
    gains = sum(len(matched[k] - old) for k, old in strict.items())
    losses = sum(len(old - matched[k]) for k, old in strict.items())
    return dict(**delta, gains_at_75=gains, losses_at_75=losses, passes=delta["safe"] and not losses)


def grid(rows, baseline, candidates, pair_scores):
    before = simple_metrics(rows, baseline)
    strict = match(rows, baseline, .75)[1]
    total = sum(len(r["objects"]) for r in rows)
    trials = []
    for arm, floor, joint in product(ARMS, FLOORS, (False, True)):
        ranker, edges = arm.split("-")
        for distinct in ((.70, .82, .94) if edges == "learned" else (1.,)):
            rule = dict(arm=arm, floor=floor, joint=joint, ranker=ranker, conflicts=edges, distinct=distinct)
            output = predictions(baseline, candidates, pair_scores, rule)
            trials.append(dict(rule=rule, **audit(rows, before, strict, output),
                               changes=sum(len(p)-len(baseline[k]) for k,p in output.items())))
    def order(t):
        return t["gain"], t["gains_at_75"], t["exact_count"], -t["changes"], not t["rule"]["joint"]

    finalist, active = {}, {}
    for arm in ARMS:
        pool = [t for t in trials if t["rule"]["arm"] == arm]
        finalist[arm] = max((t for t in pool if t["passes"]), key=order)
        probe = max((t for t in pool if t["changes"]), default=None, key=lambda t: (
            scores(t["tp"],t["fp"],total-t["tp"])["f1"], t["gain"], -t["extra_fp"], -t["changes"]))
        if probe:
            active[arm] = probe
    winner = max((t for t in finalist.values() if t["gain"] > 0), key=order, default=None)
    return dict(finalists=finalist, probes=active, winner=winner, trials=trials)


def dependencies():
    return {str(p): digest(p) for p in (Path(__file__), "scripts/experiments/ranking/overlap_candidate_features.py",
                                      "scripts/experiments/ranking/overlap_ranking_review.py", "scripts/experiments/ranking/instance_separation.py",
                                      REVIEW, WORK/"features-fit.json", WORK/"features-development.json")}


def freeze():
    if LOCK.exists():
        raise FileExistsError(LOCK)
    cv2.setNumThreads(1)
    rows, originals, proposals = inputs("fit")
    fitting = load("fit", rows, originals, proposals)
    dev, base, raw = inputs("development")
    data = load("development", dev, base, raw)
    _, _, bindings, _ = prepare_reference()
    if {r["group"] for r in rows} & {r["group"] for r in dev}:
        raise ValueError("Development and training acquisitions intersect.")
    heads, details = learn(rows, fitting, reviews())
    print("Fit", details, flush=True)
    crops, pairs = predict_evidence(dev, data, heads)
    selection = grid(dev, base, crops, pairs)
    write_json(LOCK, dict(heads=heads, fitting=details, seed=SEED, development=evaluate(dev,base,0.),
                          selection=selection, sources=dependencies(), releases=bindings))
    print("Development selection", selection["finalists"], selection["winner"], flush=True)


def sealed():
    p = read_json(LOCK)
    if p["sources"] != dependencies() or any(digest(f)!=h for f,h in p["releases"].items()):
        raise ValueError("Frozen policy training inputs changed.")
    return p


def score():
    cv2.setNumThreads(1)
    policy = sealed()
    cohorts, records = {}, {}
    for role in ("fresh", "replay"):
        rows, base, raw = inputs(role)
        data = load(role, rows, base, raw)
        crops, pairs = predict_evidence(rows, data, policy["heads"])
        before, strict = simple_metrics(rows,base), match(rows,base,.75)[1]
        measured = {}
        for kind in ("finalists", "probes"):
            for arm, t in policy["selection"][kind].items():
                pred = predictions(base, crops, pairs, t["rule"])
                measured[f"{kind}/{arm}"] = dict(delta=audit(rows,before,strict,pred),
                    metrics=evaluate(rows,pred,0.), additions={k:p[len(base[k]):] for k,p in pred.items() if len(p)>len(base[k])})
        cohorts[role] = dict(baseline=evaluate(rows,base,0.), variants=measured,
                             proposal_ceiling={r["id"]:oracle(r["objects"],crops[r["id"]]) for r in rows})
        records[role] = digest(WORK/f"features-{role}.json")
        print(role, {k:(p["delta"]["gain"],p["delta"]["extra_fp"],p["delta"]["passes"])
                     for k,p in measured.items()}, flush=True)
    winner = policy["selection"]["winner"]
    qualified = False
    if winner:
        arm = f"finalists/{winner['rule']['arm']}"
        fresh, replay = (cohorts[y]["variants"][arm]["delta"] for y in ("fresh","replay"))
        qualified = bool(fresh["gain"] > 0 and fresh["passes"] and replay["passes"])
    write_json(RESULT, dict(policy=digest(LOCK), feature_caches=records, cohorts=cohorts, qualified=qualified))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze","score"))
    {"freeze":freeze,"score":score}[parser.parse_args().stage]()
