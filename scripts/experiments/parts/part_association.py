"""Learn which visible parts belong to one syringe and test additive recovery."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.detection.additive import duplicates
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate, match, scores
from medsyringe.measurement.part_geometry import PARTS, area, containment
from scripts.experiments.crowds.native_candidate_verifier import compare, simple_metrics
from scripts.experiments.occlusion.occlusion_detector import ART, raster
from scripts.experiments.occlusion.occlusion_relocalization import prepare_reference
from scripts.experiments.parts.part_association_observations import WORK, inputs, observations
from scripts.experiments.parts.part_association_reference import REFERENCE, reviewed
from scripts.experiments.ranking.syringe_action_verifier import POLICY as ACTIONS, rank as rank_actions, sealed as action_model
from scripts.experiments.identity.syringe_assembly_model import descriptor, training_weights

LOCK = ART / "part_association_policy.json"
RESULT = ART / "part_association_results.json"
ARMS = ("box", "unlinked", "linked")
FLOORS = (.44, .52, .60, .68, .72, .76, .80, .84, .88, .92, .96, .98, 1.01)
SEED = 411


def center(box):
    return (np.asarray(box[:2]) + box[2:]) / 2


def part_labels(objects, item, part, size):
    """Match a detected part to a visible manual landmark with a distinct nearest owner."""
    p = center(item["bbox"]) * size
    ranked = []
    for i, obj in enumerate(objects):
        point = obj.get("parts", {}).get(part)
        if point is not None:
            box = np.asarray(obj["bbox"])
            diagonal = max(1., np.linalg.norm((box[2:] - box[:2]) * size))
            distance = np.linalg.norm(p - np.asarray(point) * size) / diagonal
            ranked.append((float(distance), i))
    ranked.sort()
    if not ranked or ranked[0][0] > .135 or (len(ranked) > 1 and ranked[1][0] - ranked[0][0] < .025):
        return None
    return ranked[0][1]


def part_nodes(image, observed):
    nodes = []
    for key in PARTS:
        for item in observed[key]:
            box = item["bbox"]
            normalized = np.asarray(box[2:]) - box[:2]
            geometry = None if key != "barrel" else descriptor(image, box)
            nodes.append(dict(item, id=len(nodes), name=key, center=center(box)*image.size,
                              width=normalized[0]*image.width, height=normalized[1]*image.height,
                              axis=geometry["axis"] if geometry else None))
    return nodes


def pair_vector(barrel, second):
    shift = second["center"] - barrel["center"]
    length = max(1., np.hypot(barrel["width"], barrel["height"]))
    axis = barrel["axis"]
    along, across = abs(shift @ axis) / length, abs(shift @ np.array([-axis[1], axis[0]])) / length
    return [barrel["score"], second["score"], along, across, np.linalg.norm(shift) / length,
            box_iou(barrel["bbox"], second["bbox"]), np.log(area(second["bbox"]) / area(barrel["bbox"])),
            *[float(second["name"] == name) for name in PARTS[1:]]]


def pairs(nodes):
    for a, b in product((n for n in nodes if n["name"] == "barrel"),
                        (n for n in nodes if n["name"] != "barrel")):
        yield a, b, pair_vector(a, b)


def supervised(rows, frames):
    vectors, labels, groups, by_type = [], [], [], Counter()
    audit = Counter()
    for row in rows:
        key, objects = row["id"], row["objects"]
        if not any(o.get("parts") for o in objects):
            continue
        image = raster(row)
        nodes = part_nodes(image, frames[key])
        tagged = {n["id"]: part_labels(objects, n, n["name"], image.size) for n in nodes}
        audit.update(n["name"] for n in nodes if tagged[n["id"]] is not None)
        for a, b, x in pairs(nodes):
            first, second = tagged[a["id"]], tagged[b["id"]]
            if first is None or second is None:
                continue
            vectors.append(x)
            labels.append(first == second)
            groups.append(row["group"])
            by_type[b["name"], first == second] += 1
    sizes = {label: len({group for group, y in zip(groups, labels, strict=True) if y == label})
             for label in (False, True)}
    summary = dict(same=sum(labels), different=len(labels)-sum(labels), same_groups=sizes[True],
                   different_groups=sizes[False], detections_by_part=audit,
                   pairs_by_part={f"{key}/{value}": n for (key, value), n in by_type.items()})
    return np.asarray(vectors), np.asarray(labels), groups, summary


def fit(rows, frames):
    x, y, groups, stats = supervised(rows, frames)
    if min(stats["same_groups"], stats["different_groups"]) < 2:
        return None, stats
    w = training_weights(y, groups)
    scaler = StandardScaler().fit(x, sample_weight=w)
    model = LogisticRegression(C=.35, max_iter=600, random_state=SEED)
    model.fit(scaler.transform(x), y, sample_weight=w)
    return dict(center=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coef=model.coef_[0].tolist(), bias=float(model.intercept_[0])), stats


def confidence(head, x):
    z = (np.asarray(x) - head["center"]) / head["scale"]
    return float(expit(z @ head["coef"] + head["bias"]))


def geometric_affinity(x):
    along, across = x[2:4]
    if x[-1]:
        return float(np.exp(-3.0 * across)) if along <= .8 else 0.
    return float(np.exp(-4.0 * across)) if .12 <= along <= 1.8 else 0.


def graph(nodes, head):
    return {(a["id"], b["id"]): confidence(head, x) if head else geometric_affinity(x)
            for a, b, x in pairs(nodes)}


def evidence(box, nodes, links):
    enclosed = [n for n in nodes if containment(n["bbox"], box) >= .65
                and area(n["bbox"]) <= area(box) * (.95 if n["name"] == "barrel" else .68)]
    barrels = [n for n in enclosed if n["name"] == "barrel"]
    others = [n for n in enclosed if n["name"] != "barrel"]
    independent = max((min(a["score"], b["score"]) for a in barrels for b in others), default=0.)
    best, assigned = 0., []
    for a in barrels:
        supported = [(min(a["score"], b["score"]) * links[a["id"], b["id"]], b["id"]) for b in others]
        if supported:
            score, secondary = max(supported)
            if score > best:
                best, assigned = score, [a["id"], secondary]
    return dict(unlinked=independent, linked=best, assigned=assigned)


def rank(rows, base, frames, sampled, head, action_heads):
    ranked, original = {}, {}
    actions = rank_actions(rows, base, frames, action_heads)
    for row in rows:
        k = row["id"]
        nodes = part_nodes(raster(row), sampled[k])
        links = graph(nodes, head)
        original[k] = [evidence(p["bbox"], nodes, links) for p in base[k]]
        ranked[k] = [dict(bbox=p["bbox"], score=p["score"], box=p["decisions"]["visual"]["add"],
                          **evidence(p["bbox"], nodes, links)) for p in actions[k]]
    return ranked, original


def conflict(a, b, rule):
    overlap = box_iou(a["bbox"], b["bbox"])
    if overlap >= .72:
        return True
    reliable = min(a["linked"], b["linked"]) >= rule["anchor"]
    if rule["exclusive"] and reliable:
        return bool(set(a["assigned"]) & set(b["assigned"]))
    return duplicates(a["bbox"], b["bbox"])


def apply(base, offered, incumbent_evidence, rule):
    predictions, events = {}, {}
    name, weight = rule["arm"], rule["weight"]
    for key, incumbent in base.items():
        occupied = [{**p, **q} for p, q in zip(incumbent, incumbent_evidence[key], strict=True)]
        ranked = sorted(((p["box"] * (1 - weight) + p[name] * weight, p) for p in offered[key]),
                        key=lambda q: (-q[0], -q[1]["score"]))
        picks = []
        for score, p in ranked:
            if score < rule["floor"]:
                break
            if any(conflict(p, q, rule) for q in occupied):
                continue
            occupied.append(p)
            picks.append(dict(p, fusion_score=score))
        predictions[key] = [*incumbent, *(dict(bbox=p["bbox"], score=p["score"]) for p in picks)]
        if picks:
            events[key] = picks
    return predictions, events


def audit(rows, before, strict, pred, events):
    delta = compare(rows, before, pred)
    _, matched = match(rows, pred, .75)
    losses = sum(len(v - matched[k]) for k, v in strict.items())
    gains = sum(len(v - strict[k]) for k, v in matched.items())
    return dict(**delta, strict_gain=gains, strict_loss=losses,
                additions=sum(map(len, events.values())), qualified=delta["safe"] and losses == 0)


def choose(rows, base, candidates, known):
    before, (_, strict) = simple_metrics(rows, base), match(rows, base, .75)
    trials = []
    for arm in ARMS:
        links = ((False, .25), (True, .20), (True, .30), (True, .45)) if arm == "linked" else ((False, .25),)
        weights = (0.,) if arm == "box" else (.25, .5, .75)
        for (exclusive, anchor), weight, floor in product(links, weights, FLOORS):
            rule = dict(arm=arm, exclusive=exclusive, anchor=anchor, weight=weight, floor=floor)
            pred, events = apply(base, candidates, known, rule)
            trials.append(dict(rule=rule, **audit(rows, before, strict, pred, events)))

    def order(t):
        return t["gain"], t["strict_gain"], -t["additions"], t["rule"]["floor"]

    finalists = {arm: max((t for t in trials if t["qualified"] and t["rule"]["arm"] == arm), key=order)
                 for arm in ARMS}
    active = {}
    for arm in ARMS:
        best = max((t for t in trials if t["additions"] and t["rule"]["arm"] == arm), default=None,
                   key=lambda t: (scores(t["tp"], t["fp"], sum(len(r["objects"]) for r in rows) - t["tp"])["f1"],
                                  t["gain"], -t["extra_fp"], -t["strict_loss"], -t["additions"]))
        if best:
            active[arm] = best
    winner = max((t for t in finalists.values() if t["gain"] > 0), default=None, key=order)
    return dict(trials=trials, finalists=finalists, active=active, selected=winner)


def link_audit(rows, observed, head):
    x, y, _, counts = supervised(rows, observed)
    if len(y):
        z = [confidence(head, a) if head else geometric_affinity(a) for a in x]
        counts["same_accepted_at_0.5"] = int(sum(v >= .5 and t for v, t in zip(z, y, strict=True)))
        counts["different_accepted_at_0.5"] = int(sum(v >= .5 and not t for v, t in zip(z, y, strict=True)))
    return counts


def bindings():
    return {str(p): digest(p) for p in (
        Path(__file__), Path("scripts/experiments/parts/part_association_observations.py"), REFERENCE, ACTIONS,
        WORK / "parts-fit.json", WORK / "parts-development.json",
    )}


def freeze():
    cv2.setNumThreads(1)
    if LOCK.exists():
        raise FileExistsError(LOCK)
    reference, released, sources, _ = prepare_reference()
    fit_rows, fit_frames = inputs("fit")
    dev, frames = inputs("development")
    fit_data = observations("fit", fit_rows, fit_frames)
    dev_data = observations("development", dev, frames)
    fit_rows, dev = reviewed("fit", fit_rows), reviewed("development", dev)
    base = {r["id"]: released[r["id"]] for r in dev}
    if {r["group"] for r in fit_rows} & {r["group"] for r in dev}:
        raise ValueError("Fit and development acquisitions intersect.")
    head, training = fit(fit_rows, fit_data["observations"])
    action_heads = action_model()["heads"]
    candidates, known = rank(dev, base, frames, dev_data["observations"], head, action_heads)
    selection = choose(dev, base, candidates, known)
    report = dict(fit=[r["id"] for r in fit_rows], development=[r["id"] for r in dev],
                  frame_latency={role: median(data["seconds"].values()) for role, data in
                                 (("fit", fit_data), ("development", dev_data))},
                  model=head, training=training, development_links=link_audit(dev, dev_data["observations"], head),
                  development_baseline=evaluate(dev, base, 0.), selection=selection,
                  bindings=bindings(), released_sources=sources, seed=SEED)
    write_json(LOCK, report)
    print("Pair supervision", training, "Learned", head is not None, flush=True)
    print("Development", report["development_links"], "Selected", selection["selected"], flush=True)


def sealed():
    policy = read_json(LOCK)
    if policy["bindings"] != bindings() or any(digest(p) != h for p, h in policy["released_sources"].items()):
        raise ValueError("Frozen association study inputs changed.")
    return policy


def score():
    from scripts.experiments.parts.part_association_reserve import evidence as fresh_evidence

    cv2.setNumThreads(1)
    policy = sealed()
    rows, base, frames, hashes = fresh_evidence()
    sample = observations("fresh", rows, frames)
    candidates, known = rank(rows, base, frames, sample["observations"], policy["model"], action_model()["heads"])
    before, (_, strict) = simple_metrics(rows, base), match(rows, base, .75)
    results = {}
    for kind in ("finalists", "active"):
        for arm, spec in policy["selection"][kind].items():
            pred, events = apply(base, candidates, known, spec["rule"])
            results[f"{kind}/{arm}"] = dict(rule=spec["rule"], deltas=audit(rows, before, strict, pred, events),
                                          metrics=evaluate(rows, pred, 0.), additions=events)
    selected = policy["selection"]["selected"]
    winner = results[f"finalists/{selected['rule']['arm']}"] if selected else None
    qualified = bool(winner and winner["deltas"]["qualified"] and winner["deltas"]["gain"] > 0)
    write_json(RESULT, dict(policy=digest(LOCK), inputs=hashes, parts=digest(WORK / "parts-fresh.json"),
                           baseline=evaluate(rows, base, 0.), comparisons=results, qualified=qualified,
                           median_additional_part_seconds=median(sample["seconds"].values())))
    print("Fresh", {name: a["deltas"] for name, a in results.items()}, "qualified", qualified, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "score"))
    {"freeze": freeze, "score": score}[cli.parse_args().stage]()
