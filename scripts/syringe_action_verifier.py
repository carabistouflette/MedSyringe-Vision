"""Fit and qualify a three-action verifier against the released syringe detector."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.additive import duplicates
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate, match
from scripts.complete_syringe_selection import ART, vector as visual_vector
from scripts.native_candidate_verifier import compare, reference_index, simple_metrics
from scripts.occlusion_detector import raster
from scripts.occlusion_relocalization import prepare_reference
from scripts.recrop_proposal_bank import WORK as BANK, read_bank
from scripts.recrop_verifier import fraction_inside
from scripts.relocalization_selector import anchors, evidence, pool, visible_anchors_preserved
from scripts.syringe_assembly_model import fragment_views, training_weights

WORK = Path("data/syringe-action-verifier")
POLICY = ART / "syringe_action_verifier_policy.json"
RESULT = ART / "syringe_action_verifier_results.json"
FEATURES = ("geometry", "visual")
ARMS = ("add-only", "replace-only", "combined")
CLASSES = ("add", "reject", "replace")
FLOORS = (.40, .50, .60, .70, .75, .80, .85, .90, .95, .98, 1.01)
SEED = 1289
BUDGET = 6


def context(box, incumbents):
    ranked = sorted(((box_iou(box, cur), i) for i, cur in enumerate(incumbents)), reverse=True)
    if not ranked or ranked[0][0] == 0:
        return None, [0.] * 8
    overlap, idx = ranked[0]
    current = incumbents[idx]
    a, b = np.asarray(box), np.asarray(current)
    shift = np.abs((a[:2] + a[2:] - b[:2] - b[2:]) / (a[2:] - a[:2] + b[2:] - b[:2]))
    return idx, [
        1., overlap, ranked[1][0] if len(ranked) > 1 else 0.,
        fraction_inside(box, current), fraction_inside(current, box),
        np.log(np.prod(a[2:] - a[:2]) / np.prod(b[2:] - b[:2])), *shift,
    ]


def proposal_features(candidate, image, cache, incumbents):
    box = candidate["bbox"]
    slot, relation = context(box, incumbents)
    agreement, support = cache["support"][tuple(box), candidate["view"]]
    width, height = (box[2] - box[0]) * image.width, (box[3] - box[1]) * image.height
    geometry = [
        candidate["score"], np.log1p(width), np.log1p(height), candidate["view"] == "crop",
        candidate.get("repeats", 0), agreement, support, np.log1p(len(incumbents)), *relation,
    ]
    profiles = cache["profiles"]
    involved = [box] if slot is None else [box, incumbents[slot]]
    for bounds in involved:
        if tuple(bounds) not in profiles:
            profiles[tuple(bounds)] = visual_vector(image, bounds)[3:]
    proposed = np.asarray(profiles[tuple(box)])
    previous = np.zeros_like(proposed) if slot is None else profiles[tuple(incumbents[slot])]
    return slot, (geometry, [*geometry, *proposed, *(proposed - previous)])


def expected_action(box, slot, incumbents, objects, identities):
    target = reference_index(box, objects)
    if target is None or not visible_anchors_preserved(box, objects[target]):
        return "reject"
    if target not in identities:
        return "add"
    if slot is None or identities[slot] != target:
        return "reject"
    gold = objects[target]["bbox"]
    if box_iou(box, gold) >= box_iou(incumbents[slot], gold) + .08:
        return "replace"
    return "reject"


def fit_states(objects, seeds):
    """Keep real detector geometry where available and perturb only one incumbent at a time."""
    anchor_options = [anchors(objects, i, seeds) for i in range(len(objects))]
    intact = tuple(
        next((b for b, kind in options if kind == "detector" and reference_index(b, objects) == i),
             objects[i]["bbox"]) for i, options in enumerate(anchor_options)
    )
    ids = tuple(range(len(objects)))
    yield intact, ids, "nominal"
    for i, options in enumerate(anchor_options):
        yield intact[:i] + intact[i + 1:], ids[:i] + ids[i + 1:], "missing"
        for b, origin in options:
            if b != intact[i]:
                yield intact[:i] + (b,) + intact[i + 1:], ids, origin


def feature_cache(options, extra=()):
    return dict(profiles={}, support={(tuple(p["bbox"]), p["view"]): evidence(p, options)
                                      for p in [*options, *extra]})


def fit(rows, frames):
    matrices = ([], [])
    labels, groups, ledger = [], [], []
    outcome = Counter()
    for row in rows:
        key, targets = row["id"], row["objects"]
        image, frame = raster(row), frames[key]
        options = pool(frame, BUDGET)
        extra = [dict(bbox=box, view="fragment", score=0.) for obj in targets
                 for box in fragment_views(obj, image.size)]
        extra.extend(dict(bbox=p["bbox"], view="lookalike", score=0.) for p in row.get("lookalikes", []))
        cache = feature_cache(options, extra)
        for incumbents, ids, origin in fit_states(targets, frame["seeds"]):
            for p in [*options, *extra]:
                box = p["bbox"]
                slot, values = proposal_features(p, image, cache, incumbents)
                label = expected_action(box, slot, incumbents, targets, ids)
                for xs, x in zip(matrices, values, strict=True):
                    xs.append(x)
                labels.append(label)
                groups.append(row["group"])
                outcome[f"{origin}/{p['view']}/{label}"] += 1
                ledger.append(dict(id=key, group=row["group"], incumbents=incumbents, identities=ids,
                                   source=origin, slot=slot, candidate=box, view=p["view"], action=label))
    if set(labels) != set(CLASSES):
        raise ValueError("Every action requires fit supervision.")
    weights = training_weights(labels, groups)
    heads = {}
    for name, xs in zip(FEATURES, matrices, strict=True):
        scaler = StandardScaler().fit(xs, sample_weight=weights)
        learner = LogisticRegression(C=.3, max_iter=600, random_state=SEED)
        learner.fit(scaler.transform(xs), labels, sample_weight=weights)
        if learner.classes_.tolist() != list(CLASSES) or learner.n_iter_[0] >= learner.max_iter:
            raise ValueError("Unexpected classifier output or failure to converge.")
        heads[name] = dict(center=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                           coefficients=learner.coef_.tolist(), bias=learner.intercept_.tolist())
    write_json(WORK / "fit-ledger.json", ledger)
    return heads, dict(rows=len(rows), groups=len(set(groups)), actions=Counter(labels), states=outcome)


def rank(rows, baseline, frames, heads):
    output = {}
    for row in rows:
        k = row["id"]
        options, image = pool(frames[k], BUDGET), raster(row)
        cache = feature_cache(options)
        incumbents = [p["bbox"] for p in baseline[k]]
        ranked = []
        for p in options:
            slot, vectors = proposal_features(p, image, cache, incumbents)
            decisions = {}
            for name, x in zip(FEATURES, vectors, strict=True):
                h = heads[name]
                z = (np.asarray(x) - h["center"]) / h["scale"]
                probabilities = softmax(np.asarray(h["coefficients"]) @ z + h["bias"])
                decisions[name] = dict(zip(CLASSES, map(float, probabilities), strict=True))
            ranked.append(dict(bbox=p["bbox"], score=p["score"], slot=slot, view=p["view"], decisions=decisions))
        output[k] = ranked
    return output


def apply(baseline, ranked, rule):
    predictions, events = {}, {}
    for key, original in baseline.items():
        output, changed_slots, log = list(original), set(), []
        choices = []
        for p in ranked[key]:
            scores = p["decisions"][rule["features"]]
            action = max(scores, key=scores.get)
            if action != "reject" and scores[action] >= rule[action]:
                choices.append((action, scores[action], p))
        choices.sort(key=lambda event: (event[0] != "replace", -event[1], -event[2]["score"]))
        for action, probability, p in choices:
            box, slot = p["bbox"], p["slot"]
            event = dict(action=action, score=probability, slot=slot, view=p["view"], bbox=box)
            if action == "replace":
                if slot is None or slot in changed_slots or any(
                    box_iou(box, q["bbox"]) >= .50 for j, q in enumerate(output) if j != slot
                ):
                    continue
                previous = output[slot]["bbox"]
                if not .15 <= box_iou(previous, box) < .97:
                    continue
                output[slot] = {**output[slot], "bbox": box}
                event["prior_bbox"] = previous
                changed_slots.add(slot)
            else:
                if any(box_iou(box, q["bbox"]) >= .70 for q in output):
                    continue
                if any(duplicates(box, q["bbox"]) for q in output[len(original):]):
                    continue
                if rule["overlap"] == "strict" and any(duplicates(box, q["bbox"]) for q in output):
                    continue
                output.append(dict(bbox=box, score=p["score"]))
            log.append(event)
        predictions[key] = output
        if log:
            events[key] = log
    return predictions, events


def audit(rows, baseline, before75, output, events):
    state = compare(rows, baseline, output)
    _, matched = match(rows, output, .75)
    gained = [[key, idx] for key in matched for idx in sorted(matched[key] - before75[key])]
    lost = [[key, idx] for key in matched for idx in sorted(before75[key] - matched[key])]
    actions = Counter(e["action"] for frame in events.values() for e in frame)
    return dict(**state, gains_at_75=gained, losses_at_75=lost, actions=actions, safe_at_75=state["safe"] and not lost)


def calibrate(rows, baseline, ranked):
    state, (_, before75) = simple_metrics(rows, baseline), match(rows, baseline, .75)
    trials = []
    for name, arm, overlap in product(FEATURES, ARMS, ("strict", "learned")):
        add_floors = (1.01,) if arm == "replace-only" else FLOORS
        replace_floors = (1.01,) if arm == "add-only" else FLOORS
        for a, r in product(add_floors, replace_floors):
            rule = dict(features=name, arm=arm, overlap=overlap, add=a, replace=r)
            output, events = apply(baseline, ranked, rule)
            trials.append(dict(rule=rule, **audit(rows, state, before75, output, events)))

    def order(t):
        rule = t["rule"]
        return (t["gain"], len(t["gains_at_75"]), -sum(t["actions"].values()),
                rule["add"] + rule["replace"], rule["overlap"] == "strict")

    finalists = {f"{name}/{arm}": max((t for t in trials if t["safe_at_75"] and
                 t["rule"]["arm"] == arm and t["rule"]["features"] == name), key=order)
                 for name, arm in product(FEATURES, ARMS)}
    selected = max((t for t in finalists.values() if t["gain"] or t["gains_at_75"]), key=order, default=None)
    return trials, finalists, selected


def dependencies():
    return {str(p): digest(p) for p in (
        BANK / "fit.json", BANK / "development.json", WORK / "fit-ledger.json",
        "scripts/relocalization_selector.py", "scripts/complete_syringe_selection.py",
        "scripts/recrop_proposal_bank.py", "scripts/native_candidate_verifier.py",
        "scripts/syringe_assembly_model.py",
    )}


def freeze():
    if POLICY.exists():
        raise FileExistsError(POLICY)
    cv2.setNumThreads(1)
    fitting, fit_frames = read_bank("fit")
    development, dev_frames = read_bank("development")
    dev_rows, baseline, sources, _ = prepare_reference()
    if reference_state(development) != reference_state(dev_rows) or (
        {r["group"] for r in fitting} & {r["group"] for r in development}
    ):
        raise ValueError("Inconsistent development rows or shared acquisition group.")
    heads, counts = fit(fitting, fit_frames)
    print("Fitting", counts["rows"], counts["groups"], counts["actions"], flush=True)
    ranked = rank(development, baseline, dev_frames, heads)
    trials, finalists, selected = calibrate(development, baseline, ranked)
    write_json(POLICY, dict(code=digest(__file__), inputs=dependencies(), sources=sources, seed=SEED,
                           fit=counts, heads=heads, trials=trials, finalists=finalists, winner=selected,
                           development=dict(rows=len(development), groups=len({r["group"] for r in development}),
                                            metrics=evaluate(development, baseline, 0.))))
    print("Calibrated", len(trials), "rules", "winner", selected, flush=True)


def reference_state(rows):
    return {r["id"]: (r["group"], r["sha256"], r["objects"]) for r in rows}


def sealed():
    lock = read_json(POLICY)
    if lock["code"] != digest(__file__) or lock["inputs"] != dependencies() or any(
        digest(p) != v for p, v in lock["sources"].items()
    ):
        raise ValueError("Frozen experiment inputs changed.")
    return lock


def assess(rows, baseline, frames, lock):
    ranked = rank(rows, baseline, frames, lock["heads"])
    before, (_, before75) = simple_metrics(rows, baseline), match(rows, baseline, .75)
    arms = {}
    for k, finalist in lock["finalists"].items():
        output, events = apply(baseline, ranked, finalist["rule"])
        arms[k] = dict(delta=audit(rows, before, before75, output, events), actions=events,
                       metrics=evaluate(rows, output, 0.))
    _, attainable = match(rows, {k: [*baseline[k], *p] for k, p in ranked.items()}, .50)
    oracle = {k: sorted(v - set(before["targets"][k])) for k, v in attainable.items()
              if v - set(before["targets"][k])}
    return dict(baseline=evaluate(rows, baseline, 0.), arms=arms, proposal_oracle_recovery=oracle)


def score():
    from scripts.completeness_calibration_reserve import evidence as old_evidence
    from scripts.relocalization_reserve import evidence as crowd_evidence
    from scripts.syringe_action_reserve import evidence

    cv2.setNumThreads(1)
    lock = sealed()
    rows, baseline, frames, fresh = evidence()
    public, public_base, raw, old = old_evidence()
    crowd, crowd_base, crowd_frames, earlier = crowd_evidence()
    cohorts = dict(
        reserve=assess(rows, baseline, frames, lock),
        exposed_calibration=assess(public, public_base, {k: dict(seeds=p, proposals=[]) for k, p in raw.items()}, lock),
        exposed_crowd=assess(crowd, crowd_base, crowd_frames, lock),
    )
    winner = lock["winner"]
    name = f"{winner['rule']['features']}/{winner['rule']['arm']}" if winner else None
    qualified = False
    if name:
        delta = cohorts["reserve"]["arms"][name]["delta"]
        qualified = bool((delta["gain"] > 0 or delta["gains_at_75"]) and
                         all(c["arms"][name]["delta"]["safe_at_75"] for c in cohorts.values()))
    write_json(RESULT, dict(policy=digest(POLICY), inputs={**fresh, **old, **earlier},
                           cohorts=cohorts, winner=name, qualified=qualified))
    print("Frozen winner", name, "qualification", qualified, flush=True)
    if name:
        for k, result in cohorts.items():
            print(k, result["arms"][name]["delta"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score"))
    {"freeze": freeze, "score": score}[parser.parse_args().stage]()
