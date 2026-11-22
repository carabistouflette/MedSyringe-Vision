"""Learn visible-assembly quality and compare greedy and joint box selection."""
import argparse
from collections import Counter
from functools import lru_cache
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.detection.additive import duplicates
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from scripts.experiments.ranking.instance_separation import ART, axis_profile, blocked, pair_scores, sealed as identity_policy
from scripts.experiments.ranking.instance_separation import POLICY as IDENTITY
from scripts.experiments.crowds.native_candidate_bank import archive, cohorts
from scripts.experiments.crowds.native_candidate_verifier import compare, simple_metrics
from scripts.experiments.occlusion.occlusion_detector import raster
from scripts.experiments.occlusion.occlusion_relocalization import prepare_reference
from scripts.experiments.relocalization.recrop_verifier import body_features, fraction_inside
from scripts.experiments.identity.syringe_assembly_model import REVIEW, descriptor, fragment_views, training_weights

WORK = Path("data/complete-syringe")
POLICY = ART / "complete_syringe_policy.json"
RESULT = ART / "complete_syringe_results.json"
ARMS = ("detector-greedy", "body-greedy", "body-joint")
MAX_BOXES = 14


def vector(image, box):
    shape = descriptor(image, box)
    _, _, edge, continuity, width, peak = axis_profile(image, box, shape)
    w = (box[2]-box[0])*image.width
    h = (box[3]-box[1])*image.height
    return [np.log1p(max(w,h)), np.log1p(min(w,h)), abs(np.log(w/h)),
            edge, continuity, width, peak, *body_features(image, box, shape)]


def supervise(row, bank, size, review):
    objects = row["objects"]
    yield from ((o["bbox"], True, "reference") for o in objects)
    yield from ((p["bbox"], False, "reviewed-lookalike") for p in row.get("lookalikes", []))
    yield from ((r["bbox"], False, "reviewed-fragment") for r in review if r["label"] == "fragment")
    for o in objects:
        for part in fragment_views(o, size):
            if all(box_iou(part, v["bbox"]) < .40 for v in objects):
                yield part, False, "constructed-fragment"
    for item in bank:
        box = item["bbox"]
        target, best = None, 0.
        for o in objects:
            iou = box_iou(box, o["bbox"])
            if iou > best:
                best, target = iou, o
        if target is not None and best >= .60 and fraction_inside(target["bbox"], box) >= .81:
            yield box, True, "detector-complete"
        elif best <= .10:
            yield box, False, "detector-background"
        elif target is not None and best < .48 and fraction_inside(box, target["bbox"]) >= .77:
            yield box, False, "detector-fragment"


def fit():
    fit_rows = cohorts("fit")
    bank = archive(fit_rows, "fit")
    review = read_json(REVIEW)["rows"]
    x, y, groups, history = [], [], [], []
    for row in fit_rows:
        key, photo = row["id"], raster(row)
        sources = [r for r in review if r["id"] == key]
        seen, counts = set(), Counter()
        for box, label, kind in supervise(row, bank[key], photo.size, sources):
            if tuple(box) in seen or counts[kind] >= 8:
                continue
            seen.add(tuple(box))
            counts[kind] += 1
            x.append(vector(photo, box))
            y.append(label)
            groups.append(row["group"])
            history.append(dict(image=key, group=row["group"], bbox=box, complete=label, origin=kind))
    weights = training_weights(y, groups)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = LogisticRegression(C=.4, max_iter=600, random_state=337)
    model.fit(scaler.transform(x), y, sample_weight=weights)
    write_json(WORK / "supervision.json", history)
    stats = dict(examples=len(y), complete=sum(y), groups=len(set(groups)),
                 sources=Counter(r["origin"] for r in history))
    return dict(center=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coef=model.coef_[0].tolist(), bias=float(model.intercept_[0])), stats


def quality(rows, bank, head):
    scores = {}
    for row in rows:
        key, photo = row["id"], raster(row)
        scores[key] = []
        for p in bank[key]:
            if p["score"] >= .12:
                v = (np.asarray(vector(photo, p["bbox"])) - head["center"]) / head["scale"]
                complete = float(expit(v @ head["coef"] + head["bias"]))
                scores[key].append(dict(p, complete=complete))
    return scores


def best_compatible(weights, conflicts):
    """Exact weighted independent set over the bounded candidate list."""
    bits = [sum(1 << j for j in blocked_by) for blocked_by in conflicts]

    @lru_cache(None)
    def visit(mask):
        if mask == 0:
            return 0., ()
        flag = mask & -mask
        i = flag.bit_length()-1
        left = mask ^ flag
        dropped = visit(left)
        value, chosen = visit(left & ~bits[i])
        picked = weights[i] + value, (i, *chosen)
        return picked if picked[0] >= dropped[0] else dropped

    return visit((1 << len(weights))-1)[1]


def select_boxes(original, raw, pairs, rule):
    def conflict(a, b):
        return duplicates(a["bbox"], b["bbox"]) and blocked(a["bbox"], b["bbox"], pairs, rule)

    if len(original) < rule["minimum"]:
        return list(original)
    available = []
    for p in raw:
        if p["score"] < rule["detector_floor"] or any(conflict(q, p) for q in original):
            continue
        value = p["score"] if rule["arm"] == "detector-greedy" else .30*p["score"] + .70*p["complete"]
        if value >= rule["floor"]:
            available.append((value, p))
    available.sort(key=lambda item: -item[0])
    available = available[:MAX_BOXES]
    weights = [v-rule["floor"]+.01 for v, _ in available]
    options = [p for _,p in available]
    edges = [set() for _ in options]
    for i, p in enumerate(options):
        for j in range(i+1,len(options)):
            if conflict(p, options[j]):
                edges[i].add(j)
                edges[j].add(i)
    if rule["arm"] == "body-joint":
        chosen = best_compatible(weights, edges)
    else:
        chosen = []
        for i in range(len(options)):
            if edges[i].isdisjoint(chosen):
                chosen.append(i)
    return [*original, *(dict(options[i], score=.002+.001*available[i][0],
                            detector_score=options[i]["score"], source="complete syringe selection") for i in chosen)]


def predict(base, crops, pairs, rule):
    return {k: select_boxes(p, crops[k], pairs[k], rule) for k,p in base.items()}


def grid(rows, base, crops, pairs):
    before = simple_metrics(rows, base)
    trials = []
    for arm, minimum, floor, detector, method in product(
            ARMS, (0,1,2), (.25,.35,.45,.55,.65,.75,.85,.95,1.01), (.12,.25,.40), ("overlap","axis-identity")):
        rule = dict(arm=arm, minimum=minimum, floor=floor, detector_floor=detector, method=method, distinct=.65)
        trials.append(dict(rule=rule, **compare(rows, before, predict(base, crops, pairs, rule))))
    def order(t):
        return (t["gain"], t["exact_count"], -t["extra_fp"], t["rule"]["floor"])
    finalists = {arm:max((t for t in trials if t["safe"] and t["rule"]["arm"] == arm), key=order) for arm in ARMS}
    best = max((t for t in finalists.values() if t["gain"] > 0), key=order, default=None)
    probes = {arm:dict(arm=arm,minimum=1,floor=.25,detector_floor=.25,method="axis-identity",distinct=.65)
              for arm in ARMS}
    return dict(finalists=finalists, selected=best, probes=probes, trials=trials)


def inputs():
    paths = (IDENTITY, REVIEW, WORK/"supervision.json", "scripts/experiments/ranking/instance_separation.py",
             "scripts/experiments/relocalization/recrop_verifier.py", "data/native-candidate-verifier/rf-fit.json",
             "data/native-candidate-verifier/rf-development.json")
    return {str(p):digest(p) for p in paths}


def train():
    if POLICY.exists():
        raise FileExistsError(POLICY)
    cv2.setNumThreads(1)
    head, stats = fit()
    rows, base, sources, _ = prepare_reference()
    proposals = archive(rows,"development")
    pairs = pair_scores(rows, base, proposals, identity_policy()["model"])
    decision = grid(rows, base, quality(rows, proposals, head), pairs)
    write_json(POLICY, dict(code=digest(__file__),inputs=inputs(),sources=sources,model=head,stats=stats,decision=decision))
    print(stats, decision["finalists"], decision["selected"], flush=True)


def sealed():
    data = read_json(POLICY)
    if (data["code"] != digest(__file__) or data["inputs"] != inputs()
            or any(digest(p)!=h for p,h in data["sources"].items())):
        raise ValueError("Frozen selector changed.")
    return data


def summarize(rows,base,bank,lock):
    crops = quality(rows,bank,lock["model"])
    pairs = pair_scores(rows,base,bank,identity_policy()["model"])
    pre = simple_metrics(rows,base)
    rules = {name:t["rule"] for name,t in lock["decision"]["finalists"].items()}
    rules.update({f"probe-{name}":r for name,r in lock["decision"]["probes"].items()})
    arms = {}
    for name,rule in rules.items():
        pred = predict(base,crops,pairs,rule)
        arms[name] = dict(delta=compare(rows,pre,pred),metrics=evaluate(rows,pred,0.),
                          additions={k:p[len(base[k]):] for k,p in pred.items() if len(p)>len(base[k])})
    return dict(baseline=evaluate(rows,base,0.),arms=arms,rows={r["id"]:dict(
        expected=len(r["objects"]),baseline=len(base[r["id"]])) for r in rows})


def score():
    from scripts.experiments.ranking.complete_syringe_reserve import evidence
    from scripts.experiments.ranking.instance_separation_reserve import evidence as archived
    from scripts.experiments.relocalization.relocalization_reserve import evidence as exposed

    lock = sealed()
    cv2.setNumThreads(1)
    rows, base, frames, old = exposed()
    known = summarize(rows,base,{k:p["seeds"] for k,p in frames.items()}, lock)
    past, before, raw, archive_inputs = archived()
    guard = summarize(past,before,raw,lock)
    fresh, control, proposals, current = evidence()
    reserve = summarize(fresh,control,proposals,lock)
    selected = lock["decision"]["selected"]
    winner = selected["rule"]["arm"] if selected else None
    qualified = bool(winner and reserve["arms"][winner]["delta"]["gain"]>0 and all(
                       r["arms"][winner]["delta"]["safe"] for r in (known,guard,reserve)))
    write_json(RESULT,dict(policy=digest(POLICY),inputs={**old,**archive_inputs,**current},
                          exposed=known,archive=guard,reserve=reserve,winner=winner,qualified=qualified))
    print(winner,qualified,{k:v["delta"] for k,v in reserve["arms"].items()},flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("train","score"))
    {"train":train,"score":score}[parser.parse_args().stage]()
