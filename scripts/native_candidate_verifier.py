"""Train and evaluate an additive native-crop verifier for crowded syringe scenes."""
import argparse
from collections import Counter
from itertools import combinations, product

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.additive import duplicates, overlap_fraction
from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate, match, scores
from scripts.native_candidate_bank import WORK, archive, cohorts
from scripts.occlusion_detector import ART, raster, frozen_frames
from scripts.occlusion_relocalization import prepare_reference
from scripts.occlusion_relocalization_reserve import DATA as PAST, REFERENCE as PAST_REFERENCE, refs as past_rows
from scripts.occlusion_proposals import oracle
from scripts.rfdetr_verifier import DinoEncoder, DINOV2_INFO, expanded
from scripts.sam3_study import configure_cpu

POLICY = ART / "native_candidate_verifier_policy.json"
RESULT = ART / "native_candidate_verifier_results.json"
SEED = 607
APPEARANCE = ("tight", "tight-context", "tight-context-rf")
MODES = ("box", "identity")


def reference_index(box, objects):
    """Assign a proposal only if its reference identity is visually unambiguous."""
    ranked = sorted(((box_iou(box, ref["bbox"]), i) for i, ref in enumerate(objects)), reverse=True)
    if not ranked:
        return None
    overlap, i = ranked[0]
    competing = ranked[1][0] if len(ranked) > 1 else 0.
    return i if overlap >= .50 and overlap-competing >= .08 else None


def examples(row, ranked):
    """Sample detector-shaped crops plus explicit positive and lookalike references."""
    targets = row["objects"]
    examples = [dict(bbox=p["bbox"], score=None, label=True, target=i, origin="reviewed-syringe")
                for i, p in enumerate(targets)]
    examples.extend(dict(bbox=p["bbox"], score=None, label=False, target=None,
                         origin="reviewed-lookalike") for p in row.get("lookalikes", []))
    negatives = 0
    by_target = Counter()
    for item in sorted(ranked, key=lambda p: -p["score"]):
        best = max((box_iou(item["bbox"], o["bbox"]) for o in targets), default=0.)
        target = reference_index(item["bbox"], targets)
        if target is not None and best >= .55 and by_target[target] < 4:
            examples.append(dict(**item, label=True, target=target, origin="detector-positive"))
            by_target[target] += 1
        elif best < .18 and negatives < 7:
            examples.append(dict(**item, label=False, target=None, origin="detector-negative"))
            negatives += 1
    return examples


def descriptors(encoder, rows, selected, incumbent):
    """Encode proposals from the archived source raster at two crop extents."""
    results = {}
    pending = []
    crops = []
    for row in rows:
        key = row["id"]
        image = raster(row)
        proposals = [{**p} for p in selected[key]]
        existing = [{**p} for p in incumbent.get(key, [])]
        results[key] = dict(proposals=proposals, incumbent=existing)
        for item in [*proposals, *existing]:
            box = item["bbox"]
            crops.extend((crop(image, expanded(box, .06)), crop(image, expanded(box, .33))))
            pending.append(item)
        if len(crops) >= 128:
            encode_batch(encoder, crops, pending)
    if pending:
        encode_batch(encoder, crops, pending)
    return results


def encode_batch(encoder, crops, pending):
    vectors = encoder.encode(crops).reshape(len(pending), 2, -1)
    for item, pair in zip(pending, vectors, strict=True):
        item["tight"], item["context"] = (v.tolist() for v in pair)
    crops.clear()
    pending.clear()


def acquisition_weights(groups, labels):
    groups = list(groups)
    labels = list(labels)
    count = Counter(zip(groups, labels))
    weights = np.array([1./count[group, label] for group, label in zip(groups, labels, strict=True)])
    # Equalize total mass of positive and negative groups across uneven class counts.
    for label in (False, True):
        unit = np.asarray(labels) == label
        weights[unit] *= len(weights) / (2 * weights[unit].sum())
    return weights


def classifier(x, y, groups, normalize):
    x = np.asarray(x)
    if len(set(y)) != 2:
        raise ValueError("Fitting requires both classes.")
    transform = StandardScaler().fit(x) if normalize else None
    inputs = transform.transform(x) if transform else x
    head = LogisticRegression(C=.45, max_iter=500, random_state=SEED)
    head.fit(inputs, y, sample_weight=acquisition_weights(groups, y))
    return dict(weight=head.coef_[0].tolist(), bias=float(head.intercept_[0]),
                center=transform.mean_.tolist() if transform else None,
                scale=transform.scale_.tolist() if transform else None)


def predict(head, features):
    x = np.asarray(features)
    if head["scale"] is not None:
        x = (x-head["center"])/head["scale"]
    return expit(x @ np.asarray(head["weight"]) + head["bias"])


def appearance(item, name):
    tight = np.asarray(item["tight"]) * 6.
    if name == "tight":
        return tight
    return np.r_[tight, np.asarray(item["context"]) * 6.]


def pair_features(first, second):
    a, b = (np.asarray(x["bbox"]) for x in (first, second))
    area_a, area_b = float(np.prod(a[2:]-a[:2])), float(np.prod(b[2:]-b[:2]))
    distances = np.abs((a[:2]+a[2:]-b[:2]-b[2:])/np.maximum(a[2:]-a[:2]+b[2:]-b[:2], 1e-6))
    similarity = [float(np.dot(first[key], second[key])) for key in ("tight", "context")]
    return [box_iou(a, b), overlap_fraction(a, b), *distances, abs(np.log(area_a/area_b)), *similarity]


def pair_examples(fit, frames):
    features, outcomes, groups = [], [], []
    for row in fit:
        positive = [p for p in frames[row["id"]]["proposals"] if p["target"] is not None]
        for a, b in combinations(positive, 2):
            if not duplicates(a["bbox"], b["bbox"]):
                continue
            features.append(pair_features(a, b))
            outcomes.append(a["target"] != b["target"])
            groups.append(row["group"])
    return features, outcomes, groups


def fit_appearance(fit, frames):
    heads = {}
    examples = [(item, r["group"]) for r in fit for item in frames[r["id"]]["proposals"]]
    outcomes = [item["label"] for item, _ in examples]
    groups = [group for _, group in examples]
    for model in APPEARANCE[:2]:
        matrix = [appearance(item, model) for item, _ in examples]
        heads[model] = classifier(matrix, outcomes, groups, False)
    return heads, dict(crops=Counter(item["origin"] for item, _ in examples),
                       acquisition_groups=len(set(groups)), positives=sum(outcomes), negatives=len(outcomes)-sum(outcomes))


def candidate_ranks(frames, heads):
    ranked = {}
    for key, frame in frames.items():
        items = frame["proposals"]
        out = []
        if items:
            tight, context = [predict(heads[name], [appearance(p, name) for p in items]) for name in APPEARANCE[:2]]
            for p, a, b in zip(items, tight, context, strict=True):
                out.append({**p, "rf": p["score"], "tight": p["tight"], "context": p["context"],
                            "decisions": {"rf": p["score"], "tight": float(a), "tight-context": float(b),
                                          "tight-context-rf": float(.75*b+.25*p["score"])}})
        ranked[key] = {**frame, "proposals": out}
    return ranked


def identity_scores(frames, identity):
    output = {}
    for key, values in frames.items():
        objects = [*values["incumbent"], *values["proposals"]]
        output[key] = {
            (i, j): float(predict(identity, pair_features(a, b)))
            for (i, a), (j, b) in combinations(enumerate(objects), 2)
            if duplicates(a["bbox"], b["bbox"]) and box_iou(a["bbox"], b["bbox"]) < .85
        } if identity is not None else {}
    return output


def fuse(frame, rule, probabilities):
    """Append independently verified instances and preserve all released boxes."""
    current = frame["incumbent"]
    ranked = frame["proposals"]
    result = [{k: v for k, v in p.items() if k not in ("tight", "context")} for p in current]
    chosen = list(enumerate(current))
    rank = rule["appearance"]
    candidates = sorted(enumerate(ranked, len(current)), key=lambda q: -q[1]["decisions"][rank])
    for i, p in candidates:
        if p["decisions"][rank] < rule["floor"]:
            break
        if any(duplicates(p["bbox"], prev["bbox"]) and
               (rule["distinct"] is None or probabilities.get(tuple(sorted((i, j))), 0.) < rule["distinct"])
               for j, prev in chosen):
            continue
        chosen.append((i, p))
        result.append(dict(bbox=p["bbox"], score=.002 + .001 * p["decisions"][rank],
                           source="native crop verifier", detector_score=p["rf"],
                           verifier_score=p["decisions"][rank]))
    return result


def decisions(frames, rule, probabilities):
    return {key: fuse(frame, rule, probabilities[key]) for key, frame in frames.items()}


def simple_metrics(rows, predictions):
    hits, used = match(rows, predictions, .50)
    totals = Counter(key for key, _, index, _ in hits if index is None)
    tp = sum(len(indices) for indices in used.values())
    gt = sum(len(r["objects"]) for r in rows)
    return dict(**scores(tp, sum(totals.values()), gt-tp), fp_by_frame=dict(totals),
                targets={key: sorted(indices) for key, indices in used.items()},
                count_errors={r["id"]: abs(len(predictions[r["id"]])-len(r["objects"])) for r in rows},
                exact_count=sum(len(predictions[r["id"]]) == len(r["objects"]) for r in rows))


def compare(rows, base, variants):
    current = simple_metrics(rows, variants)
    lost = any(set(indices)-set(current["targets"][key]) for key, indices in base["targets"].items())
    new_fp = any(n > base["fp_by_frame"].get(key, 0) for key, n in current["fp_by_frame"].items())
    worse_count = any(n > base["count_errors"][key] for key, n in current["count_errors"].items())
    safe = not lost and not new_fp and not worse_count
    return dict(tp=current["tp"], fp=current["fp"], exact_count=current["exact_count"],
                gain=current["tp"]-base["tp"], extra_fp=current["fp"]-base["fp"], safe=safe,
                lost_reference=lost, per_image_false_positive_regression=new_fp, count_regression=worse_count)


def clean_current(frames):
    return {k: [{a: v for a, v in p.items() if a not in ("tight", "context")} for p in frame["incumbent"]]
            for k, frame in frames.items()}


def select_rules(rows, frames, identity):
    base = simple_metrics(rows, clean_current(frames))
    pair_scores = identity_scores(frames, identity)
    trials = []
    floors = np.r_[np.arange(.15, .951, .05), .975, .99, .999, 1.001].tolist()
    for name, mode, floor in product(("rf", *APPEARANCE), (None, .85, .95, .985), floors):
        if mode is not None and identity is None:
            continue
        rule = dict(appearance=name, distinct=mode, floor=floor)
        output = decisions(frames, rule, pair_scores)
        trials.append(dict(rule=rule, **compare(rows, base, output)))
    arms = [(a, m) for a in ("rf", *APPEARANCE) for m in MODES]
    winners = {}
    for name, mode in arms:
        pool = [trial for trial in trials if trial["rule"]["appearance"] == name
                and (trial["rule"]["distinct"] is None) == (mode == "box")]
        if not pool:
            continue
        safe = [t for t in pool if t["safe"]]
        winners[f"{name}-{mode}"] = max(safe, key=lambda t: (t["gain"], t["exact_count"], -t["rule"]["floor"],
                                                            t["rule"]["distinct"] or 0.))
    eligible = [t for t in winners.values() if t["gain"] > 0]
    winner = max(eligible, key=lambda t: (t["gain"], t["exact_count"], t["rule"]["distinct"] is None),
                 default=None)
    return dict(baseline=base, trials=trials, arms=winners, selected=winner)


def checkpoints():
    return {str(path): digest(path) for path in
            (DINOV2_INFO, WORK / "rf-fit.json", WORK / "rf-development.json")}


def learn():
    if POLICY.exists():
        raise FileExistsError(POLICY)
    fit, development = cohorts("fit"), cohorts("development")
    _, original, histories, _ = prepare_reference()
    configure_cpu()
    encoder = DinoEncoder()
    raw = archive(fit, "fit")
    selected = {r["id"]: examples(r, raw[r["id"]]) for r in fit}
    training = descriptors(encoder, fit, selected, {})
    heads, summary = fit_appearance(fit, training)
    pair_matrix, pairs, groups = pair_examples(fit, training)
    identity = classifier(pair_matrix, pairs, groups, True) if len(set(pairs)) == 2 else None
    stats = dict(pairs=len(pairs), distinct=sum(pairs), duplicates=len(pairs)-sum(pairs),
                 contributing_groups=len(set(groups)),
                 distinct_groups=len({g for g, p in zip(groups, pairs, strict=True) if p}))
    # Preserve exact fitting examples and held decision scores, apart from large feature arrays.
    write_json(WORK / "training-proposals.json", selected)
    dev = descriptors(encoder, development, archive(development, "development"), original)
    ranked = candidate_ranks(dev, heads)
    choices = select_rules(development, ranked, identity)
    write_json(POLICY, dict(implementation=digest(__file__), inputs=checkpoints(), archives=histories,
                            training=summary, identity_data=stats, heads=heads, identity_head=identity,
                            decision=choices, selection="Maximize new development matches subject to no existing "
                            "target losses, no false-positive increase per image and no exact-count drop."))
    print(summary, stats, choices["selected"], flush=True)


def sealed():
    state = read_json(POLICY)
    if (state["implementation"] != digest(__file__) or state["inputs"] != checkpoints()
            or any(digest(path) != value for path, value in state["archives"].items())):
        raise ValueError("The fitted crop verification protocol changed.")
    return state


def run_evaluation(blocks, before, raw):
    state = sealed()
    configure_cpu()
    encoder = DinoEncoder()
    rows = [r for images in blocks.values() for r in images]
    frames = descriptors(encoder, rows, raw, before)
    ranked = candidate_ranks(frames, state["heads"])
    identity = identity_scores(ranked, state["identity_head"])
    cohort_results = {}
    for role, block in blocks.items():
        sub = {row["id"]: ranked[row["id"]] for row in block}
        base = clean_current(sub)
        standard = simple_metrics(block, base)
        choices = {}
        for name, trial in state["decision"]["arms"].items():
            result = decisions(sub, trial["rule"], identity)
            choices[name] = dict(baseline_delta=compare(block, standard, result), result=evaluate(block, result, 0.),
                                 supplemental={k: p[len(base[k]):] for k, p in result.items() if len(p) > len(base[k])})
        cohort_results[role] = dict(baseline=evaluate(block, base, 0.), arms=choices,
                                    oracle={r["id"]: oracle(r["objects"], raw[r["id"]])["matched"] for r in block})
    return cohort_results


def evaluate_history():
    blocks, before = frozen_frames()
    old = read_json(PAST / "released.json")
    if old["reference"] != digest(PAST_REFERENCE):
        raise ValueError("Earlier prospective outputs changed.")
    rows = cohorts("evaluation")
    before.update(old["predictions"])
    blocks["prior-relocalization-reserve"] = past_rows()
    output = run_evaluation(blocks, before, archive(rows, "evaluation"))
    write_json(RESULT, dict(policy=digest(POLICY), evidence=output,
                            scope="Retrospective replay of public, previously consulted cohorts."))
    print({key: {a: (v["baseline_delta"]["gain"], v["baseline_delta"]["extra_fp"])
                 for a, v in val["arms"].items()} for key, val in output.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("learn", "history"))
    {"learn": learn, "history": evaluate_history}[parser.parse_args().action]()
