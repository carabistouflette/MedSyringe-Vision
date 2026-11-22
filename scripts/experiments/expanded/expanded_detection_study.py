"""Train hybrid proposal heads, freeze a candidate and run held-out audits."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path
from statistics import median

import numpy as np
from joblib import dump, load as load_model
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.detection.additive import additions
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou, suppress
from medsyringe.evaluate import evaluate
from medsyringe.detection.hybrid_detection import candidates, rank, retain_baseline, rule, vectors
from medsyringe.detection.recall_guard import compare
from scripts.experiments.expanded.expanded_annotations import ANNOTATIONS
from scripts.experiments.expanded.expanded_proposals import ART, LOCK, cache, cache_file, public_rows

TRIALS = ART / "detector_trials.json"
BASE = Path("artifacts/detector_release.json")
FEATURES = Path("medsyringe/detection/hybrid_detection.py")
TREE = ART / "detector_boost.joblib"
THRESHOLDS = (.20, .30, .40, .50, .60, .70, .80, .90)
SEED = 617


def load(cohort):
    source = [r for r in read_json(ANNOTATIONS)["rows"] if r["split"] == cohort]
    if cohort != "test":
        source += public_rows(cohort)
    branches = ("xpu", "sam", "fallback-xpu") if cohort == "calibration" else ("cpu", "sam", "fallback")
    cpu, sam, fallback = (cache(cohort, branch) for branch in branches)
    floor = read_json(BASE)["cutoff"]
    frames = {}
    for row in source:
        name = row["id"]
        earlier, later, extra = cpu[name], sam[name], fallback[name]
        baseline = earlier["prior"]
        if not any(p["score"] >= floor for p in baseline):
            baseline = additions(baseline, extra["tiles"], floor, support=None, aux_cutoff=.90, iou=.30)
        branches = dict(primary=earlier["primary"], rf=earlier["rf"], sam=later["full"])
        frames[name] = dict(prior=baseline, branches=branches, proposals=candidates(branches),
                            timing=dict(old=earlier["cascade_seconds"]+extra["seconds"],
                                        primary=earlier["primary_seconds"], rf=earlier["rf_seconds"],
                                        sam=later["seconds"]))
    if len(source) != len(frames):
        raise ValueError("Duplicate reference IDs in an experiment cohort.")
    return source, frames


def fit(rows, frames, target, grouped=True):
    inputs, labels, groups = [], [], []
    for row in rows:
        proposals = frames[row["id"]]["proposals"]
        truths = [o["bbox"] for o in row["objects"]]
        inputs.extend(vectors(proposals))
        labels.extend(max((box_iou(p["bbox"], b) for b in truths), default=0.) >= target for p in proposals)
        groups.extend([row["group"]]*len(proposals))
    X, y = np.asarray(inputs), np.asarray(labels)
    sizes = Counter(groups)
    weights = np.array([1/sizes[g] if grouped else 1. for g in groups])
    for label in (True, False):
        weights[y == label] *= len(y)/(2*weights[y == label].sum())
    scale = StandardScaler().fit(X, sample_weight=weights)
    logistic = LogisticRegression(C=.5, random_state=SEED).fit(scale.transform(X), y, sample_weight=weights)
    boost = HistGradientBoostingClassifier(max_iter=65, max_leaf_nodes=9, l2_regularization=1.,
                                           random_state=SEED).fit(X, y, sample_weight=weights)
    return dict(center=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                weights=logistic.coef_[0].tolist(), bias=float(logistic.intercept_[0]),
                n=len(y), positives=int(y.sum()), target=target, grouped=grouped), boost


def probabilities(rows, head):
    scaled = (vectors(rows)-head["center"])/head["scale"]
    return 1./(1.+np.exp(-np.clip(scaled @ head["weights"] + head["bias"], -30., 30.)))


def short(result):
    return {k: result[k] for k in ("images", "objects", "tp", "fp", "fn", "precision", "recall", "f1",
                                  "ap50", "ap50_95", "exact_count", "negative_images", "correct_empty")}


def slices(rows):
    return {"all": rows, "capture": [r for r in rows if "file" in r],
            "public": [r for r in rows if "file" not in r],
            "new_frames": [r for r in rows if "file" in r and "previous_sha256" not in r]}


def apply(frame, settings, heads, boost=None):
    method, mode, cutoff = (settings[key] for key in ("method", "box", "cutoff"))
    needs = settings["channels"]
    proposals = frame["proposals"] if needs == "all" else candidates(
        {k: items if k in needs else [] for k, items in frame["branches"].items()})
    if method.startswith("logistic"):
        confidence = probabilities(proposals, heads[method])
    elif method.startswith("gradient"):
        confidence = boost[method].predict_proba(vectors(proposals))[:, 1] if len(proposals) else []
    else:
        confidence = rule(proposals, method)
    ranked = rank(proposals, confidence, mode)
    if settings["preserve"]:
        return retain_baseline(frame["prior"], ranked, cutoff, read_json(BASE)["cutoff"])
    return ranked


def locked_inputs():
    paths = (ANNOTATIONS, BASE, FEATURES, Path(__file__), Path("scripts/experiments/expanded/expanded_proposals.py"),
             Path("medsyringe/analysis/inference.py"), Path("artifacts/sam3_release.json"),
             Path("artifacts/rfdetr_verified_release.json"), Path("medsyringe/runtime/xpu_grounding.py"),
             *(cache_file("fit", branch) for branch in ("cpu", "sam", "fallback")),
             *(cache_file("calibration", branch) for branch in ("xpu", "sam", "fallback-xpu")))
    return {str(p): digest(p) for p in paths}


def select():
    if LOCK.exists():
        raise ValueError("This selection is already locked. Keep its independent evaluation unchanged.")
    train, training = load("fit")
    dev, checks = load("calibration")
    floor = read_json(BASE)["cutoff"]
    baseline = evaluate(dev, {k: f["prior"] for k, f in checks.items()}, floor)
    public = slices(dev)["public"]
    baseline_public = evaluate(public, {r["id"]: checks[r["id"]]["prior"] for r in public}, floor)
    heads, trees = {}, {}
    for target, grouped in ((.50, True), (.70, True), (.50, False)):
        suffix = f"-{int(target*100)}" + ("" if grouped else "-unweighted")
        heads["logistic"+suffix], trees["gradient"+suffix] = fit(train, training, target, grouped)
    trials = []
    controls = [("union", (name,)) for name in ("primary", "rf", "sam")]
    controls += [(m, "all") for m in ("union", "agreement", *heads, *trees)]
    controls += [(m, ("rf", "sam")) for m in ("union", "agreement")]
    for (method, channels), mode, preserve in product(controls, ("anchor", "weighted", "mask"), (False, True)):
        if preserve and len(channels) == 1:
            continue
        for cutoff in THRESHOLDS:
            if preserve and cutoff < .4:
                continue
            options = dict(method=method, channels=channels, box=mode, preserve=preserve, cutoff=cutoff)
            predictions = {k: apply(frame, options, heads, trees) for k, frame in checks.items()}
            result = evaluate(dev, predictions, cutoff)
            external = evaluate(public, {r["id"]: predictions[r["id"]] for r in public}, cutoff)
            guard = compare(baseline, result)
            eligible = (guard["passed"] and result["fp"] <= baseline["fp"] and
                        external["fp"] <= baseline_public["fp"])
            trials.append(dict(options=options, result=short(result), public=short(external),
                               guard=guard, eligible=eligible))
    def order(record):
        m = record["result"]
        return record["eligible"], m["f1"], m["recall"], m["ap50_95"], record["options"]["cutoff"]

    trials.sort(key=order, reverse=True)
    winner = next((t for t in trials if t["eligible"]), None)
    winning_method = winner["options"]["method"] if winner else None
    tree_sha256 = None
    if winning_method in trees:
        dump(trees[winning_method], TREE)
        tree_sha256 = digest(TREE)
    write_json(TRIALS, dict(baseline=short(baseline), public_baseline=short(baseline_public),
                           train_images=len(train), heads=heads, selected=winner, trials=trials))
    write_json(LOCK, dict(selected=winner, seed=SEED, heads=heads, calibration=short(baseline),
                          trials_sha256=digest(TRIALS), tree_sha256=tree_sha256, inputs=locked_inputs()))
    print("Calibration baseline", short(baseline), "selected", winner, flush=True)


def audit(cohort):
    config = read_json(LOCK)
    if config["inputs"] != locked_inputs() or digest(TRIALS) != config["trials_sha256"]:
        raise ValueError("Development artifacts changed after selection.")
    rows, frames = load(cohort)
    floor = read_json(BASE)["cutoff"]
    report = {}
    selected = config["selected"]
    trees = None
    if config["tree_sha256"]:
        if digest(TREE) != config["tree_sha256"]:
            raise ValueError("The frozen local estimator changed.")
        trees = {selected["options"]["method"]: load_model(TREE)}
    for name, subset in slices(rows).items():
        if not subset:
            continue
        baseline = evaluate(subset, {r["id"]: frames[r["id"]]["prior"] for r in subset}, floor)
        evaluation = dict(baseline=baseline)
        if selected:
            options = selected["options"]
            predictions = {r["id"]: apply(frames[r["id"]], options, config["heads"], trees) for r in subset}
            candidate = evaluate(subset, predictions, options["cutoff"])
            evaluation.update(candidate=candidate, guard=compare(baseline, candidate))
        report[name] = evaluation
    timing = {mode: median(f["timing"][mode] for f in frames.values()) for mode in ("old", "primary", "rf", "sam")}
    accepted = report["all"]
    before = short(accepted["baseline"])
    after = short(accepted["candidate"]) if selected else None
    write_json(ART / f"detector_{cohort}.json", dict(cohort=cohort, split_metrics=report,
               freeze_sha256=digest(LOCK), timing_median_seconds=timing))
    print(cohort, before, after, accepted.get("guard"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test", "reserve", "calibration"))
    args = parser.parse_args()
    select() if args.stage == "select" else audit(args.stage)


if __name__ == "__main__":
    main()
