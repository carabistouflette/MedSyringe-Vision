"""Compare transfer learning, localization, and cross-model support on locked sets."""
import argparse
from pathlib import Path

from joblib import load
import numpy as np
import torch

from medsyringe.data import digest, leakage, read_json, train_rows, write_json
from medsyringe.detectors import box_iou, suppress
from medsyringe.evaluate import best_threshold, evaluate, match
from medsyringe.localization import rerank
from medsyringe.proposal_learning import describe_proposals
from medsyringe.supervised import detector, predict
from scripts.optimize_detector import FINAL, counters
from scripts.precision_study import baseline, source
from scripts.refine_detector import ART, OLD
from scripts.supervised_study import prepare, sets

ROOT = Path(".")
SELECTION = ART / "supervised_selection.json"
NEW = ART / "supervised_external_manifest.json"
CONTROL = ART / "supervised_control.json"


def combine(grounder, direct, *, weight, box_weight, min_overlap):
    revised = []
    for g in grounder:
        overlap, _, reference = max(((box_iou(g["bbox"], s["bbox"]), i, s)
                                     for i, s in enumerate(direct)), default=(0, 0, None))
        confirm = bool(reference and overlap >= min_overlap)
        support = reference["score"] if confirm else 0.0
        box = np.array(g["bbox"])
        if confirm and box_weight:
            box = box*(1-box_weight) + np.array(reference["bbox"])*box_weight
        revised.append({"bbox": box.tolist(), "score": (1-weight)*g["score"] + weight*support})
    return suppress(revised)


def inputs(rows, split):
    stage, frozen = read_json(FINAL), read_json(OLD)
    experiment = read_json(ART / "supervised_study.json")
    path = Path(experiment["checkpoint"])
    if digest(path) != experiment["checkpoint_sha256"]:
        raise ValueError("Supervised checkpoint differs from the selected epoch.")
    model = detector(ROOT)
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    # Baseline materializes and checks public assets before either detector uses them.
    v3 = baseline(frozen, stage, rows, split)
    ssd = predict(model, prepare(rows))
    return stage, experiment, v3, ssd


def apply(rows, config, v3, ssd, *, localized=None):
    method = config["method"]
    if method == "ssd":
        return ssd
    if method == "localization":
        return localized
    if method == "fusion":
        return {key: combine(predictions[:24], ssd[key], weight=config["weight"],
                             box_weight=config["box_weight"], min_overlap=config["iou"])
                for key, predictions in v3.items()}
    return v3


def localize(rows, v3):
    report = read_json(ART / "localization_study.json")
    path = Path(report["checkpoint"])
    if digest(path) != report["checkpoint_sha256"]:
        raise ValueError("The localization checkpoint changed.")
    models = load(path)
    choice = report["selected"]
    return {row["id"]: rerank(v3[row["id"]][:24], describe_proposals(source(row), v3[row["id"]][:24]),
                             models, box_strength=choice["box_strength"],
                             quality_weight=choice["quality_weight"], overlap=.6)
            for row in rows}


def select(val):
    if SELECTION.exists():
        raise ValueError("Selection is already locked.")
    stage, experiment, v3, ssd = inputs(val, "precision-fresh")
    localized = localize(val, v3)
    base = counters(evaluate(val, v3, stage["cutoff"]))
    trials = [
        dict(method="ssd", cutoff=experiment["selected"]["cutoff"]),
        dict(method="localization", cutoff=read_json(ART / "localization_study.json")["selected"]["cutoff"]),
    ]
    for weight in (.15, .30, .45):
        for box_weight in (0, .25):
            for overlap in (.35, .5):
                config = dict(method="fusion", weight=weight, box_weight=box_weight, iou=overlap)
                pred = apply(val, config, v3, ssd)
                thresholds = sorted({p["score"] for values in pred.values() for p in values})
                # Choose the highest-F1 point among those retaining validation recall.
                # Ranking the candidates is cheaper once the score-ordered matches are known.
                assigned, _ = match(val, pred, .5)
                outcomes = {}
                tp, fp = 0, 0
                total = sum(len(r["objects"]) for r in val)
                for _, p, target, _ in assigned:
                    tp += target is not None
                    fp += target is None
                    outcomes[p["score"]] = (tp, fp)
                eligible = [t for t in thresholds if outcomes[t][0] >= base["tp"]]
                config["cutoff"] = max(eligible, key=lambda t: (
                    2*outcomes[t][0]/(total+sum(outcomes[t])), -outcomes[t][1], t
                )) if eligible else best_threshold(val, pred)
                trials.append(config)
    for trial in trials:
        prediction = apply(val, trial, v3, ssd, localized=localized)
        trial.update(counters(evaluate(val, prediction, trial["cutoff"])))
    viable = [t for t in trials if t["tp"] >= base["tp"] and t["fp"] < base["fp"]]
    choice = max(viable or trials, key=lambda t: (t["f1"], t["ap50_95"], t["precision"]))
    write_json(SELECTION, dict(chosen=choice, validation_baseline=base, validation_trials=trials,
                              has_validation_precision_gain=bool(viable),
                              external_sha256=digest(NEW), detector_sha256=digest(FINAL),
                              supervised_report_sha256=digest(ART / "supervised_study.json"),
                              localization_report_sha256=digest(ART / "localization_study.json")))
    print("Calibration", base, "chosen", choice, flush=True)


def score(rows, split):
    selection = read_json(SELECTION)
    if any(selection[key] != digest(path) for key, path in (
        ("external_sha256", NEW), ("detector_sha256", FINAL),
        ("supervised_report_sha256", ART / "supervised_study.json"),
        ("localization_report_sha256", ART / "localization_study.json")
    )):
        raise ValueError("The reference or pipeline changed after selection.")
    stage, _, v3, ssd = inputs(rows, split)
    choice = selection["chosen"]
    localized = localize(rows, v3) if choice["method"] == "localization" else None
    pred = apply(rows, choice, v3, ssd, localized=localized)
    original = evaluate(rows, v3, stage["cutoff"])
    revised = evaluate(rows, pred, choice["cutoff"])
    study = dict(baseline=original, candidate=revised, selection_sha256=digest(SELECTION))
    if split == "supervised-audit":
        ssd_cutoff = read_json(ART / "supervised_study.json")["selected"]["cutoff"]
        box_cutoff = read_json(ART / "localization_study.json")["selected"]["cutoff"]
        study["frozen_alternatives"] = {
            "ssd": counters(evaluate(rows, ssd, ssd_cutoff)),
            "localization": counters(evaluate(
                rows, localized if localized is not None else localize(rows, v3), box_cutoff
            )),
        }
    write_json(ART / f"supervised_comparison_{split}.json", study)
    print(split, counters(original), counters(revised), flush=True)


def control(rows, split, *, fit):
    frozen, stage = read_json(OLD), read_json(FINAL)
    prediction = baseline(frozen, stage, rows, split)
    if fit:
        reference = evaluate(rows, prediction, stage["cutoff"])
        scored, _ = match(rows, prediction, .5)
        correct = [p["score"] for _, p, target, _ in scored if target is not None]
        cutoff = correct[reference["tp"]-1] if reference["tp"] else stage["cutoff"]
        write_json(CONTROL, dict(cutoff=cutoff, fresh_sha256=digest(NEW),
                  baseline=counters(reference), calibration=counters(evaluate(rows, prediction, cutoff))))
    else:
        config = read_json(CONTROL)
        if config["fresh_sha256"] != digest(NEW):
            raise ValueError("External references changed since the control was fixed.")
        metrics = counters(evaluate(rows, prediction, config["cutoff"]))
        write_json(ART / "supervised_control_audit.json",
                   dict(control_sha256=digest(CONTROL), metrics=metrics))
        print("Score-only control", metrics, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "audit", "train", "historical", "control-select", "control-audit"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    training, validation, historical = sets()
    audit = read_json(NEW)["rows"]
    if leakage([*[{**row, "split": "training"} for row in training], *validation, *historical, *audit]):
        raise ValueError("The grouped experimental partitions overlap.")
    if args.stage == "control-select":
        control(validation, "precision-fresh", fit=True)
    elif args.stage == "control-audit":
        control(audit, "supervised-audit", fit=False)
    elif args.stage == "select":
        select(validation)
    elif args.stage == "audit":
        score(audit, "supervised-audit")
    elif args.stage == "historical":
        score(historical, "test")
    else:
        score(train_rows(ROOT), "train")


if __name__ == "__main__":
    main()
