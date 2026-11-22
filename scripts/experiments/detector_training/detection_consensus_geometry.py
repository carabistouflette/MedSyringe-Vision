"""Fit auxiliary box refinement on development photographs with recovered objects."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate, match
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_training.detection_consensus import MODELS, classify, features
from scripts.experiments.detector_audits.detection_crop_semantics import feature_cache, read_features
from scripts.experiments.detector_audits.detection_empty_scene_audit import LOCK, is_empty
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DEVELOPMENT, FLOOR, cache, inputs, loaded
from scripts.experiments.detector_audits.detection_open_models import open_views, proposal_path
from scripts.experiments.detector_audits.detection_semantics_audit import apply_rules
from scripts.experiments.expanded.expanded_detection_study import short

SELECTION = ART/"empty_scene_geometry_selection.json"


def refine(box, opinions, weights, selection):
    anchors = []
    for model,cutoff in zip(MODELS,(.27,.15),strict=True):
        qualified = [p for p in opinions[model] if p["score"]>=cutoff and box_iou(box,p["bbox"])>=.42]
        order = ((lambda p:(p["score"],box_iou(box,p["bbox"]))) if selection=="strongest" else
                 (lambda p:(box_iou(box,p["bbox"]),p["score"])))
        if not qualified:
            return box
        anchors.append(max(qualified,key=order)["bbox"])
    return np.average((box,*anchors),axis=0,weights=weights).tolist()


def apply(predictions,opinions,choice):
    updated = []
    for p in predictions:
        if p.get("source")!="crop consensus" or choice["name"]=="original":
            updated.append(p)
            continue
        box = refine(p["bbox"],opinions,choice["weights"],choice["match"])
        updated.append({**p,"bbox":box,"proposed_bbox_before_refinement":p["bbox"]})
    return updated


def readonly():
    decision = read_json(LOCK)
    if any(digest(file)!=expected for file,expected in decision["inputs"].items()):
        raise ValueError("The frozen recovery hypotheses have changed.")
    return decision


def development():
    if SELECTION.exists():
        raise ValueError("Geometric refinement has already been selected.")
    candidate = readonly()
    floor = read_json(FLOOR)["cutoff"]
    data = {}
    for scope in DEVELOPMENT:
        rows = inputs(scope)
        base = loaded(scope,"baseline")
        encoded = read_features(scope)
        aux = {m:open_views(scope,m)["wide"] for m in MODELS}
        opinions = {key:{m:a[key] for m,a in aux.items()} for key in base}
        raw = {}
        before = {}
        for key,frame in base.items():
            previous,_ = apply_rules(frame,encoded[key],candidate["control"])
            before[key] = previous
            raw[key] = previous
            if is_empty(frame):
                heads = candidate["heads"]
                observed = features(encoded[key],"semantic-best",heads["joint"],heads["shape"],opinions[key])
                raw[key] = classify(frame,observed,candidate["settings"])
        initial = evaluate(rows,before,floor)
        data[scope] = rows,raw,opinions,initial
    policies = [dict(name="original",weights=[1,0,0],match="closest")]
    weights = [(a,b,c) for a,b,c in product((1,2,3),(0,1,2),(0,1,2)) if b+c>0]
    policies.extend(dict(name="box fusion",weights=list(w),match=prior)
                    for w,prior in product(weights,("closest","strongest")))
    trials = []
    for choice in policies:
        outcomes = {}
        for scope,(rows,raw,opinions,baseline) in data.items():
            after = {key:apply(result,opinions[key],choice) for key,result in raw.items()}
            current = evaluate(rows,after,floor)
            previous_boxes = {(r["id"],tuple(r["prediction"]["bbox"])) for r in baseline["false_positives"]}
            new_boxes = {(r["id"],tuple(r["prediction"]["bbox"])) for r in current["false_positives"]}
            refined_ious = [iou for _,p,index,iou in match(rows,{k:[r for r in a if r["score"]>=floor] for k,a in after.items()},.5)[0]
                            if index is not None and p.get("source")=="crop consensus"]
            outcomes[scope] = dict(metrics=short(current), guard=compare(baseline,current),
                                   new_unmatched=len(new_boxes-previous_boxes), refined_iou_sum=sum(refined_ious))
        trials.append(dict(policy=choice,outcomes=outcomes))
    eligible = [t for t in trials if all(m["guard"]["passed"] and not m["new_unmatched"] for m in t["outcomes"].values())]
    best = max(eligible,key=lambda t:(sum(m["guard"]["recovered_matches"] for m in t["outcomes"].values()),
                                       sum(m["refined_iou_sum"] for m in t["outcomes"].values()),
                                       sum(m["metrics"]["ap50_95"] for m in t["outcomes"].values()),
                                       t["policy"]["name"]=="original"),default=None)
    bindings = [Path("scripts/experiments/detector_training/detection_consensus_geometry.py"),LOCK,FLOOR]
    for scope in DEVELOPMENT:
        bindings.extend([cache(scope,"baseline"),feature_cache(scope),*(proposal_path(scope,name) for name in MODELS)])
    write_json(SELECTION,dict(chosen=best,baseline=trials[0],options=trials,eligible=len(eligible),
                              inputs={str(p):digest(p) for p in bindings},
                              selection_scope="Development only. Independent deployment requires new examples "
                                              "and identical recall, false-positive and capture-count gates."))
    print("Geometric calibration",len(eligible),"/",len(trials),"safe:",best,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    development()
