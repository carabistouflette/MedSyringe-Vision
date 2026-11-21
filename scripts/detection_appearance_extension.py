"""Learn a crop verifier with the independently reviewed Open Images fit partition."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression

from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.detectors import Grounder, suppress
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.detection_crop_semantics import (DINOV2_INFO, FIT, SIGLIP, VisualSemantics, columns, feature_cache,
                                             adjust, fit_rows, local_example_boxes, read_features, scores)
from scripts.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, FLOOR, inputs, loaded
from scripts.detection_hillclimb_rank import grouped_difference, minimal, slices
from scripts.detection_openimages_additional import MANIFEST, REVIEW
from scripts.detection_semantics_audit import LOCK as INITIAL, apply_rules, crop_frame
from scripts.expanded_detection_study import short
from scripts.precision_study import source
from scripts.sam3_study import configure_cpu

WORK = DATA/"proposals"
FIT_CACHE = WORK/"openimages-fit-features.json"
BASELINE = WORK/"openimages-audit-baseline.json"
SELECTED = ART/"openimages_crop_extension_frozen.json"
RESULT = ART/"openimages_crop_extension_audit.json"
SEED = 7661


def scope(role):
    return [row for row in read_json(MANIFEST)["rows"] if row["split"]==f"openimages-{role}"]


def fit_snapshot():
    fit = scope("fit")
    old = fit_rows()
    audit = scope("audit")
    if len({r["id"] for r in (*fit,*old)})!=len(fit)+len(old) or leakage([
        *[{**r,"split":"fit"} for r in (*fit,*old)], *audit
    ]):
        raise ValueError("The appearance extension has a protected image or duplicate fit row.")
    return fit,old


def collect():
    configure_cpu()
    rows,_ = fit_snapshot()
    locked = {str(p):digest(p) for p in (MANIFEST,REVIEW,SIGLIP,DINOV2_INFO,Path(__file__))}
    key = dict(files=locked, author_groups={row["id"]:row["group"] for row in rows})
    payload = read_json(FIT_CACHE) if FIT_CACHE.exists() else dict(key=key,frames={})
    if payload["key"]!=key:
        raise ValueError("The crop-feature provenance changed.")
    classifier = grounder = None
    for row in rows:
        identifier = row["id"]
        if identifier in payload["frames"]:
            continue
        if classifier is None:
            classifier = VisualSemantics()
            grounder = Grounder(Path("."),"dino")
        raster = source(row)
        proposals = dict(floor=1.,ranked=suppress(grounder.propose(raster),.60))
        samples = local_example_boxes(row,proposals)
        # The shared crop-frame feature path uses a frame of accepted samples
        # here so the same geometry is used for reviewed positives and negatives.
        prepared = {"floor":0.,"ranked":[{"score":1.,**p} for p in samples]}
        payload["frames"][identifier] = crop_frame(classifier,raster,prepared)
        n = len(payload["frames"])
        if n%4==0 or n==len(rows):
            write_json(FIT_CACHE,payload)
            print("Open Images fit",n,"/",len(rows),flush=True)


def feature_heads(cached, groups):
    labels = []
    origin = []
    for key, frame in cached.items():
        labels.extend(p["label"] for p in frame["candidates"])
        origin.extend(groups[key] for _ in frame["candidates"])
    lengths = Counter(origin)
    weights = np.array([1/lengths[g] for g in origin])
    truth = np.asarray(labels,dtype=bool)
    for state in (True,False):
        mask = truth==state
        weights[mask] *= len(weights)/(2*weights[mask].sum())
    picked = {}
    for architecture,strength in product(("siglip","dino","joint"),(.5,2.,8.,24.)):
        values = np.vstack([columns(frame,architecture) for frame in cached.values() if frame["candidates"]])
        model = LogisticRegression(C=strength,random_state=SEED).fit(values,truth,sample_weight=weights)
        picked[f"{architecture}/{strength}"] = dict(name=architecture, head=dict(weights=model.coef_[0].tolist(),
                                                                                bias=float(model.intercept_[0])))
    return picked


def select():
    if SELECTED.exists() or BASELINE.exists() or RESULT.exists():
        raise ValueError("The Open Images audit is already frozen or has started.")
    old = read_features("fit")
    new = read_json(FIT_CACHE)
    fit,earlier = fit_snapshot()
    if (set(new["frames"])!={r["id"] for r in fit} or
        any(digest(p)!=sha for p,sha in new["key"]["files"].items())):
        raise ValueError("The new crop feature cache is incomplete or obsolete.")
    merged = {**old,**new["frames"]}
    groups = {r["id"]:r["group"] for r in (*fit,*earlier)}
    models = feature_heads(merged,groups)
    previous = read_json(INITIAL)["components"]
    visual = next((p for p in previous if p["head"] is None), None)
    floor = read_json(FLOOR)["cutoff"]
    development = {}
    for role in DEVELOPMENT:
        rows = inputs(role)
        baselines = loaded(role,"baseline")
        regions = read_features(role)
        before = {key:f["ranked"] for key,f in baselines.items()}
        after = {key:apply_rules(frame,regions[key],previous)[0] for key,frame in baselines.items()}
        with_visual = ({key:apply_rules(frame,regions[key],[visual])[0] for key,frame in baselines.items()}
                       if visual is not None else before)
        development[role] = (rows,regions,before,with_visual,minimal(rows,before,floor),minimal(rows,after,floor))
    scores_by_head = {name:{role:{key:scores(frame,model["name"],model["head"]) for key,frame in values[1].items()}
                            for role,values in development.items()} for name,model in models.items()}
    trials = []
    for model,addon,cutoff,prior in product(models,(False,True),(.15,.35,.55,.65,.8,1.1,1.45,2.0),(.23,.33,.43,.53)):
        bounds = (cutoff,-1e4,prior)
        outcomes = {}
        for role,(rows,regions,before,with_visual,plain,predecessor) in development.items():
            base = with_visual if addon else before
            hints = scores_by_head[model][role]
            result = {key:adjust(ranked,regions[key],hints[key],bounds,floor) for key,ranked in base.items()}
            final = minimal(rows,result,floor)
            outcomes[role] = dict(tp=final["tp"],fp=final["fp"],fn=final["fn"],gate=compare(plain,final),
                                  versus_previous=compare(predecessor,final),
                                  fp_delta=final["fp"]-plain["fp"],fp_delta_previous=final["fp"]-predecessor["fp"])
        trials.append(dict(model=model, visual=addon, cutoff=cutoff, proposal_floor=prior, outcomes=outcomes))
    safe = [trial for trial in trials if all(r["gate"]["passed"] and r["fp_delta"]<=0 and
                                            r["versus_previous"]["passed"] and r["fp_delta_previous"]<=0
                                            for r in trial["outcomes"].values())]

    def priority(trial):
        public = trial["outcomes"]["public-development"]
        simulator = trial["outcomes"]["phlebotomy-development"]
        return (public["versus_previous"]["recovered_matches"], -public["fp_delta_previous"],
                simulator["versus_previous"]["recovered_matches"], -simulator["fp_delta_previous"],
                public["tp"], trial["cutoff"])

    winner = max(safe,key=priority,default=None)
    components = None
    if winner:
        head = {**models[winner["model"]],"variant":winner["model"],
                "bounds":[winner["cutoff"],-1e4,winner["proposal_floor"]]}
        components = [visual,head] if winner["visual"] and visual is not None else [head]
    binding = [INITIAL,FIT_CACHE,MANIFEST,REVIEW,FLOOR,Path(__file__),feature_cache("fit"),
               *(feature_cache(r) for r in DEVELOPMENT)]
    write_json(SELECTED,dict(winner=winner,candidate=components,previous=previous,attempts=trials,
                             trial_count=len(trials),seed=SEED,fit_images=len(groups),
                             frozen_hashes={str(p):digest(p) for p in binding}))
    print("Frozen Open Images selection",len(trials),winner,flush=True)


def proof():
    lock = read_json(SELECTED)
    if any(digest(p)!=h for p,h in lock["frozen_hashes"].items()):
        raise ValueError("An Open Images selection input has changed.")
    return lock


def collect_baseline():
    from scripts.detection_dental_evaluate import runner

    configure_cpu()
    proof()
    rows = scope("audit")
    key = dict(selection_sha256=digest(SELECTED),data_sha256=digest(MANIFEST),
               ids={row["id"]:row["sha256"] for row in rows})
    payload = read_json(BASELINE) if BASELINE.exists() else dict(key=key,frames={})
    if payload["key"]!=key:
        raise ValueError("Open Images audit baseline has changed.")
    infer = close = None
    try:
        for row in rows:
            identifier = row["id"]
            if identifier in payload["frames"]:
                continue
            if infer is None:
                infer,close,_ = runner("baseline")
            start = perf_counter()
            payload["frames"][identifier] = {**infer(source(row)),"ms":(perf_counter()-start)*1000}
            write_json(BASELINE,payload)
            print("Open Images audit baseline",len(payload["frames"]),"/",len(rows),flush=True)
    finally:
        if close:
            close()


def audit():
    if RESULT.exists():
        raise ValueError("The reserved evaluation has already been recorded.")
    configure_cpu()
    used = proof()
    rows = scope("audit")
    baseline = read_json(BASELINE)
    if baseline["key"]["selection_sha256"]!=digest(SELECTED) or set(baseline["frames"])!={r["id"] for r in rows}:
        raise ValueError("The Open Images audit baseline is incomplete or stale.")
    cutoff = read_json(FLOOR)["cutoff"]
    records = {"baseline":{key:r["ranked"] for key,r in baseline["frames"].items()},"earlier":{}}
    choices = {"earlier":used["previous"]}
    if used["candidate"]:
        records["extension"]={}
        choices["extension"]=used["candidate"]
    encoder = VisualSemantics()
    durations = {}
    for row in rows:
        identifier = row["id"]
        before = baseline["frames"][identifier]
        started = perf_counter()
        crops = crop_frame(encoder,source(row),before)
        for name,settings in choices.items():
            records[name][identifier] = apply_rules(before,crops,settings)[0]
        durations[identifier]=(perf_counter()-started)*1000
    measures = {name:evaluate(rows,p,cutoff) for name,p in records.items()}
    previous = measures["earlier"]
    next_result = measures.get("extension")
    guard = compare(previous,next_result) if next_result else None
    passed = bool(next_result and guard and guard["passed"] and next_result["fp"]<=previous["fp"] and
                  compare(measures["baseline"],next_result)["passed"] and next_result["fp"]<=measures["baseline"]["fp"] and
                  (guard["recovered_matches"] or next_result["fp"]<previous["fp"]))
    write_json(RESULT,dict(passed=passed,cutoff=cutoff,computed={key:short(r) for key,r in measures.items()},
                           cases={key:dict(fp=r["false_positives"],fn=r["false_negatives"],
                                           slices=slices(rows,r,records[key],cutoff)) for key,r in measures.items()},
                           intervals={name:grouped_difference(rows,records["baseline"],values)
                                      for name,values in records.items() if name!="baseline"},
                           instance_guard=guard,selection_sha256=digest(SELECTED),durations=durations,
                           limitations="Curated positive-heavy source. One negative photo and few independently grouped scenes."))
    print("Open Images independent results",passed,{k:short(r) for k,r in measures.items()},flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("collect-fit","select","collect-baseline","audit"))
    action=parser.parse_args().action
    {"collect-fit":collect,"select":select,"collect-baseline":collect_baseline,"audit":audit}[action]()
