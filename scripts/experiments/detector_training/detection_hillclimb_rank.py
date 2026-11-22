"""Freeze a cross-model detection policy before evaluating the new public audit."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from medsyringe.detection.additive import overlap_fraction
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou, suppress
from medsyringe.evaluate import best_threshold, evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_training.detection_hillclimb_dfine import RECORD, SELECTION as DFINE_LOCK, proposal_cache
from scripts.experiments.detector_training.detection_hillclimb_experiment import (ART, AUDIT, BACKENDS, DEVELOPMENT, FLOOR, HELD_OUT,
                                                   SELECTED, cache, inputs, loaded, proposals)
from scripts.experiments.detector_training.detection_hillclimb_mix import LOG as MIX_LOG, proposal_path as mix_path
from scripts.experiments.detector_audits.detection_open_models import OPEN_MODELS, open_views, proposal_path as open_path
from scripts.experiments.expanded.expanded_detection_study import short

MODEL_COMBINATIONS = {"two-model": ("adapted-rf","dfine"), "rf-and-multidomain": ("adapted-rf","dfine-multidomain"),
                      "open-agreement": ("dino-base","owlv2")}
MODELS = (*BACKENDS[1:],"dfine","dfine-multidomain",*OPEN_MODELS,*MODEL_COMBINATIONS)
VIEWS = ("whole","wide","mixed","stable")


def data(scope, model):
    if model == "dfine-multidomain":
        return cached_frames(scope,read_json(mix_path(scope)),"training_sha256",MIX_LOG)
    if model == "dfine":
        return cached_frames(scope,read_json(proposal_cache(scope)),"checkpoint_sha256",RECORD)
    return loaded(scope,model)


def cached_frames(scope, raw, checkpoint_key, path):
    if raw["key"][checkpoint_key] != digest(path):
        raise ValueError("The proposals differ from their checkpoint.")
    if raw["key"]["rows"] != {r["id"]:r.get("derived_sha256",r["sha256"]) for r in inputs(scope)}:
        raise ValueError("The proposals have a different reference set.")
    frames = raw["frames"]
    if set(frames) != set(raw["key"]["rows"]):
        raise ValueError("The proposals are incomplete.")
    return frames


def agree(primary, secondary):
    outputs = []
    for first in primary:
        choices = [(box_iou(first["bbox"],p["bbox"]),p) for p in secondary]
        matches = [item for item in choices if item[0]>=.45]
        if not matches:
            continue
        _, second = max(matches,key=lambda item:item[0]*item[1]["score"])
        score = float(np.sqrt(first["score"]*second["score"]))
        box = [(a+b)/2 for a,b in zip(first["bbox"],second["bbox"],strict=True)]
        outputs.append(dict(bbox=box,score=score))
    return suppress(outputs,.40)


def candidate_sets(scope,model):
    if model in OPEN_MODELS:
        return open_views(scope,model)
    if model not in MODEL_COMBINATIONS:
        raw = data(scope,model)
        return {view:{name:proposals(frame,view) for name,frame in raw.items()} for view in VIEWS}
    first,second = (candidate_sets(scope,m) for m in MODEL_COMBINATIONS[model])
    return {view:{name:agree(predictions,second[view][name]) for name,predictions in first[view].items()}
            for view in VIEWS}


def augment(baseline, candidates, policy):
    floor = baseline["floor"]
    source = baseline["ranked"]
    retained = [p for p in source if p["score"]>=floor]
    if retained and policy["scope"]=="empty":
        return source
    vetoed = [p for p in source if "crop_score" in p]
    visible = [p for p in source if .26 <= p["score"] < floor and not p.get("review_reason")]
    new = []
    iou, proportional = (.35,.75) if policy["duplicate"]=="strict" else (.55,.88)
    for p in candidates:
        box = p["bbox"]
        if p["score"] < policy["threshold"] or (box[2]-box[0])*(box[3]-box[1]) > policy.get("max_area",1.):
            continue
        if any(box_iou(box,q["bbox"])>=iou or overlap_fraction(box,q["bbox"])>=proportional
               for q in (*retained,*new,*vetoed)):
            continue
        if policy["require_grounding"] and not any(box_iou(box,q["bbox"])>=.38 for q in visible):
            continue
        new.append(dict(bbox=box,score=floor,auxiliary_score=p["score"],source=policy["model"]))
    return [*retained,*new,*(p for p in source if p["score"]<floor)]


def assemble(rows, base, extra, policy):
    return {row["id"]:augment(base[row["id"]],extra[row["id"]],policy) for row in rows}


def minimal(rows, predictions, floor):
    """Score only the operating point when screening development policies."""
    from medsyringe.evaluate import match

    accepted = {key:[p for p in value if p["score"]>=floor] for key,value in predictions.items()}
    assignments, used = match(rows,accepted,.50)
    gold = {r["id"]:[o["bbox"] for o in r["objects"]] for r in rows}
    misses = [{"id":key,"box":box} for key,boxes in gold.items() for i,box in enumerate(boxes)
              if i not in used[key]]
    found = sum(map(len,used.values()))
    return dict(images=len(rows),objects=sum(map(len,gold.values())),tp=found,fn=len(misses),
                fp=len(assignments)-found,false_negatives=misses)


def review_proposals(baseline, candidates, policy):
    ranked = baseline["ranked"]
    floor = baseline["floor"]
    accepted = [p for p in ranked if p["score"]>=floor]
    original = [p for p in ranked if p["score"]<floor][:5]
    if accepted and policy["scope"]=="empty":
        return accepted,original,[]
    supplemental = []
    for proposal in candidates:
        box = proposal["bbox"]
        if proposal["score"]<policy["threshold"]:
            continue
        if any(box_iou(box,p["bbox"])>=.45 or overlap_fraction(box,p["bbox"])>=.84
               for p in (*accepted,*original,*supplemental)):
            continue
        supplemental.append(proposal)
        if len(supplemental)>=policy["limit"]:
            break
    return accepted,original,supplemental


def review_reach(rows, baselines, model_views, policy):
    before,after={},{}
    added=0
    for row in rows:
        key=row["id"]
        accepted,original,extra=review_proposals(baselines[key],model_views[key],policy)
        added+=len(extra)
        # Preserve review order and give every available suggestion equal
        # eligibility for the potential one-to-one match.
        before[key]=[{**p,"score":1.} for p in (*accepted,*original)]
        after[key]=before[key]+[{**p,"score":1.} for p in extra]
    old,new=(minimal(rows,r,1.) for r in (before,after))
    check=compare(old,new)
    return dict(previous_found=old["tp"],available_found=new["tp"],references=new["objects"],
                suggestions_added=added, reach=check)


def freeze():
    if SELECTED.exists() or DFINE_LOCK.exists():
        raise ValueError("A detection decision has already been frozen.")
    floor = read_json(FLOOR)["cutoff"]
    references = {name:inputs(name) for name in DEVELOPMENT}
    baseline = {name:data(name,"baseline") for name in DEVELOPMENT}
    base_scores = {name:evaluate(references[name],{k:v["ranked"] for k,v in baseline[name].items()},floor)
                   for name in DEVELOPMENT}
    candidates = {model:{name:candidate_sets(name,model) for name in DEVELOPMENT} for model in MODELS}
    rows = product(MODELS,VIEWS,(.025,.06,.12,.23,.40,.60,.82),(False,True),("all","empty"),
                   ("strict","separate"),(1.,.08,.025))
    trials = []
    for model,view,threshold,support,scope,duplicate,max_area in rows:
        policy = dict(model=model, view=view,threshold=threshold,require_grounding=support,
                      scope=scope,duplicate=duplicate,max_area=max_area)
        scores = {}
        for name in DEVELOPMENT:
            after = minimal(references[name],assemble(references[name],baseline[name],
                            candidates[model][name][view],policy),floor)
            guard = compare(base_scores[name],after)
            scores[name] = dict(guard=guard,tp=after["tp"],fp=after["fp"],fn=after["fn"],
                                extra_fp=after["fp"]-base_scores[name]["fp"])
        trials.append(dict(policy=policy,outcome=scores))
    def priority(trial):
        outcomes = trial["outcome"]
        return (outcomes["public-development"]["guard"]["recovered_matches"],
                sum(item["guard"]["recovered_matches"] for item in outcomes.values()),
                trial["policy"]["scope"] == "empty",
                trial["policy"]["view"] == "whole",
                -trial["policy"]["max_area"],
                trial["policy"]["threshold"])
    safe = [t for t in trials if all(v["guard"]["passed"] and v["extra_fp"]<=0 for v in t["outcome"].values())]
    winner = max(safe,key=priority,default=None)
    selected = winner if winner and any(priority(winner)[:2]) else None
    shadow = winner or max(trials,key=lambda t:(-sum(v["extra_fp"] for v in t["outcome"].values()),*priority(t)))
    standalone = {}
    for model in MODELS:
        mat = candidates[model]
        variants = {}
        for view in VIEWS:
            pooled = [r for d in references.values() for r in d]
            merged = {k:v for name in DEVELOPMENT for k,v in mat[name][view].items()}
            cutoff = best_threshold(pooled,merged)
            variants[view] = dict(cutoff=cutoff, scopes={name:short(evaluate(references[name],mat[name][view],cutoff))
                                                       for name in DEVELOPMENT})
        standalone[model]=variants
    review_trials=[]
    for model,view,confidence,scope,limit in product(MODELS,VIEWS,(.08,.16,.28,.45),("all","empty"),(1,2)):
        policy=dict(model=model,view=view,threshold=confidence,scope=scope,limit=limit)
        evaluations={name:review_reach(references[name],baseline[name],candidates[model][name][view],policy)
                     for name in DEVELOPMENT}
        review_trials.append(dict(policy=policy,outcome=evaluations))
    review_winner=max(review_trials,key=lambda t:(
        t["outcome"]["public-development"]["reach"]["recovered_matches"],
        sum(v["reach"]["recovered_matches"] for v in t["outcome"].values()),
        -sum(v["suggestions_added"] for v in t["outcome"].values()),-t["policy"]["limit"]))
    review_selected=review_winner if any(v["reach"]["recovered_matches"] for v in review_winner["outcome"].values()) else None
    files = [Path(__file__),Path("scripts/experiments/detector_training/detection_hillclimb_experiment.py"),
             Path("scripts/experiments/detector_audits/detection_open_models.py"),RECORD,MIX_LOG]
    for role in DEVELOPMENT:
        files += [cache(role,name) for name in BACKENDS]+[proposal_cache(role),mix_path(role)]
        files += [open_path(role,name) for name in OPEN_MODELS]
    bindings = {str(p):digest(p) for p in files}
    write_json(SELECTED,dict(selected=selected,shadow=shadow,trial_count=len(trials),trials=trials,
                             inputs=bindings,standalone=standalone,baseline={k:short(v) for k,v in base_scores.items()},
                             review_selected=review_selected,review_trials=review_trials))
    write_json(DFINE_LOCK,dict(comparison_sha256=digest(SELECTED)))
    print("Frozen",selected or shadow,"release-eligible",bool(selected),flush=True)


def grouped_difference(rows, before, after, seed=942):
    groups = sorted({row["group"] for row in rows})
    if len(groups) < 6:
        return None
    counts = {}
    for matches, phase in ((before,"before"),(after,"after")):
        floor = read_json(FLOOR)["cutoff"]
        group_data = []
        for group in groups:
            group_rows = [r for r in rows if r["group"]==group]
            metrics = minimal(group_rows,{row["id"]:matches[row["id"]] for row in group_rows},floor)
            group_data.append([metrics[k] for k in ("tp","fp","fn")])
        counts[phase]=np.array(group_data)
    rng=np.random.default_rng(seed)
    selected=rng.integers(len(groups),size=(2000,len(groups)))
    first,last=(counts[phase][selected].sum(axis=1) for phase in ("before","after"))
    def recall(x):
        return x[:,0]/np.maximum(1,x[:,0]+x[:,2])
    def precision(x):
        return x[:,0]/np.maximum(1,x[:,0]+x[:,1])
    return dict(groups=len(groups), samples=len(selected),
                interval_recall_delta=np.quantile(recall(last)-recall(first),[.025,.975]).tolist(),
                interval_precision_delta=np.quantile(precision(last)-precision(first),[.025,.975]).tolist())


def slices(rows, metrics, predictions, floor):
    missed={(p["id"],tuple(p["box"])) for p in metrics["false_negatives"]}
    def tally(condition):
        selected = [(row["id"],tuple(o["bbox"])) for row in rows for o in row["objects"] if condition(row,o)]
        return dict(references=len(selected), found=sum(key not in missed for key in selected))
    types=Counter()
    false_alarms=Counter()
    for row in rows:
        if row["objects"]:
            continue
        categories={obj["kind"] for obj in row.get("lookalikes",[])} or {"unspecified negative"}
        for name in categories:
            types[name]+=1
            false_alarms[name]+=any(p["score"]>=floor for p in predictions[row["id"]])
    return dict(small=tally(lambda _,o:(o["bbox"][2]-o["bbox"][0])*(o["bbox"][3]-o["bbox"][1]) < .025),
                partial=tally(lambda _,o: o.get("occluded") or o.get("truncated")),
                multiple=tally(lambda row,_:len(row["objects"])>1),
                negative={name:dict(images=count,false_alarm_images=false_alarms[name])
                          for name,count in sorted(types.items())})


def audit():
    if AUDIT.exists():
        raise ValueError("The public audit has already been recorded.")
    freeze = read_json(SELECTED)
    if any(digest(path)!=checksum for path,checksum in freeze["inputs"].items()):
        raise ValueError("The development inputs changed after the freeze.")
    chosen=freeze["shadow"]["policy"]
    floor=read_json(FLOOR)["cutoff"]
    reports={}
    review_reports={}
    for scope in HELD_OUT:
        rows=inputs(scope)
        baseline=data(scope,"baseline")
        fresh=candidate_sets(scope,chosen["model"])[chosen["view"]]
        before={key:v["ranked"] for key,v in baseline.items()}
        after=assemble(rows,baseline,fresh,chosen)
        original=evaluate(rows,before,floor)
        final=evaluate(rows,after,floor)
        groups=Counter(r["group"] for r in rows)
        reports[scope]=dict(baseline=short(original),candidate=short(final),guard=compare(original,final),
                        baseline_errors=dict(fp=original["false_positives"],fn=original["false_negatives"]),
                        candidate_errors=dict(fp=final["false_positives"],fn=final["false_negatives"]),
                        slices=dict(baseline=slices(rows,original,before,floor), candidate=slices(rows,final,after,floor)),
                        group_count=len(groups),selected_groups=dict(groups),
                        group_bootstrap=grouped_difference(rows,before,after))
        if freeze.get("review_selected"):
            option=freeze["review_selected"]["policy"]
            views=candidate_sets(scope,option["model"])
            review_reports[scope]=review_reach(rows,baseline,views[option["view"]],option)
    safe=all(v["guard"]["passed"] and v["candidate"]["fp"]<=v["baseline"]["fp"] for v in reports.values())
    recovered=reports["public-audit"]["guard"]["recovered_matches"]
    write_json(AUDIT,dict(passed=bool(freeze["selected"] and safe and recovered),
                           selection_sha256=digest(SELECTED),cases=reports, review=review_reports,
                           review_passed=("public-audit" in review_reports and
                                          review_reports["public-audit"]["reach"]["recovered_matches"]>0 and
                                          all(r["reach"]["passed"] for r in review_reports.values()))))
    print({scope:(r["baseline"],r["candidate"],r["guard"]) for scope,r in reports.items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("select","audit"))
    (freeze if parser.parse_args().action=="select" else audit)()
