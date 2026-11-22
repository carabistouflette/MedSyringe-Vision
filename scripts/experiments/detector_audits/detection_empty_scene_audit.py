"""Measure conservative recovery only in scenes originally accepted as empty."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_training.detection_consensus import MODELS, classify, features
from scripts.experiments.detector_training.detection_consensus_audit import FREEZE as CONSENSUS, cache as fresh_cache, open_predictor, proof
from scripts.experiments.detector_training.detection_consensus_regression import path as regression_cache
from scripts.experiments.detector_audits.detection_crop_semantics import VisualSemantics, read_features
from scripts.experiments.transfer.detection_dental_evaluate import runner
from scripts.experiments.detector_audits.detection_empty_scene_data import MANIFEST, REFERENCES
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, EXISTING, FLOOR, inputs, loaded
from scripts.experiments.detector_training.detection_hillclimb_rank import grouped_difference, minimal
from scripts.experiments.detector_audits.detection_open_models import open_views
from scripts.experiments.detector_audits.detection_precision_confirmation import CAPTURE, COHORTS, LOCK as PRECISION, cohort
from scripts.experiments.detector_audits.detection_semantics_audit import apply_rules, crop_frame
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

LOCK = ART / "empty_scene_selection.json"
AUDIT = ART / "empty_scene_audit_freeze.json"
RESULT = ART / "empty_scene_audit_results.json"
BACKENDS = ("baseline", *MODELS)


def file(name):
    return DATA / "proposals" / f"empty-scene-audit-{name}.json"


def is_empty(frame):
    return not any(p["score"] >= frame["floor"] for p in frame["ranked"])


def active(data, cutoff):
    return [p for p in data if p["score"] >= cutoff]


def select():
    if LOCK.exists() or any(file(name).exists() for name in BACKENDS):
        raise ValueError("The restricted recovery policy is already sealed.")
    state = proof()
    settings = state["selection"]["settings"]
    parameters = state["model_heads"]
    control = read_json(PRECISION)["components"]
    comparisons = {}
    for scope in DEVELOPMENT:
        groups = inputs(scope)
        base = loaded(scope,"baseline")
        frames = read_features(scope)
        opinions = {model: open_views(scope, model)["wide"] for model in MODELS}
        previous, candidate = {}, {}
        for key, frame in base.items():
            previous[key] = apply_rules(frame, frames[key], control)[0]
            if not is_empty(frame):
                candidate[key] = previous[key]
                continue
            candidates = features(frames[key],"semantic-best",parameters["joint"],parameters["shape"],
                                  {model:values[key] for model,values in opinions.items()})
            candidate[key] = classify(frame,candidates,settings)
        old,new = (minimal(groups,output,read_json(FLOOR)["cutoff"]) for output in (previous,candidate))
        guard = compare(old,new)
        comparisons[scope] = dict(previous={k:old[k] for k in ("tp","fp","fn")},
                                  candidate={k:new[k] for k in ("tp","fp","fn")},guard=guard)
    if not all(x["guard"]["passed"] and x["candidate"]["fp"]<=x["previous"]["fp"] and
               x["guard"]["recovered_matches"]>0 for x in comparisons.values()):
        raise ValueError("Restricted recovery did not improve both development cohorts safely.")
    sources = (CONSENSUS,PRECISION,EXISTING,FLOOR,Path("scripts/experiments/detector_audits/detection_empty_scene_audit.py"))
    write_json(LOCK,dict(inputs={str(p):digest(p) for p in sources},settings=settings,heads=parameters,
                         gate="Run two-model recovery only if the original detector reports no accepted box.",
                         control=control,development=comparisons,
                         qualification="At least one new reference match on the untouched audit. No lost instances, "
                                      "added false positives or new false alarm scenes there or in the audited regression groups. "
                                      "Every Fil Rouge count must stay exact."))
    print("Restricted recovery selected on development",comparisons,flush=True)


def freeze():
    if AUDIT.exists() or any(file(name).exists() for name in BACKENDS):
        raise ValueError("The audit must be sealed before generating any predictions.")
    decisions = (LOCK,MANIFEST,REFERENCES)
    write_json(AUDIT,dict(inputs={str(p):digest(p) for p in decisions}))
    binding()


def binding():
    keys = read_json(AUDIT)["inputs"]
    policy = read_json(LOCK)
    if any(digest(p)!=value for p,value in {**keys,**policy["inputs"]}.items()):
        raise ValueError("The frozen audit configuration or references have changed.")
    return policy,read_json(MANIFEST)["rows"]


def cache(name):
    path = file(name)
    identity = digest(AUDIT)
    records = read_json(path) if path.exists() else dict(audit_sha256=identity,frames={})
    if records["audit_sha256"]!=identity:
        raise ValueError("Proposals use a different audit binding.")
    return records


def collect(name):
    configure_cpu()
    _,rows = binding()
    tracked = cache(name)
    if name != "baseline":
        old = cache("baseline")["frames"]
        if set(old)!={r["id"] for r in rows}:
            raise ValueError("Baseline collection must finish first.")
        rows = [r for r in rows if is_empty(old[r["id"]]["ranked"])]
    needs = [r for r in rows if r["id"] not in tracked["frames"]]
    if not needs:
        if not file(name).exists():
            write_json(file(name),tracked)
        return
    if name=="baseline":
        detect,close,_=runner("baseline")
    else:
        detect,close=open_predictor(name),None
    try:
        for row in needs:
            im = source(row)
            start = perf_counter()
            result = detect(im)
            tracked["frames"][row["id"]] = dict(ranked=result,ms=(perf_counter()-start)*1000)
            write_json(file(name),tracked)
            print("Restricted scene audit",name,len(tracked["frames"]),"/",len(rows),flush=True)
    finally:
        if close:
            close()


def metrics(rows, previous, revised, floor):
    before, after = [evaluate(rows,pred,floor) for pred in (previous,revised)]
    old_alarms = {r["id"] for r in before["false_positives"]}
    new_alarms = {r["id"] for r in after["false_positives"]}
    return dict(previous=short(before),candidate=short(after),guard=compare(before,after),
                new_false_alarm_scenes=sorted(new_alarms-old_alarms),
                errors=dict(previous=before["false_positives"],candidate=after["false_positives"]))


def historical(policy, group):
    from scripts.experiments.detector_training.detection_consensus import bound

    rows,base = cohort(group)
    floor = read_json(FLOOR)["cutoff"]
    control = read_json(DATA/"proposals"/f"fixed-precision-{group}.json")["frames"]
    output = {key:frame["ranked"] for key,frame in control.items()}
    evaluated = output.copy()
    if group in CAPTURE:
        if not all(active(x["ranked"],x["floor"]) for x in base.values()):
            raise ValueError("The capture includes a frame eligible for restricted recovery.")
        return metrics(rows,output,evaluated,floor)
    if group=="fresh":
        raise ValueError("The earlier fresh group must be evaluated from its original model caches.")
    checks = read_json(regression_cache(group,"eligible"))["frames"]
    other = {name:read_json(regression_cache(group,name))["frames"] for name in MODELS}
    for key,proposals in checks.items():
        if not proposals or not is_empty(base[key]):
            continue
        record = dict(ranked=output[key],floor=base[key]["floor"])
        supported = [(p,p.get("baseline_score",p["score"]),text,learned,shape,
                      {m:bound(p["bbox"],other[m][key]) for m in MODELS}) for p,text,learned,shape in proposals]
        evaluated[key] = classify(record,supported,policy["settings"])
    return metrics(rows,output,evaluated,floor)


def run_both(policy, rows, base, other, encoder, controls=None):
    kept,revised,duration = {},{},[]
    for row in rows:
        key = row["id"]
        initial = base[key]
        picture = source(row)
        started = perf_counter()
        if is_empty(initial):
            frame = crop_frame(encoder,picture,initial)
            heads = policy["heads"]
            measured = features(frame,"semantic-best",heads["joint"],heads["shape"],
                                {name:other[name][key] for name in MODELS})
            kept[key] = initial["ranked"]
            revised[key] = classify(initial,measured,policy["settings"])
        elif controls is not None:
            kept[key]=revised[key]=controls[key]["ranked"]
        else:
            current = dict(ranked=active(initial["ranked"],initial["floor"]),floor=initial["floor"])
            measured = crop_frame(encoder,picture,current)
            kept[key]=revised[key]=apply_rules(initial,measured,policy["control"])[0]
        duration.append(perf_counter()-started)
    return kept,revised,duration


def measure():
    if RESULT.exists():
        raise ValueError("The restricted recovery study has already been scored.")
    configure_cpu()
    policy,rows = binding()
    floor = read_json(FLOOR)["cutoff"]
    base = {k:v["ranked"] for k,v in cache("baseline")["frames"].items()}
    if set(base)!={r["id"] for r in rows}:
        raise ValueError("Collect the complete sealed audit first.")
    other = {m:{k:v["ranked"] for k,v in cache(m)["frames"].items()} for m in MODELS}
    empty = {k for k,v in base.items() if is_empty(v)}
    if any(set(values)!=empty for values in other.values()):
        raise ValueError("Both recovery models must evaluate every empty scene.")
    encoder = VisualSemantics()
    before,after,times = run_both(policy,rows,base,other,encoder)
    study = metrics(rows,before,after,floor)
    study["paired_intervals"] = grouped_difference(rows,before,after)
    study["median_crop_seconds"] = median(times)
    regressions = {group:historical(policy,group) for group in COHORTS if group!="fresh"}
    fresh_rows,fresh_base = cohort("fresh")
    fresh_opinions = {m:{k:v["output"] for k,v in read_json(fresh_cache(m))["frames"].items()} for m in MODELS}
    fresh_control = read_json(DATA/"proposals"/"fixed-precision-fresh.json")["frames"]
    previous,revised,_ = run_both(policy,fresh_rows,fresh_base,fresh_opinions,encoder,controls=fresh_control)
    regressions["earlier-commons-audit"] = metrics(fresh_rows,previous,revised,floor)
    checks = [study,*regressions.values()]
    valid = all(r["guard"]["passed"] and not r["new_false_alarm_scenes"] and r["candidate"]["fp"]<=r["previous"]["fp"]
                for r in checks)
    exact = all(regressions[g]["candidate"]["exact_count"]==regressions[g]["previous"]["exact_count"] for g in CAPTURE)
    passed = valid and exact and study["guard"]["recovered_matches"] > 0
    write_json(RESULT,dict(passed=passed,audit_sha256=digest(AUDIT),fresh=study,regression=regressions))
    print("Restricted recovery qualifying result",passed,study["previous"],study["candidate"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("select","freeze","collect","measure"))
    parser.add_argument("--backend",choices=BACKENDS)
    args=parser.parse_args()
    if args.action=="collect":
        if not args.backend:
            parser.error("Model collection requires --backend.")
        collect(args.backend)
    else:
        {"select":select,"freeze":freeze,"measure":measure}[args.action]()
