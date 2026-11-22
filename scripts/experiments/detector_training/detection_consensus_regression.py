"""Verify the already selected consensus policy on the complete regression suite."""
import argparse
from pathlib import Path
from time import perf_counter

from medsyringe.detection.additive import overlap_fraction
from medsyringe.data import digest,read_json,write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_training.detection_consensus import MODELS,bound,classify,features
from scripts.experiments.detector_training.detection_consensus_audit import FREEZE,RESULT as FRESH,open_predictor,proof
from scripts.experiments.detector_audits.detection_crop_semantics import VisualSemantics
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART,DATA,FLOOR
from scripts.experiments.detector_audits.detection_precision_confirmation import CAPTURE,COHORTS,LOCK as PRECISION,cohort
from scripts.experiments.detector_audits.detection_semantics_audit import crop_frame
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.expanded.expanded_proposals import image_for
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

REGRESSION = tuple(group for group in COHORTS if group!="fresh")
RESULT = ART/"consensus_regression.json"


def path(group,stage):
    return DATA/"proposals"/f"consensus-regression-{group}-{stage}.json"


def precision_path(group):
    return DATA/"proposals"/f"fixed-precision-{group}.json"


def can_add(p,kept,prior):
    box = p["bbox"]
    return p["score"]>=prior and all(box_iou(box,q["bbox"])<.42 and overlap_fraction(box,q["bbox"])<.80 for q in kept)


def screen(group,root):
    from scripts.experiments.detector_audits.detection_crop_semantics import proposals

    configure_cpu()
    current = proof()
    settings = current["selection"]["settings"]
    control = read_json(PRECISION)["components"][0]
    if settings["guard_veto"] or control["bounds"][1]!=settings["reject"] or settings["opinions"]!="both-open":
        raise ValueError("The regression shortcut is only valid for the frozen policy and identical rejection gate.")
    reference,baseline = cohort(group)
    result = read_json(precision_path(group))
    if set(result["frames"])!=set(baseline) or result["key"]["selection_sha256"]!=digest(PRECISION):
        raise ValueError("Complete the identical preliminary veto on the same reference set.")
    tag = dict(selection_sha256=digest(FREEZE),code_sha256=digest(__file__),precision_sha256=digest(precision_path(group)))
    destination = path(group,"eligible")
    saved = read_json(destination) if destination.exists() else dict(key=tag,frames={})
    if saved["key"]!=tag:
        raise ValueError("Saved regression gate has changed inputs.")
    encoder = None
    for row in reference:
        identifier = row["id"]
        if identifier in saved["frames"]:
            continue
        old = baseline[identifier]
        cutoff = old["floor"]
        estimated = result["frames"][identifier]["ranked"]
        kept = [p for p in estimated if p["score"]>=cutoff]
        possible = [p for p in proposals(old) if p.get("baseline_score",p["score"])<cutoff and can_add(p,kept,settings["prior"])]
        measures = []
        if possible:
            if encoder is None:
                encoder = VisualSemantics()
            picture = image_for(row,root) if group in CAPTURE else source(row)
            # Geometry and the existing veto are settled. Encode only pending
            # regions that could survive the automatic duplicate check.
            subset = crop_frame(encoder,picture,{"ranked":possible,"floor":0.})
            models = current["model_heads"]
            raw = features(subset,"semantic-best",models["joint"],models["shape"],{})
            measures = [(p,text,learned,shape) for p,_,text,learned,shape,_ in raw
                        if text>=settings["text"] and learned>=settings["joint"] and
                        (not settings["shape"] or shape>=.10)]
        saved["frames"][identifier] = measures
        n = len(saved["frames"])
        if n%8==0 or n==len(reference):
            write_json(destination,saved)
            print(group,"screened",n,"/",len(reference),"with",sum(bool(v) for v in saved["frames"].values()),
                  "potential rescue scenes",flush=True)


def auxiliary(name,root):
    configure_cpu()
    proof()
    model = None
    for scope in REGRESSION:
        refs,_ = cohort(scope)
        screened = read_json(path(scope,"eligible"))
        if set(screened["frames"])!={r["id"] for r in refs}:
            raise ValueError("Potential rescue screening is incomplete.")
        checked = [row for row in refs if screened["frames"][row["id"]]]
        destination = path(scope,name)
        key = dict(screen_sha256=digest(path(scope,"eligible")),selection_sha256=digest(FREEZE))
        current = read_json(destination) if destination.exists() else dict(key=key,frames={})
        if current["key"]!=key:
            raise ValueError("The cached model proposals differ from the screening inputs.")
        for row in checked:
            identifier=row["id"]
            if identifier in current["frames"]:
                continue
            if model is None:
                model=open_predictor(name)
            im = image_for(row,root) if scope in CAPTURE else source(row)
            current["frames"][identifier]=model(im)
            write_json(destination,current)
        write_json(destination,current)
        print(scope,name,len(checked),"evaluated",flush=True)


def measure():
    if RESULT.exists():
        raise ValueError("The consensus regressions have already been scored.")
    current = proof()
    fresh = read_json(FRESH)
    settings = current["selection"]["settings"]
    floor = read_json(FLOOR)["cutoff"]
    results = {}
    for group in REGRESSION:
        refs,base=cohort(group)
        initial = {key:p["ranked"] for key,p in base.items()}
        marks = read_json(path(group,"eligible"))["frames"]
        after = {key:p["ranked"] for key,p in read_json(precision_path(group))["frames"].items()}
        details = {key:read_json(path(group,key))["frames"] for key in MODELS}
        for key,proposals in marks.items():
            if not proposals:
                continue
            frame = dict(ranked=after[key],floor=base[key]["floor"])
            keep = [(p,p["score"],1.0,0.,0.,{}) for p in frame["ranked"] if p["score"]>=frame["floor"]]
            vetted = [(p,p.get("baseline_score",p["score"]),text,learned,shape,
                        {m:bound(p["bbox"],details[m][key]) for m in MODELS}) for p,text,learned,shape in proposals]
            after[key] = classify(frame,[*keep,*vetted],settings)
        old,new = (evaluate(refs,p,floor) for p in (initial,after))
        guard = compare(old,new)
        results[group] = dict(previous=short(old),candidate=short(new),guard=guard,
                              branch_evaluated_images=sum(bool(v) for v in marks.values()),
                              errors=dict(previous=old["false_positives"],candidate=new["false_positives"]))
    safe = all(r["guard"]["passed"] and r["candidate"]["fp"]<=r["previous"]["fp"] and
               (name not in CAPTURE or r["candidate"]["exact_count"]==r["previous"]["exact_count"])
               for name,r in results.items())
    write_json(RESULT,dict(passed=bool(safe and fresh["passed"]),frozen_sha256=digest(FREEZE),
                           primary_audit_sha256=digest(FRESH),outcomes=results))
    print("Consensus regression",safe,{key:((r["previous"]["tp"],r["previous"]["fp"]),
                                           (r["candidate"]["tp"],r["candidate"]["fp"])) for key,r in results.items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("screen","auxiliary","measure"))
    parser.add_argument("--scope",choices=REGRESSION)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--backend",choices=MODELS)
    args=parser.parse_args()
    if args.action=="measure":
        measure()
    elif args.action=="screen" and args.scope and (args.scope not in CAPTURE or args.source):
        screen(args.scope,args.source)
    elif args.action=="auxiliary" and args.backend and args.source:
        auxiliary(args.backend,args.source)
    else:
        parser.error("Supply the scope and, where needed, the original capture directory or backend.")
