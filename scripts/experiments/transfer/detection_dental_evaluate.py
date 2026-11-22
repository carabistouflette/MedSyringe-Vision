"""Run the frozen detector and candidate on the blind dental challenge."""
import argparse
from pathlib import Path
from time import perf_counter

from medsyringe.detection.additive import project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import suppress
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.transfer.detection_dental_challenge import LOCK as DENTAL
from scripts.experiments.detector_training.detection_hillclimb_corpus import ART, DATA
from scripts.experiments.detector_training.detection_hillclimb_experiment import EXISTING, FLOOR, SELECTED, proposals, tiles, trainer
from scripts.experiments.detector_training.detection_hillclimb_rank import MODEL_COMBINATIONS, agree, augment, review_reach
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.detector_audits.precision_study import source

CHECK = ART/"dental_reference_review.json"
RESULT = ART/"dental_audit.json"


def task():
    freeze=read_json(SELECTED)
    references=read_json(DENTAL)
    qa=read_json(CHECK)
    if qa.get("manifest_sha256")!=digest(DENTAL):
        raise ValueError("The dental annotation QA differs from the challenge set.")
    if any(digest(p)!=checksum for p,checksum in freeze["inputs"].items()):
        raise ValueError("Development evidence changed since selection.")
    return freeze,references["rows"]


def path(branch):
    return DATA/"proposals"/f"dental-audit-{branch}.json"


def runner(branch):
    if branch in ("dfine","dfine-multidomain"):
        from scripts.experiments.detector_training.detection_hillclimb_dfine import EDGE, RECORD, RUN, pixels, predict_batches
        from scripts.experiments.detector_training.detection_hillclimb_mix import FOLDER, LOG
        from transformers import DFineForObjectDetection

        directory,proof=(RUN,RECORD) if branch=="dfine" else (FOLDER,LOG)
        lock=read_json(proof)
        if any(digest(p)!=h for p,h in lock["files"].items()):
            raise ValueError("The local D-FINE branch differs from its checkpoint.")
        model=DFineForObjectDetection.from_pretrained(directory/"best",local_files_only=True)
        def dfine(image):
            windows=list(tiles(image))
            views=[("full",pixels(image,EDGE))]+[(str(i),pixels(image.crop(window),EDGE))
                                                 for i,window in enumerate(windows)]
            predicted=predict_batches(model,views)
            tile=[{**p,"view":i,"bbox":b} for i,window in enumerate(windows)
                  for p in predicted[str(i)] if p["score"]>=.008 and (b:=project(p["bbox"],window,image.size))]
            return dict(full=predicted["full"],windows=tile)
        return dfine,None,proof
    if branch=="baseline":
        from medsyringe.runtime.grounding_backend import enable_intel
        from medsyringe.analysis.production import ProductionEngine

        engine=ProductionEngine(Path("."))
        enable_intel(engine,Path("."))
        if engine.identity.get("primary_backend")!="xpu":
            engine.close()
            raise ValueError("The dental challenge uses the verified Intel grounder.")
        def baseline(image):
            ranked,floor,_=engine.expanded.detect(image)
            return dict(ranked=ranked,floor=floor)
        return baseline,engine.close,EXISTING
    from rfdetr import RFDETRSegNano
    from scripts.experiments.transfer.rfdetr_compare import ranked_region

    model_path,edge=trainer(branch)
    model=RFDETRSegNano.from_checkpoint(str(model_path),device="cpu")
    def rf(image):
        full=suppress(list(ranked_region(model,image,edge)),.45)
        mapped=[]
        for i,window in enumerate(tiles(image)):
            for p in ranked_region(model,image.crop(window),edge):
                if p["score"]>=.008 and (b:=project(p["bbox"],window,image.size)):
                    mapped.append({**p,"bbox":b,"view":i})
        return dict(full=full,windows=mapped)
    return rf,None,model_path


def collect(branch):
    from scripts.experiments.expanded.sam3_study import configure_cpu

    configure_cpu()
    _,rows=task()
    destination=path(branch)
    current=read_json(destination) if destination.exists() else None
    if current and len(current["frames"])==len(rows):
        return
    predictor,close,proof=runner(branch)
    key=dict(selection_sha256=digest(SELECTED),dental_sha256=digest(DENTAL),
             weights_sha256=digest(proof),branch=branch,code_sha256=digest(__file__))
    payload=current or dict(key=key,frames={})
    if payload["key"]!=key:
        raise ValueError("The dental benchmark collection inputs changed.")
    frames=payload["frames"]
    try:
        for row in rows:
            identifier=row["id"]
            if identifier in frames:
                continue
            started=perf_counter()
            frames[identifier]=predictor(source(row))
            frames[identifier]["ms"]=(perf_counter()-started)*1000
            if len(frames)%4==0 or len(frames)==len(rows):
                write_json(destination,payload)
                print("Dental",branch,len(frames),"/",len(rows),flush=True)
    finally:
        if close:
            close()


def load(branch,rows):
    current=read_json(path(branch))
    if (current["key"]["selection_sha256"]!=digest(SELECTED) or
        current["key"]["dental_sha256"]!=digest(DENTAL) or
        set(current["frames"])!={r["id"] for r in rows}):
        raise ValueError("Dental predictions are missing or have stale provenance.")
    return current["frames"]


def scores(rows,baseline,option):
    model=option["model"]
    if model in MODEL_COMBINATIONS:
        a,b=(load(name,rows) for name in MODEL_COMBINATIONS[model])
        return {key:agree(proposals(v,option["view"]),proposals(b[key],option["view"])) for key,v in a.items()}
    return {key:proposals(value,option["view"]) for key,value in load(model,rows).items()}


def evaluate_dental():
    freeze,rows=task()
    base=load("baseline",rows)
    selected=freeze["shadow"]["policy"]
    model=scores(rows,base,selected)
    before={key:v["ranked"] for key,v in base.items()}
    after={key:augment(v,model[key],selected) for key,v in base.items()}
    floor=read_json(FLOOR)["cutoff"]
    a,b=(evaluate(rows,p,floor) for p in (before,after))
    check=compare(a,b)
    review=None
    if freeze.get("review_selected"):
        option=freeze["review_selected"]["policy"]
        review=review_reach(rows,base,scores(rows,base,option),option)
    write_json(RESULT,dict(baseline=short(a),candidate=short(b),guard=check,
                           errors=dict(before=a["false_negatives"],after=b["false_negatives"]),
                           review=review,passed=check["passed"] and b["fp"]<=a["fp"],
                           selection_sha256=digest(SELECTED),manifest_sha256=digest(DENTAL),
                           scope="One source with repeated specimens. No independent frame-level estimator."))
    print("Dental audit",short(a),short(b),check,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("collect","evaluate"))
    parser.add_argument("--backend",choices=("baseline","original-rf","adapted-rf","dfine","dfine-multidomain"))
    args=parser.parse_args()
    if args.action=="evaluate":
        evaluate_dental()
    elif args.backend:
        collect(args.backend)
    else:
        parser.error("Collection needs --backend.")
