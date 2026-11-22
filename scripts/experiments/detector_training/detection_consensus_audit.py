"""Score the frozen crop-consensus policy on fresh, untouched public photographs."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.detection.additive import project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import suppress
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_training.detection_consensus import LOCK, MODELS, classify, features
from scripts.experiments.detector_training.detection_consensus_corpus import MANIFEST, REVIEW
from scripts.experiments.detector_audits.detection_crop_semantics import VisualSemantics
from scripts.experiments.transfer.detection_dental_evaluate import runner
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, EXISTING, FLOOR
from scripts.experiments.detector_training.detection_hillclimb_rank import grouped_difference, slices
from scripts.experiments.detector_audits.detection_open_models import Detector, model_lock
from scripts.experiments.detector_audits.detection_semantics_audit import crop_frame
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

FREEZE = ART/"consensus_audit_freeze.json"
RESULT = ART/"consensus_audit_results.json"
BACKENDS = ("baseline",*MODELS)


def cache(backend):
    return DATA/"proposals"/f"consensus-audit-{backend}.json"


def freeze():
    if FREEZE.exists() or RESULT.exists() or any(cache(backend).exists() for backend in BACKENDS):
        raise ValueError("The consensus audit was already frozen or has already started.")
    state = read_json(LOCK)
    policy = state["candidate"]
    if not policy or any(digest(path)!=wanted for path,wanted in state["inputs"].items()):
        raise ValueError("The selected development variant is missing or obsolete.")
    paths = (LOCK,MANIFEST,REVIEW,Path(__file__),Path("scripts/experiments/detector_training/detection_consensus.py"),EXISTING,FLOOR)
    write_json(FREEZE,dict(selection=policy,model_heads=state["heads"],inputs={str(p):digest(p) for p in paths},
                          qualification="All previously matched instances and false-positive counts are preserved, "
                                        "with a meaningful improvement on the untouched audit."))


def proof():
    locked = read_json(FREEZE)
    if any(digest(path)!=expected for path,expected in locked["inputs"].items()):
        raise ValueError("Consensus selection or audited dataset has changed.")
    return locked


def rows():
    return [row for row in read_json(MANIFEST)["rows"] if row["split"]=="consensus-audit"]


def open_predictor(backend):
    detector = Detector(backend,"xpu")
    def predict(image):
        first = detector(image)
        side = min(image.size)
        x,y = round((image.width-side)*.5),round((image.height-side)*.5)
        if not x and not y:
            return suppress(first,.43)
        region = x,y,x+side,y+side
        another = detector(image.crop(region))
        focused = [{**p,"bbox":box} for p in another if (box:=project(p["bbox"],region,image.size)) is not None]
        return suppress([*first,*focused],.43)
    return predict


def collect(backend):
    configure_cpu()
    proof()
    data = rows()
    identity = dict(selection_sha256=digest(FREEZE), backend=backend,
                    images={r["id"]:r["sha256"] for r in data},
                    checkpoint=digest(EXISTING) if backend=="baseline" else model_lock(backend))
    destination = cache(backend)
    artifact = read_json(destination) if destination.exists() else dict(key=identity,frames={})
    if artifact["key"]!=identity:
        raise ValueError("Cached predictions are from another detector or partition.")
    image_predictor = close = None
    try:
        for row in data:
            key = row["id"]
            if key in artifact["frames"]:
                continue
            if not image_predictor:
                if backend=="baseline":
                    image_predictor,close,_ = runner("baseline")
                else:
                    image_predictor = open_predictor(backend)
            image = source(row)
            start = perf_counter()
            result = image_predictor(image)
            artifact["frames"][key] = dict(output=result,ms=(perf_counter()-start)*1000)
            done = len(artifact["frames"])
            if done%3==0 or done==len(data):
                write_json(destination,artifact)
                print("Fresh consensus audit",backend,done,"/",len(data),flush=True)
    finally:
        if close:
            close()


def measure():
    if RESULT.exists():
        raise ValueError("This untouched audit has already been measured.")
    configure_cpu()
    state = proof()
    data = rows()
    floor = read_json(FLOOR)["cutoff"]
    stored = {backend:read_json(cache(backend)) for backend in BACKENDS}
    ids = {r["id"] for r in data}
    if any(item["key"]["selection_sha256"]!=digest(FREEZE) or set(item["frames"])!=ids for item in stored.values()):
        raise ValueError("One or more detector caches are incomplete or belong to another frozen audit.")
    measurements = {backend:{key:item["output"] for key,item in evidence["frames"].items()}
                    for backend,evidence in stored.items()}
    baseline = measurements["baseline"]
    after = {}
    elapsed = {}
    crop = VisualSemantics()
    for row in data:
        key = row["id"]
        image = source(row)
        start = perf_counter()
        frame = crop_frame(crop,image,baseline[key])
        heads = state["model_heads"]
        local = features(frame,"semantic-best",heads["joint"],heads["shape"],{
            name:measurements[name][key] for name in MODELS})
        after[key] = classify(baseline[key],local,state["selection"]["settings"])
        elapsed[key]=(perf_counter()-start)*1000
    before={key:item["ranked"] for key,item in baseline.items()}
    prev,next_ = (evaluate(data,pred,floor) for pred in (before,after))
    guard = compare(prev,next_)
    passed = guard["passed"] and next_["fp"]<=prev["fp"] and (guard["recovered_matches"] or next_["fp"]<prev["fp"])
    write_json(RESULT,dict(passed=bool(passed),scope="Fresh positive and contextual negative scenes. Previously scored "
                           "lookalikes are separate regression material.",selection_sha256=digest(FREEZE),
                           groups=dict(Counter(r["group"] for r in data)),release=short(prev),candidate=short(next_),guard=guard,
                           errors=dict(before=dict(fp=prev["false_positives"],fn=prev["false_negatives"]),
                                       after=dict(fp=next_["false_positives"],fn=next_["false_negatives"])),
                           paired_intervals=grouped_difference(data,before,after),
                           slices=dict(release=slices(data,prev,before,floor),candidate=slices(data,next_,after,floor)),
                           median_ms={**{name:median(v["ms"] for v in evidence["frames"].values())
                                         for name,evidence in stored.items()},"crop_features":median(elapsed.values())}))
    print("Fresh consensus audit",passed,short(prev),short(next_),guard,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","collect","measure"))
    parser.add_argument("--backend",choices=BACKENDS)
    args = parser.parse_args()
    if args.action=="collect":
        if not args.backend:
            parser.error("Collection requires --backend.")
        collect(args.backend)
    else:
        (freeze if args.action=="freeze" else measure)()
