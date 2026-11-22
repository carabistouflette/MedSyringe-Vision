"""Evaluate the two pre-existing frozen policies on a separate test-only medical source."""
import argparse
from pathlib import Path
from time import perf_counter

from medsyringe.data import digest, read_json, write_json
from scripts.experiments.detector_training.detection_consensus_audit import MODELS, open_predictor
from scripts.experiments.detector_audits.detection_crop_semantics import VisualSemantics
from scripts.experiments.transfer.detection_dental_evaluate import runner
from scripts.experiments.detector_audits.detection_empty_scene_audit import LOCK as RECOVERY, is_empty, metrics
from scripts.experiments.detector_audits.detection_empty_scene_screened import compare_set
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, EXISTING, FLOOR
from scripts.experiments.detector_audits.detection_medbin_data import MANIFEST, REVIEW
from scripts.experiments.detector_audits.detection_precision_confirmation import LOCK as PRECISION
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

LOCK = ART/"medbin_held_out_freeze.json"
RESULT = ART/"medbin_detector_audit.json"
BACKENDS = ("baseline",*MODELS)


def path(name):
    return DATA/"proposals"/f"medbin-held-out-{name}.json"


def freeze():
    if LOCK.exists() or any(path(b).exists() for b in BACKENDS):
        raise ValueError("The medical-source comparison has already begun.")
    sources = (Path("scripts/experiments/detector_audits/detection_medbin_audit.py"),MANIFEST,REVIEW,RECOVERY,PRECISION,EXISTING,FLOOR)
    write_json(LOCK,dict(inputs={str(p):digest(p) for p in sources},
                         policy="Compare the released appearance gate and the unchanged restricted recovery. "
                                "A model gain requires at least one new true match and no lost matches or added false "
                                "positives. The source is a single test-only domain."))


def frozen():
    seal=read_json(LOCK)
    if any(digest(p)!=h for p,h in seal["inputs"].items()):
        raise ValueError("An audit model setting or reference has changed.")
    return read_json(MANIFEST)["rows"]


def bound(name):
    dest = path(name)
    exact = digest(LOCK)
    loaded = read_json(dest) if dest.exists() else dict(freeze_sha256=exact,frames={})
    if loaded["freeze_sha256"]!=exact:
        raise ValueError("The saved model outputs differ from the selected dataset.")
    return loaded


def collect(model):
    configure_cpu()
    rows = frozen()
    old = bound("baseline")["frames"]
    if model!="baseline":
        if set(old)!={r["id"] for r in rows}:
            raise ValueError("Complete the baseline run before restricting the secondary passes.")
        rows=[r for r in rows if is_empty(old[r["id"]]["result"])]
    stored=bound(model)
    todo=[r for r in rows if r["id"] not in stored["frames"]]
    if not todo:
        if not path(model).exists():
            write_json(path(model),stored)
        return
    if model=="baseline":
        predict,close,_=runner(model)
    else:
        predict,close=open_predictor(model),None
    try:
        for row in todo:
            im=source(row)
            begin=perf_counter()
            result=predict(im)
            stored["frames"][row["id"]]=dict(result=result,seconds=perf_counter()-begin)
            write_json(path(model),stored)
            print("MedBin",model,len(stored["frames"]),"/",len(rows),flush=True)
    finally:
        if close is not None:
            close()


def measure():
    if RESULT.exists():
        raise ValueError("The test-only cohort has already been scored.")
    configure_cpu()
    refs=frozen()
    policy=read_json(RECOVERY)
    data={m:{k:x["result"] for k,x in bound(m)["frames"].items()} for m in BACKENDS}
    base=data.pop("baseline")
    empty={k for k,v in base.items() if is_empty(v)}
    if set(base)!={r["id"] for r in refs} or any(set(extra)!=empty for extra in data.values()):
        raise ValueError("Model caches do not cover all applicable sealed rows.")
    control,proposed,elapsed=compare_set(policy,refs,base,data,VisualSemantics(),None)
    cutoff=read_json(FLOOR)["cutoff"]
    previous={k:v["ranked"] for k,v in base.items()}
    first=metrics(refs,previous,control,cutoff)
    result=metrics(refs,control,proposed,cutoff)
    qualifies=bool(result["guard"]["passed"] and result["guard"]["recovered_matches"] and
                   result["candidate"]["fp"]<=result["previous"]["fp"] and not result["new_false_alarm_scenes"])
    write_json(RESULT,dict(passed=qualifies,freeze_sha256=digest(LOCK),scope="One acquisition source with "
                           "unresolved same-specimen dependence. No cross-population precision inference.",
                           precision_gate=first,recovery=result,crop_study=elapsed))
    print("MedBin test",qualifies,first["previous"],first["candidate"],result["candidate"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","collect","measure"))
    parser.add_argument("--backend",choices=BACKENDS)
    args=parser.parse_args()
    if args.action=="collect":
        if not args.backend:
            parser.error("Model collection requires --backend.")
        collect(args.backend)
    else:
        (freeze if args.action=="freeze" else measure)()
