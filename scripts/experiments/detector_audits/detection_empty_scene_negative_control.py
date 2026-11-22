"""Verify restricted recovery against the deployed precision gate's lookalike audit."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, write_json
from scripts.experiments.detector_training.detection_consensus_audit import open_predictor
from scripts.experiments.detector_audits.detection_crop_semantics import VisualSemantics
from scripts.experiments.detector_audits.detection_empty_scene_audit import LOCK, MODELS, is_empty, metrics, run_both
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, FLOOR
from scripts.experiments.detector_audits.detection_precision_negative_audit import CACHE, FREEZE, MANIFEST
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

SEAL = ART / "empty_scene_negative_control_freeze.json"
RESULT = ART / "empty_scene_negative_control.json"


def freeze():
    if SEAL.exists() or RESULT.exists() or any(path(m).exists() for m in MODELS):
        raise ValueError("A control run already exists.")
    inputs = (Path("scripts/experiments/detector_audits/detection_empty_scene_negative_control.py"),LOCK,FREEZE,MANIFEST,CACHE)
    write_json(SEAL,dict(inputs={str(p):digest(p) for p in inputs},
                         qualification="Every scene correctly empty under the deployed precision policy must remain empty."))


def path(model):
    return DATA/"proposals"/f"empty-scene-negative-{model}.json"


def setup():
    configure_cpu()
    seal = read_json(SEAL)
    if any(digest(p)!=value for p,value in seal["inputs"].items()):
        raise ValueError("The control inputs were changed.")
    rows = read_json(MANIFEST)["rows"]
    prior = read_json(CACHE)["frames"]
    base = {k:dict(ranked=v["before"],floor=v["floor"]) for k,v in prior.items()}
    filtered = {k:dict(ranked=v["after"]) for k,v in prior.items()}
    return rows,base,filtered


def collect(model):
    rows,base,_ = setup()
    records = read_json(path(model)) if path(model).exists() else dict(seal_sha256=digest(SEAL),frames={})
    if records["seal_sha256"]!=digest(SEAL):
        raise ValueError("Old proposals differ from this control's seal.")
    missing = [r for r in rows if is_empty(base[r["id"]]) and r["id"] not in records["frames"]]
    if not missing:
        return
    detector = open_predictor(model)
    for row in missing:
        records["frames"][row["id"]] = detector(source(row))
        write_json(path(model),records)
        print("Negative recovery regression",model,len(records["frames"]),flush=True)


def measure():
    if RESULT.exists():
        raise ValueError("This control was already measured.")
    rows,base,filtered = setup()
    opinions = {model:read_json(path(model)) for model in MODELS}
    expected = {k for k,v in base.items() if is_empty(v)}
    if any(o["seal_sha256"]!=digest(SEAL) or set(o["frames"])!=expected for o in opinions.values()):
        raise ValueError("Complete both independent detector passes before scoring.")
    old,new,_ = run_both(read_json(LOCK),rows,base,{m:v["frames"] for m,v in opinions.items()},
                         VisualSemantics(),controls=filtered)
    result = metrics(rows,old,new,read_json(FLOOR)["cutoff"])
    passed = not result["new_false_alarm_scenes"] and result["previous"]["fp"]>=result["candidate"]["fp"]
    write_json(RESULT,dict(passed=passed,freeze_sha256=digest(SEAL),outcome=result))
    print("Restricted negative recovery regression",passed,result["new_false_alarm_scenes"],flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","collect","measure"))
    parser.add_argument("--backend",choices=MODELS)
    args = parser.parse_args()
    if args.action == "collect":
        if not args.backend:
            parser.error("Collection requires --backend.")
        collect(args.backend)
    else:
        (freeze if args.action=="freeze" else measure)()
