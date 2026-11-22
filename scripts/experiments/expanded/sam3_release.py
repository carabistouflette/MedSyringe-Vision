"""Verify the isolated SAM 3 runtime against the calibrated fallback."""
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.detection.additive import additions
from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.runtime.local_models import ImageWorker
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_training.optimize_detector import FINAL, counters
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_evaluate import ART, CONTROLS, FROZEN, get_prepared, hashes

RELEASE=ART/"sam3_release.json"
RUNNER=ART/"sam3_runtime.json"
EXECUTABLE="data/sam3/xpu-env/bin/python"


def main():
    selection=read_json(FROZEN)
    if selection["inputs"]!=hashes() or not selection["selected"]:
        raise ValueError("A frozen calibration is required.")
    options=selection["selected"]["options"]
    if options["view"]!="full_and_tiles" or options["support"] is not None or selection["device"]!="xpu":
        raise ValueError("The tested runtime requires the verified full-image and tile policy.")
    prepared={}
    for name in CONTROLS:
        audit=read_json(ART/f"sam3_audit_{name}.json")
        if (audit["frozen_sha256"]!=digest(FROZEN) or not audit["guard"]["passed"]
                or audit["candidate"]["fp"]>audit["baseline"]["fp"]):
            raise ValueError(f"SAM 3 regression in {name}.")
        prepared[name]=audit
    if prepared["fil_rouge"]["candidate"]["fn"] or prepared["fil_rouge"]["guard"]["recovered_matches"]<1:
        raise ValueError("The verified fallback must recover the missed Fil rouge images.")
    config=dict(edge=selection["edge"],device=selection["device"],cutoff=options["aux_cutoff"])
    worker=ImageWorker(Path("."),Path(EXECUTABLE),"sam3",config)
    checks={}
    floor=read_json(FINAL)["cutoff"]
    for name,audit in prepared.items():
        rows,base=get_prepared(name)
        output=dict(base)
        timings=[]
        for row in rows:
            identifier=row["id"]
            if any(p["score"]>=floor for p in base[identifier]):
                continue
            started=perf_counter()
            candidates=worker.run(source(row))
            output[identifier]=additions(base[identifier],candidates,floor,support=None,
                                         aux_cutoff=config["cutoff"],iou=.30)
            elapsed=perf_counter()-started
            timings.append(elapsed)
            print(name,identifier,round(elapsed,2),"s",flush=True)
        actual=evaluate(rows,output,floor)
        guard=compare(audit["candidate"],actual)
        if not guard["passed"] or any(actual[k]!=audit["candidate"][k] for k in ("tp","fp","fn")):
            raise ValueError(f"Local worker parity failed in {name}.")
        checks[name]=dict(metrics=counters(actual),guard=guard,images=len(rows),
                          fallback_uses=len(timings),median_fallback_seconds=median(timings) if timings else None)
    write_json(RUNNER,dict(passed=True,selection_sha256=digest(FROZEN),cohorts=checks))
    paths=("scripts/experiments/expanded/sam3_study.py","scripts/workers/local_model_worker.py","medsyringe/runtime/local_models.py",
           "medsyringe/detection/additive.py")
    valid=[p for p in paths if Path(p).exists()]
    write_json(RELEASE,dict(id="sam3-local-fallback",executable=EXECUTABLE,options=config,
                           evidence=str(RUNNER),evidence_sha256=digest(RUNNER),
                           code={p:digest(p) for p in valid}))
    print("Activated",RELEASE,flush=True)


if __name__=="__main__":
    main()
