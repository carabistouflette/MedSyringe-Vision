"""Measure scene false alarms on photographer-disjoint, verified lookalikes."""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

from medsyringe.data import digest, read_json, write_json
from scripts.detection_appearance_extension import SELECTED, proof
from scripts.detection_crop_semantics import VisualSemantics
from scripts.detection_dental_evaluate import runner
from scripts.detection_hillclimb_corpus import ART, DATA
from scripts.detection_lookalike_audit_data import MANIFEST
from scripts.detection_semantics_audit import apply_rules, crop_frame
from scripts.precision_study import source
from scripts.sam3_study import configure_cpu

BASE = DATA/"proposals"/"lookalike-only-baseline.json"
RESULT = ART/"external_lookalike_audit.json"


def binding():
    return dict(manifest_sha256=digest(MANIFEST),variant_sha256=digest(SELECTED),script_sha256=digest(__file__))


def collect():
    configure_cpu()
    proof()
    reference = read_json(MANIFEST)["rows"]
    cfg = binding()
    current = read_json(BASE) if BASE.exists() else dict(key=cfg,frames={})
    if current["key"]!=cfg:
        raise ValueError("The lookalike baseline belongs to a different frozen policy.")
    predict = close = None
    try:
        for row in reference:
            ident = row["id"]
            if ident in current["frames"]:
                continue
            if predict is None:
                predict,close,_ = runner("baseline")
            start = perf_counter()
            current["frames"][ident] = {**predict(source(row)),"ms":(perf_counter()-start)*1000}
            write_json(BASE,current)
            print("Lookalike baseline",len(current["frames"]),"/",len(reference),flush=True)
    finally:
        if close:
            close()


def evaluate():
    if RESULT.exists():
        raise ValueError("The independent specificity results are already recorded.")
    configure_cpu()
    state = proof()
    cache = read_json(BASE)
    rows = read_json(MANIFEST)["rows"]
    if cache["key"]!=binding() or set(cache["frames"])!={r["id"] for r in rows} or any(r["objects"] for r in rows):
        raise ValueError("A scored negative audit must contain the same verified empty targets.")
    options = {"baseline":None,"earlier":state["previous"]}
    if state["candidate"]:
        options["extension"]=state["candidate"]
    encoder = VisualSemantics()
    outcomes = {name:{} for name in options}
    details = []
    for row in rows:
        ident = row["id"]
        frame = cache["frames"][ident]
        regions = crop_frame(encoder,source(row),frame)
        candidates = {}
        for name,steps in options.items():
            ranked = apply_rules(frame,regions,steps)[0] if steps else frame["ranked"]
            detections = [p for p in ranked if p["score"]>=frame["floor"]]
            outcomes[name][ident] = len(detections)
            candidates[name] = detections
        details.append(dict(id=ident,lookalike_type=row["lookalike_type"],predictions=candidates))
    metrics = {name:dict(images=len(rows),groups=len({r["group"] for r in rows}),boxes=sum(out.values()),
                         false_alarm_scenes=sum(value>0 for value in out.values()),
                         specificity=sum(value==0 for value in out.values())/len(rows)) for name,out in outcomes.items()}
    gained = {name:sorted(key for key,v in outcomes[name].items() if v>0 and not outcomes["baseline"][key])
              for name in options if name!="baseline"}
    write_json(RESULT,dict(inputs=binding(),summary=metrics,new_false_alarm_scenes=gained,
                            cases=details,categories=dict(Counter(r["lookalike_type"] for r in rows)),
                            scope="Negative-only cohort. Reports image specificity and number of false boxes."))
    print("Lookalike audit",metrics,"new false alarms",gained,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("collect","evaluate"))
    (collect if parser.parse_args().action=="collect" else evaluate)()
