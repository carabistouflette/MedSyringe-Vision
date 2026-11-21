"""Compare the text-only worker with the qualified operating decisions and encoder."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.appearance_runtime import RELEASE
from medsyringe.data import digest, read_json, write_json
from scripts.appearance_text_encoder import TextMarginEncoder
from scripts.appearance_worker import explain
from scripts.detection_crop_semantics import VisualSemantics
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.detection_precision_confirmation import CAPTURE, COHORTS, cohort
from scripts.detection_precision_negative_audit import CACHE, MANIFEST
from scripts.detection_semantics_audit import apply_rules, crop_frame
from scripts.expanded_proposals import image_for
from scripts.precision_study import source
from scripts.sam3_study import configure_cpu

OUTPUT = ART/"appearance_runtime_parity.json"
SCOPES = (*COHORTS, "independent-negatives")


def old_records(scope):
    if scope == "independent-negatives":
        saved = read_json(CACHE)["frames"]
        before = {key:dict(ranked=p["before"],floor=p["floor"]) for key,p in saved.items()}
        expected = {key:p["after"] for key,p in saved.items()}
        return read_json(MANIFEST)["rows"],before,expected
    rows,before = cohort(scope)
    expected = read_json(DATA/"proposals"/f"fixed-precision-{scope}.json")["frames"]
    return rows,before,{key:p["ranked"] for key,p in expected.items()}


def pairs(scopes, unchanged):
    for scope in scopes:
        refs,before,expected = old_records(scope)
        for row in refs:
            key = row["id"]
            camera = scope in CAPTURE
            picture = image_for(row,unchanged) if camera else source(row)
            yield scope,key,picture,before[key],expected[key]


def classify(picture, initial, model, policy):
    cutoff = initial["floor"]
    retained = [p for p in initial["ranked"] if p["score"]>=cutoff]
    if not retained:
        return initial["ranked"], None
    block = crop_frame(model,picture,dict(ranked=retained,floor=cutoff))
    ranked,margins = apply_rules(initial,block,policy)
    return ranked,margins


def verify(source_dir):
    if OUTPUT.exists():
        raise ValueError("This fixed runtime parity run already exists.")
    configure_cpu()
    decision = read_json(RELEASE)
    policy = decision["components"]
    if decision["mode"]!="veto-only" or {p["name"] for p in policy}!={"semantic-best"}:
        raise ValueError("The qualification has a different precision gate.")
    tick = perf_counter()
    model = TextMarginEncoder()
    fast_load = perf_counter()-tick
    covers = {scope:dict(images=0,encoded=0,changed=[]) for scope in SCOPES}
    reference = None
    timing = {"previous":[],"text_only":[]}
    trial = 0
    exact_margins = True
    for scope,key,picture,before,expected in pairs(SCOPES,source_dir):
        start = perf_counter()
        outputs,margins = classify(picture,before,model,policy)
        elapsed = perf_counter()-start
        measured = covers[scope]
        measured["images"] += 1
        measured["encoded"] += margins is not None
        if explain(before,outputs)!=explain(before,expected):
            measured["changed"].append(key)
        if margins is not None and scope not in CAPTURE and trial<20:
            if reference is None:
                tick = perf_counter()
                reference = VisualSemantics()
                slow_load = perf_counter()-tick
                classify(picture,before,reference,policy)
            tick = perf_counter()
            _,reference_margins = classify(picture,before,reference,policy)
            timing["previous"].append(perf_counter()-tick)
            timing["text_only"].append(elapsed)
            exact_margins = exact_margins and margins == reference_margins
            trial += 1
        if measured["images"]%20==0 or not measured["images"]%3 and scope not in CAPTURE:
            print("Appearance parity",scope,measured["images"],len(measured["changed"]),flush=True)
    passed = not any(r["changed"] for r in covers.values()) and exact_margins and trial==20
    write_json(OUTPUT,dict(passed=passed,scope=covers,exact_margins=exact_margins,timed_parity_images=trial,
                           seconds=dict(previous_load=slow_load,text_only_load=fast_load,
                                        previous_median=median(timing["previous"]),
                                        text_only_median=median(timing["text_only"])),
                           code_sha256=digest("scripts/appearance_text_encoder.py"),previous_release_sha256=digest(RELEASE)))
    print("Exact appearance parity",passed,covers,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    verify(parser.parse_args().source)
