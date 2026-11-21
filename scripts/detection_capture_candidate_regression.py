"""Replay the frozen candidate on every expanded Fil Rouge reference."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.detection_crop_semantics import VisualSemantics
from scripts.detection_hillclimb_corpus import ART, DATA
from scripts.detection_hillclimb_experiment import FLOOR
from scripts.detection_semantics_audit import LOCK, apply_rules, crop_frame, evidence
from scripts.expanded_detection_study import load, short
from scripts.expanded_proposals import image_for
from scripts.sam3_study import configure_cpu

COHORTS = ("fit","calibration","test")
RESULT = ART/"crop_semantics_capture_regression.json"
PRIOR = Path("artifacts/expanded/crop_verification_full_capture.json")
REVIEW = ART/"fil_rouge_box_review.json"
CACHE = DATA/"proposals"/"frozen-crop-rule-expanded-capture.json"


def replay(source):
    configure_cpu()
    proof = evidence()
    original = read_json(PRIOR)
    if not original["passed"] or any(original["results"][k]["removed"] for k in COHORTS):
        raise ValueError("The cached capture proposals differ from the deployed verifier.")
    floor = read_json(FLOOR)["cutoff"]
    reviewed_cases = {r["id"]:r["alternative"] for r in read_json(REVIEW)["entries"]}
    datasets = {}
    for scope in COHORTS:
        rows,frames=load(scope)
        rows = [r for r in rows if "file" in r]
        datasets[scope] = (rows,{r["id"]:frames[r["id"]]["prior"] for r in rows})
    key = dict(selection=digest(LOCK),baseline_capture=digest(PRIOR),review=digest(REVIEW),source=digest(__file__),
               rows={r["id"]:r["sha256"] for rows,_ in datasets.values() for r in rows})
    saved = read_json(CACHE) if CACHE.exists() else dict(key=key,frames={})
    if saved["key"]!=key:
        raise ValueError("The capture regression was started with different inputs.")
    encoder = None
    cases = saved["frames"]
    for scope,(rows,baseline) in datasets.items():
        for row in rows:
            name = row["id"]
            if name in cases:
                continue
            if encoder is None:
                encoder = VisualSemantics()
            initial = dict(ranked=baseline[name],floor=floor)
            started = perf_counter()
            regions = crop_frame(encoder,image_for(row,source),initial)
            next_result,_ = apply_rules(initial,regions,proof["components"])
            cases[name] = dict(ranked=next_result,ms=(perf_counter()-started)*1000)
            if len(cases)%6==0 or len(cases)==len(key["rows"]):
                write_json(CACHE,saved)
                print(scope,len(cases),"/",len(key["rows"]),flush=True)
    outputs = {}
    for scope,(rows,baseline) in datasets.items():
        next_result = {r["id"]:cases[r["id"]]["ranked"] for r in rows}
        initial,after = (evaluate(rows,version,floor) for version in (baseline,next_result))
        amended = [{**r,"objects":[{**r["objects"][0],"bbox":reviewed_cases[r["id"]]}]}
                   if r["id"] in reviewed_cases else r for r in rows]
        review_base,review_after = (evaluate(amended,version,floor) for version in (baseline,next_result))
        outputs[scope] = dict(previous=short(initial),candidate=short(after),guard=compare(initial,after),
                              revised_reference_sensitivity=dict(previous=short(review_base),candidate=short(review_after),
                                                                 guard=compare(review_base,review_after)),
                              added_median_seconds=median(cases[r["id"]]["ms"] for r in rows)/1000)
    passed = all(r["guard"]["passed"] and r["candidate"]["exact_count"]==r["previous"]["exact_count"] and
                 r["candidate"]["fp"]<=r["previous"]["fp"] and r["revised_reference_sensitivity"]["guard"]["passed"] and
                 r["revised_reference_sensitivity"]["candidate"]["fp"]<=r["revised_reference_sensitivity"]["previous"]["fp"]
                 for r in outputs.values())
    write_json(RESULT,dict(passed=passed,outcomes=outputs,selection_sha256=digest(LOCK),cache_sha256=digest(CACHE),
                           scope="Same-session complete capture replay. Both original boxes and the separately reviewed-sensitive boxes."))
    print("Capture release gate",passed,outputs,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    replay(parser.parse_args().source)
