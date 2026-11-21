"""Independently qualify a fixed precision-only verifier, with no rescue proposals."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.detection_consensus_audit import FREEZE, cache as consensus_cache, rows as consensus_rows
from scripts.detection_crop_semantics import VisualSemantics, feature_cache, read_features
from scripts.detection_appearance_extension import BASELINE as OPENIMAGES, scope as openimages_rows
from scripts.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, EXISTING, FLOOR, inputs, loaded
from scripts.detection_hillclimb_rank import grouped_difference, minimal
from scripts.detection_lookalike_audit import BASE as LOOKALIKES
from scripts.detection_lookalike_audit_data import MANIFEST as LOOKALIKE_MANIFEST
from scripts.detection_semantics_audit import apply_rules, baseline as saved_baseline, crop_frame, rows_for
from scripts.expanded_detection_study import load, short
from scripts.expanded_proposals import image_for
from scripts.precision_study import source
from scripts.sam3_study import configure_cpu

LOCK = ART/"precision_only_confirmation_freeze.json"
RESULT = ART/"precision_only_confirmation_audit.json"
CAPTURE = ("fit","calibration","test")
COHORTS = ("fresh", "commons-regression", "previous-reserve", "openimages", "lookalikes", "dental", *CAPTURE)


def freeze():
    if LOCK.exists() or RESULT.exists() or (ART/"consensus_audit_results.json").exists():
        raise ValueError("A precision confirmation exists or the new public cohort was already scored.")
    components = [dict(name="semantic-best",variant="precision-only",head=None,bounds=[10000.,-.005,1.])]
    floor = read_json(FLOOR)["cutoff"]
    development = {}
    for scope in DEVELOPMENT:
        rows, baseline, features = inputs(scope),loaded(scope,"baseline"),read_features(scope)
        original = minimal(rows,{key:f["ranked"] for key,f in baseline.items()},floor)
        revised = minimal(rows,{key:apply_rules(f,features[key],components)[0] for key,f in baseline.items()},floor)
        development[scope] = dict(before={key:original[key] for key in ("tp","fp","fn")},
                                  after={key:revised[key] for key in ("tp","fp","fn")},guard=compare(original,revised))
    if not all(r["guard"]["passed"] and r["after"]["fp"]<=r["before"]["fp"] for r in development.values()):
        raise ValueError("The precision-only control did not preserve development instances.")
    paths = (Path(__file__),Path("scripts/detection_crop_semantics.py"),Path("scripts/detection_semantics_audit.py"),
             FREEZE,EXISTING,FLOOR,*(feature_cache(scope) for scope in DEVELOPMENT))
    write_json(LOCK,dict(components=components,development=development,inputs={str(p):digest(p) for p in paths},
                         proof="Fresh public cohort as the qualification set. Previously scored data are regression controls. "
                               "No rescued boxes. No lost matches or additional false positives anywhere."))
    print("Fixed precision-only confirmation",development,flush=True)


def verify():
    lock = read_json(LOCK)
    if any(digest(path)!=hash_ for path,hash_ in lock["inputs"].items()):
        raise ValueError("A frozen qualification input changed.")
    return lock


def cohort(name):
    if name=="fresh":
        frames = read_json(consensus_cache("baseline"))["frames"]
        rows = consensus_rows()
        before = {key:value["output"] for key,value in frames.items()}
        return rows,before
    if name=="openimages":
        return openimages_rows("audit"),read_json(OPENIMAGES)["frames"]
    if name=="lookalikes":
        return read_json(LOOKALIKE_MANIFEST)["rows"],read_json(LOOKALIKES)["frames"]
    if name in CAPTURE:
        rows,frames = load(name)
        floor = read_json(FLOOR)["cutoff"]
        rows = [r for r in rows if "file" in r]
        return rows,{r["id"]:dict(ranked=frames[r["id"]]["prior"],floor=floor) for r in rows}
    internal = "public-audit" if name=="commons-regression" else name
    return rows_for(internal),saved_baseline(internal)


def collect(name, unaltered=None):
    configure_cpu()
    policy = verify()
    refs,before = cohort(name)
    path = DATA/"proposals"/f"fixed-precision-{name}.json"
    key = dict(selection_sha256=digest(LOCK),roots={r["id"]:r["sha256"] for r in refs},
               before_ranks={r["id"]:str([(p["bbox"],p["score"]) for p in before[r["id"]]["ranked"]]) for r in refs})
    saved = read_json(path) if path.exists() else dict(key=key,frames={})
    if saved["key"]!=key:
        raise ValueError("The precision gate had different baseline predictions or image inputs.")
    predictor = None
    for row in refs:
        identifier = row["id"]
        if identifier in saved["frames"]:
            continue
        initial = before[identifier]
        accepted = [p for p in initial["ranked"] if p["score"]>=initial["floor"]]
        mark = perf_counter()
        if accepted:
            if predictor is None:
                predictor = VisualSemantics()
            picture = image_for(row,unaltered) if name in CAPTURE else source(row)
            patches = crop_frame(predictor,picture,dict(ranked=accepted,floor=initial["floor"]))
            ranked,_ = apply_rules(initial,patches,policy["components"])
        else:
            ranked = initial["ranked"]
        saved["frames"][identifier] = dict(ranked=ranked,ms=(perf_counter()-mark)*1000)
        done = len(saved["frames"])
        if done%8==0 or done==len(refs):
            write_json(path,saved)
            print("Fixed precision gate",name,done,"/",len(refs),flush=True)


def measure():
    if RESULT.exists():
        raise ValueError("The fixed precision-only qualification has already been measured.")
    verify()
    evidence = {}
    cutoff = read_json(FLOOR)["cutoff"]
    for name in COHORTS:
        data,first = cohort(name)
        path = DATA/"proposals"/f"fixed-precision-{name}.json"
        values = read_json(path)
        if values["key"]["selection_sha256"]!=digest(LOCK) or set(values["frames"])!=set(first):
            raise ValueError(f"The precision predictions are incomplete or stale for {name}.")
        before = {key:f["ranked"] for key,f in first.items()}
        after = {key:f["ranked"] for key,f in values["frames"].items()}
        previous,candidate = (evaluate(data,received,cutoff) for received in (before,after))
        check = compare(previous,candidate)
        evidence[name] = dict(previous=short(previous),candidate=short(candidate),guard=check,
                              median_extra_seconds=median(f["ms"] for f in values["frames"].values())/1000,
                              paired_groups=grouped_difference(data,before,after) if name=="fresh" else None,
                              errors=dict(previous=previous["false_positives"],candidate=candidate["false_positives"]))
    safe = all(r["guard"]["passed"] and r["candidate"]["fp"]<=r["previous"]["fp"] and
               (name not in CAPTURE or r["candidate"]["exact_count"]==r["previous"]["exact_count"])
               for name,r in evidence.items())
    fresh = evidence["fresh"]
    passed = safe and fresh["candidate"]["fp"]<fresh["previous"]["fp"]
    write_json(RESULT,dict(passed=passed,selection_sha256=digest(LOCK),outcomes=evidence))
    print("Precision-only qualification",passed,{name:((r["previous"]["tp"],r["previous"]["fp"]),
                                                        (r["candidate"]["tp"],r["candidate"]["fp"]))
                                                   for name,r in evidence.items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","collect","measure"))
    parser.add_argument("--scope",choices=COHORTS)
    parser.add_argument("--source",type=Path)
    args=parser.parse_args()
    if args.action=="collect":
        if not args.scope or (args.scope in CAPTURE and not args.source):
            parser.error("Collection requires --scope. Capture partitions also require --source.")
        collect(args.scope,args.source)
    else:
        (freeze if args.action=="freeze" else measure)()
