"""Score the frozen decision after rejecting geometrically unsupported candidates."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, write_json
from scripts.detection_consensus import MODELS, agree_with, bound, classify, features
from scripts.detection_consensus_audit import cache as earlier_cache
from scripts.detection_crop_semantics import PADDINGS, VisualSemantics, proposals
from scripts.detection_empty_scene_audit import AUDIT, LOCK, RESULT, binding, cache, historical, is_empty, metrics
from scripts.detection_empty_scene_negative_control import (
    RESULT as NEGATIVE, SEAL, path as negative_path, setup as negative_setup,
)
from scripts.detection_hillclimb_experiment import ART, DATA, FLOOR
from scripts.detection_hillclimb_rank import grouped_difference
from scripts.detection_precision_confirmation import CAPTURE, COHORTS, cohort
from scripts.detection_semantics_audit import apply_rules, crop_frame
from scripts.precision_study import source
from scripts.rfdetr_verifier import expanded

PROOF = ART/"empty_scene_screened_execution.json"


def eligible_frames(picture, original, opinions, settings, encoder):
    candidates = proposals(original)
    eligible = [p for p in candidates if p["score"] >= settings["prior"] and agree_with(
        {model:bound(p["bbox"], row) for model,row in opinions.items()},settings["opinions"])]
    vectors = dict(siglip=[],dino=[],prompt=[])
    patches = [crop(picture,expanded(p["bbox"],padding)) for padding in PADDINGS for p in eligible]
    for offset in range(0,len(patches),8):
        for name,value in encoder.encode(patches[offset:offset+8]).items():
            vectors[name].extend(value.tolist())
    return dict(candidates=eligible,values=vectors),len(candidates)


def compare_set(policy,rows,base,opinions,reader,old):
    baseline,result,latencies={},{},[]
    total=screened=0
    for row in rows:
        identifier=row["id"]
        original=base[identifier]
        start=perf_counter()
        if is_empty(original):
            picture=source(row)
            supporting={m:values[identifier] for m,values in opinions.items()}
            eligible,considered=eligible_frames(picture,original,supporting,policy["settings"],reader)
            total += considered
            screened += len(eligible["candidates"])
            info=features(eligible,"semantic-best",policy["heads"]["joint"],policy["heads"]["shape"],supporting)
            baseline[identifier]=original["ranked"]
            result[identifier]=classify(original,info,policy["settings"])
        elif old is not None:
            baseline[identifier]=result[identifier]=old[identifier]["ranked"]
        else:
            picture=source(row)
            likely=[p for p in original["ranked"] if p["score"]>=original["floor"]]
            accepted=crop_frame(reader,picture,dict(ranked=likely,floor=original["floor"]))
            baseline[identifier]=result[identifier]=apply_rules(original,accepted,policy["control"])[0]
        latencies.append(perf_counter()-start)
        if len(baseline)%5==0 or len(baseline)==len(rows):
            print("Screened crop scoring",len(baseline),"/",len(rows),total,screened,flush=True)
    return baseline,result,dict(seconds_median=median(latencies),possible_before=total,possible_after=screened)


def qualify():
    if PROOF.exists() or RESULT.exists() or NEGATIVE.exists():
        raise ValueError("An existing scoring attempt already recorded the frozen holdout.")
    negative_rows,negative_base,negative_only=negative_setup()
    policy,refs=binding()
    cutoff=read_json(FLOOR)["cutoff"]
    stored={name:cache(name)["frames"] for name in ("baseline",*MODELS)}
    base={k:v["ranked"] for k,v in stored["baseline"].items()}
    aux={m:{k:v["ranked"] for k,v in stored[m].items()} for m in MODELS}
    if set(base)!={r["id"] for r in refs} or any(set(a)!={k for k,v in base.items() if is_empty(v)} for a in aux.values()):
        raise ValueError("Complete the full original detector collection before scoring.")
    reader=VisualSemantics()
    before,proposed,fresh_time=compare_set(policy,refs,base,aux,reader,None)
    holdout=metrics(refs,before,proposed,cutoff)
    holdout["paired_intervals"]=grouped_difference(refs,before,proposed)
    regressions={name:historical(policy,name) for name in COHORTS if name!="fresh"}
    refs,base=cohort("fresh")
    previous_aux={m:{k:v["output"] for k,v in read_json(earlier_cache(m))["frames"].items()} for m in MODELS}
    only=read_json(DATA/"proposals/fixed-precision-fresh.json")["frames"]
    earlier,combined,previous_time=compare_set(policy,refs,base,previous_aux,reader,only)
    regressions["earlier-commons-audit"]=metrics(refs,earlier,combined,cutoff)
    constraints=all(r["guard"]["passed"] and not r["new_false_alarm_scenes"] and
                    r["candidate"]["fp"]<=r["previous"]["fp"] for r in (holdout,*regressions.values()))
    unchanged=all(regressions[g]["candidate"]["exact_count"]==regressions[g]["previous"]["exact_count"] for g in CAPTURE)
    passed=constraints and unchanged and holdout["guard"]["recovered_matches"]>0
    negatives={m:read_json(negative_path(m)) for m in MODELS}
    expected={k for k,v in negative_base.items() if is_empty(v)}
    if any(item["seal_sha256"]!=digest(SEAL) or set(item["frames"]) != expected for item in negatives.values()):
        raise ValueError("Complete both detector passes on the negative control.")
    old,new,negative_time=compare_set(policy,negative_rows,negative_base,
                                      {m:x["frames"] for m,x in negatives.items()},reader,negative_only)
    neg=metrics(negative_rows,old,new,cutoff)
    negatives_pass=not neg["new_false_alarm_scenes"] and neg["candidate"]["fp"]<=neg["previous"]["fp"]
    # The full batch scorers may still be running. Never overwrite sealed results.
    if RESULT.exists() or NEGATIVE.exists():
        raise ValueError("The original scorer finished while the audit was running.")
    write_json(PROOF,dict(selected_policy_sha256=digest(LOCK), code_sha256=digest(Path(__file__)),
                          unchanged=["operating thresholds","model decisions","reference boundaries","qualification criteria"],
                          execution="Apply each proposal's frozen raw-score and agreement requirements before encoding "
                                    "its crop. Unsupported proposals cannot be accepted under the fixed rule.",
                          timings=dict(fresh=fresh_time,previous=previous_time,negative=negative_time)))
    write_json(RESULT,dict(passed=passed,audit_sha256=digest(AUDIT),fresh=holdout,regression=regressions,
                           screened_execution_sha256=digest(PROOF)))
    write_json(NEGATIVE,dict(passed=negatives_pass,freeze_sha256=digest(SEAL),outcome=neg,
                             screened_execution_sha256=digest(PROOF)))
    print("Restricted recovery",passed,holdout["previous"],holdout["candidate"],"negatives passed",negatives_pass,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    qualify()
