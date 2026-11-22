"""Freeze and audit a local crop policy at the existing automatic operating point."""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_audits.detection_crop_semantics import (ART, DECISION, DINOV2_INFO, FIT, PADDINGS, SIGLIP, VisualSemantics,
                                             adjust, expanded, feature_cache, proposals, read_features, scores)
from scripts.experiments.transfer.detection_dental_challenge import LOCK as DENTAL
from scripts.experiments.detector_training.detection_hillclimb_experiment import (DATA, DEVELOPMENT, FLOOR, HELD_OUT, SELECTED, cache, inputs,
                                                   loaded, manifest)
from scripts.experiments.detector_training.detection_hillclimb_rank import grouped_difference, minimal, slices
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

LOCK = ART / "crop_semantics_frozen.json"
RESULT = ART / "crop_semantics_audit.json"
ZERO = ART / "crop_semantics_zero_shot.json"
DENTAL_REVIEW = ART / "dental_reference_review.json"
SCOPES = (*HELD_OUT,"dental")


def freeze():
    if LOCK.exists() or RESULT.exists() or any(cache(scope,"baseline").exists() for scope in HELD_OUT):
        raise ValueError("A crop freeze already exists or the protected baseline was opened.")
    trial_reports = (ZERO,DECISION) if DECISION.exists() else (ZERO,)
    available = {}
    for path in trial_reports:
        result = read_json(path)
        decision = result["decision"]
        if decision:
            name = decision["variant"]
            specs = [dict(name=name.split("/")[0], variant=name, head=result["heads"][name],
                          bounds=decision["bounds"])]
            available[path] = dict(policy=decision, components=specs, report=str(path))
    if not available:
        raise ValueError("No crop rule preserves the development true positives and false-positive ceiling.")

    def priority(item):
        decision = item["policy"]
        p = decision["roles"]["public-development"]
        s = decision["roles"]["phlebotomy-development"]
        return (p["guard"]["recovered_matches"], -p["extra_fp"],
                s["guard"]["recovered_matches"], -s["extra_fp"], -len(item["components"]))

    crossed = []
    if ZERO in available and DECISION in available:
        visual = available[ZERO]["components"][0]
        fit = available[DECISION]["components"][0]
        if fit["head"] is not None:
            crossed = joint_rules(visual,fit)
    candidates = [*available.values(),*(r for r in crossed if all(
        a["guard"]["passed"] and a["extra_fp"]<=0 for a in r["policy"]["roles"].values()))]
    decision = max(candidates,key=priority)
    required = [Path(__file__),Path("scripts/experiments/detector_audits/detection_crop_semantics.py"),*trial_reports,SIGLIP,DINOV2_INFO,FLOOR,
                manifest("public-audit"),Path("artifacts/sam3_reserve.json"),DENTAL,DENTAL_REVIEW,
                *(feature_cache(role) for role in DEVELOPMENT)]
    if DECISION.exists():
        required += [FIT,feature_cache("fit")]
    files = {str(p):digest(p) for p in required}
    write_json(LOCK,dict(policy=decision["policy"],components=decision["components"],
                        combined_development_trials=crossed,
                        selection_inputs=files, scope="Both development partitions, before all reserved inference.",
                        acceptance="Every prior matched instance is preserved in each reserved cohort, false positives "
                                   "do not increase in any, and public-audit precision or recall improves."))
    print("Frozen crop rule",decision["policy"],flush=True)


def apply_rules(original, regions, components):
    ranked = original["ranked"]
    margins = {}
    for rule in components:
        value = scores(regions,rule["name"],rule["head"])
        ranked = adjust(ranked,regions,value,rule["bounds"],original["floor"])
        margins[rule["variant"]] = [float(s) for s in value]
    return ranked,margins


def joint_rules(visual, learned):
    veto = {**visual,"bounds":[1e4,*visual["bounds"][1:]]}
    rescue = {**learned,"bounds":[learned["bounds"][0],-1e4,learned["bounds"][2]]}
    modes = {"visual-veto-plus-learned-rescue":[veto,rescue],
             "visual-and-learned-evidence":[visual,rescue]}
    found = {name:{} for name in modes}
    cutoff = read_json(FLOOR)["cutoff"]
    for role in DEVELOPMENT:
        rows = inputs(role)
        base = loaded(role,"baseline")
        frames = read_features(role)
        initial = minimal(rows,{key:f["ranked"] for key,f in base.items()},cutoff)
        for name,parts in modes.items():
            rec = {key:apply_rules(value,frames[key],parts)[0] for key,value in base.items()}
            end = minimal(rows,rec,cutoff)
            found[name][role] = dict(tp=end["tp"],fp=end["fp"],fn=end["fn"],guard=compare(initial,end),
                                    extra_fp=end["fp"]-initial["fp"])
    return [dict(policy=dict(variant=name,roles=found[name]),components=parts) for name,parts in modes.items()]


def evidence():
    policy = read_json(LOCK)
    if any(digest(p)!=value for p,value in policy["selection_inputs"].items()):
        raise ValueError("The crop policy or a frozen input has changed.")
    return policy


def rows_for(scope):
    return read_json(DENTAL)["rows"] if scope=="dental" else inputs(scope)


def baseline(scope):
    if scope=="dental":
        from scripts.experiments.transfer.detection_dental_evaluate import load

        return load("baseline", rows_for(scope))
    return loaded(scope,"baseline")


def output(scope):
    return DATA/"proposals"/f"{scope}-frozen-crop-rule.json"


def crop_frame(encoder,image,ranked):
    candidates = proposals(ranked)
    patches = [crop(image,expanded(p["bbox"],padding)) for padding in PADDINGS for p in candidates]
    return encode_frame(encoder, candidates, patches)


def encode_frame(encoder, candidates, patches):
    vectors = {name:[] for name in ("siglip","dino","prompt")}
    for i in range(0,len(patches),8):
        for name,value in encoder.encode(patches[i:i+8]).items():
            vectors[name].extend(value.tolist())
    return dict(candidates=candidates,values=vectors)


def collect(scope):
    configure_cpu()
    if not SELECTED.exists():
        raise ValueError("The detector families must be frozen before collecting reserved predictions.")
    policy = evidence()
    frames = baseline(scope)
    rows = rows_for(scope)
    key = dict(freeze_sha256=digest(LOCK),reference_hashes={r["id"]:r["sha256"] for r in rows})
    target = output(scope)
    current = read_json(target) if target.exists() else dict(key=key,frames={})
    if current["key"]!=key:
        raise ValueError("Audit crop predictions were generated from different inputs.")
    runner = None
    for row in rows:
        identifier = row["id"]
        if identifier in current["frames"]:
            continue
        if runner is None:
            runner = VisualSemantics()
        before = frames[identifier]
        image = source(row)
        started = perf_counter()
        regions = crop_frame(runner,image,before)
        result,margins = apply_rules(before,regions,policy["components"])
        current["frames"][identifier] = dict(ranked=result,ms=(perf_counter()-started)*1000,
                                            candidate_scores=[dict(bbox=p["bbox"],margins={name:values[i] for name,values in margins.items()})
                                                              for i,p in enumerate(regions["candidates"])])
        n = len(current["frames"])
        if n%4==0 or n==len(rows):
            write_json(target,current)
            print(scope,"crop audit",n,"/",len(rows),flush=True)


def measure():
    if RESULT.exists():
        raise ValueError("The independent crop audit has already been recorded.")
    proof = evidence()
    cutoff = read_json(FLOOR)["cutoff"]
    findings = {}
    for scope in SCOPES:
        rows = rows_for(scope)
        previous = baseline(scope)
        crop_outputs = read_json(output(scope))
        if crop_outputs["key"]["freeze_sha256"] != digest(LOCK) or set(crop_outputs["frames"]) != set(previous):
            raise ValueError("Reserved crop inference was incomplete or used another policy.")
        before = {key:obj["ranked"] for key,obj in previous.items()}
        after = {key:obj["ranked"] for key,obj in crop_outputs["frames"].items()}
        old,new = (evaluate(rows,values,cutoff) for values in (before,after))
        guard = compare(old,new)
        findings[scope] = dict(previous=short(old),candidate=short(new),guard=guard,
                               errors=dict(before={"fp":old["false_positives"],"fn":old["false_negatives"]},
                                           after={"fp":new["false_positives"],"fn":new["false_negatives"]}),
                               added_crop_ms={k:obj["ms"] for k,obj in crop_outputs["frames"].items()},
                               slices=dict(previous=slices(rows,old,before,cutoff),candidate=slices(rows,new,after,cutoff)),
                               groups=dict(Counter(r["group"] for r in rows)),
                               paired_group_bootstrap=grouped_difference(rows,before,after) if scope=="public-audit" else None)
    public = findings["public-audit"]
    improvement = public["guard"]["recovered_matches"]>0 or public["candidate"]["fp"]<public["previous"]["fp"]
    passed = improvement and all(r["guard"]["passed"] and r["candidate"]["fp"]<=r["previous"]["fp"]
                                 for r in findings.values())
    write_json(RESULT,dict(passed=passed,selection_sha256=digest(LOCK),qualification=proof["acceptance"],
                           outcomes=findings))
    print("Crop audit",passed,{scope:(r["previous"],r["candidate"]) for scope,r in findings.items()},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","collect","measure"))
    parser.add_argument("--scope",choices=SCOPES)
    args = parser.parse_args()
    if args.action=="collect":
        if not args.scope:
            parser.error("collect requires --scope")
        collect(args.scope)
    elif args.action=="freeze":
        freeze()
    else:
        measure()


if __name__=="__main__":
    main()
