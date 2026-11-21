"""Select independent corroboration from existing ranked development proposals."""
from itertools import product
from pathlib import Path

from medsyringe.data import digest, read_json, write_json
from medsyringe.recall_guard import compare
from scripts.detection_consensus import bound, classify, features
from scripts.detection_crop_semantics import feature_cache, read_features
from scripts.detection_empty_scene_audit import LOCK as PREVIOUS, is_empty
from scripts.detection_exemplar_comparison import destination
from scripts.detection_hillclimb_experiment import ART, DEVELOPMENT, FLOOR, cache, inputs, loaded
from scripts.detection_hillclimb_rank import minimal
from scripts.detection_open_models import open_views, proposal_path
from scripts.detection_semantics_audit import apply_rules

LOCK = ART/"auxiliary_choice_development.json"
OPEN = ("dino-base","owlv2")
MM_VARIANTS = ("context-full","title-wide","phrases-full")


def development():
    if LOCK.exists():
        raise ValueError("These development choices are already fixed.")
    previous = read_json(PREVIOUS)
    floor = read_json(FLOOR)["cutoff"]
    stages = {}
    for scope in DEVELOPMENT:
        refs,base = inputs(scope),loaded(scope,"baseline")
        encoded = read_features(scope)
        auxiliary = {m:open_views(scope,m)["wide"] for m in OPEN}
        mm = read_json(destination("openmmlab",scope))["frames"]
        kept = {}
        crops = {}
        for k,frame in base.items():
            kept[k] = apply_rules(frame,encoded[k],previous["control"])[0]
            if not is_empty(frame):
                continue
            heads = previous["heads"]
            opinions = {m:v[k] for m,v in auxiliary.items()}
            evidence = features(encoded[k],"semantic-best",heads["joint"],heads["shape"],opinions)
            candidates = [(p,prior,text,joint,shape,support,{
                name:bound(p["bbox"],mm[k]["proposals"][name]) for name in MM_VARIANTS
            }) for p,prior,text,joint,shape,support in evidence]
            crops[k] = candidates
        stages[scope] = (refs,base,kept,crops,minimal(refs,kept,floor))
    trials = []
    settings = previous["settings"]
    options = product(MM_VARIANTS, (.15,.25,.35,.45), ("old-pair","dino","owl","either"),
                      (.26,.36), (.65,1.0), (-.008,.005,.015))
    for arm,raw,partner,prior,joint,text in options:
        fused = dict(settings,prior=prior,joint=joint,text=text)
        outcomes = {}
        for scope,(refs,base,kept,crops,control) in stages.items():
            ranked = kept.copy()
            for key,items in crops.items():
                examined = []
                for p,previous_score,description,confidence,appearance,support,matching in items:
                    dino, owl = support["dino-base"]>=.27, support["owlv2"]>=.15
                    agrees = {"old-pair":dino and owl,"dino":dino,"owl":owl,"either":dino or owl}[partner]
                    if matching[arm]>=raw and agrees:
                        examined.append((p,previous_score,description,confidence,appearance,{"dino-base":1.,"owlv2":1.}))
                ranked[key] = classify(base[key],examined,fused)
            score = minimal(refs,ranked,floor)
            outcomes[scope] = dict(tp=score["tp"],fp=score["fp"],fn=score["fn"],
                                   new_errors=score["fp"]-control["fp"],retention=compare(control,score))
        trials.append(dict(policy=dict(arm=arm,minimum=raw,partner=partner,settings=fused),outcomes=outcomes))
    safe = [r for r in trials if all(x["new_errors"]<=0 and x["retention"]["passed"] for x in r["outcomes"].values())]
    successes = [r for r in safe if sum(v["retention"]["recovered_matches"] for v in r["outcomes"].values())>0]
    def strength(row):
        a=row["outcomes"]["public-development"]
        b=row["outcomes"]["phlebotomy-development"]
        return a["tp"],-a["fp"],b["tp"],-b["fp"],row["policy"]["minimum"]
    winner = max(successes,key=strength,default=None)
    paths = [Path("scripts/detection_auxiliary_choice.py"),PREVIOUS,FLOOR]
    for scope in DEVELOPMENT:
        paths.extend((cache(scope,"baseline"),feature_cache(scope),destination("openmmlab",scope),
                      *(proposal_path(scope,m) for m in OPEN)))
    write_json(LOCK,dict(winner=winner,policies=len(trials),safe=len(safe),trial_results=trials,
                         established_recovery=previous["development"],
                         inputs={str(p):digest(p) for p in paths}))
    print("Independent corroboration",len(trials),"configurations",len(safe),"without regressions",winner,flush=True)


if __name__=="__main__":
    development()
