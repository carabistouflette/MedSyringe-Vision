"""Fit a lookalike-aware verifier for incremental syringe detections."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.appearance_runtime import RELEASE
from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate, match
from medsyringe.pipeline_extensions import verified_config
from scripts.native_candidate_verifier import compare, simple_metrics
from scripts.occlusion_detector import ART
from scripts.overlap_candidate_features import inputs as cached_inputs
from scripts.syringe_assembly_model import training_weights
from scripts.syringe_object_features import (
    CLASSES, EXPOSED_RESULT, ROUTES, TARGET, UPSTREAM, exposed, feature_store, fitting,
)
from scripts.syringe_object_references import REFERENCE, WORK, references

LOCK = ART / "syringe_object_verifier_policy.json"
RESULT = ART / "syringe_object_verifier_results.json"
ARMS = ("targeted-text", "binary", "typed", "typed-hybrid")
SEED = 284
LOGIT_FLOORS = (-99., -4., -3., -2.5, -2., -1.75, -1.5, -1.25, -1., -.75, -.50, -.25, 0., .25, .50, .75, 1.)
TEXT_FLOORS = (-99., -.030, -.024, -.018, -.015, -.012, -.009, -.006, -.004, -.002, 0., .002,
               .004, .006, .009, .012, .015, .018, .025, .035)


def appearance(p):
    return 6.*np.asarray(p["features"]).ravel()


def margins(p):
    by_crop = np.asarray(p["targeted"])
    return np.subtract(by_crop[:,[0]],by_crop[:,1:])


def fit_models(rows, frames):
    samples = [(p,r["group"]) for r in rows for p in frames[r["id"]]]
    visual = np.stack([appearance(p) for p,_ in samples])
    text = np.stack([margins(p).ravel() for p,_ in samples])
    kind = [p["label"] for p,g in samples]
    groups = [g for p,g in samples]
    group_support = {t:len({g for p,g in samples if p["label"]==t}) for t in CLASSES}
    if any(not n for n in group_support.values()):
        raise ValueError("Every verifier class needs a reviewed training source.")
    scaler = StandardScaler().fit(text)
    targets = dict(binary=[p==TARGET for p in kind], typed=kind)
    targets["typed-hybrid"] = kind
    results = {}
    for name,y in targets.items():
        xs = np.column_stack((visual,scaler.transform(text))) if name=="typed-hybrid" else visual
        w = training_weights(y,groups)
        model = LogisticRegression(C=.4,max_iter=600,random_state=SEED)
        model.fit(xs,y,sample_weight=w)
        if max(model.n_iter_) == model.max_iter:
            raise ValueError(f"The {name} verifier failed to converge.")
        results[name] = dict(weight=model.coef_.tolist(),bias=model.intercept_.tolist(),
                             classes=model.classes_.tolist())
    counts = dict(
        examples=len(samples), counts=Counter(kind), groups=group_support, total_groups=len(set(groups)),
        origins=Counter(p["origin"] for p,g in samples),
    )
    return dict(heads=results, center=scaler.mean_.tolist(), scale=scaler.scale_.tolist()), counts


def score_candidates(frames, model):
    result = {}
    for key,items in frames.items():
        result[key] = {}
        for p in items:
            box = tuple(p["bbox"])
            x = appearance(p)
            semantic = margins(p)
            rank = {"targeted-text":float(np.min(semantic,axis=1).max()),
                    "release-veto":p["release_margin"]}
            for name,head in model["heads"].items():
                inputs = x if name!="typed-hybrid" else np.r_[
                    x, (semantic.ravel()-model["center"])/model["scale"]]
                z = np.asarray(head["weight"]) @ inputs + head["bias"]
                if name=="binary":
                    rank[name] = float(z[0])
                    continue
                i = head["classes"].index(TARGET)
                rank[name] = float(z[i]-max(v for j,v in enumerate(z) if j!=i))
                rank[f"{name}_strongest_lookalike"] = max(
                    (c for c in head["classes"] if c!=TARGET), key=lambda c:z[head["classes"].index(c)])
            result[key][box] = rank
    return result


def predictions(base, offers, scored, arm, floor):
    accept, rejected = {}, {}
    for key,previous in base.items():
        accept[key] = list(previous)
        for p in offers[key]:
            state = scored[key][tuple(p["bbox"])]
            if arm=="off" or state[arm] >= floor:
                accept[key].append(p)
            else:
                rejected.setdefault(key,[]).append(dict(p,evidence=state))
    return accept,rejected


def guard(rows, base, unfiltered, accepted):
    hold = compare(rows,base,accepted)
    losses = {}
    for iou, prior in unfiltered.items():
        current = match(rows,accepted,iou)[1]
        losses[str(iou)] = {k:sorted(ids-current[k]) for k,ids in prior.items() if ids-current[k]}
    return dict(**hold, lost_recovered=losses, preserves_recovery=not any(losses.values()))


def thresholds(rows, base, routes, ranked):
    current = simple_metrics(rows,base)
    trials, finalists = {}, {}
    for route in ROUTES:
        offers = routes[route]
        base_and_recovery = {k:[*v,*offers[k]] for k,v in base.items()}
        frozen = {t:match(rows,base_and_recovery,t)[1] for t in (.50,.75)}
        pool = []
        for arm,floor in chain_rules():
            outcome, dropped = predictions(base,offers,ranked,arm,floor)
            pool.append(dict(arm=arm,floor=floor,rejected=sum(map(len,dropped.values())),
                             **guard(rows,current,frozen,outcome)))
        trials[route] = pool
        finalists[route] = {
            arm:max((t for t in pool if t["arm"]==arm and t["preserves_recovery"]),
                    key=lambda a:(-a["fp"],a["exact_count"],-a["rejected"],-a["floor"]))
            for arm in ARMS
        }
    eligible = [dict(route=k,**r) for k,g in finalists.items() for r in g.values()
                if r["safe"] and r["gain"]>0 and r["rejected"]>0]
    chosen = max(eligible,key=lambda t:(t["gain"],t["exact_count"],t["rejected"]),default=None)
    return dict(trials=trials,finalists=finalists,winner=chosen)


def chain_rules():
    for name in ARMS:
        for layout in (TEXT_FLOORS if name=="targeted-text" else LOGIT_FLOORS):
            yield name,layout


def bindings():
    paths = (Path(__file__), Path("scripts/syringe_object_features.py"), REFERENCE, RELEASE, UPSTREAM,
             WORK/"features-fit.json", WORK/"features-development.json")
    return {str(p):digest(p) for p in paths}


def partition_audit(fit, development, reserve):
    cohorts = dict(fit=fit, development=development, reserve=reserve)
    keys = ("id", "group", "sha256", "source_url")
    provenance = {role: {key: {r[key] for r in rows if r.get(key)} for key in keys}
                  for role, rows in cohorts.items()}
    overlaps = {}
    for left, right in combinations(cohorts, 2):
        shared = {key: len(provenance[left][key] & provenance[right][key]) for key in keys}
        if any(shared.values()):
            raise ValueError(f"Shared provenance between {left} and {right}: {shared}.")
        overlaps[f"{left}/{right}"] = shared
    return {role: dict(images=len(rows), groups=len(provenance[role]["group"]))
            for role, rows in cohorts.items()}, overlaps


def freeze():
    if LOCK.exists():
        raise FileExistsError(LOCK)
    rows,_ = fitting()
    dev = cached_inputs("development")[0]
    cohort_sizes, overlaps = partition_audit(rows, dev, references("fresh"))
    train = feature_store("fit",rows)
    validation = feature_store("development",dev)
    model, counts = fit_models(rows,train["frames"])
    score = score_candidates(validation["frames"],model)
    rules = thresholds(dev,validation["base"],validation["routes"],score)
    metrics = {k:evaluate(dev,{key:[*p,*v[key]] for key,p in validation["base"].items()},0.)
               for k,v in validation["routes"].items()}
    write_json(LOCK,dict(model=model,fitting=counts,selection=rules,seed=SEED,bindings=bindings(),
                         partitions=cohort_sizes,overlaps=overlaps,
                         development=dict(base=evaluate(dev,validation["base"],0.),routes=metrics)))
    print("Training",counts,flush=True)
    for route,arms in rules["finalists"].items():
        print(route,{k:(v["tp"],v["fp"],v["rejected"],v["safe"]) for k,v in arms.items()},flush=True)
    print("Selected",rules["winner"],flush=True)


def sealed():
    state = read_json(LOCK)
    if state["bindings"] != bindings():
        raise ValueError("The fixed verifier inputs changed.")
    return state


def measure(rows, packet, frozen, release_floor):
    base = packet["base"]
    metrics = simple_metrics(rows,base)
    ranked = score_candidates(packet["frames"],frozen["model"])
    routes = {}
    for route, offered in packet["routes"].items():
        baseline = {k:[*p,*offered[k]] for k,p in base.items()}
        accepted = {iou:match(rows,baseline,iou)[1] for iou in (.50,.75)}
        finalists = frozen["selection"]["finalists"][route]
        choices = {**{key:r["floor"] for key,r in finalists.items()},"off":-99.,"release-veto":release_floor}
        routes[route] = {}
        for arm,floor in choices.items():
            output,rejected = predictions(base,offered,ranked,arm,floor)
            routes[route][arm] = dict(metrics=evaluate(rows,output,0.),rejected=rejected,
                                      delta=guard(rows,metrics,accepted,output))
    ledger = {r["id"]:[dict(bbox=list(box),**p) for box,p in ranked[r["id"]].items()] for r in rows}
    return dict(baseline=evaluate(rows,base,0.),routes=routes,evidence=ledger)


def score():
    lock = sealed()
    release = verified_config(Path.cwd(),RELEASE)
    rule = release["components"][0]
    if release.get("mode")!="veto-only" or rule["name"]!="semantic-best" or rule["head"] is not None:
        raise ValueError("This diagnostic requires the issued semantic-only veto.")
    threshold = rule["bounds"][1]
    results = {}
    for role,rows in (("fresh",references("fresh")),("exposed",exposed()[0])):
        cache = feature_store(role,rows)
        results[role] = measure(rows,cache,lock,threshold)

    choice = lock["selection"]["winner"]
    qualified = False
    if choice:
        fresh,exposed_delta = (v["routes"][choice["route"]][choice["arm"]]["delta"] for v in results.values())
        qualified = bool(fresh["gain"]>0 and all(d["safe"] and d["preserves_recovery"]
                                                 for d in (fresh,exposed_delta)))
    write_json(RESULT,dict(policy=digest(LOCK),exposure_source=digest(EXPOSED_RESULT),
                          feature_digests={role:digest(WORK/f"features-{role}.json") for role in results},
                          winner=choice,qualified=qualified,cohorts=results))
    for name,cohort in results.items():
        for route,arms in cohort["routes"].items():
            print(name,route,{k:(v["metrics"]["tp"],v["metrics"]["fp"],v["delta"]["preserves_recovery"])
                              for k,v in arms.items()},flush=True)
    print("Qualified",qualified,flush=True)


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("freeze","score"))
    {"freeze":freeze,"score":score}[cli.parse_args().stage]()
