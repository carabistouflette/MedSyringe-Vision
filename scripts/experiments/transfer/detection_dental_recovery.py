"""Ablate dental-target crop scoring using separate acquisition groups."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import suppress
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_audits.detection_crop_semantics import PADDINGS, SIGLIP, VisualSemantics, columns
from scripts.experiments.transfer.detection_dental_evaluate import runner
from scripts.experiments.transfer.detection_dental_transfer_data import MANIFEST
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, EXISTING, FLOOR, cache, inputs, loaded
from scripts.experiments.detector_training.detection_hillclimb_rank import minimal
from scripts.experiments.detector_audits.detection_precision_confirmation import cohort
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.transfer.rfdetr_verifier import DINOV2_INFO, expanded
from scripts.experiments.expanded.sam3_study import configure_cpu


LOCK = ART / "dental_recovery_selection.json"
RESULT = ART / "dental_recovery_audit.json"
GENERAL = (*DEVELOPMENT, "fresh", "commons-regression", "previous-reserve", "openimages", "lookalikes")
POSITIVE = (
    "a photo of a metal dental aspirating syringe with a ring for the thumb",
    "a photo of a dental local anesthetic cartridge syringe with a piston",
    "a gloved dentist holding a dental injection syringe with a visible cartridge",
    "a glass and metal hypodermic syringe with a plunger",
)
NEGATIVE = (
    "a dental mirror with a circular reflective head", "a dental pick or periodontal probe",
    "a dental drill handpiece with an angled head", "a laboratory pipette or vial",
    "a pair of surgical forceps or scissors", "a gloved hand holding a dental tool",
    "an intravenous catheter without a syringe", "a row of metal tools on a medical tray",
)


def references(scope):
    if scope in DEVELOPMENT:
        return inputs(scope)
    if scope in (*GENERAL, "dental"):
        return cohort(scope)[0]
    return [r for r in read_json(MANIFEST)["rows"] if r["split"] == scope]


def output(scope, kind):
    return DATA / "proposals" / f"dental-transfer-{scope}-{kind}.json"


def baseline(scope):
    if scope in (*GENERAL, "dental"):
        return loaded(scope, "baseline") if scope in DEVELOPMENT else cohort(scope)[1]
    path = output(scope, "baseline")
    current = read_json(path)
    if current["manifest_sha256"] != digest(MANIFEST):
        raise ValueError("The dental transfer dataset has changed.")
    if set(current["frames"]) != {r["id"] for r in references(scope)}:
        raise ValueError("Dental transfer detector cache is incomplete.")
    return current["frames"]


def collect(scope):
    if scope == "audit" and not LOCK.exists():
        raise ValueError("Freeze development choices before audit inference.")
    configure_cpu()
    destination = output(scope, "baseline")
    known = read_json(destination) if destination.exists() else dict(
        manifest_sha256=digest(MANIFEST), detector_sha256=digest(EXISTING), frames={})
    if (known["manifest_sha256"], known["detector_sha256"]) != (digest(MANIFEST), digest(EXISTING)):
        raise ValueError("The dental transfer cache has stale inputs.")
    data = [r for r in references(scope) if r["id"] not in known["frames"]]
    if not data:
        return
    detect, close, _ = runner("baseline")
    try:
        for row in data:
            known["frames"][row["id"]] = detect(source(row))
            write_json(destination, known)
            print("Dental transfer", scope, len(known["frames"]), flush=True)
    finally:
        close()


class DentalEvidence(VisualSemantics):
    def __init__(self):
        super().__init__()
        prompts = self.processor(text=[*POSITIVE, *NEGATIVE], padding="max_length", truncation=True, return_tensors="pt")
        with torch.inference_mode():
            new = self.model.get_text_features(**prompts)
            self.dental_prompts = torch.nn.functional.normalize(new).numpy()


def candidate_regions(frame):
    pending = [{**p, "score":p.get("original_score", p["score"])} for p in frame["ranked"]
               if p["score"] < frame["floor"] and p.get("original_score", p["score"]) >= .20]
    return suppress(pending, .65)[:18]


def feature_frame(reader, image, candidates):
    patches = [crop(image, expanded(p["bbox"], padding)) for padding in PADDINGS for p in candidates]
    vectors = dict(siglip=[], dino=[])
    for i in range(0, len(patches), 8):
        values = reader.encode(patches[i:i+8])
        for name in vectors:
            vectors[name].extend(values[name].tolist())
    return dict(candidates=candidates, values=vectors)


def collect_features(scope):
    if scope not in ("fit", "development", *DEVELOPMENT) and not LOCK.exists():
        raise ValueError("Freeze dental recovery before processing a reserved group.")
    configure_cpu()
    fitting = scope == "fit"
    base = {} if fitting else baseline(scope)
    path = output(scope, "features")
    refs = references(scope)
    key = dict(manifest_sha256=digest(MANIFEST), code_sha256=digest(Path(__file__)),
               siglip_sha256=digest(SIGLIP), dino_sha256=digest(DINOV2_INFO),
               images={r["id"]:r["sha256"] for r in refs})
    result = read_json(path) if path.exists() else dict(key=key, frames={})
    if result["key"] != key:
        raise ValueError("Dental feature inputs changed.")
    encoder = None
    for row in refs:
        identity = row["id"]
        if identity in result["frames"]:
            continue
        if fitting:
            proposed = [{**item, "label":label} for name,label in (("objects",1),("lookalikes",0))
                        for item in row[name]]
        else:
            frame = base[identity]
            proposed = candidate_regions(frame) if not any(p["score"] >= frame["floor"] for p in frame["ranked"]) else []
        picture = source(row) if proposed else None
        if picture is not None and encoder is None:
            encoder = DentalEvidence()
            result["prompts"] = encoder.dental_prompts.tolist()
        block = feature_frame(encoder, picture, proposed)
        if fitting and proposed:
            # Rotate the same approved fit crops after the source-group split.
            # One extra view limits specimen overrepresentation.
            rotated = [{**p, "bbox":[1-p["bbox"][2],1-p["bbox"][3],1-p["bbox"][0],1-p["bbox"][1]]}
                       for p in proposed]
            second = feature_frame(encoder, picture.rotate(180), rotated)
            block = dict(candidates=[*block["candidates"],*second["candidates"]], values={name:
                         [*block["values"][name][:len(proposed)],*second["values"][name][:len(proposed)],
                          *block["values"][name][len(proposed):],*second["values"][name][len(proposed):]]
                         for name in block["values"]})
        result["frames"][identity] = block
        if len(result["frames"]) % 3 == 0 or len(result["frames"]) == len(refs):
            write_json(path, result)
            print("Dental features", scope, len(result["frames"]), "/", len(refs), flush=True)


def measured(scope):
    stored = read_json(output(scope, "features"))
    if set(stored["frames"]) != {r["id"] for r in references(scope)}:
        raise ValueError(f"Incomplete feature extraction in {scope}.")
    return stored


def heads(fit):
    weights = np.array([1/len(frame["candidates"]) for frame in fit.values() for _ in frame["candidates"]])
    labels = np.array([p["label"] for frame in fit.values() for p in frame["candidates"]])
    for cls in (0,1):
        weights[labels == cls] *= len(weights)/(2*weights[labels == cls].sum())
    result = {}
    for name in ("dino", "siglip", "joint"):
        x = np.vstack([columns(f, name) for f in fit.values()])
        for strength in (.5, 4., 20.):
            model = LogisticRegression(C=strength, random_state=123).fit(x, labels, sample_weight=weights)
            result[f"{name}/{strength}"] = dict(name=name, weights=model.coef_[0].tolist(), bias=float(model.intercept_[0]))
    dino = np.vstack([columns(f, "dino") for f in fit.values()])
    dino /= np.linalg.norm(dino, axis=1, keepdims=True)
    result["prototype"] = dict(name="prototype", positive=dino[labels==1].tolist(), negative=dino[labels==0].tolist())
    result["prompt"] = dict(name="prompt")
    return result


def evidence(frame, models, prompts):
    candidates = frame["candidates"]
    if not candidates:
        return np.array([]),{name:np.array([]) for name in models}
    cols = {name:columns(frame, name) for name in ("dino","siglip","joint")}
    sig = np.array(frame["values"]["siglip"]).reshape(len(PADDINGS),len(candidates),-1)
    agreements = sig @ prompts.T
    text = (agreements[:,:,:len(POSITIVE)].max(axis=2) - agreements[:,:,len(POSITIVE):].max(axis=2)).max(axis=0)
    signals = {}
    for name,model in models.items():
        kind = model["name"]
        if kind == "prompt":
            signals[name] = text*10
        elif kind == "prototype":
            vectors = cols["dino"]/np.linalg.norm(cols["dino"],axis=1,keepdims=True)
            yes, no = (vectors @ np.array(model[k]).T for k in ("positive","negative"))
            signals[name] = 10*(yes.max(axis=1)-no.max(axis=1))
        else:
            signals[name] = cols[kind] @ np.array(model["weights"]) + model["bias"]
    return text,signals


def apply(frame, regions, text, signals, policy):
    score = signals[policy["head"]]
    chosen = [(p,float(value),float(margin)) for p,value,margin in zip(regions,score,text,strict=True)
              if p["score"]>=policy["prior"] and value>=policy["confidence"] and margin>=policy["semantics"]]
    if not chosen:
        return frame["ranked"]
    proposal,value,margin = max(chosen,key=lambda p:(p[1],p[0]["score"]))
    return [dict(bbox=proposal["bbox"], score=frame["floor"], source="dental crop recovery",
                 recognition_margin=value, semantic_margin=margin), *frame["ranked"]]


def select():
    if LOCK.exists() or output("audit", "baseline").exists():
        raise ValueError("Dental selection is frozen or reserved inference has already run.")
    fit = measured("fit")
    models = heads(fit["frames"])
    prompts = np.array(fit["prompts"])
    floor = read_json(FLOOR)["cutoff"]
    scopes = ("development",*DEVELOPMENT)
    prepared = {}
    for scope in scopes:
        refs,base = references(scope),baseline(scope)
        candidates = measured(scope)["frames"]
        signals = {key:evidence(v,models,prompts) for key,v in candidates.items()}
        original = minimal(refs,{key:f["ranked"] for key,f in base.items()},floor)
        prepared[scope] = refs,base,candidates,signals,original
    search = product(models, (.20,.30,.42,.55), (0.,.08,.17,.32,.6,1.0), (-.025,-.01,0.,.012,.025))
    outcomes = []
    for head,prior,confidence,semantics in search:
        policy = dict(head=head,prior=prior,confidence=confidence,semantics=semantics)
        results = {}
        for scope,(refs,base,candidates,signals,original) in prepared.items():
            predictions = {key:apply(frame,candidates[key]["candidates"],*signals[key],policy) for key,frame in base.items()}
            outcome = minimal(refs,predictions,floor)
            results[scope] = dict(tp=outcome["tp"],fp=outcome["fp"],fn=outcome["fn"],
                                  fp_delta=outcome["fp"]-original["fp"],guard=compare(original,outcome))
        outcomes.append(dict(policy=policy, results=results))
    safe = [x for x in outcomes if all(row["guard"]["passed"] and row["fp_delta"] <= 0 for row in x["results"].values())]
    improved = [x for x in safe if x["results"]["development"]["guard"]["recovered_matches"] > 0]
    best = max(improved, key=lambda r:(r["results"]["development"]["tp"],
                                      sum(x["tp"] for x in r["results"].values()),
                                      r["policy"]["head"]=="prompt",r["policy"]["semantics"]),default=None)
    alternate = {}
    for head in models:
        choices = [x for x in improved if x["policy"]["head"]==head]
        alternate[head] = max(choices,key=lambda r:r["results"]["development"]["tp"],default=None)
    relative = (Path("scripts/experiments/transfer/detection_dental_recovery.py"),MANIFEST,EXISTING,FLOOR,SIGLIP,DINOV2_INFO,
                *(output(scope,"features") for scope in ("fit",*scopes)),
                output("development","baseline"),*(cache(scope,"baseline") for scope in DEVELOPMENT))
    write_json(LOCK,dict(selected=best, safe_choices=len(safe), trials=len(outcomes), best_by_ablation=alternate,
                         development_baseline={k:{x:v[-1][x] for x in ("tp","fp","fn")} for k,v in prepared.items()},
                         prompts=fit["prompts"], model=models[best["policy"]["head"]] if best else None,
                         inputs={str(path):digest(path) for path in relative if path.exists()}))
    print("Dental recognition",len(outcomes),"policies,",len(improved),"safe improvements",best,flush=True)


def measure():
    from scripts.experiments.expanded.expanded_detection_study import short

    if RESULT.exists():
        raise ValueError("The dental recovery policy has already been scored.")
    locked = read_json(LOCK)
    if any(digest(path) != value for path,value in locked["inputs"].items()):
        raise ValueError("Frozen dental development inputs were modified.")
    if locked["selected"] is None:
        raise ValueError("No improvement qualified on development.")
    policy = locked["selected"]["policy"]
    models = {policy["head"]:locked["model"]}
    prompts = np.array(locked["prompts"])
    floor = read_json(FLOOR)["cutoff"]
    results = {}
    for scope in ("audit","dental",*(r for r in GENERAL if r not in DEVELOPMENT)):
        reference,base = references(scope),baseline(scope)
        features = measured(scope)["frames"]
        start = perf_counter()
        after = {key:apply(frame,features[key]["candidates"],*evidence(features[key],models,prompts),policy)
                 for key,frame in base.items()}
        old,new = (evaluate(reference,p,floor) for p in ({key:f["ranked"] for key,f in base.items()},after))
        guard = compare(old,new)
        results[scope] = dict(previous=short(old),candidate=short(new),guard=guard,
                              metric_seconds=perf_counter()-start,new_errors=new["false_positives"])
    safe = all(r["guard"]["passed"] and r["candidate"]["fp"] <= r["previous"]["fp"] for r in results.values())
    ok = safe and results["audit"]["guard"]["recovered_matches"] > 0
    write_json(RESULT,dict(passed=ok,selection_sha256=digest(LOCK),outcomes=results,
                           capture_preservation="Only empty baseline scenes can change. All 177 Fil Rouge images already have an automatic instance."))
    print("Dental holdout",ok,{k:(v["previous"]["tp"],v["candidate"]["tp"],v["candidate"]["fp"]) for k,v in results.items()},
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "features", "select", "measure"))
    parser.add_argument("--scope", choices=("fit","development","audit","dental",*GENERAL))
    args = parser.parse_args()
    if args.action in ("collect","features"):
        if not args.scope or (args.action=="collect" and args.scope not in ("development","audit")):
            parser.error("Choose a supported dataset partition.")
        (collect if args.action=="collect" else collect_features)(args.scope)
    else:
        (select if args.action=="select" else measure)()
