"""Measure independent local crop semantics for uncertain detector proposals."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import ImageOps
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoProcessor

from medsyringe.additive import overlap_fraction
from medsyringe.attributes import crop
from medsyringe.data import digest, leakage, read_json, write_json
from medsyringe.detectors import box_iou, suppress
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, FLOOR, cache, inputs, loaded
from scripts.detection_hillclimb_rank import minimal
from scripts.expanded_detection_study import short
from scripts.precision_study import source
from scripts.rfdetr_verifier import DINOV2_INFO, DinoEncoder, expanded
from scripts.sam3_study import configure_cpu

MODEL = Path("data/models/siglip2-base")
MODEL_ID = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP = ART / "siglip2_crop_backbone.json"
FIT = ART / "public_additional_fit.json"
DECISION = ART / "crop_semantics_development.json"
PADDINGS = (.10, .35)
SYRINGES = ("a photo of a medical syringe with a barrel and plunger",
            "a close-up photo of a hypodermic syringe", "a used disposable injection syringe",
            "a plastic syringe with its needle cap on", "a syringe installed in a medical infusion pump")
LOOKALIKES = ("a person holding a medical item", "a pen or marker", "a pipette or laboratory tube",
              "an insulin injection pen", "a loose hypodermic needle without its syringe",
              "an injection vial or medicine bottle", "a photo of a medical instrument without a syringe",
              "a medical infusion pump without a syringe", "a test tube or glass vial",
              "tape or tubing or clear plastic packaging", "a stapler or pencil")


def lock_weights():
    if not SIGLIP.exists():
        required = ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json",
                    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model")
        write_json(SIGLIP, dict(source=MODEL_ID, revision=REVISION, license="Apache-2.0",
                               files={name:digest(MODEL/name) for name in required}))
    lock = read_json(SIGLIP)
    if any(digest(MODEL/name) != checksum for name,checksum in lock["files"].items()):
        raise ValueError("SigLIP 2 model bytes do not match the local lock.")


class VisualSemantics:
    def __init__(self):
        lock_weights()
        self.processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=True).eval()
        self.dino = DinoEncoder()
        prompts = self.processor(text=[*SYRINGES,*LOOKALIKES], padding="max_length", truncation=True,
                                 return_tensors="pt")
        with torch.inference_mode():
            self.text = torch.nn.functional.normalize(self.model.get_text_features(**prompts))

    @torch.inference_mode()
    def encode(self, patches):
        resized = [ImageOps.pad(patch, (224,224), color=(124,116,104)) for patch in patches]
        batch = self.processor(images=resized, return_tensors="pt")
        vectors = torch.nn.functional.normalize(self.model.get_image_features(**batch))
        text = vectors @ self.text.T
        positive,negative = text[:,:len(SYRINGES)],text[:,len(SYRINGES):]
        summary = torch.stack((positive.max(dim=1).values,negative.max(dim=1).values,
                               positive.mean(dim=1),negative.mean(dim=1)), dim=1)
        return dict(siglip=vectors.numpy(), prompt=summary.numpy(), dino=self.dino.encode(patches))


def proposals(frame):
    ranked = frame["ranked"]
    floor = frame["floor"]
    accepted = [p for p in ranked if p["score"] >= floor]
    medium = [p for p in ranked if p["score"] < floor and p.get("original_score",p["score"]) >= .23]
    pending = suppress([{**p,"score":p.get("original_score",p["score"]),"baseline_score":p["score"]}
                        for p in medium],.65)[:16]
    return [*accepted,*pending]


def local_example_boxes(row, frame):
    truths = [o["bbox"] for o in row["objects"]]
    candidates = [{"bbox":box,"label":True,"origin":"annotation"} for box in truths]
    candidates += [{"bbox":o["bbox"],"label":False,"origin":"lookalike"} for o in row.get("lookalikes",[])]
    known = [p["bbox"] for p in candidates]
    for p in proposals(frame):
        box = p["bbox"]
        best = max((box_iou(box,q) for q in truths),default=0.)
        if (best < .15 or best >= .50) and all(box_iou(box,q)<.78 for q in known):
            candidates.append({**p,"label":best>=.50,"origin":"proposal"})
            known.append(box)
    return candidates[:10]


def fit_rows():
    base = read_json(FIT)["rows"]
    old = [r for r in read_json(ART / "multidomain_training_inputs.json")["rows"]
           if r["training_domain"] == "public-previous-fit"]
    protected = [r for cohort in (*DEVELOPMENT, "public-audit", "previous-reserve") for r in inputs(cohort)]
    protected.extend(read_json(ART / "dental_manifest.json")["rows"])
    eligible = [{**r,"split":"fit"} for r in (*base,*old)]
    if len({r["id"] for r in eligible}) != len(eligible) or leakage([
        *eligible,*[{**r,"split":"protected"} for r in protected]
    ]):
        raise ValueError("Crop fitting rows overlap protected images.")
    return eligible


def feature_cache(role):
    return DATA / "proposals" / f"{role}-crop-semantics.json"


def collector(role):
    from medsyringe.detectors import Grounder
    from scripts.expanded_proposals import image_for

    configure_cpu()
    lock_weights()
    fitting = role == "fit"
    rows = fit_rows() if fitting else inputs(role)
    base = {} if fitting else loaded(role,"baseline")
    lock = {"siglip":digest(SIGLIP),"dino":digest(DINOV2_INFO),"source":digest(__file__),
            "inputs":digest(FIT if fitting else cache(role,"baseline")),
            "images":{r["id"]:r.get("derived_sha256",r["sha256"]) for r in rows}}
    path = feature_cache(role)
    saved = read_json(path) if path.exists() else dict(key=lock,frames={})
    if saved["key"] != lock:
        raise ValueError("Crop features were produced from different inputs.")
    encoder = None
    grounder = None
    for row in rows:
        key = row["id"]
        if key in saved["frames"]:
            continue
        if encoder is None:
            encoder = VisualSemantics()
            grounder = Grounder(Path("."),"dino") if fitting else None
        started = perf_counter()
        image = image_for(row,Path(".")) if "file" in row else source(row)
        if fitting:
            detections = {"floor":1.,"ranked":suppress(grounder.propose(image),.60)}
            candidates = local_example_boxes(row,detections)
        else:
            candidates = proposals(base[key])
        patches = [crop(image,expanded(p["bbox"],padding)) for padding in PADDINGS for p in candidates]
        vectors = {name:[] for name in ("siglip","dino","prompt")}
        for start in range(0,len(patches),8):
            for name,values in encoder.encode(patches[start:start+8]).items():
                vectors[name].extend(values.tolist())
        saved["frames"][key] = dict(candidates=candidates,values=vectors,ms=(perf_counter()-started)*1000)
        if len(saved["frames"]) % 4 == 0 or len(saved["frames"]) == len(rows):
            write_json(path,saved)
            print(role,len(saved["frames"]),"/",len(rows),flush=True)


def read_features(role):
    result = read_json(feature_cache(role))
    if result["key"]["source"] != digest(__file__) or set(result["frames"]) != set(result["key"]["images"]):
        raise ValueError("The crop feature cache is incomplete or obsolete.")
    return result["frames"]


def columns(frame, name):
    count = len(frame["candidates"])
    if not count:
        return np.empty((0,0))
    if name.startswith("semantic"):
        values = np.asarray(frame["values"]["prompt"]).reshape(len(PADDINGS),count,-1)
        margin = ((values[:,:,0]-values[:,:,1]) if name == "semantic-best" else
                  (values[:,:,2]-values[:,:,3]))
        return margin.max(axis=0)
    keys = ("siglip","dino") if name == "joint" else (name,)
    features = []
    for key in keys:
        values = np.asarray(frame["values"][key]).reshape(len(PADDINGS),count,-1)
        features.append(values.mean(axis=0))
    return np.column_stack(features)


def train(fit, name, strength):
    examples = [(key,p["label"]) for key,frame in fit.items() for p in frame["candidates"]]
    counts = Counter(examples)
    groups = {r["id"]:r["group"] for r in fit_rows()}
    group_sizes = Counter({group:sum(n for (key,_),n in counts.items() if groups[key]==group)
                           for group in set(groups.values())})
    labels = np.array([label for _,label in examples])
    weights = np.array([1/group_sizes[groups[key]] for key,_ in examples])
    for state in (True,False):
        weights[labels==state] *= len(weights)/(2*weights[labels==state].sum())
    x = np.vstack([columns(frame,name) for frame in fit.values() if frame["candidates"]])
    learner = LogisticRegression(C=strength, random_state=1761).fit(x,labels,sample_weight=weights)
    return dict(weights=learner.coef_[0].tolist(),bias=float(learner.intercept_[0]))


def scores(frame, name, head):
    value = columns(frame,name)
    return value if head is None or not value.size else value @ np.asarray(head["weights"]) + head["bias"]


def adjust(ranked, frame, value, bounds, floor):
    add, reject, previous = bounds
    retained = [p for p in ranked if p["score"] >= floor]
    proposed = []
    rejected = {tuple(p["bbox"]) for p,s in zip(frame["candidates"],value,strict=True)
                if p.get("baseline_score",p["score"]) >= floor and s < reject}
    retained = [p for p in retained if tuple(p["bbox"]) not in rejected]
    for p,s in zip(frame["candidates"],value,strict=True):
        if (p.get("baseline_score",p["score"]) >= floor or p["score"] < previous or s < add
            or any(box_iou(p["bbox"],q["bbox"]) >= .42 or overlap_fraction(p["bbox"],q["bbox"]) >= .80
                   for q in (*retained,*proposed))):
            continue
        proposed.append({"score":floor,"bbox":p["bbox"],"semantic_margin":float(s)})
    return [*retained,*proposed,*(p for p in ranked if p["score"]<floor)]


def compare_variants(include_fit):
    floor = read_json(FLOOR)["cutoff"]
    references = {role:inputs(role) for role in DEVELOPMENT}
    originals = {role:{key:f["ranked"] for key,f in loaded(role,"baseline").items()} for role in DEVELOPMENT}
    matrices = {role:read_features(role) for role in DEVELOPMENT}
    before = {role:evaluate(references[role],originals[role],floor) for role in DEVELOPMENT}
    alternatives = {name:None for name in ("semantic-best","semantic-mean")}
    if include_fit:
        fit = read_features("fit")
        for name, strength in product(("siglip","dino","joint"),(.1,2.,20.)):
            alternatives[f"{name}/{strength}"] = train(fit,name,strength)
    trials = []
    for variant,head in alternatives.items():
        name = variant.split("/")[0]
        model_scores = {role:{key:scores(frame,name,head) for key,frame in matrices[role].items()}
                        for role in DEVELOPMENT}
        cutoffs = (-.03, -.015, -.005, .0, .006, .014, .022, .035, .05) if head is None else (
                   -1.0,-.5,-.15,0.,.15,.35,.65,1.,1.5,2.2,3.)
        thresholds = (*cutoffs, 1e4)
        for add,reject,prior in product(thresholds,(-1e4,*cutoffs),(.23,.33,.43,.53,.63)):
            if reject > add:
                continue
            bounds = (add,reject,prior)
            metrics = {}
            for role in DEVELOPMENT:
                predictions = {key:adjust(ranked,matrices[role][key],model_scores[role][key],bounds,floor)
                               for key,ranked in originals[role].items()}
                measure = minimal(references[role],predictions,floor)
                metrics[role] = {**{k:measure[k] for k in ("tp","fp","fn")},
                                 "guard":compare(before[role],measure),"extra_fp":measure["fp"]-before[role]["fp"]}
            trials.append(dict(variant=variant,bounds=bounds,roles=metrics))
    def objective(trial):
        public = trial["roles"]["public-development"]
        clinic = trial["roles"]["phlebotomy-development"]
        return (public["guard"]["recovered_matches"],-public["extra_fp"],
                clinic["guard"]["recovered_matches"],-clinic["extra_fp"], trial["bounds"][1] == -1e4)

    feasible = [t for t in trials if all(result["guard"]["passed"] and result["extra_fp"]<=0
                                         for result in t["roles"].values())]
    choice = max(feasible,key=objective,default=None)
    leaders = {name:max((t for t in feasible if t["variant"]==name),key=objective,default=None)
               for name in alternatives}
    path = DECISION if include_fit else ART / "crop_semantics_zero_shot.json"
    write_json(path,dict(baselines={name:short(m) for name,m in before.items()},trial_count=len(trials),
                         decision=choice,leaders=leaders,heads=alternatives,
                         feature_sha256={role:digest(feature_cache(role)) for role in (*DEVELOPMENT,*(('fit',) if include_fit else ()))},
                         trials=trials))
    print("Crop semantics", len(trials),"variants; best",choice,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("collect","zero-shot","compare"))
    parser.add_argument("--role",choices=(*DEVELOPMENT,"fit"))
    args = parser.parse_args()
    if args.action == "collect":
        if not args.role:
            parser.error("collect requires --role")
        collector(args.role)
    else:
        compare_variants(args.action == "compare")


if __name__ == "__main__":
    main()
