"""Cache fixed and targeted crop evidence for the same experimental recoveries."""
import argparse
from collections import Counter
from itertools import chain
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import ImageOps
from transformers import AutoModel, AutoProcessor

from medsyringe.runtime.appearance_runtime import RELEASE as APPEARANCE
from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.runtime.pipeline_extensions import verified_config
from scripts.experiments.detector_audits.detection_crop_semantics import LOOKALIKES, MODEL, PADDINGS, SYRINGES, lock_weights
from scripts.experiments.crowds.native_candidate_verifier import reference_index
from scripts.experiments.occlusion.occlusion_detector import raster
from scripts.experiments.ranking.overlap_candidate_features import (
    annotate_embedding, candidate_pool, inputs as cached_inputs, load,
)
from scripts.experiments.ranking.overlap_ranking import LOCK as UPSTREAM, RESULT as EXPOSED_RESULT, predict_evidence, select, sealed
from scripts.experiments.transfer.rfdetr_verifier import DINOV2_INFO, DinoEncoder, expanded
from scripts.experiments.expanded.sam3_study import configure_cpu
from scripts.experiments.identity.syringe_object_references import REFERENCE, WORK, references

TARGET = "syringe"
CLASSES = (TARGET, "pipette", "tube", "needle", "other")
ROUTES = ("calibrated", "active")
PROMPTS = {
    "syringe": (
        "a medical syringe barrel and a push-pull plunger joined together",
        "a transparent plastic hypodermic syringe with its rubber piston",
        "a medical syringe with fingers covering part of the barrel",
        "a dental cartridge syringe with finger loops and a push rod",
        "a prefilled syringe with visible graduated barrel and needle cap",
    ),
    "pipette": (
        "a handheld adjustable laboratory micropipette with a removable plastic tip",
        "a scientific laboratory pipette with a thick handle and a narrow tip",
        "a graduated glass serological pipette without a sliding plunger",
        "a plastic transfer pipette with an integral squeezable bulb",
        "a laboratory pipette controller holding a graduated transfer tube",
    ),
    "tube": (
        "a blood specimen tube fitted inside a vacuum blood collection holder",
        "a sealed cylindrical blood collection tube with a colored cap",
        "a glass or plastic laboratory test tube with closed rounded bottom",
    ),
    "needle": (
        "a detached medical needle with a colored hub, separate from a syringe",
        "an exposed needle or cannula connected to medical tubing only",
        "a set of standalone capped hypodermic needles",
    ),
    "other": (
        "a dental mirror, scaler, drill or probe with a metal handle",
        "a specimen vial or a small glass medicine bottle",
        "a medical pen-like instrument or plastic packaging",
        "only part of a medical syringe, a loose cap or a disconnected rod",
    ),
}


def family(label):
    text = label.casefold()
    if "pipette" in text or "pipettor" in text:
        return "pipette"
    if "tube" in text or "vacutainer" in text:
        return "tube"
    if "needle" in text or "cannula" in text:
        return "needle"
    return "other"


class CropSemantics:
    """One local SigLIP 2 model with both unchanged release and higher-specificity prompts."""

    def __init__(self):
        lock_weights()
        self.processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(MODEL, local_files_only=True, use_safetensors=True).eval()
        self.groups = (len(SYRINGES), len(LOOKALIKES), *(len(PROMPTS[k]) for k in CLASSES))
        texts = [*SYRINGES, *LOOKALIKES, *chain.from_iterable(PROMPTS[k] for k in CLASSES)]
        tokens = self.processor(text=texts, padding="max_length", truncation=True, return_tensors="pt")
        with torch.inference_mode():
            self.text = torch.nn.functional.normalize(self.model.get_text_features(**tokens))

    @torch.inference_mode()
    def encode(self, crops):
        result = []
        for i in range(0, len(crops), 12):
            resized = [ImageOps.pad(patch, (224, 224), color=(124, 116, 104)) for patch in crops[i:i+12]]
            inputs = self.processor(images=resized, return_tensors="pt")
            image = torch.nn.functional.normalize(self.model.get_image_features(**inputs))
            logits = torch.split(image @ self.text.T, self.groups, dim=1)
            result.append(torch.stack([t.max(dim=1).values for t in logits], dim=1).numpy())
        return np.concatenate(result, axis=0)


def class_for_proposal(p, row):
    box = p["bbox"]
    if reference_index(box, row["objects"]) is not None:
        return TARGET
    if max((box_iou(box,o["bbox"]) for o in row["objects"]), default=0.) >= .12:
        return None
    items = row.get("lookalikes", [])
    best = max(items, key=lambda q: box_iou(box,q["bbox"]), default=None)
    if best and box_iou(box,best["bbox"]) >= .35:
        return family(best["kind"])
    return "other"


def fit_samples(row, proposed):
    """Cap repeated candidates per source. Keep clean typed annotations as anchor examples."""
    manual = [dict(bbox=o["bbox"], label=family(o["kind"]), origin="annotated-lookalike", manual=True)
              for o in row.get("lookalikes", [])]
    boxes = [dict(bbox=o["bbox"], label=TARGET, origin="annotated-syringe", manual=True)
             for o in row["objects"]]
    if len(boxes) > 4:
        boxes = boxes[:4]
    seen = Counter()
    for p in proposed:
        group = class_for_proposal(p, row)
        if group is None or p["score"] < .12:
            continue
        target = reference_index(p["bbox"],row["objects"]) if group==TARGET else None
        index = (group, target)
        if seen[index] >= (1 if group == "other" else 2):
            continue
        seen[index] += 1
        boxes.append(dict(bbox=p["bbox"], label=group, origin="detector-proposal", features=p["features"]))
    return manual + boxes


def fitting():
    rows, base, frames = cached_inputs("fit")
    bank = load("fit", rows, base, frames)
    added = references("fit")
    requests = {r["id"]:fit_samples(r,bank.get(r["id"],{}).get("proposed",[])) for r in [*rows,*added]}
    return [*rows,*added], requests


def rules():
    upstream = sealed()["selection"]
    return {mode: upstream[key]["rank-learned"]["rule"]
            for mode,key in zip(ROUTES,("finalists","probes"),strict=True)}


def gather(role, encoder=None):
    if role == "development":
        rows, base, raw = cached_inputs(role)
        extracted = load(role, rows, base, raw)
    elif role == "fresh":
        from scripts.experiments.identity.syringe_object_reserve import evidence

        rows, base, raw, _ = evidence()
        extracted = {}
        for row in rows:
            key, image = row["id"], raster(row)
            original = [dict(bbox=p["bbox"], score=p["score"], corroboration=0., view="released") for p in base[key]]
            candidates = candidate_pool(raw[key])
            requested = [*original, *candidates]
            found = annotate_embedding(encoder, image, requested) if requested else []
            extracted[key] = dict(base=found[:len(original)], proposed=found[len(original):])
    else:
        raise ValueError(f"No new proposals for {role}.")
    ranked, relation = predict_evidence(rows,extracted,sealed()["heads"])
    routes = {role:{k:select(b,ranked[k],relation[k],r) for k,b in base.items()} for role,r in rules().items()}
    bank = {key:{tuple(v["bbox"]):v["features"] for v in p["proposed"]} for key,p in extracted.items()}
    return rows, base, requests_for(rows,routes,bank), routes


def requests_for(rows, routes, lookup):
    out = {}
    for row in rows:
        k = row["id"]
        dedup = {tuple(p["bbox"]):p for frame in routes.values() for p in frame[k]}
        out[k] = [dict(bbox=p["bbox"],features=lookup[k][box]) for box,p in dedup.items()]
    return out


def exposed():
    output = read_json(EXPOSED_RESULT)["cohorts"]
    rows, base, lookup = [], {}, {}
    additions = {k:{} for k in ROUTES}
    for cohort in ("fresh", "replay"):
        refs, initial, raw = cached_inputs(cohort)
        frames = load(cohort,refs,initial,raw)
        rows.extend(refs)
        base.update(initial)
        lookup.update({key:{tuple(v["bbox"]):v["features"] for v in p["proposed"]} for key,p in frames.items()})
        for route,arm in zip(ROUTES,("finalists/rank-learned","probes/rank-learned"),strict=True):
            a = output[cohort]["variants"][arm]["additions"]
            additions[route].update({r["id"]:a.get(r["id"],[]) for r in refs})
    return rows,base,requests_for(rows,additions,lookup),additions


def encode_samples(photo, samples, dino, semantics, release):
    if not samples:
        return []
    manual = [i for i,p in enumerate(samples) if p.get("manual")]
    features = {}
    if manual:
        encoded = annotate_embedding(dino,photo,[samples[i] for i in manual])
        features = {i:p["features"] for i,p in zip(manual,encoded,strict=True)}
    paddings = (.035,.24,*PADDINGS) if release else (.035,.24)
    views = [crop(photo,expanded(p["bbox"],pad)) for p in samples for pad in paddings]
    logits = semantics.encode(views).reshape(len(samples),len(paddings),-1)
    out = []
    for i,(p,x) in enumerate(zip(samples,logits,strict=True)):
        item = dict(p,features=features.get(i,p.get("features")),
                    targeted=np.round(x[:2,2:],6).tolist())
        if release:
            # The production rule uses the maximum across the two production crop sizes.
            item["release_margin"] = float(np.max(x[2:,0]-x[2:,1]))
        out.append(item)
    return out


def source_state(role, rows):
    state = dict(code=digest(__file__), reference=digest(REFERENCE), upstream=digest(UPSTREAM),
                dino=digest(DINOV2_INFO), release=digest(APPEARANCE),
                rows={r["id"]: r.get("derived_sha256", r["sha256"]) for r in rows}, role=role)
    if role == "fresh":
        from scripts.experiments.identity.syringe_object_reserve import PROPOSALS, RELEASE

        state["stage_outputs"] = {str(path): digest(path) for path in (PROPOSALS, RELEASE)}
    if role == "exposed":
        state["exposed_result"] = digest(EXPOSED_RESULT)
    return state


def collect(roles):
    configure_cpu()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    verified_config(Path.cwd(),APPEARANCE)
    dino, semantics = DinoEncoder(), CropSemantics()
    for role in roles:
        path = WORK/f"features-{role}.json"
        if path.exists():
            raise FileExistsError(path)
        if role=="fit":
            rows, requests = fitting()
            base, routes = {},{}
        elif role=="exposed":
            rows,base,requests,routes = exposed()
        else:
            rows,base,requests,routes = gather(role,dino)
        binding = source_state(role,rows)
        processed = {}
        for i,r in enumerate(rows,1):
            key=r["id"]
            processed[key]=encode_samples(raster(r),requests[key],dino,semantics,role!="fit")
            if i%20==0 or i==len(rows):
                print(role,i,len(rows),"crops",sum(map(len,processed.values())),flush=True)
        write_json(path,dict(binding=binding,frames=processed,base=base,routes=routes))


def feature_store(role,rows):
    packet=read_json(WORK/f"features-{role}.json")
    if packet["binding"]!=source_state(role,rows) or set(packet["frames"])!={r["id"] for r in rows}:
        raise ValueError(f"Missing or stale evidence for {role}.")
    return packet


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("roles",nargs="+",choices=("fit","development","fresh","exposed"))
    collect(cli.parse_args().roles)
