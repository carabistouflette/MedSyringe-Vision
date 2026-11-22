"""Compare crop verification models for RF-DETR additions only."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
from itertools import product
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
import torch
from torchvision.transforms.functional import normalize, pil_to_tensor

from medsyringe.measurement.attributes import crop
from medsyringe.data import author_key, dhash, digest, leakage, read_json, train_rows, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from medsyringe.detection.verification import BACKBONE, CropEncoder, classifier_scores
from scripts.experiments.detector_training.optimize_detector import FINAL, counters
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.transfer.rfdetr_compare import (EXTERNAL, SELECTION, auxiliary, baseline_predictions,
                                    combine, manifest_paths, public_calibration_rows)
from scripts.experiments.transfer.rfdetr_transfer import ARCHIVE, ART, AUDIT, MANIFEST, WORK, polygons, public_clip
from scripts.experiments.transfer.supervised_study import sets

PENS = ART / "rfdetr_pen_manifest.json"
PUBLIC = ART / "rfdetr_syringe_reserve.json"
RESERVE = ART / "rfdetr_verifier_reserve.json"
DECISION = ART / "rfdetr_verifier_selection.json"
DINOV2 = Path("data/models/dinov2-small")
DINOV2_INFO = ART / "auxiliary_crop_backbone.json"
SEED = 8040


def known_rows():
    training, validation, historical = sets()
    rows = [*training, *validation, *historical, *read_json(MANIFEST)["rows"],
            *read_json(AUDIT)["rows"]]
    for name in (EXTERNAL, ART / "supervised_external_manifest.json"):
        rows.extend(read_json(name)["rows"])
    return rows


def prepare_reserve():
    public = read_json(MANIFEST)
    if digest(ARCHIVE) != public["archive_sha256"]:
        raise ValueError("Zenodo archive checksum mismatch")
    rows = known_rows()
    used_clips = {public_clip(row["member"]) for row in rows if "member" in row and
                  re.match(r"frame_\d+_", PurePosixPath(row["member"]).name)}
    dhashes = [int(row["dhash"], 16) for row in rows if row.get("dhash")]
    parents, seen, excluded = set(), [], Counter()
    folder = WORK / "verifier-reserve"
    folder.mkdir(exist_ok=True)
    with ZipFile(ARCHIVE) as archive:
        members = set(archive.namelist())
        ordered = sorted(members, key=lambda name: (name.split("/")[0] != "test",
                         sha256(f"{SEED}:{name}".encode()).hexdigest()))
        for name in ordered:
            path = PurePosixPath(name)
            if path.parent.name != "images" or path.suffix.lower() != ".jpg":
                continue
            try:
                clip = public_clip(path)
            except ValueError:
                continue
            if clip in used_clips:
                continue
            parent = path.stem.split(".rf.")[0]
            if parent in parents:
                excluded["same_source_frame"] += 1
                continue
            label = str(path.parent.parent / "labels" / (path.stem + ".txt"))
            if label not in members:
                continue
            contents = archive.read(name)
            with Image.open(BytesIO(contents)) as image:
                visual = dhash(image)
                width, height = image.size
            if any((int(visual,16)^previous).bit_count() <= 4 for previous in dhashes):
                excluded["near_hash"] += 1
                continue
            checksum = sha256(contents).hexdigest()
            identifier = f"zenodo-verifier-{checksum[:16]}"
            target = folder / (identifier+".jpg")
            target.write_bytes(contents)
            objects = list(polygons(archive.read(label).decode(), public["class_id"]))
            seen.append(dict(id=identifier, group="zenodo-clip-"+clip, split="verifier-reserve",
                             image=str(target.relative_to("data")), member=name, license="CC BY 4.0",
                             width=width, height=height, sha256=checksum, dhash=visual, objects=objects))
            dhashes.append(int(visual,16))
            parents.add(parent)
    if leakage([*[{**r, "split": "consulted"} for r in rows], *seen]):
        raise ValueError("Reserved images collide with earlier experiments")
    write_json(RESERVE, dict(rows=seen, skipped=dict(excluded), archive_sha256=public["archive_sha256"],
                            consulted_audit_sha256=digest(AUDIT), seed=SEED))
    print(len(seen), "unseen frames", sum(len(r["objects"]) for r in seen), "boxes",
          len({r["group"] for r in seen}), "clips", flush=True)


class DinoEncoder:
    def __init__(self):
        from transformers import Dinov2Model

        lock = read_json(DINOV2_INFO)
        if {name: digest(DINOV2/name) for name in lock["files"]} != lock["files"]:
            raise ValueError("DINOv2 checkpoint checksum mismatch")
        self.model = Dinov2Model.from_pretrained(DINOV2, local_files_only=True).eval()

    @torch.inference_mode()
    def encode(self, images):
        if not images:
            return np.zeros((0,384), dtype=np.float32)
        tensors = [normalize(pil_to_tensor(ImageOps.pad(im, (168,168), color=(124,116,104))).float()/255,
                             (.485,.456,.406), (.229,.224,.225)) for im in images]
        batches = torch.stack(tensors).split(16)
        features = [self.model(pixel_values=batch).last_hidden_state[:,0] for batch in batches]
        return torch.nn.functional.normalize(torch.cat(features)).numpy()


def expanded(box, pad):
    l, t, r, b = box
    x, y = pad*(r-l), pad*(b-t)
    return [max(l-x,0), max(t-y,0), min(r+x,1), min(b+y,1)]


def windows(encoder, rows, boxes, padding):
    records = []
    for row in rows:
        image_boxes = boxes.get(row["id"], [])
        if image_boxes:
            im = source(row)
            records.extend((row["id"], crop(im, expanded(bbox, padding))) for bbox in image_boxes)
    features = {}
    for start in range(0, len(records), 16):
        chunk = records[start:start+16]
        encodings = encoder.encode([patch for _, patch in chunk])
        for (key, _), vector in zip(chunk, encodings, strict=True):
            features.setdefault(key, []).append(vector)
    return {key: np.asarray(vectors) for key, vectors in features.items()}


def inputs(cohort):
    if cohort == "calibration":
        return [*read_json(manifest_paths()["calibration"])["rows"], *public_calibration_rows()]
    if cohort in ("pen_validation", "pen_reserve"):
        return [r for r in read_json(PENS)["rows"] if r["split"] == cohort]
    if cohort == "fresh":
        return read_json(RESERVE)["rows"]
    if cohort == "syringe_reserve":
        return read_json(PUBLIC)["rows"]
    if cohort == "previous_zenodo":
        return read_json(AUDIT)["rows"]
    if cohort == "fil_rouge":
        return train_rows(Path("."))
    return read_json(manifest_paths()[cohort])["rows"]


PARTITIONS = dict(fil_rouge="train", historical="test", previous_external="additive-audit",
                  previous_zenodo="rfdetr-reserve", fresh="rfdetr-verifier-reserve",
                  pen_reserve="rfdetr-verifier-pens", syringe_reserve="rfdetr-verifier-commons",
                  pen_validation="rfdetr-verifier-pen-cal")
AUDITS = ("previous_zenodo", "previous_external", "fil_rouge", "historical",
          "fresh", "pen_reserve", "syringe_reserve")


def baseline_for(cohort, rows):
    if cohort == "calibration":
        project, public = [], []
        for row in rows:
            (public if row["id"].startswith("zenodo-") else project).append(row)
        return {**baseline_predictions(project,"precision-fresh"),
                **baseline_predictions(public,"rfdetr-public-validation")}
    return baseline_predictions(rows, PARTITIONS[cohort])


def model_proposals(cohort, rows):
    locked = read_json(SELECTION)["locked"]
    run, view = locked["run"], locked["view"]
    caches = {"previous_zenodo":"fresh"}
    cached = caches.get(cohort,cohort)
    views = ("full", "full_and_tiles") if cohort=="calibration" else (view,)
    return auxiliary(rows, cached, run, views)["predictions"][view]


def ensure_pens():
    rows = read_json(PENS)["rows"]
    reserve = read_json(PUBLIC)["rows"]
    if leakage([*[{**r, "split":"consulted"} for r in known_rows()], *rows, *reserve]):
        raise ValueError("A pen photo shares a source group with another split")
    if any(r["objects"] or not r["pens"] or not r.get("author") for r in rows):
        raise ValueError("Pen scenes require verified negative labels and known provenance")
    return rows


def examples():
    from scripts.experiments.transfer.rfdetr_transfer import prepare_public
    data, _ = prepare_public()
    old, _, _ = sets()
    # The original box set complements the tiny Zenodo objects. Limit repeated
    # scenes using the same burst representatives as the adaptation experiment.
    ids = set(read_json(ART/"rfdetr_adapt_data.json")["train"]["ids"])
    positives = [r for r in [*data["train"], *old] if r["objects"] and
                 (r["id"].startswith("zenodo-") or r["id"] in ids)]
    negatives = [r for r in ensure_pens() if r["split"] == "pen_training"]
    return positives, negatives


def crop_examples(encoder, positive, negative, padding):
    boxes = {r["id"]: [o["bbox"] for o in r["objects"]] for r in positive}
    boxes.update({r["id"]: r["pens"] for r in negative})
    result = windows(encoder,[*positive,*negative],boxes,padding)
    features, targets, weights = [], [], []
    for rows, label in ((positive,1),(negative,0)):
        for row in rows:
            values = result[row["id"]]
            features.extend(values)
            targets.extend([label]*len(values))
            weights.extend([1/(len(values)*len(rows))]*len(values))
    return np.asarray(features), targets, np.asarray(weights)*len(features)


def extras(predictions, floor):
    return {k:[p for p in proposals if p.get("source")=="auxiliary" and p["score"]>=floor]
            for k,proposals in predictions.items()}


def apply(predictions, embeddings, head):
    out={}
    for key, proposals in predictions.items():
        values = iter(classifier_scores(embeddings[key],head["weights"],head["bias"])) if key in embeddings else iter(())
        out[key] = [p for p in proposals if p.get("source") != "auxiliary" or
                    next(values) >= head["cutoff"]]
    return out


def get_encoder(name):
    return (DinoEncoder() if name == "dino" else
            CropEncoder(Path("."),read_json(ART/"precision_candidate.json")["backbone_sha256"]))


def fit():
    ensure_pens()
    positives, negatives=examples()
    rows=inputs("calibration")
    base=baseline_for("calibration",rows)
    rf=model_proposals("calibration",rows)
    cutoff=read_json(FINAL)["cutoff"]
    locked=read_json(SELECTION)["locked"]
    hybrid=combine(base,rf,cutoff,locked["options"])
    reference=evaluate(rows,hybrid,cutoff)
    added=extras(hybrid,cutoff)
    proposal_boxes={k:[p["bbox"] for p in v] for k,v in added.items()}
    pen_rows=inputs("pen_validation")
    pen_boxes={r["id"]:r["pens"] for r in pen_rows}
    trials, candidates, controls = [], [], {}
    for encoder_name in ("mobile","dino"):
        encoder=get_encoder(encoder_name)
        for padding in (0.,.18):
            train_X,y,w=crop_examples(encoder,positives,negatives,padding)
            candidate_features=windows(encoder,rows,proposal_boxes,padding)
            distractor_features=windows(encoder,pen_rows,pen_boxes,padding)
            if encoder_name=="mobile" and padding==0:
                old=read_json(ART/"precision_head.json")
                old_stats=evaluate(rows,apply(hybrid,candidate_features,old),cutoff)
                penalties=[m for feature in distractor_features.values()
                           for m in classifier_scores(feature,old["weights"],old["bias"])]
                controls["prior_verifier"]=dict(hybrid=counters(old_stats),guard=compare(reference,old_stats),
                                                rejected_pen_crops=sum(float(x)<old["cutoff"] for x in penalties),
                                                pen_crops=len(penalties))
            for regularization in (1.,10.):
                fit_model=LogisticRegression(C=regularization,random_state=SEED)
                fit_model.fit(train_X,y,sample_weight=w)
                head=dict(weights=fit_model.coef_[0].tolist(),bias=float(fit_model.intercept_[0]))
                margins={k:classifier_scores(v,head["weights"],head["bias"]) for k,v in candidate_features.items()}
                pens=np.concatenate([classifier_scores(v,head["weights"],head["bias"])
                                     for v in distractor_features.values()])
                values=sorted({float(x) for v in margins.values() for x in v}|set(pens.tolist()))
                for threshold in (values[0]-1,*[(a+b)/2 for a,b in zip(values,values[1:])],values[-1]+1):
                    prototype=dict(**head,cutoff=threshold)
                    score=evaluate(rows,apply(hybrid,candidate_features,prototype),cutoff)
                    guard=compare(reference,score)
                    total=int((pens<threshold).sum())
                    trial=dict(encoder=encoder_name,padding=padding,C=regularization,cutoff=threshold,
                               rejected_pen_crops=total,pen_crops=len(pens),hybrid=counters(score),guard=guard)
                    trials.append(trial)
                    if guard["passed"] and total:
                        candidates.append((trial,prototype))
    def rank(item):
        trial,_=item
        return (trial["rejected_pen_crops"],-trial["hybrid"]["fp"],
                trial["encoder"]=="mobile",trial["padding"]==0, -trial["C"])
    if not candidates:
        raise ValueError("No pen rejection configuration preserved the calibration recoveries")
    choice,head=max(candidates,key=rank)
    write_json(DECISION,dict(choice=choice,head=head,trials=trials,controls=controls,calibration=counters(reference),
                           seed=SEED,positive_ids=[r["id"] for r in positives],
                           negative_ids=[r["id"] for r in negatives],data_hashes=data_hashes()))
    print("Selected",choice,flush=True)


def data_hashes():
    paths=(PENS,RESERVE,PUBLIC,DINOV2_INFO,BACKBONE,SELECTION,FINAL,Path(__file__),
           ART/"precision_head.json")
    return {str(p):digest(p) for p in paths}


def audit(cohort):
    record=read_json(DECISION)
    if record["data_hashes"] != data_hashes():
        raise ValueError("Locked verifier inputs changed")
    rows=inputs(cohort)
    base=baseline_for(cohort,rows)
    rf=model_proposals(cohort,rows)
    floor=read_json(FINAL)["cutoff"]
    hybrid=combine(base,rf,floor,read_json(SELECTION)["locked"]["options"])
    added=extras(hybrid,floor)
    encoder=get_encoder(record["choice"]["encoder"])
    embeddings=windows(encoder,rows,{k:[p["bbox"] for p in v] for k,v in added.items()},
                       record["choice"]["padding"])
    checked=apply(hybrid,embeddings,record["head"])
    base_stats=evaluate(rows,base,floor)
    hybrid_stats=evaluate(rows,hybrid,floor)
    verified_stats=evaluate(rows,checked,floor)
    control=dict(baseline=base_stats,hybrid=hybrid_stats,verified=verified_stats,
                 guard=compare(base_stats,verified_stats),extra_guard=compare(hybrid_stats,verified_stats))
    if cohort == "pen_reserve":
        pen_features=windows(encoder,rows,{r["id"]:r["pens"] for r in rows},record["choice"]["padding"])
        scores=[float(m) for v in pen_features.values() for m in
                classifier_scores(v,record["head"]["weights"],record["head"]["bias"])]
        control["verified_negative_crops"]=dict(rejected=sum(m<record["head"]["cutoff"] for m in scores),
                                                total=len(scores))
    write_json(ART/f"rfdetr_verifier_{cohort}.json",dict(**control,protocol_sha256=digest(DECISION)))
    print(cohort,*(counters(s) for s in (base_stats,hybrid_stats,verified_stats)),flush=True)


def summarize():
    results = {}
    for cohort in AUDITS:
        evidence = read_json(ART/f"rfdetr_verifier_{cohort}.json")
        if evidence["protocol_sha256"] != digest(DECISION):
            raise ValueError("Verifier audit came from a different protocol")
        results[cohort] = {k: counters(evidence[k]) for k in ("baseline","hybrid","verified")}
        results[cohort].update({k:evidence[k] for k in ("guard","extra_guard")})
    preserving = all(e["guard"]["passed"] and e["extra_guard"]["passed"] for e in results.values())
    no_extra_fp = all(e["verified"]["fp"]<=e["baseline"]["fp"] for e in results.values())
    fresh_recovery = any(results[k]["verified"]["tp"]>results[k]["baseline"]["tp"]
                         for k in ("fresh","syringe_reserve"))
    write_json(ART/"rfdetr_verifier_release_decision.json",dict(
        eligible_for_release=preserving and no_extra_fp and fresh_recovery,
        preserved_all_matches=preserving, no_extra_false_positives=no_extra_fp,
        fresh_recovery=fresh_recovery, cohorts=results, protocol_sha256=digest(DECISION)))
    print("Verification gate", preserving, no_extra_fp, fresh_recovery, flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare-reserve","fit","base","audit","summarize"])
    parser.add_argument("--cohort",choices=AUDITS)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "prepare-reserve":
        prepare_reserve()
    elif args.stage == "fit":
        fit()
    elif args.stage == "summarize":
        summarize()
    elif not args.cohort:
        parser.error("A cohort is required")
    elif args.stage == "base":
        baseline_for(args.cohort,inputs(args.cohort))
    else:
        audit(args.cohort)


if __name__=="__main__":
    main()
