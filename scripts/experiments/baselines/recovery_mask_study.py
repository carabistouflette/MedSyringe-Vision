"""Verify weak proposals using text and box-conditioned SAM 3 on padded crops."""
import argparse
from itertools import product
from pathlib import Path

import torch
from PIL import ImageOps

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.detection.recall_guard import compare
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.expanded.expanded_proposals import ART, image_for, rows_for
from scripts.experiments.baselines.recovery_study import FLOOR_PATH, LOCK as PRIMARY, WORK, dataset, decide
from scripts.experiments.transfer.rfdetr_verifier import expanded
from scripts.experiments.expanded.sam3_study import Segmenter, configure_cpu, mask_geometry

LOCK = ART / "additive_mask_selection.json"
AUDIT = ART / "additive_mask_regression.json"


def bindings(split):
    files = (Path("scripts/experiments/baselines/recovery_mask_study.py"), Path("artifacts/sam3_checkpoint.json"), PRIMARY,
             WORK / f"{split}.json")
    return {str(path): digest(path) for path in files}


@torch.inference_mode()
def local_masks(segmenter, image, box):
    bounds = expanded(box, .26)
    x,y,r,b = bounds
    # Match the actual integer image crop used by attributes.crop.
    from math import ceil, floor

    l, t = floor(x*image.width), floor(y*image.height)
    rw, bh = ceil(r*image.width), ceil(b*image.height)
    x,y,r,b = l/image.width, t/image.height, rw/image.width, bh/image.height
    region = ImageOps.contain(crop(image, bounds), (1024,1024))
    ratios = [((box[i]-(x,y)[i%2])/(r-x,b-y)[i%2]) for i in range(4)]
    original_sizes = (*region.size, *region.size)
    prompt_box = [a*width for a,width in zip(ratios, original_sizes, strict=True)]
    comparisons = {}
    for method, prompt in (("text", {}), ("guided", {"input_boxes": [[prompt_box]], "input_boxes_labels": [[1]]})):
        inputs = segmenter.processor(images=region, text="syringe", **prompt, return_tensors="pt",
                    size={"height":segmenter.edge, "width":segmenter.edge}).to(segmenter.device)
        outputs = segmenter.model(**inputs)
        detections = segmenter.processor.post_process_instance_segmentation(outputs, threshold=.10)[0]
        best = None
        for score, mask in zip(detections["scores"].tolist(), detections["masks"], strict=True):
            mask_box, fill = mask_geometry(mask)
            if mask_box is None or not .06 <= fill <= .92:
                continue
            overlap = box_iou(mask_box, ratios)
            if overlap < .40:
                continue
            full_mask_box = [x+mask_box[0]*(r-x), y+mask_box[1]*(b-y),
                             x+mask_box[2]*(r-x), y+mask_box[3]*(b-y)]
            result = dict(score=score, mask_bbox=full_mask_box, overlap=overlap, mask_area=fill)
            if valid_box(full_mask_box) and (best is None or score > best["score"]):
                best = result
        comparisons[method] = best
    return comparisons


def collect(source, split):
    if split != "calibration" and not LOCK.exists():
        raise ValueError("Freeze the mask policy before collection on regression groups.")
    target = WORK / f"masks-{split}.json"
    key = bindings(split)
    payload = read_json(target) if target.exists() else dict(inputs=key, cases={})
    if payload["inputs"] != key:
        raise ValueError("The proposal-guided mask inputs changed.")
    selections = read_json(PRIMARY)
    policy, heads = selections["shadow"]["policy"], selections["models"]
    original = dataset(split)
    rows = {r["id"]: r for r in rows_for(split)}
    floor = read_json(FLOOR_PATH)["cutoff"]
    eligible = {}
    for identifier, frame in original["frames"].items():
        result, proposal = decide(frame, policy, heads, floor, shadow=split == "calibration")
        if proposal:
            eligible[identifier] = result[0]
    if set(eligible) == set(payload["cases"]):
        if not target.exists():
            write_json(target, payload)
        return
    model = Segmenter(504, "xpu")
    for identifier, proposal in eligible.items():
        if identifier in payload["cases"]:
            continue
        image = image_for(rows[identifier], source)
        payload["cases"][identifier] = local_masks(model, image, proposal["bbox"])
        if len(payload["cases"]) % 4 == 0 or len(payload["cases"]) == len(eligible):
            write_json(target, payload)
            print(split, len(payload["cases"]), "/", len(eligible), "mask pairs", flush=True)


def verified(cases, settings):
    if cases is None:
        return None
    required = ("text","guided") if settings["method"] == "agreement" else (settings["method"],)
    matches = [cases[method] for method in required]
    if any(not p or p["score"] < settings["confidence"] for p in matches):
        return None
    if len(matches) == 2 and box_iou(matches[0]["mask_bbox"], matches[1]["mask_bbox"]) < .58:
        return None
    return max(matches, key=lambda p: p["score"])["mask_bbox"]


def score(split, settings, shadow):
    data = dataset(split)
    masks = read_json(WORK / f"masks-{split}.json")
    if masks["inputs"] != bindings(split):
        raise ValueError("The mask cache changed since extraction.")
    gate = read_json(PRIMARY)
    floor = read_json(FLOOR_PATH)["cutoff"]
    rows = data["references"]
    baseline, result, proposed, rejected = {}, {}, {}, {}
    for row in rows:
        identifier = row["id"]
        frame = data["frames"][identifier]
        baseline[identifier] = [] if shadow else frame["accepted"]
        result[identifier] = baseline[identifier]
        pred, candidate = decide(frame, gate["shadow"]["policy"], gate["models"], floor, shadow)
        box = verified(masks["cases"].get(identifier), settings)
        if not candidate:
            continue
        if box:
            accepted = {**pred[0], "bbox": box} if settings["box"] == "mask" else pred[0]
            result[identifier] = [accepted]
            proposed[identifier] = max((box_iou(accepted["bbox"], o["bbox"]) for o in row["objects"]), default=0.)
        else:
            rejected[identifier] = candidate["iou"]
    before, after = evaluate(rows, baseline, floor), evaluate(rows, result, floor)
    return dict(baseline=short(before), candidate=short(after), guard=compare(before, after),
                added_iou=proposed, rejected_iou=rejected)


def select():
    if LOCK.exists():
        raise ValueError("This mask study is already frozen.")
    trials = []
    for method, confidence, box in product(("text", "guided", "agreement"), (.35,.55,.70,.85), ("mask","proposal")):
        policy = dict(method=method, confidence=confidence, box=box)
        result = score("calibration", policy, True)
        trials.append(dict(policy=policy, tp=result["candidate"]["tp"], fp=result["candidate"]["fp"]))
    rank = lambda t: (-t["fp"], t["tp"], t["policy"]["method"] == "agreement", t["policy"]["confidence"])
    choice = max((t for t in trials if t["tp"] and not t["fp"]), key=rank, default=None)
    shadow = choice or max(trials, key=rank)
    write_json(LOCK, dict(selected=choice, shadow=shadow, trials=trials, inputs=bindings("calibration")))
    print("Mask recovery calibration", shadow, flush=True)


def audit():
    selected = read_json(LOCK)
    if selected["inputs"] != bindings("calibration"):
        raise ValueError("The frozen mask prerequisites changed.")
    scores = {split: score(split, selected["shadow"]["policy"], False) for split in ("test","reserve")}
    safe = all(s["guard"]["passed"] and s["candidate"]["fp"] <= s["baseline"]["fp"] for s in scores.values())
    improved = any(s["guard"]["recovered_matches"] > 0 for s in scores.values())
    write_json(AUDIT, dict(passed=bool(selected["selected"] and safe and improved), scores=scores,
                           selection_sha256=digest(LOCK), scope="Existing reserves, regression only."))
    for split, result in scores.items():
        print(split, result["candidate"], result["guard"], result["added_iou"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "select", "audit"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--split", choices=("calibration", "test", "reserve"))
    args = parser.parse_args()
    configure_cpu()
    if args.stage == "collect":
        if not args.source or not args.split:
            parser.error("Collection requires --source and --split.")
        collect(args.source, args.split)
    else:
        (select if args.stage == "select" else audit)()
