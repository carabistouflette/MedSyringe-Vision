"""Use SAM 3 visual prompts to localize existing detections on development data."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

from PIL import ImageOps

from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.detection_crop_semantics import feature_cache, read_features
from scripts.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, cache, inputs, loaded
from scripts.detection_precision_confirmation import LOCK as APPEARANCE
from scripts.detection_semantics_audit import apply_rules
from scripts.precision_study import source
from scripts.rfdetr_verifier import expanded
from scripts.sam3_study import LOCK, REVISION, WEIGHTS, configure_cpu, mask_geometry

OUTPUT = ART / "guided_geometry_development.json"
PADDINGS = (0., .15)
EDGE = 504


class GuidedMasks:
    def __init__(self):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor, Sam3VideoConfig

        manifest = read_json(LOCK)
        if manifest["revision"] != REVISION or any(digest(WEIGHTS / k) != h for k, h in manifest["files"].items()):
            raise ValueError("SAM 3 assets differ from the official locked checkpoint.")
        config = Sam3VideoConfig.from_pretrained(WEIGHTS, local_files_only=True)
        config.image_size = EDGE
        configured = config.tracker_config
        configured.prompt_encoder_config.image_size = EDGE
        configured.vision_config.backbone_feature_sizes = [[EDGE * n // 14] * 2 for n in (4, 2, 1)]
        model, loading = Sam3TrackerModel.from_pretrained(
            WEIGHTS, config=config, local_files_only=True, attn_implementation="sdpa", output_loading_info=True)
        if loading["missing_keys"] or loading.get("mismatched_keys"):
            raise ValueError(f"Uninitialized promptable segmentation weights: {loading['missing_keys']}")
        self.model = model.eval().to("xpu")
        self.processor = Sam3TrackerProcessor.from_pretrained(
            WEIGHTS, target_size=EDGE, local_files_only=True)
        self.torch = torch

    def predict(self, image, boxes):
        picture = ImageOps.contain(image, (EDGE, EDGE))
        native = [picture.width, picture.height] * 2
        prompts = [[q * n for q, n in zip(expanded(box, pad), native, strict=True)]
                   for pad in PADDINGS for box in boxes]
        inputs = self.processor(images=picture, input_boxes=[prompts], return_tensors="pt",
                                size={"height": EDGE, "width": EDGE})
        pixels = inputs["pixel_values"].to(device="xpu", dtype=self.model.dtype)
        with self.torch.inference_mode():
            predicted = self.model(pixel_values=pixels, input_boxes=inputs["input_boxes"].to("xpu"),
                                   multimask_output=True)
            scores = predicted.iou_scores.cpu()
            masks = self.processor.post_process_masks(predicted.pred_masks.cpu(), inputs["original_sizes"])[0]
        found = [dict(mask_bbox=box, coverage=coverage, quality=float(score))
                 for hypotheses, values in zip(masks, scores[0], strict=True)
                 for score, mask in [max(zip(values, hypotheses, strict=True), key=lambda pair: pair[0])]
                 for box, coverage in [mask_geometry(mask)]]
        return [found[i:i+len(boxes)] for i in range(0, len(found), len(boxes))]


def references(scope):
    ranked = loaded(scope, "baseline")
    regions = read_features(scope)
    policy = read_json(APPEARANCE)["components"]
    return {
        k: [p for p in apply_rules(row, regions[k], policy)[0] if p["score"] >= row["floor"]]
        for k, row in ranked.items()
    }


def path(scope):
    return DATA / "proposals" / f"guided-geometry-{scope}.json"


def binding(scope):
    files = [Path(__file__), LOCK, APPEARANCE, cache(scope, "baseline"), feature_cache(scope)]
    return {str(p): digest(p) for p in files}


def collect(scope, limit):
    configure_cpu()
    before = references(scope)
    key = binding(scope)
    dest = path(scope)
    record = read_json(dest) if dest.exists() else dict(key=key, frames={})
    if record["key"] != key:
        raise ValueError("The guided-mask checkpoint differs from the development inputs.")
    rows = [r for r in inputs(scope) if r["id"] not in record["frames"]]
    model = None
    for row in rows[:limit]:
        name = row["id"]
        original = before[name]
        seconds = 0.
        geometry = []
        if original:
            if model is None:
                model = GuidedMasks()
            image = source(row)
            start = perf_counter()
            geometry = model.predict(image, [p["bbox"] for p in original])
            seconds = perf_counter() - start
        record["frames"][name] = dict(masks=geometry, seconds=seconds)
        write_json(dest, record)
        print("Guided geometry", scope, len(record["frames"]), "/", len(before), flush=True)


def refine(prior, alternatives, policy):
    if not prior:
        return prior
    quality, overlap, blend, which = policy
    selected = []
    for original, mask in zip(prior, alternatives[which], strict=True):
        old = original["bbox"]
        new = mask["mask_bbox"]
        if new is None or mask["quality"] < quality or box_iou(old, new) < overlap:
            selected.append(original)
        else:
            selected.append({**original, "bbox": [(1-blend)*a+blend*b for a, b in zip(old, new, strict=True)]})
    return selected


def measure():
    if OUTPUT.exists():
        raise ValueError("The development geometry comparison is already measured.")
    comparison = {}
    for scope in DEVELOPMENT:
        rows, old = inputs(scope), references(scope)
        present = read_json(path(scope))
        if present["key"] != binding(scope) or set(present["frames"]) != set(old):
            raise ValueError("Run every development input under the same model configuration first.")
        comparison[scope] = (rows, old, present["frames"], evaluate(rows, old, 0.))
    safe = []
    parameters = list(product((.65, .8, .9), (.45, .65, .8), (.25, .5, 1.), range(len(PADDINGS))))
    for choice in parameters:
        current = {}
        for scope, (rows, old, features, initial) in comparison.items():
            predictions = {k: refine(old[k], f["masks"], choice) for k, f in features.items()}
            candidate = evaluate(rows, predictions, 0.)
            guard = compare(initial, candidate)
            if not guard["passed"] or candidate["fp"] > initial["fp"]:
                break
            current[scope] = dict(tp=candidate["tp"], fp=candidate["fp"], fn=candidate["fn"],
                                  ap75=candidate["ap_by_iou"]["0.75"], map=candidate["ap50_95"], guard=guard)
        if len(current) == len(comparison):
            safe.append(dict(choice=choice, scopes=current))
    cross_domains = {}
    for scope, (_, _, frames, initial) in comparison.items():
        times = [f["seconds"] for f in frames.values() if f["masks"]]
        cross_domains[scope] = dict(tp=initial["tp"], fp=initial["fp"], fn=initial["fn"],
                                    ap75=initial["ap_by_iou"]["0.75"], map=initial["ap50_95"],
                                    median_mask_seconds=median(times) if times else None)
    preferred = sorted(safe, key=lambda trial: (
        sum(v["tp"] for v in trial["scopes"].values()),
        min(v["map"] - cross_domains[k]["map"] for k, v in trial["scopes"].items()),
        sum(v["map"] for v in trial["scopes"].values())), reverse=True)
    write_json(OUTPUT, dict(baseline=cross_domains, trials=len(parameters), qualified_on_development=len(safe),
                            settings=["minimum_mask_quality", "minimum_old_box_iou", "mask_blend", "padding_index"],
                            paddings=PADDINGS, best=preferred[:5],
                            scope="Accepted input boxes only. No change to automatic counts. The comparison "
                                  "uses only development data and a locked class-agnostic SAM 3 checkpoint."))
    print("Guided geometry", len(safe), "development candidates preserving recalled instances", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "measure"))
    parser.add_argument("--scope", choices=DEVELOPMENT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "measure":
        measure()
    elif args.scope and (args.limit is None or args.limit > 0):
        collect(args.scope, args.limit)
    else:
        parser.error("Collection requires a development cohort and an optional positive limit.")
