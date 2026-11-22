"""Test class-agnostic mask proposals on protected detection development images."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from PIL import ImageOps

from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou, suppress
from scripts.experiments.detector_audits.detection_guided_geometry import EDGE, GuidedMasks, references
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, inputs
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import LOCK, configure_cpu, mask_geometry

STUDY = ART / "mask_retrieval_development.json"
GRIDS = (8, 16)
BATCH = 16
QUALITY_FLOOR = .55
MAX_PROPOSALS = 100


class MaskRetriever(GuidedMasks):
    def predict_grid(self, image):
        torch = self.torch
        picture = ImageOps.contain(image, (EDGE, EDGE))
        inputs = self.processor(images=picture, return_tensors="pt", size={"height": EDGE, "width": EDGE})
        pixels = inputs["pixel_values"].to(device="xpu", dtype=self.model.dtype)
        proposals = {}
        with torch.inference_mode():
            embedding = self.model.get_image_embeddings(pixels)
            for edge in GRIDS:
                centers = [(x + .5) * EDGE / edge for x in range(edge)]
                points = list(product(centers, repeat=2))
                rows = self.decode(picture, embedding, points)
                proposals[str(edge)] = suppress(rows, .45)[:MAX_PROPOSALS]
        return proposals

    def decode(self, picture, embedding, points):
        torch = self.torch
        for start in range(0, len(points), BATCH):
            batch = torch.tensor(points[start:start+BATCH], dtype=torch.float32, device="xpu")[None, :, None, :]
            outputs = self.model(image_embeddings=embedding, input_points=batch, multimask_output=True)
            quality = outputs.iou_scores.flatten()
            selected = quality >= QUALITY_FLOOR
            masks = outputs.pred_masks.flatten(end_dim=2)[selected].cpu()
            for logit, score in zip(masks, quality[selected].cpu(), strict=True):
                box, area = mask_geometry(logit)
                if box is None or not .00005 <= area <= .22:
                    continue
                high = int((logit > .5).sum())
                low = int((logit > -.5).sum())
                stability = high / max(low, 1)
                locations = (logit > 0).nonzero().numpy()
                if len(locations) < 10:
                    continue
                scale = np.array([picture.height, picture.width]) / np.array(logit.shape)
                geometry = np.linalg.eigvalsh(np.cov(locations * scale, rowvar=False))
                elongation = float(np.sqrt(geometry[-1] / max(.1, geometry[0])))
                yield dict(bbox=box, score=float(score), stability=stability,
                           mask_area=area, elongation=elongation)


def path(scope):
    return DATA / "proposals" / f"mask-retrieval-{scope}.json"


def binding(scope):
    return dict(code=digest(Path(__file__)), front_end=digest("scripts/experiments/detector_audits/detection_guided_geometry.py"),
                checkpoint=digest(LOCK), grids=GRIDS,
                sources={r["id"]: r.get("derived_sha256", r["sha256"]) for r in inputs(scope)})


def collect(scope, limit):
    configure_cpu()
    binding_ = {**binding(scope), "grids": list(GRIDS)}
    current = read_json(path(scope)) if path(scope).exists() else dict(key=binding_, frames={})
    if current["key"] != binding_:
        raise ValueError("The points, source images or model changed since collection began.")
    todo = [r for r in inputs(scope) if r["id"] not in current["frames"]]
    predictor = None
    for row in todo[:limit]:
        if predictor is None:
            predictor = MaskRetriever()
        picture = source(row)
        start = perf_counter()
        new = predictor.predict_grid(picture)
        current["frames"][row["id"]] = dict(grids=new, ms=1000*(perf_counter()-start))
        write_json(path(scope), current)
        print("Class-agnostic masks", scope, len(current["frames"]), "/", len(binding_["sources"]), flush=True)


def coverage(rows, frames, old, grid, elongation, stable):
    total = reachable = found = covered_with_masks = 0
    for row in rows:
        key = row["id"]
        independent = [p for p in frames[key]["grids"][str(grid)]
                       if p["elongation"] >= elongation and p["stability"] >= stable]
        total += len(independent)
        for target in row["objects"]:
            if any(box_iou(p["bbox"], target["bbox"]) >= .50 for p in old[key]):
                found += 1
            elif any(box_iou(p["bbox"], target["bbox"]) >= .50 for p in independent):
                reachable += 1
            if any(box_iou(p["bbox"], target["bbox"]) >= .50 for p in independent):
                covered_with_masks += 1
    return dict(baseline_reached=found, candidate_reached=covered_with_masks,
                extra_reachable=reachable, total_candidates=total)


def measure():
    if STUDY.exists():
        raise ValueError("This development proposal audit is already measured.")
    scopes = {}
    for scope in DEVELOPMENT:
        rows, old = inputs(scope), references(scope)
        current = read_json(path(scope))
        if current["key"] != {**binding(scope), "grids": list(GRIDS)} or set(current["frames"]) != set(old):
            raise ValueError("The sealed development inputs were not collected in full.")
        frames = current["frames"]
        options = [dict(grid=grid, min_elongation=ratio, min_stability=stability,
                        diagnostics=coverage(rows, frames, old, grid, ratio, stability))
                   for grid, ratio, stability in product(GRIDS, (1., 2., 4.), (0., .85, .95))]
        scopes[scope] = dict(median_pair_ms=median(f["ms"] for f in frames.values()), options=options)
    write_json(STUDY, dict(scopes=scopes, checkpoint_sha256=digest(LOCK),
                           interpretation="Reference reach is oracle overlap with candidate boxes on development. "
                                          "The candidates have no class predictions and are not automatic detections."))
    print("Audited mask proposals on both development domains.", flush=True)


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
        parser.error("Provide a development scope and an optional positive limit.")
