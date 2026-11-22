"""Compare pixel-mask identity and box overlap in crowded development scenes."""
import argparse
from itertools import product
from time import perf_counter

import numpy as np
from PIL import ImageOps

from medsyringe.additive import duplicates
from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou, suppress
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.appearance_text_encoder import TextMarginEncoder
from scripts.detection_crowd_consensus import ARMS, SCOPES, inputs_for
from scripts.detection_guided_geometry import EDGE, GuidedMasks
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.expanded_detection_study import short
from scripts.precision_study import source
from scripts.rfdetr_verifier import expanded
from scripts.sam3_study import LOCK, configure_cpu

CACHE = DATA / "proposals/mask-identity-development.json"
OUTPUT = ART / "mask_identity_development.json"
MAX_CANDIDATES = 6


def shape_candidates(first, second):
    hypotheses = []
    for primary in first:
        if primary["score"] < .20:
            continue
        overlaps = [(box_iou(primary["bbox"], p["bbox"]), p) for p in second if p["score"] >= .20]
        iou, peer = max(overlaps, key=lambda item: (item[0], item[1]["score"]), default=(0., None))
        if iou < .55:
            continue
        region = [(a + b) / 2 for a, b in zip(primary["bbox"], peer["bbox"], strict=True)]
        hypotheses.append(dict(bbox=region, score=min(primary["score"], peer["score"]), support_iou=iou))
    return suppress(hypotheses, .70)[:MAX_CANDIDATES]


class IdentityMasks(GuidedMasks):
    def instances(self, image, boxes):
        picture = ImageOps.contain(image, (EDGE, EDGE))
        size = [picture.width, picture.height] * 2
        prompts = [[coord * dimension for coord, dimension in zip(box, size, strict=True)]
                   for box in boxes]
        inputs = self.processor(images=picture, input_boxes=[prompts], return_tensors="pt",
                                size={"height": EDGE, "width": EDGE})
        pixels = inputs["pixel_values"].to(device="xpu", dtype=self.model.dtype)
        with self.torch.inference_mode():
            result = self.model(pixel_values=pixels, input_boxes=inputs["input_boxes"].to("xpu"),
                                multimask_output=True)
            scores = result.iou_scores.cpu()[0]
            masks = self.processor.post_process_masks(result.pred_masks.cpu(), inputs["original_sizes"])[0]
        return [(float(values[index]), (group[index] > 0).numpy()) for group, values in zip(masks, scores, strict=True)
                for index in [int(values.argmax())]]


def identity_overlap(one, two):
    minimum = min(np.count_nonzero(one), np.count_nonzero(two))
    return float(np.count_nonzero(one & two) / minimum) if minimum else 1.


def collect():
    configure_cpu()
    model = None
    stored = read_json(CACHE) if CACHE.exists() else {}
    for scope in SCOPES:
        rows, original, forecasts, hashes = inputs_for(scope)
        binding = dict(inputs=hashes, model=digest(LOCK), method=digest(__file__))
        saved = stored.setdefault(scope, dict(binding=binding, frames={}))
        if saved["binding"] != binding:
            raise ValueError("The mask check changed since its first output.")
        for row in rows:
            key = row["id"]
            if key in saved["frames"]:
                continue
            known = original[key]
            candidates = []
            if len(known) >= 2:
                views = ("suppressed", "deskewed") if scope == "parallel" else ("suppressed",)
                for view in views:
                    candidates.extend(shape_candidates(*(forecasts[a][key][view] for a in ARMS)))
                candidates = suppress(candidates, .70)[:MAX_CANDIDATES]
            observed = []
            start = perf_counter()
            if candidates:
                if model is None:
                    model = IdentityMasks()
                masks = model.instances(source(row), [p["bbox"] for p in [*known, *candidates]])
                bases, extra = masks[:len(known)], masks[len(known):]
                for candidate, (quality, pixels) in zip(candidates, extra, strict=True):
                    observed.append(dict(**candidate, quality=min(quality, *(q for q, _ in bases)),
                                         overlap=max(identity_overlap(pixels, other) for _, other in bases),
                                         box_duplicate=any(duplicates(p["bbox"], candidate["bbox"]) for p in known)))
            saved["frames"][key] = dict(candidates=observed, seconds=perf_counter() - start)
            write_json(CACHE, stored)
            print("Mask identity", scope, len(saved["frames"]), "/", len(rows), flush=True)


def comparison(rows, before, information, option):
    quality, overlap, sem, method = option
    candidate = {
        key: [*before[key], *(dict(bbox=p["bbox"], score=.1) for p in ranked
                             if p["quality"] >= quality and p["margin"] >= sem and
                             (p["overlap"] <= overlap if method == "mask" else not p["box_duplicate"]))][:len(before[key])+1]
        for key, ranked in information.items()
    }
    earlier, current = (evaluate(rows, boxes, 0.) for boxes in (before, candidate))
    guard = compare(earlier, current)
    spoiled = any(len(before[r["id"]]) == len(r["objects"]) and len(candidate[r["id"]]) != len(r["objects"])
                  for r in rows)
    return dict(outcome=short(current), guard=guard, spoiled=spoiled,
                safe=guard["passed"] and not spoiled and current["fp"] <= earlier["fp"])


def measure():
    if OUTPUT.exists():
        raise ValueError("The mask separation development suite is already measured.")
    configure_cpu()
    mask_data = read_json(CACHE)
    options = tuple(product((.70, .82, .90), (.15, .30, .50), (-.005, .01), ("mask", "box")))
    ranked, scopes = [], {}
    encoder = None
    for scope in SCOPES:
        rows, known, _, hashes = inputs_for(scope)
        record = mask_data[scope]
        if (record["binding"] != dict(inputs=hashes, model=digest(LOCK), method=digest(__file__))
                or set(record["frames"]) != set(known)):
            raise ValueError("Mask identity evidence is incomplete or incompatible.")
        results = {}
        for row in rows:
            key = row["id"]
            entries = record["frames"][key]["candidates"]
            if entries:
                if encoder is None:
                    encoder = TextMarginEncoder()
                image = source(row)
                feature = encoder.encode([crop(image, expanded(p["bbox"], .08)) for p in entries])["prompt"]
                for p, margin in zip(entries, feature[:, 0] - feature[:, 1], strict=True):
                    p["margin"] = float(margin)
            results[key] = entries
        scopes[scope] = rows, known, results
    for option in options:
        evidence = {scope: comparison(*vals, option) for scope, vals in scopes.items()}
        if all(v["safe"] for v in evidence.values()):
            ranked.append(dict(option=option, results=evidence))
    ranked.sort(key=lambda row: (sum(v["guard"]["recovered_matches"] for v in row["results"].values()),
                                 row["option"][-1] == "box"), reverse=True)
    write_json(OUTPUT, dict(variants=len(options), safe_development=len(ranked), leading=ranked[:12],
                            collection_sha256=digest(CACHE),
                            scope="Mask identity is measured independently of bounding-box identity. Scope is "
                                  "development only. Each scene receives one addition at most."))
    print("Mask identity", len(ranked), "/", len(options), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "measure"))
    (collect if parser.parse_args().action == "collect" else measure)()
