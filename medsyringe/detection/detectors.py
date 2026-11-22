"""Offline prompt-based detectors and a learned proposal re-ranker."""
import argparse
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import ImageEnhance, ImageOps
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from medsyringe.data import digest, read_image, read_json, write_json

MODEL_SOURCES = {
    "dino": ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71",
             ("tokenizer.json", "vocab.txt")),
    "owl": ("google/owlv2-base-patch16", "2a1560802f8cf3c408fec9b809d705f56a2f7146",
            ("merges.txt", "vocab.json")),
}
PROMPT = "syringe"
RAW_MINIMUM = 0.02
NMS_IOU = 0.6
THREADS = 4


def model_files(name):
    return ("config.json", "preprocessor_config.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "model.safetensors",
            *MODEL_SOURCES[name][2])


def model_hashes(root, name):
    return {file: digest(root / "data/models" / name / file) for file in model_files(name)}


def box_iou(a, b):
    ax, ay, ar, ab = a
    bx, by, br, bb = b
    area = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
    union = (ar - ax) * (ab - ay) + (br - bx) * (bb - by) - area
    return area / union if union > 0 else 0.0


def suppress(predictions, overlap=NMS_IOU):
    kept = []
    for prediction in sorted(predictions, key=lambda pred: -pred["score"]):
        if all(box_iou(prediction["bbox"], prev["bbox"]) <= overlap for prev in kept):
            kept.append(prediction)
    return kept[:100]


def features(prediction):
    x, y, r, b = prediction["bbox"]
    w, h = r - x, b - y
    score = np.clip(prediction["score"], 1e-6, 1 - 1e-6)
    return [math.log(score / (1-score)), math.log(w*h), abs(math.log(w/h))]


def rerank(predictions, checkpoint):
    mean, scale, weight = (np.array(checkpoint[key]) for key in ("mean", "scale", "weights"))
    ranked = []
    for prediction in predictions:
        vector = (features(prediction) - mean) / scale
        logit = float(vector @ weight + checkpoint["intercept"])
        adjusted = 1 / (1 + math.exp(-max(-50, min(50, logit))))
        # Preserve some open-domain ranking when the single-session learner extrapolates.
        ranked.append({**prediction, "raw_score": prediction["score"],
                       "score": (prediction["score"] + adjusted) / 2})
    return ranked


class Grounder:
    def __init__(self, root, name):
        torch.set_num_threads(THREADS)
        self.name = name
        path = root / "data/models" / name
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=True)
        if name == "dino":
            self.processor.image_processor.size = {"shortest_edge": 480, "longest_edge": 800}
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            path, local_files_only=True, use_safetensors=True
        ).eval()

    @torch.inference_mode()
    def propose(self, image):
        prompt = f"{PROMPT}." if self.name == "dino" else [[f"a photo of a {PROMPT}"]]
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        output = self.model(**inputs)
        size = [(image.height, image.width)]
        if self.name == "dino":
            raw = self.processor.post_process_grounded_object_detection(
                output, inputs.input_ids, threshold=RAW_MINIMUM, target_sizes=size
            )[0]
        else:
            raw = self.processor.post_process_grounded_object_detection(
                output, threshold=RAW_MINIMUM, target_sizes=size
            )[0]
        scale = np.array([image.width, image.height, image.width, image.height])
        candidates = []
        for box, score in zip(raw["boxes"].numpy(), raw["scores"].tolist()):
            x, y, r, b = np.clip(box/scale, 0, 1).tolist()
            if r-x >= 0.005 and b-y >= 0.005:
                candidates.append({"bbox": [x, y, r, b], "score": score})
        return sorted(candidates, key=lambda pred: -pred["score"])[:100]


def cached_proposals(root, name, split, variant, rows):
    cache_file = root / "data/cache" / f"{name}-{split}-{variant}.json"
    config = dict(model=name, weights=model_hashes(root, name), split=split, variant=variant,
                  prompt=PROMPT, dino_resolution=[480, 800], minimum=RAW_MINIMUM,
                  rows=[[r["id"], r.get("derived_sha256", r["sha256"])] for r in rows],
                  runtime={"detectors.py": digest(__file__), "data.py": digest(Path(__file__).parents[1] / "data.py")})
    predictions = {}
    if cache_file.exists():
        saved = read_json(cache_file)
        if saved["config"] == config:
            predictions = saved["predictions"]
    remaining = [row for row in rows if row["id"] not in predictions]
    if remaining:
        detector = Grounder(root, name)
        for i, row in enumerate(remaining, 1):
            path = root / "data" / row["image"]
            if digest(path) != row.get("derived_sha256", row["sha256"]):
                raise ValueError(f"Image checksum mismatch: {row['id']}")
            im = read_image(path)
            if variant == "augmented":
                brightness = 0.78 if int(row["id"][-1]) % 2 else 1.20
                im = ImageEnhance.Brightness(ImageOps.mirror(im)).enhance(brightness)
            started = perf_counter()
            proposals = detector.propose(im)
            predictions[row["id"]] = {"proposals": proposals, "seconds": perf_counter() - started}
            if i == 1 or i % 5 == 0 or i == len(remaining):
                write_json(cache_file, dict(config=config, predictions=predictions))
                print(name, split, variant, i, "/", len(remaining), flush=True)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODEL_SOURCES)
    parser.add_argument("split", choices=["train", "validation", "test"])
    parser.add_argument("--variant", choices=["original", "augmented"], default="original")
    args = parser.parse_args()
    root = Path(".")
    if args.split == "train":
        rows = [r for r in read_json(root / "artifacts/source_inventory.json") if r["split"] == "train"]
    else:
        rows = read_json(root / "artifacts" / f"{args.split}_manifest.json")["rows"]
    cached_proposals(root, args.model, args.split, args.variant, rows)
