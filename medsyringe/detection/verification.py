"""Lightweight crop classifier for syringe lookalikes."""
from pathlib import Path

import numpy as np
from PIL import ImageOps
import torch
from torchvision.models import mobilenet_v3_small
from torchvision.transforms.functional import normalize, pil_to_tensor

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_json

BACKBONE = Path("data/models/verifier/mobilenet_v3_small-047dcff4.pth")
MODEL_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"


class CropEncoder:
    def __init__(self, root, checksum):
        path = root / BACKBONE
        if digest(path) != checksum:
            raise ValueError("Crop encoder checkpoint checksum mismatch.")
        net = mobilenet_v3_small()
        net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        self.features = net.features.eval()
        self.pool = net.avgpool

    @torch.inference_mode()
    def encode(self, images):
        if not images:
            return np.zeros((0, 576), dtype=np.float32)
        batch = []
        for image in images:
            padded = ImageOps.pad(image, (160, 160), color=(124, 116, 104))
            batch.append(normalize(pil_to_tensor(padded).float()/255,
                                   (.485, .456, .406), (.229, .224, .225)))
        values = self.pool(self.features(torch.stack(batch))).flatten(1)
        return torch.nn.functional.normalize(values).numpy()


def classifier_scores(values, weights, bias):
    return np.asarray(values) @ np.asarray(weights) + bias


def eligible(ranked, cutoff):
    return [p for p in ranked if p["score"] >= cutoff]


def filter_proposals(candidates, scores, cutoff):
    if len(candidates) != len(scores):
        raise ValueError("Every proposal needs one verification score.")
    result = [{"verification_score": float(value), **p} for p, value in zip(candidates, scores)]
    return [p for p in result if p["verification_score"] >= cutoff], [
        p for p in result if p["verification_score"] < cutoff
    ]


class CropVerifier:
    def __init__(self, root, release):
        path = root / "artifacts/precision_head.json"
        if digest(path) != release["head_sha256"]:
            raise ValueError("Crop verifier head checksum mismatch.")
        self.head = read_json(path)
        self.encoder = CropEncoder(root, release["backbone_sha256"])

    def __call__(self, image, candidates):
        patches = [crop(image, p["bbox"]) for p in candidates]
        values = self.encoder.encode(patches)
        scores = classifier_scores(values, self.head["weights"], self.head["bias"])
        return filter_proposals(candidates, scores, self.head["cutoff"])
