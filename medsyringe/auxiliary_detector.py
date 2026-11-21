"""Verified RF-DETR additions for the frozen local detector."""
from pathlib import Path

from PIL import Image
import torch
from torchvision.transforms.functional import normalize, to_tensor

from .additive import additions, overlapping_tiles, project
from .attributes import crop
from .data import digest, read_json, valid_box
from .detectors import suppress
from .verification import CropEncoder, classifier_scores

MANIFEST = Path("artifacts/rfdetr_verified_release.json")
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


class VerifiedRecovery:
    def __init__(self, root, config):
        if digest(root / config["weights"]) != config["weights_sha256"]:
            raise ValueError("The auxiliary detector checkpoint changed.")
        self.network = torch.jit.load(str(root / config["weights"]), map_location="cpu").eval()
        self.encoder = CropEncoder(root, config["encoder_sha256"])
        self.edge = config["resolution"]
        self.options = config["options"]
        self.head = config["head"]

    @torch.inference_mode()
    def _region(self, image):
        square = image.resize((self.edge, self.edge), Image.Resampling.BILINEAR)
        tensor = normalize(to_tensor(square), MEAN, STD)
        boxes, logits = self.network(tensor.unsqueeze(0))
        scores = logits[0].sigmoid().flatten()
        order = torch.argsort(scores, descending=True, stable=True)[:100]
        # Keep the foreground class from the publisher's top query/class pairs.
        order = order[(order % 2 == 0) & (scores[order] > .001)]
        x, y, width, height = boxes[0, order // 2].unbind(-1)
        xyxy = torch.stack((x-width/2, y-height/2, x+width/2, y+height/2), -1).clip(0, 1)
        return [dict(bbox=box, score=score) for box, score in zip(xyxy.tolist(), scores[order].tolist())
                if valid_box(box)]

    def propose(self, image):
        candidates = suppress(self._region(image), .45)
        for window in overlapping_tiles(image.size, .66):
            for proposal in self._region(image.crop(window)):
                box = project(proposal["bbox"], window, image.size)
                if box:
                    candidates.append({**proposal, "bbox": box})
        return suppress(candidates, .45)

    def apply(self, image, ranked, cutoff):
        support = self.options["support"]
        if support is not None and not any(support <= p["score"] < cutoff for p in ranked):
            return ranked
        picked = additions(ranked, self.propose(image), cutoff, **self.options)
        extra = [p for p in picked if p.get("source")=="auxiliary"]
        if not extra:
            return ranked
        values = self.encoder.encode([crop(image, p["bbox"]) for p in extra])
        margins = classifier_scores(values, self.head["weights"], self.head["bias"])
        accepted = [{**p, "verification_score": float(score)} for p,score in zip(extra,margins,strict=True)
                    if score >= self.head["cutoff"]]
        split = next(i for i,p in enumerate(picked) if p.get("source")=="auxiliary")
        return [*picked[:split], *accepted, *[p for p in picked[split:] if p.get("source")!="auxiliary"]]


def load_verified_recovery(root, base_path):
    path = root / MANIFEST
    if not path.exists():
        return None
    config = read_json(path)
    evidence = root / config["evidence"]
    if (digest(Path(__file__)) != config["runtime_sha256"] or
            digest(base_path) != config["base_sha256"] or
            digest(evidence) != config["evidence_sha256"] or
            not read_json(evidence)["passed"]):
        raise ValueError("The supplemental detector lacks verified release evidence.")
    return VerifiedRecovery(root, config)
