"""Review weak, uncorroborated detections using a frozen DINOv2 crop verifier."""
from pathlib import Path

import numpy as np
from PIL import ImageOps

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest
from medsyringe.detection.detectors import box_iou
from medsyringe.runtime.expanded_runtime import WORKER_ENTRY
from medsyringe.runtime.local_models import ImageWorker
from medsyringe.runtime.pipeline_extensions import verified_config
from medsyringe.detection.verification import eligible

RELEASE = Path("artifacts/expanded/crop_verifier_release.json")


def supports(box, proposals):
    return any(p["score"] >= .35 and box_iou(box, p["bbox"]) >= .45 for p in proposals)


class CropVerificationRuntime:
    def __init__(self, engine, root, config):
        from transformers import Dinov2Model
        from scripts.experiments.transfer.rfdetr_verifier import DINOV2, DinoEncoder

        class LocalEncoder(DinoEncoder):
            def __init__(self):
                self.model = Dinov2Model.from_pretrained(root / DINOV2, local_files_only=True).eval()

        self.engine = engine
        self.encoder = LocalEncoder()
        self.weights = np.asarray(config["head"]["weights"])
        self.bias = config["head"]["bias"]
        self.cutoff = config["cutoff"]
        self.support_worker = ImageWorker(root, WORKER_ENTRY, "sam3",
            {key: engine.sam3.options[key] for key in ("edge", "device")})

    def detect(self, image):
        from scripts.experiments.transfer.rfdetr_verifier import expanded

        engine = self.engine
        primary = engine.rank_detections(image)
        ranked, floor, sam_status = engine.baseline_detections(image, primary)
        accepted = eligible(ranked, floor)
        if not accepted:
            return ranked, floor, sam_status
        crops = [crop(image, expanded(p["bbox"], .08)) for p in accepted]
        scores = self.encoder.encode(crops)@self.weights + self.bias
        probabilities = 1/(1+np.exp(-np.clip(scores, -30., 30.)))
        weak = [(p, float(score)) for p, score in zip(accepted, probabilities, strict=True)
                if score < self.cutoff]
        if not weak:
            return ranked, floor, sam_status
        try:
            rf = engine.recovery._region(image)
            undecided = [(p, score, sum(supports(p["bbox"], branch) for branch in (primary, rf)))
                         for p, score in weak]
            sam = (self.support_worker.run(ImageOps.contain(image, (1500, 1500)), operation="detect")
                   if any(votes == 1 for _, _, votes in undecided) else [])
        except RuntimeError:
            return ranked, floor, sam_status
        rejected = {id(p): score for p, score, votes in undecided if
                    votes < 2 and votes + supports(p["bbox"], sam) < 2}
        needs_review = [
            {**p, "score": floor*.01, "original_score": p["score"], "crop_score": rejected[id(p)],
             "review_reason": "Weak crop evidence without agreement from two detectors"}
            for p in accepted if id(p) in rejected]
        retained = [p for p in accepted if id(p) not in rejected]
        return retained + needs_review + [p for p in ranked if p["score"] < floor], floor, sam_status

    def close(self):
        self.support_worker.close()


def install_crop_verifier(engine, root):
    config = verified_config(root, RELEASE)
    if config is None:
        return None
    if engine.identity != config["model"] or engine.expanded.detector is not None:
        raise ValueError("The crop verifier does not match the current baseline pipeline.")
    runtime = CropVerificationRuntime(engine, root, config)
    engine.expanded.detect = runtime.detect
    engine.identity["crop_verification_sha256"] = digest(root / RELEASE)
    return runtime
