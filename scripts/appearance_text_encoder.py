"""Encode the appearance veto without unreferenced shape features."""
from PIL import ImageOps
import torch
from transformers import AutoModel, AutoProcessor

from scripts.detection_crop_semantics import LOOKALIKES, MODEL, SYRINGES, VisualSemantics, lock_weights


class TextMarginEncoder:
    """Encode only image-text scores when the policy never uses shape features."""
    def __init__(self):
        lock_weights()
        self.processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(
            MODEL, local_files_only=True, use_safetensors=True).eval()
        prompts = self.processor(text=[*SYRINGES, *LOOKALIKES], padding="max_length", truncation=True,
                                 return_tensors="pt")
        with torch.inference_mode():
            self.text = torch.nn.functional.normalize(self.model.get_text_features(**prompts))

    @torch.inference_mode()
    def encode(self, patches):
        resized = [ImageOps.pad(patch, (224, 224), color=(124, 116, 104)) for patch in patches]
        batch = self.processor(images=resized, return_tensors="pt")
        vectors = torch.nn.functional.normalize(self.model.get_image_features(**batch))
        pairs = vectors @ self.text.T
        target, counterexamples = pairs[:, :len(SYRINGES)], pairs[:, len(SYRINGES):]
        summary = torch.stack((target.max(dim=1).values, counterexamples.max(dim=1).values,
                               target.mean(dim=1), counterexamples.mean(dim=1)), dim=1)
        return {"prompt": summary.numpy()}


def detector_encoder(release):
    text_only = release.get("mode") == "veto-only" and all(
        entry["name"] in ("semantic-best", "semantic-mean") and entry["head"] is None
        for entry in release["components"]
    )
    return TextMarginEncoder() if text_only else VisualSemantics()
