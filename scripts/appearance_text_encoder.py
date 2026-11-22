"""Encode the appearance veto without unreferenced shape features."""
from PIL import Image, ImageOps
import torch
from transformers import AutoModel, AutoProcessor

from medsyringe.attributes import crop
from scripts.detection_crop_semantics import (LOOKALIKES, MODEL, PADDINGS, SYRINGES,
                                             VisualSemantics, expanded, lock_weights)


PATCH_EDGE = 224


def prepare_patch(patch):
    return ImageOps.pad(patch, (PATCH_EDGE, PATCH_EDGE), color=(124, 116, 104))


def pack_patches(image, candidates):
    count = len(PADDINGS) * len(candidates)
    if image.mode != "RGB" or not count or image.width * image.height <= count * PATCH_EDGE ** 2:
        return None
    packed = Image.new("RGB", (PATCH_EDGE, count * PATCH_EDGE))
    boxes = (expanded(p["bbox"], padding) for padding in PADDINGS for p in candidates)
    for row, box in enumerate(boxes):
        patch = prepare_patch(crop(image, box))
        packed.paste(patch, (0, row * PATCH_EDGE))
    return packed


def unpack_patches(packed, candidates):
    count = len(PADDINGS) * len(candidates)
    if packed.size != (PATCH_EDGE, count * PATCH_EDGE):
        raise ValueError("The appearance crop payload does not match its proposals.")
    return [packed.crop((0, row * PATCH_EDGE, PATCH_EDGE, (row + 1) * PATCH_EDGE))
            for row in range(count)]


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
        resized = [prepare_patch(patch) for patch in patches]
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
