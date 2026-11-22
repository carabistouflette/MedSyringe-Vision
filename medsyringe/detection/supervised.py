"""Transfer learning for a compact one-class syringe detector."""
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSD
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

from medsyringe.data import digest

SSD_BASE = "data/models/ssd/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"


def empty_scene_loss(logits, targets, top=8):
    empty = [i for i, target in enumerate(targets) if not len(target["labels"])]
    if not empty:
        return logits.sum() * 0
    errors = -F.log_softmax(logits[empty], dim=-1)[..., 0]
    return errors.topk(min(top, errors.shape[1]), dim=1).values.mean()


def _with_empty_scene_loss(model, targets, outputs, anchors, matches):
    loss = SSD.compute_loss(model, targets, outputs, anchors, matches)
    loss["classification"] = loss["classification"] + model.empty_weight * empty_scene_loss(
        outputs["cls_logits"], targets
    )
    return loss


def detector(root, base_checksum=None, empty_weight=0.0):
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None,
                                           num_classes=91, detections_per_img=100)
    if base_checksum is not None:
        path = root / SSD_BASE
        if digest(path) != base_checksum:
            raise ValueError("Pretrained SSD weights differ from the experiment.")
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    classifier = model.head.classification_head
    classifier.num_columns = 2
    for block in classifier.module_list:
        original = block[-1]
        anchors = original.out_channels // 91
        projection = nn.Conv2d(original.in_channels, anchors*2, kernel_size=1)
        if base_checksum is not None:
            with torch.no_grad():
                weights = original.weight.view(anchors, 91, *original.weight.shape[1:])
                biases = original.bias.view(anchors, 91)
                newer = projection.weight.view(anchors, 2, *projection.weight.shape[1:])
                new_bias = projection.bias.view(anchors, 2)
                newer[:, 0] = weights[:, 0]
                newer[:, 1] = weights[:, 1:].mean(dim=1)
                new_bias[:, 0] = biases[:, 0]
                new_bias[:, 1] = biases[:, 1:].mean(dim=1)
        block[-1] = projection
    model.backbone.requires_grad_(False)
    if empty_weight:
        model.empty_weight = empty_weight
        model.compute_loss = MethodType(_with_empty_scene_loss, model)
    return model


def tensor(image):
    return pil_to_tensor(image.resize((320, 320), Image.Resampling.BILINEAR)).float()/255


@torch.inference_mode()
def predict(model, images, batch=4, edge=320):
    model.eval()
    output = {}
    for start in range(0, len(images), batch):
        rows = images[start:start+batch]
        results = model([x for _, x in rows])
        for (identifier, _), prediction in zip(rows, results):
            boxes = (prediction["boxes"]/edge).clamp(0, 1).tolist()
            output[identifier] = [{"bbox": box, "score": score}
                                  for box, score in zip(boxes, prediction["scores"].tolist())
                                  if box[0] < box[2] and box[1] < box[3]]
    return output
