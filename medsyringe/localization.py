"""Learn box-coordinate corrections and rank proposals by expected overlap."""
import numpy as np

from .detectors import suppress


def relative_offsets(boxes, targets):
    scales = np.maximum(boxes[:, 2:] - boxes[:, :2], 1e-6)
    return (targets - boxes)/np.tile(scales, (1, 2))


def correct_boxes(boxes, offsets, strength):
    scales = np.tile(boxes[:, 2:] - boxes[:, :2], (1, 2))
    output = np.clip(boxes + scales * np.clip(offsets, -.30, .30) * strength, 0, 1)
    valid = (output[:, 0] < output[:, 2]) & (output[:, 1] < output[:, 3])
    return np.where(valid[:, None], output, boxes)


def rerank(proposals, features, models, *, box_strength, quality_weight, overlap):
    if not proposals:
        return []
    inputs = np.column_stack(features)
    boxes = np.array([p["bbox"] for p in proposals])
    if box_strength:
        boxes = correct_boxes(boxes, models["box"].predict(inputs), box_strength)
    scores = np.array([p["score"] for p in proposals])
    if quality_weight:
        quality = np.clip(models["quality"].predict(inputs), 0, 1)
        scores = (1-quality_weight)*scores + quality_weight*quality
    return suppress([{**p, "bbox": box.tolist(), "score": float(score)}
                     for p, box, score in zip(proposals, boxes, scores)], overlap)
