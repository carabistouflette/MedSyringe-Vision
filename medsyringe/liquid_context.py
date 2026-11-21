"""Reject weak color evidence that matches the nearby background."""
import cv2
import numpy as np


def contextual_support(frame, support, minimum_separation):
    if not support.any():
        return support
    region = frame["lab"]
    sample = np.median(region[support], axis=0)
    if float(np.linalg.norm(sample[1:])) >= 32.:
        return support
    around_liquid = cv2.dilate(support.astype(np.uint8), np.ones((39, 39), dtype=np.uint8)).astype(bool)
    around_silhouette = cv2.dilate(frame["mask"], np.ones((5, 5), dtype=np.uint8)).astype(bool)
    neighbors = around_liquid & ~around_silhouette & (region[..., 0] >= 15) & (region[..., 0] <= 95)
    if neighbors.sum() >= 64:
        context = np.median(region[neighbors], axis=0)
        separation = float(np.linalg.norm(sample[1:]-context[1:]))
    else:
        separation = float(np.linalg.norm(sample[1:])) - 12.
    return support if separation >= minimum_separation else np.zeros_like(support)
