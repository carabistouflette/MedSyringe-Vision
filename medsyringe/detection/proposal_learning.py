"""Local proposal learning with geometry and optional crop appearance."""
from time import perf_counter

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from medsyringe.detection.detectors import box_iou, features


VARIANTS = {
    "logistic-geometry": ("logistic", False, False, False),
    "logistic-appearance": ("logistic", True, False, False),
    "logistic-appearance-augmented": ("logistic", True, True, False),
    "logistic-appearance-balanced": ("logistic", True, True, True),
    "trees-geometry-augmented": ("trees", False, True, True),
    "trees-appearance-augmented": ("trees", True, True, True),
    "boosted-appearance-augmented": ("boosted", True, True, True),
}


def describe_proposals(image, proposals):
    sample = image.copy()
    sample.thumbnail((600, 600))
    rgb = np.asarray(sample)
    height, width = rgb.shape[:2]
    geometry, texture = [], []
    for proposal in proposals:
        l, t, r, b = proposal["bbox"]
        geometry.append(features(proposal))
        region = rgb[int(t*height):max(int(t*height)+1, int(np.ceil(b*height))),
                     int(l*width):max(int(l*width)+1, int(np.ceil(r*width)))]
        patch = cv2.resize(region, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        magnitude, angle = cv2.cartToPolar(gx, gy)
        bins = (angle * (8/np.pi)).astype(int) % 8
        sectors = []
        for y in (0, 16):
            for x in (0, 16):
                selection = np.s_[y:y+16, x:x+16]
                hist = np.bincount(bins[selection].ravel(),
                                   weights=magnitude[selection].ravel(), minlength=8)
                sectors.extend(hist/(hist.sum()+1e-6))
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        texture.append([*sectors, *hsv[..., 1:].mean(axis=(0, 1))/255,
                        *hsv[..., 1:].std(axis=(0, 1))/255, float(magnitude.mean())])
    return np.array(geometry).reshape(-1, 3), np.array(texture).reshape(-1, 37)


def labels_for(proposals, boxes):
    return np.array([any(box_iou(p["bbox"], box) >= .5 for box in boxes) for p in proposals])


def fit_variant(name, frames):
    family, appearance, augment, burst_balanced = VARIANTS[name]
    chosen = [frame for frame in frames if augment or not frame["augmented"]]
    geometry, texture, labels = (np.concatenate([f[key] for f in chosen])
                                 for key in ("geometry", "texture", "labels"))
    X = np.column_stack((geometry, texture)) if appearance else geometry
    weights = np.ones(len(labels))
    if burst_balanced:
        sizes = {}
        for frame in chosen:
            burst = frame["burst"]
            sizes[burst] = sizes.get(burst, 0) + len(frame["labels"])
        weights = np.concatenate([np.full(len(f["labels"]), 1/sizes[f["burst"]])
                                  for f in chosen])
    for label in (False, True):
        mask = labels == label
        weights[mask] *= len(labels)/(2*weights[mask].sum())
    started = perf_counter()
    if family == "logistic":
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, random_state=607))
        estimator.fit(X, labels, logisticregression__sample_weight=weights)
    else:
        if family == "trees":
            estimator = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=12,
                                             max_depth=9, n_jobs=3, random_state=607)
        else:
            estimator = HistGradientBoostingClassifier(max_iter=70, max_leaf_nodes=9,
                                                       l2_regularization=1, random_state=607)
        estimator.fit(X, labels, sample_weight=weights)
    return dict(estimator=estimator, appearance=appearance, metrics=dict(
        fitting_seconds=perf_counter()-started, training_proposals=len(labels),
        positive_proposals=int(labels.sum()), augmented=augment, burst_balanced=burst_balanced,
        family=family, appearance=appearance, training_seed=607))


def adjust_scores(proposals, descriptors, models, weight):
    if not proposals or not models:
        return proposals
    geometry, texture = descriptors
    probabilities = np.mean([
        model["estimator"].predict_proba(
            np.column_stack((geometry, texture)) if model["appearance"] else geometry
        )[:, 1] for model in models
    ], axis=0)
    return [{**p, "score": (1-weight)*p["score"] + weight*float(probability)}
            for p, probability in zip(proposals, probabilities)]
