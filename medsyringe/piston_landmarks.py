"""Locate shoulder, piston rim and flange using dense features and barrel geometry."""
from itertools import product

import cv2
import numpy as np

from .fill import pixels_at

LANDMARKS = ("shoulder", "piston", "flange")


def features(frame, method):
    if method == "dense":
        return frame["grid"]
    color = cv2.resize(frame["lab"], (16, 16), interpolation=cv2.INTER_AREA)/[100., 60., 60.]
    return np.concatenate([frame["grid"], color], axis=-1)


def point_options(grid, mask, head):
    heat = grid @ head["weights"] + head["bias"]
    valid = cv2.resize(mask, (16, 16), interpolation=cv2.INTER_AREA) > .10
    peaks = (heat >= cv2.dilate(heat, np.ones((3, 3), np.uint8))) & valid
    order = sorted(zip(*np.nonzero(peaks)), key=lambda p: heat[p], reverse=True)[:4]
    return [((np.asarray([x,y])+.5)/16, float(heat[y,x])) for y,x in order]


def geometry(frame, model):
    grid = features(frame, model["method"])
    mask = frame["mask"].astype(np.float32)
    h, w = mask.shape
    points = [point_options(grid, mask, model["heads"][key]) for key in LANDMARKS]
    scale = np.array([w, h])
    candidate = []
    for (a, sa), (b, sb), (c, sc) in product(*points):
        a, b, c = a*scale, b*scale, c*scale
        length = np.linalg.norm(c-a)
        if not (.29*max(w,h) <= length <= max(w,h)*1.13):
            continue
        axis = (c-a)/length
        tangent = np.array([-axis[1], axis[0]])
        along, across = (b-a)@axis, abs((b-a)@tangent)
        if along < -.045*length or along > .96*length or across > .14*length:
            continue
        rank = sa+sb+sc - 12*across/length
        candidate.append((rank, a, a+max(0., along)*axis, c, min(sa,sb,sc)))
    if not candidate:
        return None
    rank, a, b, c, confidence = max(candidate, key=lambda p: p[0])
    return dict(shoulder=a, piston=b, flange=c, score=confidence, rank=rank)


def refine_rim(frame, local, min_drop):
    """Verify a dark transverse rim in the neighborhood of the learned piston."""
    a, b, c = (local[k] for k in LANDMARKS)
    length = np.linalg.norm(c-a)
    axis = (c-a)/length
    tangent = np.array([-axis[1], axis[0]])
    offsets = np.linspace(-.060, .060, 37)*length
    tracks = np.linspace(-.090, .090, 13)*length
    levels = np.array([-.018, -.009, .011, .023])*length
    centers = b+offsets[:, None]*axis
    sample = centers[:, None, None, :] + tracks[None, :, None, None]*tangent + levels[None, None, :, None]*axis
    profile = pixels_at(frame["lab"], sample.reshape((-1, 13*4, 2))).reshape((len(offsets), 13, 4, 3))[...,0]
    support = pixels_at(frame["mask"].astype(np.float32), sample.reshape((-1, 13*4, 2))).reshape((len(offsets), 13, 4))
    # Use only the interior portion of the outline. Confirm a dark rear side
    # across multiple adjacent tracks, independently of liquid chroma.
    inside = support.min(axis=2) > .85
    front = profile[:, :, :2].mean(axis=2)
    back = profile[:, :, 2:].mean(axis=2)
    scores = []
    for i, valid in enumerate(inside):
        if valid.sum() < 7 or np.median(back[i,valid]) > 52.:
            continue
        drops = front[i,valid]-back[i,valid]
        contrast = np.median(drops)
        if contrast >= min_drop and np.mean(drops >= .55*min_drop) >= .74:
            scores.append((float(contrast - 4*abs(offsets[i])/(.06*length)), i, float(contrast)))
    if not scores:
        return None
    rank, index, contrast = max(scores)
    alternatives = [offsets[i] for s, i, _ in scores if s >= max(min_drop, .80*rank)]
    if alternatives and np.ptp(alternatives) > .06*length:
        return None
    p = centers[index]
    fraction = (p-a)@axis/length
    if fraction < -.025 or fraction > .98:
        return None
    return {**local, "piston": p, "verified_contrast": contrast, "fraction": float(fraction)}


def lumen_support(frame, local):
    a, b, c = (local[k] for k in LANDMARKS)
    length = np.linalg.norm(c-a)
    axis = (c-a)/length
    tangent = np.array([-axis[1], axis[0]])
    portion = (b-a)@axis
    mask = frame["mask"].astype(bool)
    rows, cols = np.indices(mask.shape)
    axial = (cols-a[0])*axis[0] + (rows-a[1])*axis[1]
    width = (cols-a[0])*tangent[0] + (rows-a[1])*tangent[1]
    mid = mask & (axial >= .26*length) & (axial <= .54*length)
    if portion < .08*length or mid.sum() < 25:
        return np.zeros_like(mask)
    radius = .66*np.quantile(abs(width[mid]), .75)
    return mask & (axial >= .08*portion) & (axial <= .85*portion) & (np.abs(width) <= radius)
