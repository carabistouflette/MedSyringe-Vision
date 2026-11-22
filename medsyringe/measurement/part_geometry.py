"""Describe the relative order of candidate syringe components."""
from itertools import product

import numpy as np

PARTS = ("barrel", "flange", "rod", "seal")
PROMPTS = {
    "barrel": "syringe barrel",
    "flange": "syringe finger flange",
    "rod": "syringe plunger rod",
    "seal": "rubber piston seal inside a syringe",
}


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def containment(part, body):
    l, t, r, b = [max(part[0], body[0]), max(part[1], body[1]),
                  min(part[2], body[2]), min(part[3], body[3])]
    return max(0., r - l) * max(0., b - t) / max(1e-9, area(part))


def geometry(candidate, parts, points, width, height):
    """Return part evidence and geometry without treating missing pieces as background labels."""
    extent = area(candidate)
    options = {}
    for key in PARTS:
        options[key] = [entry for entry in parts[key] if containment(entry["bbox"], candidate) >= .65
                        and area(entry["bbox"]) <= extent * (.94 if key == "barrel" else .58)][:3]
    axis = np.array([1., 0.])
    if len(points) >= 4:
        eigenvalues, directions = np.linalg.eigh(np.cov(points.T))
        axis = directions[:, -1]
    cross = np.array([-axis[1], axis[0]])
    scale = np.array([width, height])
    size = scale * (np.asarray(candidate[2:]) - candidate[:2])
    length = max(1., np.linalg.norm(size))
    center = {key: [np.array([(r["bbox"][0] + r["bbox"][2]) / 2,
                              (r["bbox"][1] + r["bbox"][3]) / 2]) * scale for r in entries]
              for key, entries in options.items()}

    def alignment(first, second):
        delta = second - first
        return abs(delta @ axis) >= .10 * length and abs(delta @ cross) <= .14 * length

    pair = 0.
    triple = 0.
    for i, j in product(range(len(center["barrel"])), range(len(center["rod"]))):
        barrel, rod = center["barrel"][i], center["rod"][j]
        if not alignment(barrel, rod):
            continue
        strength = min(options["barrel"][i]["score"], options["rod"][j]["score"])
        pair = max(pair, strength)
        for k, flange in enumerate(center["flange"]):
            left, right = (flange - barrel) @ axis, (rod - flange) @ axis
            if left * right > 0 and abs((flange - barrel) @ cross) <= .14 * length:
                triple = max(triple, min(strength, options["flange"][k]["score"]))
    scores = [max((p["score"] for p in options[key]), default=0.) for key in PARTS]
    return [*scores, pair, triple], options
