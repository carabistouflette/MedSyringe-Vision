"""Screen split boundaries for shared photos that survive crop and rotation."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import MiniBatchKMeans

from medsyringe.data import digest, read_image, read_json, write_json
from scripts.detection_context_data import RELEASE as CONTEXT
from scripts.detection_crowd_source_data import MANIFEST as CROWD
from scripts.detection_empty_scene_data import MANIFEST as NEW_SCENES
from scripts.detection_hillclimb_experiment import ART, HELD_OUT, inputs
from scripts.detection_medbin_data import MANIFEST as MEDBIN
from scripts.detection_real_box_transfer import DENTAL, OPENIMAGES, benchmark_sources
from scripts.detection_device_scene_data import MANIFEST as NATIVE

OUTPUT = ART / "visual_lineage_screen.json"
NEIGHBORS = 8
MAX_FEATURES = 384
VOCABULARY = 320
SEED = 607


def corpus():
    fit_real, fit_simulator, development = benchmark_sources()
    manifests = (
        (OPENIMAGES, "openimages-audit"), (DENTAL, "audit"), (NEW_SCENES, None), (MEDBIN, None),
        (NATIVE, None), (CROWD, None), (ART / "consensus_audit_manifest.json", None),
        (ART / "external_lookalike_manifest.json", None), (ART / "precision_negative_manifest.json", None),
    )
    reserve = [r for scope in HELD_OUT for r in inputs(scope)]
    reserve.extend(r for manifest, split in manifests for r in read_json(manifest)["rows"]
                   if split is None or r["split"] == split)
    for row in read_json(CONTEXT)["rows"]:
        (development if row["split"] == "device-context-development" else reserve).append(row)
    grouped = defaultdict(list)
    for role, rows in (
        ("fit", [*fit_real, *fit_simulator]), ("development", development), ("reserve", reserve)
    ):
        for row in rows:
            grouped[row["sha256"]].append(dict(row=row, role=role))
    collisions = [dict(sha256=key, roles=sorted(roles), ids=[item["row"]["id"] for item in records])
                  for key, records in grouped.items()
                  if len(roles := {item["role"] for item in records}) > 1]
    return {key: records[0] for key, records in grouped.items()}, collisions


def representation(picture, detector):
    frame = ImageOps.contain(picture, (576, 576)).convert("L")
    points, descriptors = detector.detectAndCompute(np.asarray(frame), None)
    if descriptors is None:
        return dict(points=np.empty((0, 2), dtype=np.float32), descriptors=np.empty((0, 128), np.float32))
    positions = np.array([p.pt for p in points], np.float32) / np.float32(frame.size)
    rootsift = np.sqrt(descriptors / descriptors.sum(axis=1, keepdims=True).clip(min=1e-12))
    return dict(points=positions, descriptors=rootsift)


def verification(first, second, matcher):
    if min(len(first["points"]), len(second["points"])) < 16:
        return None
    pairs = matcher.knnMatch(first["descriptors"], second["descriptors"], k=2)
    pairs = [a for a, b in pairs if a.distance < .73 * b.distance]
    pairs.sort(key=lambda p: p.distance)
    unique = {}
    for match in pairs:
        unique.setdefault(match.trainIdx, match)
    pairs = list(unique.values())
    if len(pairs) < 16:
        return None
    left = np.asarray([first["points"][p.queryIdx] for p in pairs])
    right = np.asarray([second["points"][p.trainIdx] for p in pairs])
    transform, keep = cv2.findHomography(left, right, cv2.RANSAC, .007, maxIters=1500, confidence=.995)
    if transform is None or not np.isfinite(transform).all():
        return None
    inlier = keep.ravel() > 0
    count = int(inlier.sum())
    if count < 16 or count / len(pairs) < .55:
        return None
    areas = [float(cv2.contourArea(cv2.convexHull(positions[inlier]))) for positions in (left, right)]
    if min(areas) < .045:
        return None
    return dict(inliers=count, ratio=count / len(pairs), normalized_areas=areas)


def run():
    if OUTPUT.exists():
        raise ValueError("The visual lineage screen has already run.")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(SEED)
    detector = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    matcher = cv2.BFMatcher()
    entries, collisions = corpus()
    keys = sorted(entries)
    fitted_controls = []
    feature = {}
    for key in keys:
        item = entries[key]
        row = item["row"]
        path = Path("data") / row["image"]
        if not path.is_file() or digest(path) != row.get("derived_sha256", key):
            raise ValueError(f"Absent or changed source for the visual audit: {row['id']}.")
        picture = read_image(path)
        feature[key] = representation(picture, detector)
        if item["role"] == "fit" and len(fitted_controls) < 12 and len(feature[key]["points"]) >= 240:
            x, y = picture.size
            crop = picture.crop((round(x*.17), round(y*.13), round(x*.86), round(y*.91)))
            crop = crop.rotate(27, Image.Resampling.BICUBIC, expand=True)
            alternate = representation(crop, detector)
            alternate_key = "control/" + key
            feature[alternate_key] = alternate
            fitted_controls.append((key, alternate_key))

    descriptors = np.concatenate([feature[k]["descriptors"] for k in keys if entries[k]["role"] == "fit"])
    chosen = np.random.default_rng(SEED).choice(len(descriptors), min(50000, len(descriptors)), replace=False)
    words = MiniBatchKMeans(VOCABULARY, batch_size=2048, random_state=SEED, n_init=2, max_iter=48)
    words.fit(descriptors[chosen])
    all_keys = [*keys, *(alternate for _, alternate in fitted_controls)]
    histograms = np.vstack([np.bincount(words.predict(feature[k]["descriptors"]), minlength=VOCABULARY)
                            if len(feature[k]["descriptors"]) else np.zeros(VOCABULARY)
                            for k in all_keys])
    frequencies = (histograms[:len(keys)] > 0).sum(axis=0)
    weights = np.log1p(len(keys) / (1 + frequencies))
    vectors = np.sqrt(histograms) * weights
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -1.)
    roles = np.asarray([entries[k]["role"] for k in keys])
    candidates = set()
    for index, role in enumerate(roles):
        for other in sorted(set(roles) - {role}):
            allowed = np.flatnonzero(roles == other)
            ordered = allowed[np.argsort(-similarity[index, allowed], kind="stable")[:NEIGHBORS]]
            candidates.update(tuple(sorted((index, int(i)))) for i in ordered)
    findings = []
    for i, j in sorted(candidates):
        k, candidate = keys[i], keys[j]
        if evidence := verification(feature[k], feature[candidate], matcher):
            rows = [entries[t]["row"] for t in (k, candidate)]
            findings.append(dict(ids=[r["id"] for r in rows], groups=[r["group"] for r in rows],
                                 sha256=[k, candidate], splits=[roles[i], roles[j]], **evidence))
    controls = []
    for offset, (key, alternate) in enumerate(fitted_controls):
        i, j = len(keys) + offset, keys.index(key)
        neighbors = np.argsort(-similarity[i, :len(keys)], kind="stable")[:NEIGHBORS]
        controls.append(dict(id=entries[key]["row"]["id"], shortlisted=j in neighbors,
                             geometry=verification(feature[alternate], feature[key], matcher)))
    write_json(OUTPUT, dict(
        images=dict(Counter(roles)), source_hashes={k: entries[k]["role"] for k in keys},
        checked_split_pairs=len(candidates), findings=findings, exact_cross_split_hashes=collisions,
        transformed_fit_controls=controls,
        groups_with_matches=[list(pair) for pair in sorted({tuple(p["splits"]) for p in findings})],
        code_sha256=digest(__file__),
        method="RootSIFT bags of visual words shortlist neighbors in each other partition. "
               "Local keypoint geometry requires at least 16 spatially distributed homography inliers. "
               "These checks can detect shared pixels under cropping and rotation. "
               "They cannot establish separate specimens or unseen foundation-model pretraining."))
    print("Lineage audit", len(keys), "images,", len(candidates), "pairs,", len(findings), "flags", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
