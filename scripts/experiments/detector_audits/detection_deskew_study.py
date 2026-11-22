"""Test orientation-aligned inference before overlap suppression on crowded development images."""
import argparse
from math import atan2, cos, degrees, radians, sin
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from medsyringe.runtime.appearance_runtime import RELEASE
from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import evaluate
from medsyringe.runtime.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.detection.recall_guard import compare
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA
from scripts.experiments.detector_audits.detection_mosaic_study import run_baseline
from scripts.experiments.detector_audits.detection_parallel_development import OUTPUT as PARALLEL
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

CACHE = DATA / "proposals/parallel-deskew-development.json"
OUTPUT = ART / "parallel_deskew_development_results.json"


def orientation(image):
    picture = ImageOps.contain(image, (832, 832)).convert("L")
    edges = cv2.Canny(np.asarray(picture), 55, 120)
    base = min(picture.size)
    lines = cv2.HoughLinesP(edges, 1., np.pi/180., threshold=23,
                            minLineLength=max(12, base//9), maxLineGap=max(3, base//55))
    if lines is None:
        return None
    delta = lines[:, 0, 2:] - lines[:, 0, :2]
    members = [(float(np.hypot(dx, dy)), (degrees(atan2(dy,dx))+90)%180-90)
               for dx,dy in delta if dx or dy]
    if not members:
        return None
    totals = np.zeros(36, dtype=float)
    for weight, angle in members:
        totals[int((angle+90)//5) % len(totals)] += weight
    smoothed = totals + np.roll(totals, 1) + np.roll(totals, -1)
    dominant = (int(np.argmax(smoothed))+.5)*5-90
    supported = [(length, angle) for length,angle in members
                 if abs((angle-dominant+90)%180-90)<8]
    if len(supported) < 3 or sum(v[0] for v in supported) < .22*sum(p[0] for p in members):
        return None
    axial_y = sum(weight * sin(radians(2 * angle)) for weight, angle in supported)
    axial_x = sum(weight * cos(radians(2 * angle)) for weight, angle in supported)
    angle = degrees(atan2(axial_y, axial_x)) / 2
    return round(angle, 1) if 12 <= abs(angle) <= 78 else None


def restored_box(box, original, rotated, angle):
    x, y, right, bottom = box
    cx, cy = original[0]/2, original[1]/2
    piv_x, piv_y = rotated[0]/2, rotated[1]/2
    c, s = cos(radians(angle)), sin(radians(angle))
    points = []
    for a,b in ((x,y),(right,y),(right,bottom),(x,bottom)):
        tx,ty = a*rotated[0]-piv_x, b*rotated[1]-piv_y
        points.append(((c*tx-s*ty+cx)/original[0],(s*tx+c*ty+cy)/original[1]))
    bounds = [max(0.,min(v[0] for v in points)),max(0.,min(v[1] for v in points)),
              min(1.,max(v[0] for v in points)),min(1.,max(v[1] for v in points))]
    return bounds if valid_box(bounds) else None


def aligned_detection(image, amount, engine):
    fill = tuple(int(v) for v in np.asarray(image)[0, 0, :3])
    deskewed = image.rotate(amount, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
    attempts = run_baseline(deskewed, engine)
    return [{**p,"bbox":box} for p in attempts if (
        box:=restored_box(p["bbox"],image.size,deskewed.size,amount))]


def maximum_matches(targets, proposals):
    if not targets or not proposals:
        return 0
    edges = [[box_iou(obj["bbox"], p["bbox"]) >= .50 for p in proposals] for obj in targets]
    matches = maximum_bipartite_matching(csr_matrix(edges), perm_type="column")
    return int(np.count_nonzero(matches != -1))


def binding():
    return dict(inputs=digest(PARALLEL), active_release=digest(RELEASE), implementation=digest(__file__))


def collect(limit):
    configure_cpu()
    rows=read_json(PARALLEL)["rows"]
    key=binding()
    saved=read_json(CACHE) if CACHE.exists() else dict(key=key,frames={})
    if saved["key"]!=key:
        raise ValueError("The parallel-scene inference does not match its inputs.")
    todo=[r for r in rows if r["id"] not in saved["frames"]]
    root=Path.cwd()
    engine=MedSyringeEngine(root)
    try:
        configure_device(engine,root,"xpu")
        if engine.identity.get("primary_backend")!="xpu":
            raise ValueError("Use the verified Intel environment for this comparison.")
        for row in todo[:limit]:
            picture=source(row)
            tick=perf_counter()
            baseline=run_baseline(picture,engine)
            amount=orientation(picture)
            detected=aligned_detection(picture,amount,engine) if amount is not None else baseline
            # The known synthetic transform is a localization ceiling and never enters selection.
            aligned=aligned_detection(picture,-row["nominal_angle"],engine)
            saved["frames"][row["id"]]=dict(baseline=baseline,estimated_angle=amount,
                                           estimated=detected,oracle=aligned,seconds=perf_counter()-tick)
            write_json(CACHE,saved)
            print("Parallel orientation",len(saved["frames"]),"/",len(rows),flush=True)
    finally:
        engine.close()


def measure():
    rows=read_json(PARALLEL)["rows"]
    saved=read_json(CACHE)
    frames=saved["frames"]
    if saved["key"]!=binding() or set(frames)!={r["id"] for r in rows} or OUTPUT.exists():
        raise ValueError("Run every fixed development scene before measuring.")
    variants = ("baseline", "estimated", "oracle")
    measured={kind:evaluate(rows,{k:v[kind] for k,v in frames.items()},0.) for kind in variants}
    cases = {}
    for row in rows:
        frame = frames[row["id"]]
        targets = row["objects"]
        cases[row["id"]] = dict(expected=len(targets), estimated_angle=frame["estimated_angle"],
                               computed_angle=-row["nominal_angle"], views={
                                   kind: dict(count=len(frame[kind]),
                                              maximum_matches=maximum_matches(targets, frame[kind]))
                                   for kind in variants})
    write_json(OUTPUT,dict(outcomes={k:short(v) for k,v in measured.items()},
                           preservation=compare(measured["baseline"],measured["estimated"]),
                           estimated_views=sum(f["estimated_angle"] is not None for f in frames.values()),
                           median_seconds=median(f["seconds"] for f in frames.values()),
                           cases=cases,
                           input_sha256=digest(PARALLEL),predictions_sha256=digest(CACHE),
                           interpretation="Synthetic overlap stress. The oracle uses the known rendering rotation. "
                                          "The automatically estimated angle uses only image pixels. Optional maximum "
                                          "matching measures box assignment ambiguity with ranked AP kept separately."))
    print({k:short(v) for k,v in measured.items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("collect","measure"))
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.action=="collect":
        collect(args.limit)
    else:
        measure()
