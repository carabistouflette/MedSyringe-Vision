"""Evaluate glare-resistant color measurements with barrel-aligned support."""
import argparse
from itertools import product
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import ImageOps

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.measurement.liquid_appearance import appearance
from medsyringe.measurement.liquid_color import describe_region, display_rgb, lab_pixels
from medsyringe.measurement.liquid_region import HEAD, RegionFeatures
from scripts.experiments.expanded.expanded_color_context import controls
from scripts.experiments.expanded.expanded_liquid import ART, LOCK as FIRST_LOCK, mask_syringe, records, support
from scripts.experiments.expanded.expanded_proposals import image_for
from scripts.experiments.expanded.sam3_study import Segmenter, configure_cpu

LOCK = ART / "liquid_structure_selection.json"
AUDIT = ART / "liquid_structure_regression.json"


def screen(frame, region, width_fraction, min_run, cut_glare):
    mask = frame["mask"].astype(bool)
    if not region.any() or not mask.any():
        return region
    lab = frame["lab"]
    if cut_glare:
        selected = lab[region]
        brightness = np.median(selected[:, 0])
        chroma = np.linalg.norm(lab[..., 1:], axis=-1)
        region = region & ~((lab[..., 0] >= min(94., brightness+14.)) &
                            (chroma < np.median(chroma[region])*.85))
    y, x = mask.nonzero()
    coordinates = np.column_stack((x, y))
    _, basis = np.linalg.eigh(np.cov(coordinates.T))
    longitudinal = coordinates@basis[:, 1]
    transverse = coordinates@basis[:, 0]
    bins = np.linspace(*np.quantile(longitudinal, [.02, .98]), 19)
    locations = np.digitize(longitudinal, bins)
    marked = region[y, x]
    accepted = []
    run = []
    for index in range(1, len(bins)):
        inside = locations == index
        section = inside & marked
        if np.sum(section) >= 9 and np.sum(section)/np.sum(inside) >= width_fraction:
            axis_width = np.ptp(np.quantile(transverse[inside], [.08, .92]))
            colored_width = np.ptp(np.quantile(transverse[section], [.08, .92]))
            if colored_width >= width_fraction*axis_width:
                run.append(index)
                continue
        if len(run) >= min_run:
            accepted.extend(run)
        run = []
    if len(run) >= min_run:
        accepted.extend(run)
    chosen = np.zeros_like(region)
    keep = marked & np.isin(locations, accepted)
    chosen[y[keep], x[keep]] = True
    return chosen


def score(items, strategy, model, legacy):
    cases = {}
    for row, frame in items:
        logits = cv2.resize(frame["grid"] @ legacy["weights"] + legacy["bias"], frame["mask"].shape[::-1])
        initial = appearance(frame["raw"], frame["mask"], logits >= legacy["cutoff"])
        region = support(frame, model["model"], model["cutoff"])
        if strategy:
            region = screen(frame, region, **strategy)
        observed = describe_region(frame["rgb"], region, str(frame["profile"]))
        expected = frame["distance"] is not None
        case = dict(expected=expected, legacy=initial, candidate=observed)
        if expected:
            near = frame["distance"] < .023
            reference = np.median(frame["lab"][near], axis=0)
            for label, measure in (("legacy", initial), ("candidate", observed)):
                if measure["lightness"]:
                    x, y, r, b = measure["region"]
                    h, w = near.shape
                    coverage = near[int(y*h):int(np.ceil(b*h)), int(x*w):int(np.ceil(r*w))].sum()/near.sum()
                    if coverage >= .35:
                        case[label+"_error"] = float(abs(measure["lightness"]["lstar"]-reference[0]))
            if observed["apparent_color"]:
                case["candidate_delta_e76"] = float(np.linalg.norm(observed["apparent_color"]["lab"]-reference))
        cases[row["id"]] = case
    return cases


def counts(cases, method):
    errors = [r[method+"_error"] for r in cases.values() if method+"_error" in r]
    return dict(images=len(cases), supported=len(errors), median_lstar_error=median(errors) if errors else None,
                false_reports=sum(not r["expected"] and bool(r[method]["lightness"]) for r in cases.values()))


def inputs():
    paths = (__file__, HEAD, FIRST_LOCK, "artifacts/expanded/liquid_traces.json",
             "artifacts/expanded/color_context_freeze.json", "scripts/experiments/expanded/expanded_liquid.py")
    return {str(p): digest(p) for p in paths}


def select():
    if LOCK.exists():
        raise ValueError("The appearance candidate is already frozen.")
    items = list(records("calibration"))
    model = read_json(FIRST_LOCK)["winner"]
    legacy = read_json(HEAD)
    cases = score(items, None, model, legacy)
    base = counts(cases, "legacy")
    trials = []
    for width, run, glare in product((.14, .22, .31), (2, 3), (False, True)):
        strategy = dict(width_fraction=width, min_run=run, cut_glare=glare)
        evidence = score(items, strategy, model, legacy)
        trials.append(dict(strategy=strategy, counts=counts(evidence, "candidate")))
    key = lambda t: (t["counts"]["supported"], -t["counts"]["false_reports"],
                     -t["counts"]["median_lstar_error"], t["strategy"]["min_run"])
    choice = max((t for t in trials if t["counts"]["supported"] >= base["supported"] and
                  t["counts"]["median_lstar_error"] <= base["median_lstar_error"]+1.), key=key, default=None)
    write_json(LOCK, dict(inputs=inputs(), legacy=base, unguarded=counts(cases, "candidate"), trials=trials,
                          selected=choice, shadow=choice or max(trials, key=key)))
    print("Calibration", base, counts(cases, "candidate"), choice, flush=True)


def audit():
    freeze = read_json(LOCK)
    if freeze["inputs"] != inputs():
        raise ValueError("The appearance study inputs changed.")
    model = read_json(FIRST_LOCK)["winner"]
    legacy = read_json(HEAD)
    strategy = freeze["shadow"]["strategy"]
    clear = score(list(records("test")), strategy, model, legacy)
    segmenter, encoder = Segmenter(504, "xpu"), RegionFeatures("xpu")
    public, unguarded = {}, {}
    for row in controls():
        image = ImageOps.contain(crop(image_for(row, Path(".")), row["objects"][0]["bbox"]), (420, 420))
        display, profile = display_rgb(image)
        rgb = np.asarray(display)
        frame = dict(raw=np.asarray(image.convert("RGB")), rgb=rgb, profile=profile, lab=lab_pixels(rgb),
                     mask=mask_syringe(segmenter, image), grid=encoder.extract(image), distance=None)
        public.update(score([(row, frame)], strategy, model, legacy))
        unguarded.update(score([(row, frame)], None, model, legacy))
    totals = {scope: {method: counts(cases, method) for method in ("legacy", "candidate")}
              for scope, cases in (("clear", clear), ("public", public))}
    totals["public"]["unguarded"] = counts(unguarded, "candidate")
    passed = bool(freeze["selected"] and all(r["candidate"]["false_reports"] == 0
                                           for r in totals.values()))
    write_json(AUDIT, dict(passed=passed, totals=totals, strategy=strategy, clear=clear, public=public,
                           selection_sha256=digest(LOCK), scope="Previously consulted appearance controls."))
    print("Appearance regression", passed, totals, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "audit"))
    configure_cpu()
    (select if parser.parse_args().stage == "select" else audit)()
