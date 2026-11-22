"""Compare square zoom and elongated-strip views for thin syringe proposals."""
import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

from medsyringe.detection.additive import project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import suppress
from scripts.experiments.crowds.native_candidate_bank import (
    MAX_CANDIDATES, MINIMUM_SCORE, NMS_IOU, archive, cohorts,
)
from scripts.experiments.crowds.native_candidate_reserve import references as archived_native
from scripts.experiments.occlusion.occlusion_detector import EDGE, INITIAL, raster
from scripts.experiments.occlusion.occlusion_relocalization_reserve import refs as archived_refinement
from scripts.experiments.transfer.rfdetr_compare import ranked_region

WORK = Path("data/syringe-assembly-study")
MODES = ("standard", "fine-squares", "strips", "fine-and-strips")
WINDOWS = {"fine-squares": (.43, .43, 3, 3),
           "horizontal": (1., .28, 1, 5), "vertical": (.28, 1., 5, 1)}


def windows(size, rule):
    fw, fh, nx, ny = rule
    width, height = size
    tw, th = max(1, round(fw*width)), max(1, round(fh*height))
    xs = [round(i*(width-tw)/max(1, nx-1)) for i in range(nx)]
    ys = [round(i*(height-th)/max(1, ny-1)) for i in range(ny)]
    return [(x, y, x+tw, y+th) for x, y in product(sorted(set(xs)), sorted(set(ys)))]


def proposals(model, picture, mode):
    views = WINDOWS if mode == "strips" else (mode,)
    if mode == "strips":
        views = ("horizontal", "vertical")
    results, count = [], 0
    for name in views:
        for window in windows(picture.size, WINDOWS[name]):
            count += 1
            for proposal in ranked_region(model, picture.crop(window), EDGE):
                if proposal["score"] >= MINIMUM_SCORE and (bbox := project(proposal["bbox"], window, picture.size)):
                    results.append({**proposal, "bbox": bbox, "view": name})
    return results, count


def variant_pools(base, square, strips):
    return dict(standard=base, **{
        "fine-squares": suppress([*base, *square], NMS_IOU)[:MAX_CANDIDATES],
        "strips": suppress([*base, *strips], NMS_IOU)[:MAX_CANDIDATES],
        "fine-and-strips": suppress([*base, *square, *strips], NMS_IOU)[:MAX_CANDIDATES],
    })


def inputs(role):
    if role == "development":
        rows = cohorts("development")
        return rows, archive(rows, "development")
    if role == "history":
        evaluation = cohorts("evaluation")
        return [*archived_refinement(), *archived_native()], {
            **archive(evaluation, "evaluation"), **archive(archived_native(), "prospective")
        }
    from scripts.experiments.identity.syringe_assembly_reserve import references
    rows = references()
    return rows, archive(rows, "assembly-reserve")


def key_for(role, rows):
    return dict(checkpoint=read_json(INITIAL)["checkpoint_sha256"], rows={r["id"]: r["sha256"] for r in rows},
                protocol=digest(__file__), role=role, windows=WINDOWS)


def collect(role):
    import torch
    from rfdetr import RFDETRNano

    rows, original = inputs(role)
    key = key_for(role, rows)
    # JSON uses arrays for the geometry tuples.
    key["windows"] = {k: list(v) for k, v in WINDOWS.items()}
    path = WORK / f"tiles-{role}.json"
    output = read_json(path) if path.exists() else dict(key=key, frames={})
    if output["key"] != key:
        raise ValueError("Tiles already exist for another protocol.")
    pending = [r for r in rows if r["id"] not in output["frames"]]
    if not pending:
        return
    info = read_json(INITIAL)
    if digest(info["checkpoint"]) != key["checkpoint"]:
        raise ValueError("Shared RF-DETR checkpoint changed.")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model = RFDETRNano.from_checkpoint(info["checkpoint"], device="cpu")
    for i, row in enumerate(pending, 1):
        picture = raster(row)
        started = perf_counter()
        square, square_views = proposals(model, picture, "fine-squares")
        square_seconds = perf_counter() - started
        started = perf_counter()
        strips, strip_views = proposals(model, picture, "strips")
        strip_seconds = perf_counter() - started
        output["frames"][row["id"]] = dict(
            pools=variant_pools(original[row["id"]], square, strips),
            extra_views={"fine-squares": square_views, "strips": strip_views},
            extra_seconds={"fine-squares": square_seconds, "strips": strip_seconds})
        if i % 4 == 0 or i == len(pending):
            write_json(path, output)
            print(role, len(output["frames"]), "/", len(rows), flush=True)


def observations(role, rows):
    evidence = read_json(WORK / f"tiles-{role}.json")
    key = key_for(role, rows)
    key["windows"] = {k: list(v) for k, v in WINDOWS.items()}
    if evidence["key"] != key or set(evidence["frames"]) != {r["id"] for r in rows}:
        raise ValueError(f"Missing or altered raster evidence for {role}.")
    return evidence["frames"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("development", "history", "prospective"))
    collect(parser.parse_args().role)
