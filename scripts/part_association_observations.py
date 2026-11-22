"""Collect pinned local SAM part proposals for the assembly association study."""
import argparse
from pathlib import Path
from time import perf_counter

from medsyringe.additive import project
from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detectors import suppress
from medsyringe.part_geometry import PARTS, PROMPTS, area
from scripts.occlusion_detector import ART, raster
from scripts.recrop_proposal_bank import read_bank
from scripts.rfdetr_verifier import expanded
from scripts.sam3_study import LOCK, Segmenter, configure_cpu

WORK = Path("data/part-association")
EDGE = 756


def inputs(role):
    if role == "fresh":
        from scripts.part_association_reserve import frozen_inputs

        return frozen_inputs()
    rows, frames = read_bank(role)
    # Include every reviewed crowd scene, then cover additional groups and hard negatives
    # with a bounded compute budget. Ordering uses the pre-existing RF proposal bank.
    selected = [r for r in rows if len(r["objects"]) > 1 or
                (role == "fit" and any(o.get("parts") for o in r["objects"]))]
    limits = ((False, 12),) if role == "fit" else ((True, 12), (False, 12))
    for positive, limit in limits:
        seen = {r["group"] for r in selected}
        remaining = [r for r in rows if bool(r["objects"]) == positive and r not in selected]
        ranked = sorted(remaining, key=lambda r: (
            -max((p["score"] for p in frames[r["id"]]["seeds"]), default=0.), r["id"]
        ))
        for r in ranked:
            if r["group"] not in seen:
                selected.append(r)
                seen.add(r["group"])
                limit -= 1
                if not limit:
                    break
    return selected, {r["id"]: frames[r["id"]] for r in selected}


def frame_binding(rows, frames):
    return dict(code=digest(__file__), sam=digest(LOCK), edge=EDGE, rows={
        r["id"]: r.get("derived_sha256", r["sha256"]) for r in rows
    }, proposal_keys={r["id"]: frames[r["id"]]["seeds"][:5] for r in rows})


def windows(im, frame):
    yield (0, 0, *im.size)
    seeds = [p["bbox"] for p in frame["seeds"][:5] if p["score"] >= .12]
    if seeds:
        region = expanded([min(p[0] for p in seeds), min(p[1] for p in seeds),
                           max(p[2] for p in seeds), max(p[3] for p in seeds)], .25)
        if .025 <= area(region) <= .66:
            yield tuple(round(z*n) for z, n in zip(region, im.size*2, strict=True))


def prompt_parts(sam, patch, tokens):
    import torch

    with torch.inference_mode():
        pixels = sam.processor(images=patch, return_tensors="pt",
                               size={"height": EDGE, "width": EDGE}).pixel_values.to(sam.device)
        features = sam.model.get_vision_features(pixel_values=pixels)
        for label in PARTS:
            result = sam.processor.post_process_instance_segmentation(
                sam.model(vision_embeds=features, **tokens[label]), threshold=.14,
                target_sizes=[(EDGE, EDGE)])[0]
            for box, score in zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist(), strict=True):
                rect = [max(0., min(1., k / EDGE)) for k in box]
                if valid_box(rect):
                    yield label, rect, score


def collect(roles):
    configure_cpu()
    sam = None
    for role in roles:
        rows, frames = inputs(role)
        path = WORK / f"parts-{role}.json"
        binding = frame_binding(rows, frames)
        state = read_json(path) if path.exists() else dict(binding=binding, observations={}, seconds={})
        if state["binding"] != binding:
            raise ValueError(f"Part cache moved for {role}.")
        pending = [r for r in rows if r["id"] not in state["observations"]]
        if not pending:
            continue
        if sam is None:
            sam = Segmenter(EDGE, "xpu")
            tokens = {key: sam.processor(text=value, return_tensors="pt").to(sam.device)
                      for key, value in PROMPTS.items()}
        for i, row in enumerate(pending, 1):
            key, im = row["id"], raster(row)
            started = perf_counter()
            found = {k: [] for k in PARTS}
            for j, window in enumerate(windows(im, frames[key])):
                for label, local, score in prompt_parts(sam, im.crop(window), tokens):
                    if box := project(local, window, im.size):
                        found[label].append(dict(bbox=box, score=score, view=j))
            state["observations"][key] = {k: suppress(v, .55)[:18] for k, v in found.items()}
            state["seconds"][key] = round(perf_counter() - started, 6)
            if i % 3 == 0 or i == len(pending):
                write_json(path, state)
                print(role, len(state["observations"]), len(rows), flush=True)


def observations(role, rows, frames):
    path = WORK / f"parts-{role}.json"
    packet = read_json(path)
    if packet["binding"] != frame_binding(rows, frames) or set(packet["observations"]) != {r["id"] for r in rows}:
        raise ValueError(f"Incomplete or changed SAM observations for {role}.")
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roles", nargs="+", choices=("fit", "development", "fresh"))
    collect(parser.parse_args().roles)
