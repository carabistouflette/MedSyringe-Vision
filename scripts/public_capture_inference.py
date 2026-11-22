"""Compare whole, aspect-preserving and tiled public-detector inference."""
from pathlib import Path
from statistics import median
from time import perf_counter

from PIL import ImageOps

from medsyringe.additive import overlapping_tiles, project
from medsyringe.data import digest, read_json, valid_box, write_json
from medsyringe.detectors import suppress
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.detection_hillclimb_experiment import ART
from scripts.expanded_detection_study import short
from scripts.precision_study import source
from scripts.public_augmentation_evaluate import CACHE, EDGE, FLOOR, bound, validation
from scripts.rfdetr_compare import ranked_region
from scripts.sam3_study import configure_cpu

OUTPUT = ART / "public_capture_inference_development.json"
CANDIDATES = CACHE / "capture-views.json"


def letterboxed(model, picture):
    padded = ImageOps.contain(picture, (EDGE, EDGE))
    left = (EDGE - padded.width) // 2
    top = (EDGE - padded.height) // 2
    border = picture.resize((1, 1)).getpixel((0, 0))
    canvas = ImageOps.expand(padded, (left, top, EDGE - padded.width - left, EDGE - padded.height - top),
                             fill=border)
    for proposal in ranked_region(model, canvas, EDGE):
        w, h = padded.size
        box = [min(1., max(0., (coord * EDGE - offset) / size)) for coord, offset, size in zip(
            proposal["bbox"], (left, top, left, top), (w, h, w, h), strict=True)]
        if valid_box(box):
            yield {**proposal, "bbox": box}


def compare_views():
    from rfdetr import RFDETRNano

    if OUTPUT.exists():
        raise FileExistsError("The development view comparison already exists.")
    configure_cpu()
    rows = validation()
    checkpoint, binding = bound("capture", rows)
    whole_path = CACHE / "capture.json"
    whole = read_json(whole_path)
    if whole["binding"] != binding or set(whole["frames"]) != set(rows):
        raise ValueError("Missing whole-frame references from the matched trial.")
    key = dict(implementation=digest(__file__), whole=digest(whole_path), checkpoint=binding["checkpoint"])
    saved = read_json(CANDIDATES) if CANDIDATES.exists() else dict(binding=key, frames={})
    if saved["binding"] != key:
        raise ValueError("Inference evidence differs from this comparison.")
    model = RFDETRNano.from_checkpoint(checkpoint, device="cpu")
    for identifier, row in rows.items():
        if identifier in saved["frames"]:
            continue
        picture = source(row)
        started = perf_counter()
        resized = list(letterboxed(model, picture))
        letterbox_seconds = perf_counter() - started
        started = perf_counter()
        tiled = []
        for window in overlapping_tiles(picture.size):
            for proposal in ranked_region(model, picture.crop(window), EDGE):
                box = project(proposal["bbox"], window, picture.size)
                if box is not None:
                    tiled.append({**proposal, "bbox": box})
        saved["frames"][identifier] = dict(letterbox=suppress(resized, .45),
                                           tiled=suppress(tiled, .45),
                                           letterbox_seconds=letterbox_seconds,
                                           tile_seconds=perf_counter() - started)
        write_json(CANDIDATES, saved)
    base = {identifier: values["ranked"] for identifier, values in whole["frames"].items()}
    views = {"whole": base}
    for view in ("letterbox", "tiled"):
        alternative = {identifier: values[view] for identifier, values in saved["frames"].items()}
        views[view] = alternative
        views[f"whole_plus_{view}"] = {
            identifier: suppress([*base[identifier], *other], .45) for identifier, other in alternative.items()}
    domains = {r.get("domain", "transparent") for r in rows.values()}
    subsets = {"all": list(rows.values()), **{
        domain: [row for row in rows.values() if row.get("domain", "transparent") == domain]
        for domain in sorted(domains)}}
    reference = evaluate(subsets["all"], base, FLOOR)
    results = {name: {domain: short(evaluate(group, ranked, FLOOR))
                      for domain, group in subsets.items()} for name, ranked in views.items()}
    retention = {name: compare(reference, evaluate(subsets["all"], ranked, FLOOR))
                 for name, ranked in views.items()}
    write_json(OUTPUT, dict(results=results, retention=retention, floor=FLOOR,
                            median_extra_seconds={key: median(frame[key] for frame in saved["frames"].values())
                                                  for key in ("letterbox_seconds", "tile_seconds")},
                            binding=key, view_cache_sha256=digest(CANDIDATES),
                            scope="Real development photos. The fixed capture checkpoint and floor match the "
                                  "training-family comparison. No final test photos or Fil Rouge photos included."))
    print({name: {k: row["all"][k] for k in ("tp", "fp", "f1", "ap50")} for name, row in results.items()},
          flush=True)


if __name__ == "__main__":
    compare_views()
