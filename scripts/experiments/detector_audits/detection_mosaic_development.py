"""Build crowded stress images from photos that already have a development role."""
from collections import Counter
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps

from medsyringe.data import digest, leakage, read_json, valid_box, write_json
from scripts.experiments.detector_training.detection_context_data import RELEASE as CONTEXT
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, inputs
from scripts.experiments.public_data.detection_public_benchmark import order, sheet
from scripts.experiments.transfer.detection_real_box_transfer import benchmark_sources
from scripts.experiments.detector_audits.precision_study import source

FOLDER = DATA / "crowded-development-mosaics"
MANIFEST = ART / "crowded_mosaic_development.json"
SEED = 64122
EDGE = 512
DESIGNS = (0, 1, 2, 3, 4) * 6


def inputs_for_mosaic():
    fit, simulated, _ = benchmark_sources()
    development = inputs("public-development") + [
        r for r in read_json(CONTEXT)["rows"] if r["split"] == "device-context-development"]
    bounded = [{**r, "split": "train"} for r in fit + simulated]
    validation = [{**r, "split": "development"} for r in development]
    if leakage([*bounded, *validation]):
        raise ValueError("A captured reference overlaps a fitting image.")
    return development


def choose(positive, negative, count, random):
    items = random.sample(positive, count) + random.sample(negative, 4-count)
    random.shuffle(items)
    return items


def assemble(parents, random):
    side = 2 * EDGE
    canvas = Image.new("RGB", (side, side), (210, 210, 211))
    boxes, placement = [], []
    for slot, (source_row, raster) in enumerate(parents):
        scale = random.choice((.77, .89, 1.))
        maximum = round(EDGE*scale)
        picture = ImageOps.contain(raster, (maximum, maximum), Image.Resampling.LANCZOS)
        local_x = random.randint(0, EDGE-picture.width)
        local_y = random.randint(0, EDGE-picture.height)
        x, y = (slot % 2)*EDGE+local_x, (slot // 2)*EDGE+local_y
        canvas.paste(picture, (x, y))
        placement.append(dict(id=source_row["id"], image_sha256=source_row["sha256"],
                              group=source_row["group"], x=x, y=y, size=list(picture.size)))
        for box in source_row["objects"]:
            left, top, right, bottom = box["bbox"]
            mapped = [(x+left*picture.width)/side, (y+top*picture.height)/side,
                      (x+right*picture.width)/side, (y+bottom*picture.height)/side]
            if not valid_box(mapped):
                raise ValueError("A reference transformed outside the synthetic image.")
            boxes.append(dict(kind="syringe", bbox=mapped, source_id=source_row["id"]))
    return canvas, boxes, placement


def build():
    if FOLDER.exists() or MANIFEST.exists():
        raise ValueError("The crowd development set has already been generated.")
    rows = inputs_for_mosaic()
    pairs = [(r, ImageOps.contain(source(r), (EDGE, EDGE), Image.Resampling.LANCZOS))
             for r in sorted(rows, key=lambda item: order(item["id"]))]
    positive = [(r, i) for r, i in pairs if len(r["objects"]) == 1]
    negative = [(r, i) for r, i in pairs if not r["objects"]]
    generator = Random(SEED)
    records = []
    with TemporaryDirectory(prefix="medsyringe-crowd-mosaics-", dir=DATA) as temp:
        stage = Path(temp) / "mosaics"
        stage.mkdir()
        for index, nominal in enumerate(DESIGNS, 1):
            parents = choose(positive, negative, nominal, generator)
            raster, objects, placed = assemble(parents, generator)
            if len(objects) != nominal:
                raise ValueError("Unexpected mosaic count.")
            name = f"mosaic-development-{index:03d}"
            file = stage / f"{name}.jpg"
            raster.save(file, format="JPEG", quality=95, subsampling=0)
            records.append(dict(id=name, objects=objects, sha256=digest(file),
                                image=str((FOLDER / file.name).relative_to("data")),
                                title="Synthetic development mosaic", split="mosaic-development",
                                parents=placed, group="dependent-development-sources"))
        stage.rename(FOLDER)
    write_json(MANIFEST, dict(seed=SEED, placement_code_sha256=digest(__file__), rows=records,
                              parent_pool_sha256={r["id"]: r["sha256"] for r in rows},
                              distribution=dict(Counter(len(r["objects"]) for r in records)),
                              purpose="Development-only count and scale stress. Montages retain separate photo "
                                      "rectangles. Parent images recur and their original acquisition groups are recorded. "
                                      "These scenes are synthetic and are excluded from independent test outcomes."))
    sheet(records[:12], "mosaic-development-sample", 4, 3)
    print("Generated", len(records), "development montages with exact transformed references.", flush=True)


if __name__ == "__main__":
    build()
