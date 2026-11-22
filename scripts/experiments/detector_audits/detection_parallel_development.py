"""Generate overlapping-box, disjoint-mask stress scenes from development masks."""
from itertools import combinations
from math import atan2, degrees
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from medsyringe.data import digest, write_json
from medsyringe.detection.detectors import box_iou
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, inputs, manifest
from scripts.experiments.public_data.detection_public_benchmark import sheet
from scripts.experiments.detector_audits.precision_study import source

OUTPUT = ART / "parallel_device_development.json"
FOLDER = DATA / "parallel-development"
SOURCE = "phlebotomy-development"
SEED = 20261121
SIDE = 576
COUNTS = (2, 3, 4) * 6


def fit_axis(mask):
    y, x = np.nonzero(mask)
    centered = np.column_stack((x-x.mean(), y-y.mean()))
    values, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    major = vectors[:, -1]
    angle = (degrees(atan2(major[1], major[0])) + 90) % 180 - 90
    return angle, float(np.sqrt(values[-1] / max(values[0], 1e-4)))


def isolated(row, label):
    picture = source(row)
    mask = Image.new("L", picture.size)
    poly = label["polygon"]
    vertices = [(x*picture.width,y*picture.height) for x,y in zip(poly[::2],poly[1::2],strict=True)]
    ImageDraw.Draw(mask).polygon(vertices, fill=255)
    bounds = mask.getbbox()
    region = mask.crop(bounds)
    angle, slender = fit_axis(np.asarray(region))
    if slender < 2.3 or max(region.size) < 80:
        return None
    foreground = picture.crop(bounds)
    foreground.putalpha(region)
    straight = foreground.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    reference = region.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
    tight = reference.getbbox()
    if not tight:
        return None
    return dict(source=row, foreground=straight.crop(tight), mask=reference.crop(tight))


def panel(examples, angle, generator):
    source_colours = (142, 168, 196, 218)
    background = generator.choice(source_colours)
    rgb = Image.new("RGB", (SIDE, SIDE), (background, background, background))
    scene = Image.new("RGBA", (SIDE, SIDE))
    masks = []
    cursor = SIDE//2 - sum(h for _, _, h in examples)//2 - 6*(len(examples)-1)
    for item, width, height in examples:
        pos = (SIDE//2-width//2+generator.randint(-14,14), cursor)
        foreground = item["foreground"].resize((width,height), Image.Resampling.BICUBIC)
        scene.alpha_composite(foreground, pos)
        mask = Image.new("L", (SIDE, SIDE))
        mask.paste(item["mask"].resize((width,height), Image.Resampling.NEAREST), pos)
        masks.append((item["source"], mask.rotate(angle, Image.Resampling.NEAREST)))
        cursor += height + generator.randint(3,10)
    rotated = scene.rotate(angle, resample=Image.Resampling.BICUBIC)
    rgb.paste(rotated.convert("RGB"), (0, 0), rotated.getchannel("A"))
    observations = []
    occupied = np.zeros((SIDE,SIDE),dtype=bool)
    for row, mask in masks:
        current = np.asarray(mask) > 0
        if np.any(current & occupied):
            return None
        occupied |= current
        rect = mask.getbbox()
        if rect is None or any(v==0 for v in rect[:2]) or any(v==SIDE for v in rect[2:]):
            return None
        observations.append(dict(kind="syringe", bbox=[x/SIDE for x in rect],
                                 source_id=row["id"], source_group=row["group"],
                                 source_sha256=row["sha256"], raster_mask_pixels=int(current.sum())))
    overlap = max((box_iou(a["bbox"],b["bbox"]) for a,b in combinations(observations,2)), default=0.)
    if overlap < .42:
        return None
    return rgb, observations, overlap


def build():
    if OUTPUT.exists() or FOLDER.exists():
        raise ValueError("The parallel development corpus already exists.")
    sources = inputs(SOURCE)
    pool = [sample for r in sources for label in r["objects"] if (sample:=isolated(r,label)) is not None]
    if len(pool) < 6:
        raise ValueError("This mask split has too few well-resolved source devices.")
    generator = Random(SEED)
    audit = []
    with TemporaryDirectory(prefix="medsyringe-parallel-", dir=DATA) as temporary:
        stage=Path(temporary)/"parallel"
        stage.mkdir()
        for i,count in enumerate(COUNTS,1):
            produced = None
            for _ in range(120):
                items = generator.sample(pool,count)
                samples = []
                for item in items:
                    w,h=item["foreground"].size
                    ratio=min(2.6,generator.uniform(245,325)/w)
                    samples.append((item, round(w*ratio), round(h*ratio)))
                angle=generator.choice((-55,-37,-28,27,39,53)) + generator.uniform(-2,2)
                produced=panel(samples,angle,generator)
                if produced is not None:
                    break
            if produced is None:
                raise ValueError("The fixed parameters could not produce a valid unoccluded stress input.")
            image,targets,overlap=produced
            key=f"parallel-dev-{i:03d}"
            raster=stage/f"{key}.jpg"
            image.save(raster,quality=96,subsampling=0)
            audit.append(dict(id=key,sha256=digest(raster),image=str((FOLDER/raster.name).relative_to("data")),
                              title="Synthetic parallel devices", objects=targets, nominal_angle=angle,
                              maximum_reference_aabb_iou=overlap, split="parallel-development",
                              group="dependent-simulator-development"))
        stage.rename(FOLDER)
    write_json(OUTPUT,dict(seed=SEED, rows=audit, source_sha256=digest(manifest(SOURCE)),
                           generator_sha256=digest(__file__),
                           role="Development only. Disjoint foreground masks and deliberately overlapping "
                                "axis-aligned boxes from known simulator-development polygons. "
                                "Real-image inference is a separate audit."))
    sheet(audit[:12],"parallel-development",4,3)
    print("Generated",len(audit),"parallel mask scenes.",flush=True)


if __name__=="__main__":
    build()
