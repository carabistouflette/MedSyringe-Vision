"""Create a matched-budget, split-first masked composition training ablation."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter
from random import Random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from medsyringe.data import digest, read_image, read_json, write_json
from scripts.detection_hillclimb_corpus import ART, COCO, DATA, LOCK
from scripts.detection_hillclimb_train import INITIAL, INITIAL_RECORD
from scripts.precision_study import source
from scripts.rfdetr_transfer import ANNOTATIONS

SOURCE = DATA/"whole-frame"
SETUP = ART/"composite_ablation_data.json"
POSITIVES = ART/"public_additional_fit.json"
OPEN_IMAGES = ART/"openimages_additional_manifest.json"
ARMS = ("repeated", "composite")
SEED = 6813251
EXTRA = 64


def verify_coco():
    raw = read_json(COCO)
    if raw["corpus_sha256"] != digest(LOCK):
        raise ValueError("The source corpus is not the sealed simulator fit and development set.")
    for role in ("train","valid"):
        if digest(SOURCE/role/ANNOTATIONS)!=raw[role]["annotation_sha256"]:
            raise ValueError("A simulator split changed.")
    return read_json(SOURCE/"train"/ANNOTATIONS)


def source_mask(picture, polygons):
    output = Image.new("L",picture.size)
    draw = ImageDraw.Draw(output)
    for polygon in polygons:
        coords = list(zip(polygon[::2],polygon[1::2],strict=True))
        draw.polygon(coords, fill=255)
    bounds = output.getbbox()
    if not bounds:
        raise ValueError("An existing annotation supplies an empty mask.")
    return picture.crop(bounds),output.crop(bounds)


def polygons_from_mask(mask):
    result = []
    contours,_ = cv2.findContours(np.asarray(mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for trace in contours:
        if len(trace)<3 or cv2.contourArea(trace)<2:
            continue
        result.append(trace.reshape(-1,2).astype(float).ravel().tolist())
    return result


def composite(front,mask,background,rng):
    scene = background.copy()
    scene.thumbnail((480,480),Image.Resampling.LANCZOS)
    angle = rng.uniform(-85.,85.)
    front = front.rotate(angle,Image.Resampling.BICUBIC,expand=True)
    mask = mask.rotate(angle,Image.Resampling.NEAREST,expand=True)
    target = min(max(scene.size)*rng.uniform(.12,.72),min(scene.size)*.72)
    scale = min(2.,target/max(front.size))
    size = tuple(max(1,round(v*scale)) for v in front.size)
    front = ImageEnhance.Brightness(front.resize(size,Image.Resampling.LANCZOS)).enhance(rng.uniform(.8,1.2))
    mask = mask.resize(size,Image.Resampling.NEAREST)
    xy = tuple(rng.randint(0,large-small) for small,large in zip(mask.size,scene.size,strict=True))
    if rng.random()<.33:
        # A local copy of the background acts as an occluder. The mask
        # records the visible pixels only.
        nominal = int(np.count_nonzero(np.asarray(mask)))
        cut = max(1,round(.12*mask.width))
        x = rng.randrange(max(1,mask.width-cut))
        visible = mask.copy()
        visible.paste(0,(x,0,min(x+cut,mask.width),mask.height))
        if np.count_nonzero(np.asarray(visible)) >= .75*nominal:
            mask = visible
    canvas = Image.new("L",scene.size)
    canvas.paste(mask,xy)
    regions = polygons_from_mask(canvas)
    bounds = canvas.getbbox()
    if not regions or bounds is None or min(bounds[2]-bounds[0],bounds[3]-bounds[1])<3:
        raise ValueError("A synthetic sample lacks a valid visible target.")
    scene.paste(front,xy,mask.filter(ImageFilter.GaussianBlur(.45)))
    return scene,bounds,regions,int(np.count_nonzero(np.asarray(canvas)))


def copy_frame(reference,target):
    if not target.exists():
        target.hardlink_to(reference.resolve())
    elif digest(reference)!=digest(target):
        raise ValueError("The matched arms have inconsistent source images.")


def build():
    destination = DATA/"composite-study"
    if SETUP.exists() or destination.exists():
        raise ValueError("The composition experiment dataset already exists.")
    with TemporaryDirectory(dir=DATA,prefix="composite-staging-") as temporary:
        stage = Path(temporary)/"study"
        result = write_datasets(stage)
        stage.rename(destination)
    write_json(SETUP,result)
    print("Matched datasets",result["arms"],flush=True)


def write_datasets(stage):
    coco = verify_coco()
    background_rows = [r for path in (POSITIVES,OPEN_IMAGES) for r in read_json(path)["rows"]
                       if r["split"]=="fit" and not r["objects"]]
    if len({r["group"] for r in background_rows})<12:
        raise ValueError("Insufficient independent fitting background sources.")
    backgrounds = [(r,source(r)) for r in background_rows]
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    eligible = [row for row in coco["images"] if len(by_image[row["id"]])==1 and
                by_image[row["id"]][0]["segmentation"] and
                max(by_image[row["id"]][0]["bbox"][2:])>=45 and min(by_image[row["id"]][0]["bbox"][2:])>=9]
    rng = Random(SEED)
    selected = rng.sample(eligible,EXTRA)
    assignments = []
    for arm in ARMS:
        root = stage/arm
        root.mkdir(parents=True)
        (root/"train").mkdir()
        (root/"valid").symlink_to((SOURCE/"valid").resolve(),target_is_directory=True)
        for row in coco["images"]:
            copy_frame(SOURCE/"train"/row["file_name"],root/"train"/row["file_name"])
    arms = {arm:dict(images=coco["images"].copy(),annotations=coco["annotations"].copy()) for arm in ARMS}
    next_image = max(row["id"] for row in coco["images"])+1
    next_ann = max(row["id"] for row in coco["annotations"])+1
    for i,row in enumerate(selected):
        reference = SOURCE/"train"/row["file_name"]
        foreground,alpha = source_mask(read_image(reference),by_image[row["id"]][0]["segmentation"])
        background,raster = backgrounds[i%len(backgrounds)]
        picture,box,contours,area = composite(foreground,alpha,raster,rng)
        x0,y0,x1,y1 = box
        ancestry = dict(image_id=next_image+i,foreground=row["file_name"],foreground_sha256=digest(reference),
                        background_id=background["id"],background_group=background["group"],
                        background_sha256=background["sha256"],background_license=background["license"])
        assignments.append(ancestry)
        for arm in ARMS:
            original = arm=="repeated"
            filename = f"extra-{i:03d}"+(reference.suffix if original else ".png")
            here = stage/arm/"train"/filename
            if original:
                copy_frame(reference,here)
                ann = by_image[row["id"]][0]
                metadata = {**row}
            else:
                picture.save(here)
                ann = dict(bbox=[x0,y0,x1-x0,y1-y0],area=area,segmentation=contours,category_id=1,iscrowd=0)
                metadata = dict(width=picture.width,height=picture.height)
            arms[arm]["images"].append({**metadata,"id":next_image+i,"file_name":filename})
            arms[arm]["annotations"].append({**ann,"id":next_ann+i,"image_id":next_image+i})
    found = {}
    for arm,parts in arms.items():
        path = stage/arm/"train"/ANNOTATIONS
        write_json(path,{**coco,**parts})
        found[arm] = dict(annotation_sha256=digest(path),images=len(parts["images"]),targets=len(parts["annotations"]))
    if len({(m["images"],m["targets"]) for m in found.values()})!=1:
        raise ValueError("The control and treatment must have matched sample sizes.")
    return dict(seed=SEED,arms=found,provenance=assignments, code_sha256=digest(Path(__file__)),
                         source_sha256=digest(COCO),negative_manifests={str(p):digest(p) for p in (POSITIVES,OPEN_IMAGES)},
                         design="Only source and background fit groups. All development and audit photos remain untouched. "
                                "The matched control repeats each selected foreground's actual scene once. "
                                "The quality check limits foreground enlargement to 2x and artificial occlusion to 25%.")


def train(arm):
    import torch
    from rfdetr import RFDETRSegNano

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    data = DATA/"composite-study"/arm
    lock = read_json(SETUP)
    if digest(data/"train"/ANNOTATIONS)!=lock["arms"][arm]["annotation_sha256"]:
        raise ValueError("Training inputs changed.")
    verify_coco()
    if read_json(INITIAL_RECORD)["checkpoint_sha256"]!=digest(INITIAL):
        raise ValueError("The archived initial checkpoint changed.")
    out = DATA/"runs"/f"mask-composite-{arm}"
    record = ART/f"rfdetr_composite_{arm}_run.json"
    if record.exists() and read_json(record)["status"]=="completed":
        raise ValueError("The matched arm has already completed.")
    config = dict(dataset_dir=str(data),output_dir=str(out),device="cpu",epochs=1,batch_size=4,
                  resolution=240,seed=SEED,lr=.00012,lr_encoder=0,num_workers=0,num_sanity_val_steps=0,
                  use_ema=False,multi_scale=False,save_dataset_grids=False,run_test=False,
                  progress_bar=None,train_log_on_step=False)
    prior = dict(dataset_sha256=digest(SETUP),warm_start_sha256=digest(INITIAL),configuration=config)
    write_json(record,dict(status="started",**prior))
    model = RFDETRSegNano(device="cpu",pretrain_weights=str(INITIAL),segmentation_head=True,
                         freeze_encoder=True,num_classes=1)
    begin=perf_counter()
    model.train(**config)
    checkpoint=out/"checkpoint_best_total.pth"
    shutil.copyfile(out/"metrics.csv",ART/f"rfdetr_composite_{arm}_metrics.csv")
    write_json(record,dict(status="completed",seconds=perf_counter()-begin,checkpoint=str(checkpoint),
                            checkpoint_sha256=digest(checkpoint),**prior))
    print("Finished CPU matched arm",arm,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("build","train"))
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.action=="build":
        build()
    elif args.arm:
        train(args.arm)
    else:
        parser.error("Choose the matched training arm.")
