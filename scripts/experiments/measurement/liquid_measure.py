"""Localize the syringe silhouette and measure the appearance of its colored interior."""
import argparse
from pathlib import Path
from functools import lru_cache
from time import perf_counter

import cv2
import numpy as np
from PIL import Image,ImageOps

from medsyringe.measurement.liquid_appearance import appearance,relative_opacity,unavailable



@lru_cache(maxsize=1)
def localizer(device):
    from medsyringe.measurement.liquid_region import LiquidRegion

    return LiquidRegion(device)


def measure_crop(model,image):
    import torch

    crop=ImageOps.contain(image.convert("RGB"),(420,420))
    with torch.inference_mode():
        record=model.processor(images=crop,text="syringe",
                               size={"height":model.edge,"width":model.edge},return_tensors="pt").to(model.device)
        result=model.processor.post_process_instance_segmentation(model.model(**record),threshold=.48)[0]
    chosen=None
    for mask,score in zip(result["masks"].cpu().numpy(),result["scores"].cpu().tolist(),strict=True):
        area=float(mask.mean())
        if area<.15 or area>.95:
            continue
        rank=(score*area,mask,score)
        if chosen is None or rank[0]>chosen[0]:
            chosen=rank
    if chosen is None:
        return unavailable("The syringe mask could not be validated.")
    _,mask,score=chosen
    mask=cv2.resize(mask.astype(np.uint8),crop.size,interpolation=cv2.INTER_NEAREST)
    prior=localizer(model.device).mask(crop)
    stats=appearance(np.asarray(crop),mask,prior)
    stats["syringe_mask_score"]=score
    return stats


def study(cohort):
    from medsyringe.data import read_json,write_json
    from scripts.experiments.expanded.sam3_study import Segmenter,configure_cpu

    configure_cpu()
    model=Segmenter(504,"xpu")
    manifest=Path(f"artifacts/marking_{cohort}_manifest.json")
    files=read_json(manifest)["files"]
    report={}
    for row in files:
        if not row.get("image"):
            continue
        started=perf_counter()
        with Image.open(row["image"]) as image:
            observed=measure_crop(model,image)
        report[row["id"]]=dict(**observed,seconds=perf_counter()-started)
        print(row["id"],observed.get("status"),observed.get("lightness"),flush=True)
    output=Path(f"artifacts/liquid_{cohort}_results.json")
    write_json(output,dict(scope="Same-session visible color measurements. Absolute optical opacity is unavailable.",
                           cohort=cohort,frames=report))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_subparsers(dest="action",required=True)
    exp=action.add_parser("study")
    exp.add_argument("cohort",choices=("reference","detected"))
    pair=action.add_parser("opacity")
    pair.add_argument("observed",type=Path)
    pair.add_argument("clear_reference",type=Path)
    pair.add_argument("--matched-setup",action="store_true")
    args=parser.parse_args()
    if args.action=="study":
        study(args.cohort)
    else:
        with Image.open(args.observed) as observed,Image.open(args.clear_reference) as reference:
            print(relative_opacity(np.asarray(observed.convert("RGB")),np.asarray(reference.convert("RGB")),
                                   matched_setup=args.matched_setup))


if __name__=="__main__":
    main()
