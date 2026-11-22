"""Evaluate the official WeDetect Tiny weights in an isolated research experiment."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import suppress
from medsyringe.evaluate import evaluate
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, DEVELOPMENT, inputs
from scripts.experiments.expanded.expanded_detection_study import short

UPSTREAM = DATA/"WeDetect"
IMPLEMENTATION = UPSTREAM/"deploy/test_coco_pytorch.py"
RAW = Path("data/models/wedetect-tiny/wedetect_tiny.pth")
TENSORS = RAW.with_name("wedetect_tiny_tensors.pth")
TOKENIZER = Path("data/models/wedetect-tokenizer")
LOCK = ART/"wedetect_tiny_backbone.json"
RESULT = ART/"wedetect_development.json"
REFS = {
    "source": "42c34c6742a22f26280bb0e59f95a701baee1a32",
    "checkpoint": "3b9d0670cbf300d0b48f900ae87a6c4785d88939",
    "tokenizer": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
}
ENGLISH = ("syringe", "hypodermic syringe", "dental cartridge syringe")
# The publisher's vision tower is trained with localized XLM-R prompt tokens.
# These are the literal token strings for the same three class names.
LOCALIZED = ("\u6ce8\u5c04\u5668", "\u76ae\u4e0b\u6ce8\u5c04\u5668", "\u7259\u79d1\u9ebb\u9189\u6ce8\u5c04\u5668")


class IgnoredHistory:
    """Placeholder for non-inference OpenMMLab training statistics."""


def statistic_placeholder(*_values):
    return None


def history_attribute(owner, key):
    if owner is not IgnoredHistory or key not in {"min","max","mean","current"}:
        raise ValueError("Unexpected attribute in the official training statistics.")
    return statistic_placeholder


def weights():
    if LOCK.exists():
        raise ValueError("The weights already have a conversion record.")
    if TENSORS.exists():
        raise FileExistsError(TENSORS)
    numpy_history = [
        (np._core.multiarray._reconstruct,"numpy.core.multiarray._reconstruct"),
        (np._core.multiarray.scalar,"numpy.core.multiarray.scalar"),
        np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType,
        np.dtypes.Int32DType, np.dtypes.Int64DType, np.dtypes.BoolDType,
    ]
    allowed = [*numpy_history, (IgnoredHistory,"mmengine.logging.history_buffer.HistoryBuffer"),
               (history_attribute,"builtins.getattr")]
    with torch.serialization.safe_globals(allowed):
        published = torch.load(RAW,map_location="cpu",weights_only=True)
    state = published["state_dict"]
    if not state or not all(isinstance(value,torch.Tensor) for value in state.values()):
        raise ValueError("The converted state must contain tensors exclusively.")
    torch.save({"state_dict":state},TENSORS)
    write_json(LOCK,dict(source=REFS,license="GPL-3.0 for the publisher checkpoint",tensors=len(state),
                         files={str(path):digest(path) for path in (RAW,TENSORS,IMPLEMENTATION,*(TOKENIZER.glob("*.json")),
                                                                  *(TOKENIZER.glob("*.model")))},
                         loading="Weights-only deserialization. Known training history values are ignored. "
                                 "The converted checkpoint contains only the model tensor dictionary."))
    print("Converted",len(state),"tensor entries",flush=True)


def prepare():
    record = read_json(LOCK)
    if any(digest(asset)!=checksum for asset,checksum in record["files"].items()):
        raise ValueError("Weights, tokenizer or upstream source differ from the conversion record.")
    spec = spec_from_file_location("wedetect_inference",IMPLEMENTATION)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    language = module.XLMRobertaLanguageBackbone(str(TOKENIZER),str(TENSORS)).eval()
    with torch.inference_mode():
        signals = dict(english=torch.nn.functional.normalize(language(list(ENGLISH)),dim=-1),
                       publisher_language=torch.nn.functional.normalize(language(list(LOCALIZED)),dim=-1))
    model = module.SimpleYOLOWorldDetector("tiny",score_thr=.012,nms_iou=.45)
    module.load_vision_checkpoint(model,str(TENSORS))
    return model.eval(), signals


def proposals(model, prompts, row):
    path = Path("data")/row["image"]
    if digest(path)!=row.get("derived_sha256",row["sha256"]):
        raise ValueError(f"Development raster {row['id']} changed.")
    from PIL import Image
    with Image.open(path) as opened:
        width,height = opened.size
    factor = np.array([width,height,width,height])
    with torch.inference_mode():
        predicted = model([str(path)],prompts)[0]
    items = [dict(bbox=np.clip(box/factor,0.,1.).tolist(),score=float(score))
             for box,score in zip(predicted["bboxes"].cpu().numpy(),predicted["scores"].cpu().numpy(),strict=True)
             if box[2]-box[0]>=.005*width and box[3]-box[1]>=.005*height]
    return suppress(items,.45)[:32]


def collect(role):
    os.environ["HF_HUB_OFFLINE"] = "1"
    key = dict(weights_sha256=digest(LOCK), code_sha256=digest(Path(__file__)), images={
        r["id"]:r["sha256"] for r in inputs(role)})
    path = DATA/"proposals"/f"wedetect-{role}.json"
    saved = read_json(path) if path.exists() else dict(key=key,frames={})
    if saved["key"]!=key:
        raise ValueError("The saved WeDetect experiment used different inputs.")
    rows = [r for r in inputs(role) if r["id"] not in saved["frames"]]
    if not rows:
        return
    model, signals = prepare()
    for row in rows:
        begin = perf_counter()
        options = {name:proposals(model,prompt,row) for name,prompt in signals.items()}
        saved["frames"][row["id"]] = dict(options=options,ms=(perf_counter()-begin)*1000)
        write_json(path,saved)
        print("WeDetect", role,len(saved["frames"]),flush=True)


def measure():
    if RESULT.exists():
        raise ValueError("WeDetect development has previously been evaluated.")
    measured = {}
    for role in DEVELOPMENT:
        rows = inputs(role)
        frames = read_json(DATA/"proposals"/f"wedetect-{role}.json")["frames"]
        if set(frames)!={r["id"] for r in rows}:
            raise ValueError("Complete WeDetect inference before comparison.")
        options = ("english","publisher_language")
        measured[role] = dict(median_ms=median(v["ms"] for v in frames.values()),
                              variants={name:short(evaluate(rows,{k:v["options"][name] for k,v in frames.items()},.2))
                                        for name in options})
    write_json(RESULT,dict(development=measured,weights_sha256=digest(LOCK),scope="All threshold-free ranking "
                           "metrics refer to development only. The nominal cutoff of 0.20 is diagnostic."))
    print("WeDetect compared labels and ranking on both development cohorts.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("weights","collect","measure"))
    parser.add_argument("--scope",choices=DEVELOPMENT)
    args=parser.parse_args()
    if args.action=="collect":
        if not args.scope:
            parser.error("Choose a development cohort.")
        collect(args.scope)
    else:
        (weights if args.action=="weights" else measure)()
