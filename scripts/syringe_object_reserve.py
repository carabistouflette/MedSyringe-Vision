"""Collect fixed RF-DETR proposals and release outputs after verifier calibration."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, write_json
from medsyringe.appearance_runtime import RELEASE as RUNTIME
from scripts.native_candidate_bank import views
from scripts.occlusion_detector import INITIAL, raster
from scripts.recrop_proposal_bank import crop_proposals
from scripts.syringe_object_references import REFERENCE, WORK, references as role_references
from scripts.syringe_object_verifier import LOCK, sealed

PROPOSALS = WORK / "fresh-rf.json"
RELEASE = WORK / "fresh-release.json"


def references():
    return role_references("fresh")


def state(path):
    sealed()
    binding = dict(reference=digest(REFERENCE), policy=digest(LOCK),
                   rf=read_json(INITIAL)["checkpoint_sha256"], runtime=digest(RUNTIME), code=digest(__file__))
    state = read_json(path) if path.exists() else dict(binding=binding, frames={})
    if state["binding"] != binding:
        raise ValueError(f"Wrong acquisition or policy evidence in {path}.")
    return state


def collect():
    import torch
    from rfdetr import RFDETRNano

    cache = state(PROPOSALS)
    cfg = read_json(INITIAL)
    if digest(cfg["checkpoint"]) != cfg["checkpoint_sha256"]:
        raise ValueError("Unexpected RF-DETR checkpoint.")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model = RFDETRNano.from_checkpoint(cfg["checkpoint"], device="cpu")
    for row in references():
        k = row["id"]
        if k not in cache["frames"]:
            image = raster(row)
            cache["frames"][k] = crop_proposals(model, image, views(model, image))
            write_json(PROPOSALS, cache)
            print(k, "collected", flush=True)


def baseline():
    from medsyringe.grounding_backend import analyze
    from medsyringe.hardware import configure_device
    from medsyringe.pipeline import MedSyringeEngine
    from scripts.sam3_study import configure_cpu

    cache = state(RELEASE)
    configure_cpu()
    with_engine = MedSyringeEngine(Path.cwd())
    try:
        configure_device(with_engine, Path.cwd(), "xpu")
        for row in references():
            k = row["id"]
            if k not in cache["frames"]:
                result = analyze(with_engine, raster(row), False)
                check = result["detection_checks"]
                if check["status"] != "available" or check["primary_backend"] != "xpu":
                    raise ValueError("Released detector unavailable.")
                cache["frames"][k] = result["objects"]
                write_json(RELEASE, cache)
                print(k, len(result["objects"]), flush=True)
    finally:
        with_engine.close()


def frozen_inputs():
    rows, frames = references(), state(PROPOSALS)["frames"]
    if set(frames) != {r["id"] for r in rows}:
        raise ValueError("Missing holdout RF-DETR proposals.")
    return rows, frames


def evidence():
    rows, frames = frozen_inputs()
    released = state(RELEASE)["frames"]
    if set(released) != set(frames):
        raise ValueError("Missing released holdout outputs.")
    return rows, released, frames, {str(p): digest(p) for p in (REFERENCE, RELEASE, PROPOSALS)}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("collect", "baseline"))
    {"collect": collect, "baseline": baseline}[cli.parse_args().stage]()
