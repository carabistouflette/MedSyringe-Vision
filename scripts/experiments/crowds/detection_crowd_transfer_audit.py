"""Measure whether matched crowd training transfers to natural photographs."""
import argparse
from pathlib import Path
from time import perf_counter

from PIL import Image

from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import suppress
from medsyringe.evaluate import evaluate
from scripts.experiments.crowds.detection_crowd_source_data import MANIFEST as NATURAL
from scripts.experiments.crowds.detection_crowd_training_evaluate import EDGE, FLOOR, MODELS, OUTPUT as DEVELOPMENT
from scripts.experiments.crowds.detection_crowd_training_evaluate import records, state
from scripts.experiments.detector_audits.detection_deskew_study import orientation, restored_box
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA
from scripts.experiments.detector_audits.detection_transparent_followup import archive_image, archived
from scripts.experiments.expanded.expanded_detection_study import short
from scripts.datasets.package_detection_evidence import MANIFEST, NATIVE, confined
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.transfer.rfdetr_compare import ranked_region
from scripts.experiments.expanded.sam3_study import configure_cpu

PLAN = ART / "crowd_training_real_transfer_freeze.json"
CACHE = DATA / "proposals/crowd-training-real-transfer.json"
OUTPUT = ART / "crowd_training_real_transfer.json"


def signature(archive):
    settings = records()
    manifest = read_json(archive / MANIFEST)
    native = manifest["cohorts"][NATIVE]
    if digest(archive / confined(native["path"])) != native["sha256"]:
        raise ValueError("The native photographic cohort differs from the archived reference.")
    return dict(files={str(p): digest(p) for p in (DEVELOPMENT, NATURAL, __file__)},
                weights={model: state(model, settings)[1]["checkpoint"] for model in MODELS},
                native_sha256=native["sha256"], score_cutoff=FLOOR, nms_iou=.45)


def seal(archive):
    if PLAN.exists() or CACHE.exists():
        raise ValueError("The real transfer study has already been opened.")
    write_json(PLAN, {**signature(archive), "scope": (
        "Run all three locked research checkpoints at their already reported development score. "
        "Use the original and dominant-axis views on the same images. Native device scenes are regression "
        "diagnostics. Four Commons acquisitions were annotated without candidate outputs. "
        "This standalone transfer comparison does not select a deployment gate.")})
    print("Real crowded scenes registered with all three weights and native raster controls.", flush=True)


def inputs(archive):
    native, _ = archived(archive, NATIVE)
    return dict(native=native, commons=read_json(NATURAL)["rows"])


def registered(archive):
    proof = read_json(PLAN)
    observed = signature(archive)
    if {key: proof[key] for key in observed} != observed:
        raise ValueError("The fixed transfer models or reference data changed.")
    return proof


def collect(archive):
    from rfdetr import RFDETRNano

    proof = registered(archive)
    configure_cpu()
    sealed = digest(PLAN)
    saved = read_json(CACHE) if CACHE.exists() else dict(seal=sealed, result={})
    if saved["seal"] != sealed:
        raise ValueError("The local cache is not bound to this comparison.")
    groups = inputs(archive)
    development = records()
    for name in MODELS:
        checkpoint, _ = state(name, development)
        model = RFDETRNano.from_checkpoint(checkpoint, device="cpu")
        if model.class_names != ["syringe"] or model.model_config.resolution != EDGE:
            raise ValueError("Unexpected class or input size in a paired training checkpoint.")
        cohorts = saved["result"].setdefault(name, {})
        for cohort, rows in groups.items():
            frames = cohorts.setdefault(cohort, {})
            for row in rows:
                identifier = row["id"]
                if identifier in frames:
                    continue
                picture = archive_image(archive, row) if cohort == "native" else source(row)
                started = perf_counter()
                original = suppress(list(ranked_region(model, picture, EDGE)), proof["nms_iou"])
                angle = orientation(picture)
                rotated = original
                if angle is not None:
                    image = picture.rotate(angle, Image.Resampling.BICUBIC, expand=True,
                                           fillcolor=picture.getpixel((0, 0)))
                    boxes = suppress(list(ranked_region(model, image, EDGE)), proof["nms_iou"])
                    rotated = [{**p, "bbox": geometry} for p in boxes
                               if (geometry := restored_box(p["bbox"], picture.size, image.size, angle))]
                frames[identifier] = dict(original=original, aligned=rotated, angle=angle,
                                          seconds=perf_counter() - started)
                write_json(CACHE, saved)
            print("Real transfer", name, cohort, len(frames), "/", len(rows), flush=True)
        del model


def measure(archive):
    if OUTPUT.exists():
        raise ValueError("This audit has already been scored.")
    registered(archive)
    cache = read_json(CACHE)
    if cache["seal"] != digest(PLAN):
        raise ValueError("The transfer outputs differ from the registered study.")
    findings = {}
    for name in MODELS:
        findings[name] = {}
        for cohort, rows in inputs(archive).items():
            frames = cache["result"][name][cohort]
            if set(frames) != {r["id"] for r in rows}:
                raise ValueError("The transfer audit is missing a real photograph.")
            findings[name][cohort] = {method: {
                "all": short(evaluate(rows, {k: f[method] for k, f in frames.items()}, FLOOR)),
                "crowded": short(evaluate([r for r in rows if len(r["objects"]) >= 2],
                                           {k: f[method] for k, f in frames.items()}, FLOOR))
            } for method in ("original", "aligned")}
    write_json(OUTPUT, dict(outcomes=findings, scope="Standalone external transfer of three fixed local RF-DETR "
                                                     "weights. Natural cohorts remain descriptive by source.",
                            cache_sha256=digest(CACHE), freeze_sha256=digest(PLAN)))
    print("Real transfer scored", {k: v["commons"]["aligned"]["crowded"]["tp"] for k, v in findings.items()},
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "collect", "measure"))
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    {"seal": seal, "collect": collect, "measure": measure}[args.action](args.archive)
