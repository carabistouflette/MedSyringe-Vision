"""Run a paired CPU and Intel XPU regression of full local inference."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from PIL import Image

from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.inference import Engine
from scripts.expanded_proposals import cache, image_for, rows_for
from scripts.sam3_study import configure_cpu

ART = Path("artifacts/expanded")
CONTROLS = (("IMG_3203", 0), ("IMG_3203", 180), ("IMG_3316", 0), ("IMG_3316", 180),
            ("IMG_3185", 0), ("IMG_3229", 0), ("IMG_3322", 0))


def run(source, backend):
    configure_cpu()
    capture = {r["id"]: r for split in ("fit", "calibration", "test")
               for r in rows_for(split) if "file" in r}
    prior = cache("reserve", "cpu")
    floor = read_json("artifacts/detector_release.json")["cutoff"]
    control = next(row for row in rows_for("reserve") if "file" not in row and not row["objects"]
                   and not any(p["score"] >= floor for p in prior[row["id"]]["prior"]))
    cases = [(capture[name], angle) for name, angle in CONTROLS] + [(control, 0)]
    started = perf_counter()
    engine = Engine(Path("."))
    try:
        if backend == "xpu":
            from medsyringe.xpu_grounding import AcceleratedGrounder

            engine.detector = AcceleratedGrounder(engine.detector)
        initialization_ms = (perf_counter() - started)*1000
        outputs = []
        for row, angle in cases:
            image = image_for(row, source)
            if angle:
                image = image.transpose(Image.Transpose.ROTATE_180)
            result = engine.analyze(image, ocr=True)
            outputs.append(dict(id=row["id"], rotation=angle, sha256=row["sha256"], result=result))
            print(backend, row["id"], angle, result["accepted_count"], round(result["elapsed_ms"]), flush=True)
    finally:
        engine.close()
    write_json(ART / f"runtime_pair_{backend}.json", dict(backend=backend, initialization_ms=initialization_ms,
               script_sha256=digest(__file__), outputs=outputs))


def signature(obj):
    capacity = (obj["inscriptions"].get("capacity") or {}).get("value")
    fill = obj.get("fill_level") or {}
    appearance = obj.get("liquid_appearance") or {}
    lightness = (appearance.get("lightness") or {}).get("lstar")
    return dict(capacity=capacity, scale=obj["scale"], color=obj["color"], fill_ml=fill.get("ml"),
                fill_status=fill.get("status"), lightness=lightness)


def compare():
    cpu, xpu = (read_json(ART / f"runtime_pair_{backend}.json") for backend in ("cpu", "xpu"))
    comparisons = []
    for a, b in zip(cpu["outputs"], xpu["outputs"], strict=True):
        keys = ("id", "rotation", "sha256")
        if any(a[k] != b[k] for k in keys):
            raise ValueError("CPU and XPU inputs differ.")
        lhs, rhs = a["result"], b["result"]
        objects = []
        for before, after in zip(lhs["objects"], rhs["objects"]):
            first, second = signature(before), signature(after)
            light = first["lightness"], second["lightness"]
            appearance_ok = light[0] == light[1] or (None not in light and abs(light[0]-light[1]) <= 1.)
            stable = appearance_ok and all(first[k] == second[k] for k in first if k != "lightness")
            objects.append(dict(iou=box_iou(before["bbox"], after["bbox"]), stable=stable,
                                cpu=first, xpu=second))
        comparisons.append(dict(id=a["id"], rotation=a["rotation"], cpu_ms=lhs["elapsed_ms"],
                                xpu_ms=rhs["elapsed_ms"], cpu_count=lhs["accepted_count"],
                                xpu_count=rhs["accepted_count"], objects=objects,
                                cpu_sam3=lhs["sam3_status"], xpu_sam3=rhs["sam3_status"]))
    warm = comparisons[1:]
    medians = {backend: median(row[f"{backend}_ms"] for row in warm) for backend in ("cpu", "xpu")}
    passed = all(row["cpu_count"] == row["xpu_count"] and
                 all(pair["stable"] and pair["iou"] >= .98 for pair in row["objects"]) for row in comparisons)
    write_json(ART / "runtime_pair.json", dict(passed=passed, warm_median_ms=medians,
               initialization_ms={r["backend"]: r["initialization_ms"] for r in (cpu, xpu)}, cases=comparisons,
               scope="Fixed regression controls. Grounding DINO device varies. SAM and OCR use the same workers."))
    print(passed, medians)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cpu", "xpu", "compare"))
    parser.add_argument("--source", type=Path)
    settings = parser.parse_args()
    if settings.stage == "compare":
        compare()
    elif settings.source:
        run(settings.source, settings.stage)
    else:
        parser.error("--source is required for inference.")


if __name__ == "__main__":
    main()
