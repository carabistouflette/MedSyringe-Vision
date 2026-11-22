"""Compare outputs and latency of resident SAM worker arrangements on local inputs."""
import argparse
from pathlib import Path
from time import perf_counter

from PIL import ImageOps

from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.runtime.grounding_backend import analyze, enable_intel
from medsyringe.pipeline import MedSyringeEngine
from medsyringe.runtime.shared_sam import share
from scripts.experiments.expanded.sam3_study import configure_cpu

ROOT = Path.cwd()
ART = ROOT / "artifacts/detection_hillclimb"
IMPLEMENTATION = (Path(__file__).resolve(), ROOT / "medsyringe/runtime/shared_sam.py", ROOT / "scripts/workers/shared_sam_worker.py",
                  ROOT / "scripts/workers/shared_sam_worker.sh", ROOT / "artifacts/sam3_release.json",
                  ART / "appearance_release.json")


def discarded_time(report):
    return {k: v for k, v in report.items() if k != "elapsed_ms"}


def trial(mode, device, images, output, ocr, long_side):
    configure_cpu()
    engine = MedSyringeEngine(ROOT)
    try:
        if device == "xpu":
            enable_intel(engine, ROOT)
            if engine.identity.get("primary_backend") != "xpu":
                raise ValueError("The requested Intel path did not activate.")
        else:
            for worker in (engine.sam3, engine.crop_verifier.support_worker):
                worker.options = {**worker.options, "device": "cpu"}
        if mode == "shared":
            share(engine, ROOT)
            support = engine.crop_verifier.support_worker
            support.run = engine.detection_checks.observe(support.run, "SAM corroboration")
        scenarios = []
        for index, file in enumerate(images):
            image = read_image(file)
            region = ImageOps.contain(image, (long_side, long_side))
            start = perf_counter()
            values = discarded_time(analyze(engine, region, ocr))
            if values["detection_checks"]["status"] != "available":
                raise RuntimeError(f"Unqualified detection on {file.name}: {values['detection_checks']}")
            scenario = dict(index=index, image=file.name, sha256=digest(file), ocr=ocr, long_side=long_side,
                            seconds=perf_counter()-start, report=values)
            scenarios.append(scenario)
            write_json(output, dict(mode=mode, device=device, cases=scenarios, completed=False))
            print("SAM residency", mode, file.name, round(scenario["seconds"], 2), flush=True)
        write_json(output, dict(mode=mode, device=device, cases=scenarios, completed=True))
    finally:
        engine.close()


def compare(before, after, output):
    old, new = read_json(before), read_json(after)
    if not old["completed"] or not new["completed"] or old["device"] != new["device"]:
        raise ValueError("Compare complete runs on the same requested device.")
    proof = []
    for a, b in zip(old["cases"], new["cases"], strict=True):
        comparable = ("index", "image", "sha256", "ocr", "long_side")
        if any(a[k] != b[k] for k in comparable):
            raise ValueError("The paired workloads used different images or analysis options.")
        differences = [k for k, v in a["report"].items() if b["report"].get(k) != v]
        proof.append(dict(image=a["image"], sha256=a["sha256"], matches=not differences,
                          different_fields=differences, previous_seconds=a["seconds"], shared_seconds=b["seconds"],
                          counts=[a["report"]["accepted_count"], b["report"]["accepted_count"]],
                          supplemental=a["report"]["sam3_status"], ocr=a["ocr"]))
    write_json(output, dict(passed=bool(proof) and all(r["matches"] for r in proof), device=old["device"],
                            source_caches=[digest(before),digest(after)],
                            code={str(p.relative_to(ROOT)): digest(p) for p in IMPLEMENTATION},
                            trials=proof))
    print("Identical analysis reports", sum(r["matches"] for r in proof), "/", len(proof), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("separate", "shared", "compare"))
    parser.add_argument("--device", choices=("cpu", "xpu"), default="xpu")
    parser.add_argument("--image", action="append", type=Path, default=[])
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--long-side", type=int, default=1600)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "compare" and args.previous and args.candidate:
        compare(args.previous, args.candidate, args.output)
    elif args.mode != "compare" and args.image and args.long_side > 0:
        trial(args.mode, args.device, args.image, args.output, args.ocr, args.long_side)
    else:
        parser.error("Choose paired inputs for comparison or images for a trial.")
