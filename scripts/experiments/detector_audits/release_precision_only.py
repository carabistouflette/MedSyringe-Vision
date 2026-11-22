"""Stage and verify a precision-only local detector extension after qualification."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from medsyringe.runtime.appearance_runtime import RELEASE, install_appearance
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.runtime.grounding_backend import analyze, enable_intel
from medsyringe.pipeline import MedSyringeEngine
from scripts.experiments.detector_audits.appearance_release import RECIPES, expected_boxes, sandbox_assets
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA, FLOOR
from scripts.experiments.detector_audits.detection_precision_confirmation import LOCK, cohort
from scripts.experiments.detector_audits.detection_precision_negative_audit import CACHE, MANIFEST, RESULT
from scripts.experiments.detector_audits.precision_study import source
from scripts.experiments.expanded.sam3_study import configure_cpu

DRAFT = ART/"precision_only_release_candidate.json"
LIVE = ART/"precision_only_live_check.json"
OPTIMIZED = ART/"precision_text_only_candidate.json"
OPTIMIZED_LIVE = ART/"precision_text_only_live_check.json"
PARITY = ART/"appearance_runtime_parity.json"
GRADUATION = Path("artifacts/fill/graduation_capacity_release.json")
SIGLIP = ART/"siglip2_crop_backbone.json"
DINO = Path("artifacts/auxiliary_crop_backbone.json")


def stage():
    result = read_json(RESULT)
    if not result["passed"] or result["selection_sha256"]!=digest(LOCK):
        raise ValueError("A precision-only extension requires passing reserved and capture evidence.")
    saved = read_json(GRADUATION)
    baseline = {**saved["baseline"],"graduation_capacity_sha256":digest(GRADUATION)}
    verified_models = {}
    for folder,lock in (("data/models/siglip2-base",SIGLIP),("data/models/dinov2-small",DINO)):
        verified_models.update({str(Path(folder)/name):value for name,value in read_json(lock)["files"].items()})
    draft = dict(id="fixed-precision-crop-gate", baseline=baseline, components=read_json(LOCK)["components"],
                 mode="veto-only", evidence=str(RESULT),evidence_sha256=digest(RESULT),
                 code={name:digest(name) for name in RECIPES},models=verified_models)
    write_json(DRAFT,draft)


def optimize():
    original = read_json(DRAFT)
    trial = read_json(PARITY)
    if (not trial["passed"] or trial["previous_release_sha256"] != digest(DRAFT) or
        trial["code_sha256"] != digest("scripts/workers/appearance_text_encoder.py")):
        raise ValueError("The text-only worker lacks a matching approval result.")
    write_json(OPTIMIZED,dict(
        **{k:v for k,v in original.items() if k!="code"},
        code={name:digest(name) for name in RECIPES}, encoder="text-only",
        runtime_parity=dict(path=str(PARITY),sha256=digest(PARITY)),
    ))


def check(capture, manifest=DRAFT, output=LIVE):
    configure_cpu()
    policy = read_json(manifest)
    if policy["evidence_sha256"]!=digest(RESULT):
        raise ValueError("The candidate and precision qualification have changed.")
    publication = read_json(RESULT)
    controls = publication["corrected_scene_ids"][:2]
    references = {r["id"]:r for r in read_json(MANIFEST)["rows"]}
    cache = read_json(DATA/"proposals/fixed-precision-fresh.json")["frames"]
    floor = read_json(FLOOR)["cutoff"]
    negatives = read_json(CACHE)["frames"]
    known = [(key,source(references[key]),expected_boxes({"ranked":negatives[key]["after"]},floor),False)
             for key in controls]
    present = next(row for row in cohort("fresh")[0] if any(
                   box_iou(o["bbox"],box)>=.5 for o in row["objects"] for box in expected_boxes(cache[row["id"]],floor)))
    known.append((present["id"],source(present),expected_boxes(cache[present["id"]],floor),False))
    selected = {r["file"]:r for split in ("fit","calibration","test") for r in cohort(split)[0]}
    filename = "IMG_3316.jpeg"
    meta = selected[filename]
    path = capture/filename
    if digest(path)!=meta["sha256"]:
        raise ValueError("Capture image differs from its sealed reference.")
    capture_values = read_json(DATA/"proposals"/f"fixed-precision-{meta['split']}.json")["frames"]
    known.append((filename,read_image(path),expected_boxes(capture_values[meta["id"]],floor),True))
    outcomes = []
    with TemporaryDirectory(prefix="medsyringe-precision-release-") as temporary:
        root = Path(temporary)
        sandbox_assets(root)
        write_json(root/RELEASE,policy)
        engine = MedSyringeEngine(root)
        worker = getattr(engine,"appearance",None) or install_appearance(engine,root)
        try:
            enable_intel(engine,root)
            if engine.identity.get("primary_backend")!="xpu":
                raise ValueError("The live check did not activate the audited Intel environment.")
            for name,image,expected,ocr in known:
                start = perf_counter()
                result = analyze(engine,image,ocr)
                boxes = [p["bbox"] for p in result["objects"]]
                matches = len(boxes)==len(expected) and all(box_iou(a,b)>=.98 for a,b in zip(boxes,expected,strict=True))
                outcome = dict(id=name,expected_count=len(expected),live_count=len(boxes),matches=matches,
                               worker_status=worker.status,elapsed_seconds=perf_counter()-start,backend="xpu")
                if ocr:
                    outcome["nominal_capacity"] = result["objects"][0]["inscriptions"]["capacity"] if boxes else None
                    outcome["fill"] = result["objects"][0]["fill_level"] if boxes else None
                outcomes.append(outcome)
                print(name,matches,worker.status,flush=True)
        finally:
            worker.close()
            engine.close()
    passed = all(m["matches"] and not m["worker_status"].startswith("baseline retained") for m in outcomes)
    write_json(output,dict(passed=passed,cases=outcomes,release_candidate_sha256=digest(manifest)))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("stage","optimize","check"))
    parser.add_argument("--source",type=Path)
    parser.add_argument("--text-only",action="store_true")
    args = parser.parse_args()
    if args.action=="stage":
        stage()
    elif args.action=="optimize":
        optimize()
    elif args.source:
        if args.text_only:
            check(args.source,OPTIMIZED,OPTIMIZED_LIVE)
        else:
            check(args.source)
    else:
        parser.error("Live verification requires --source.")
