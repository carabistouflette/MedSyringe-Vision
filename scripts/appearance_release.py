"""Qualify the crop candidate on reserved data and verify its installed local worker."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from medsyringe.appearance_runtime import RELEASE, install_appearance
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.grounding_backend import analyze, enable_intel
from medsyringe.pipeline import MedSyringeEngine
from scripts.detection_capture_candidate_regression import CACHE, RESULT as CAPTURE_PROOF
from scripts.detection_appearance_extension import RESULT as OPEN_IMAGES_PROOF
from scripts.detection_hillclimb_experiment import ART, FLOOR, inputs
from scripts.detection_lookalike_audit import RESULT as LOOKALIKES_PROOF
from scripts.detection_semantics_audit import LOCK, RESULT as AUDIT, output
from scripts.precision_study import source
from scripts.sam3_study import configure_cpu

QUALIFICATION = ART/"appearance_release_qualification.json"
DRAFT = ART/"appearance_release_candidate.json"
SMOKE = ART/"appearance_release_runtime.json"
CAPTURE = Path("artifacts/expanded/annotations.json")
RECIPES = (
    "medsyringe/appearance_runtime.py", "medsyringe/local_models.py", "scripts/appearance_worker.py",
    "scripts/appearance_worker.sh", "scripts/detection_crop_semantics.py", "scripts/detection_semantics_audit.py",
    "scripts/rfdetr_verifier.py", "scripts/sam3_study.py",
)


def stage():
    crop = read_json(AUDIT)
    replay = read_json(CAPTURE_PROOF)
    stage_sha = digest(LOCK)
    if not crop["passed"] or not replay["passed"] or any(
            record["selection_sha256"]!=stage_sha for record in (crop,replay)):
        raise ValueError("The candidate must pass both reserved and complete capture release gates.")
    extended = read_json(OPEN_IMAGES_PROOF)
    base, candidate = (extended["computed"][k] for k in ("baseline","earlier"))
    lost = [miss for miss in extended["cases"]["earlier"]["fn"] if miss not in extended["cases"]["baseline"]["fn"]]
    if lost or candidate["fp"]>base["fp"] or candidate["f1"]<base["f1"]:
        raise ValueError("The candidate regressed on the independent Open Images cohort.")
    lookalikes = read_json(LOOKALIKES_PROOF)
    if (lookalikes["new_false_alarm_scenes"]["earlier"] or
        lookalikes["summary"]["earlier"]["boxes"]>lookalikes["summary"]["baseline"]["boxes"]):
        raise ValueError("The candidate added false alarms in the independent lookalike cohort.")
    model = read_json("artifacts/fill/graduation_capacity_release.json")
    identity = {**model["baseline"], "graduation_capacity_sha256":digest("artifacts/fill/graduation_capacity_release.json")}
    locked = {str(path):digest(path) for path in (AUDIT,CAPTURE_PROOF,LOCK,CAPTURE,OPEN_IMAGES_PROOF,LOOKALIKES_PROOF)}
    qualification = dict(passed=True, scope="Reserved Commons and dental results, historical group and complete capture replay.",
                         inputs=locked)
    write_json(QUALIFICATION, qualification)
    models = {}
    for folder, path in (("data/models/siglip2-base",ART/"siglip2_crop_backbone.json"),
                         ("data/models/dinov2-small",Path("artifacts/auxiliary_crop_backbone.json"))):
        for name,value in read_json(path)["files"].items():
            models[str(Path(folder)/name)] = value
    config = dict(id="guarded-crop-appearance", baseline=identity, components=read_json(LOCK)["components"],
                  evidence=str(QUALIFICATION),evidence_sha256=digest(QUALIFICATION),
                  code={name:digest(name) for name in RECIPES},models=models)
    write_json(DRAFT, config)


def sandbox_assets(directory):
    directory.mkdir(exist_ok=True)
    (directory/".venv").symlink_to(Path(".venv").resolve(),target_is_directory=True)
    for cache in ("data", "artifacts"):
        (directory/cache).mkdir()
        for path in Path(cache).iterdir():
            if path.name in {"workspace",ART.name}:
                continue
            (directory/path).symlink_to(path.resolve(),target_is_directory=path.is_dir())
    (directory/ART).mkdir()
    for path in ART.iterdir():
        if path!=RELEASE:
            (directory/path).symlink_to(path.resolve(),target_is_directory=path.is_dir())


def expected_boxes(frame, cutoff):
    return [p["bbox"] for p in frame["ranked"] if p["score"]>=cutoff]


def runtime(capture):
    configure_cpu()
    choice = read_json(DRAFT)
    nested = read_json(AUDIT)["outcomes"]["public-audit"]
    recovered = {(p["id"],tuple(p["box"])) for p in nested["errors"]["before"]["fn"]} - {
                 (p["id"],tuple(p["box"])) for p in nested["errors"]["after"]["fn"]}
    rows = {r["id"]:r for r in inputs("public-audit")}
    before = read_json(output("public-audit"))["frames"]
    floor = read_json(FLOOR)["cutoff"]
    empty = [key for key in rows if not rows[key]["objects"] and not expected_boxes(before[key],floor)]
    controls = [(key,source(rows[key]),before[key],False) for key in sorted({key for key,_ in recovered})[:2]]
    if empty:
        key = sorted(empty)[0]
        controls.append((key,source(rows[key]),before[key],False))
    known = {r["file"]:r for r in read_json(CAPTURE)["rows"]}
    capture_cache = read_json(CACHE)["frames"]
    for file in ("IMG_3316.jpeg","IMG_3190.jpeg"):
        row = known[file]
        path = capture/file
        if digest(path)!=row["sha256"]:
            raise ValueError("The original smoke-test photo changed.")
        controls.append((file,read_image(path),capture_cache[row["id"]],True))
    reports = []
    with TemporaryDirectory(prefix="medsyringe-appearance-release-") as work:
        root = Path(work)
        sandbox_assets(root)
        write_json(root/RELEASE,choice)
        engine = MedSyringeEngine(root)
        verifier = None
        try:
            verifier = install_appearance(engine,root)
            enable_intel(engine,root)
            if engine.identity.get("primary_backend")!="xpu":
                raise ValueError("Runtime parity needs the audited Intel grounder.")
            for label,image,expected,ocr in controls:
                start = perf_counter()
                actual = analyze(engine,image,ocr)
                expected_here = expected_boxes(expected,floor)
                matches = len(expected_here)==len(actual["objects"]) and all(
                    box_iou(x,y["bbox"])>=.98 for x,y in zip(expected_here,actual["objects"],strict=True))
                reports.append(dict(id=label,boxes=actual["objects"],matches=matches,
                                    detection_verified=verifier.status=="local crop verification applied",
                                    ocr=ocr,seconds=perf_counter()-start))
                print(label,matches,verifier.status,flush=True)
        finally:
            if verifier is not None:
                verifier.close()
            engine.close()
    passed = bool(reports and all(r["matches"] and r["detection_verified"] for r in reports))
    write_json(SMOKE,dict(passed=passed,cases=reports,manifest_sha256=digest(DRAFT),
                          scope="Real source images through the Intel application engine and the pinned local CPU crop worker."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("stage","runtime"))
    parser.add_argument("--source",type=Path)
    args = parser.parse_args()
    if args.action=="stage":
        stage()
    elif args.source:
        runtime(args.source)
    else:
        parser.error("Runtime needs --source.")
