"""Freeze and evaluate the crowded-scene SAM proposal arm on archived rasters."""
import argparse
from pathlib import Path

from medsyringe.additive import project
from medsyringe.attributes import crop
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.detectors import suppress
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.appearance_text_encoder import TextMarginEncoder
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.detection_mask_retrieval import MaskRetriever
from scripts.detection_mask_verification import OUTPUT as DEVELOPMENT, candidates, detect, support
from scripts.detection_open_models import OPEN_MODELS, Detector, model_lock
from scripts.expanded_detection_study import short
from scripts.package_detection_evidence import MANIFEST, NATIVE, PUBLIC
from scripts.rfdetr_verifier import expanded
from scripts.runtime_bundle import confined
from scripts.sam3_study import configure_cpu

SELECTION = ART / "crowded_mask_recovery_seal.json"
EXECUTION = ART / "crowded_mask_recovery_execution.json"
RESULT = ART / "crowded_mask_recovery_audit.json"
CACHE = DATA / "proposals/crowded-mask-recovery.json"
SCOPES = (*PUBLIC, NATIVE)
OPTION = [.55, -.005, 1., "external", "multiple", .90]


def load_cases(bundle):
    manifest = read_json(bundle / MANIFEST)
    results = {}
    for name in SCOPES:
        meta = manifest["cohorts"][name]
        source = bundle / confined(meta["path"])
        if digest(source) != meta["sha256"]:
            raise ValueError(f"Archive cohort changed: {name}.")
        record = read_json(source)
        rows = record["rows"]
        if name == NATIVE:
            before = {k: f["actual"] for k, f in record["observations"].items()}
        else:
            before = {k: [p for p in boxes if p["score"] >= record["floor"]]
                      for k, boxes in record["candidate"].items()}
        results[name] = rows, before
    return results


def seal(bundle):
    if SELECTION.exists() or CACHE.exists() or RESULT.exists():
        raise ValueError("This scheduled crowded-scene comparison is already sealed.")
    study = read_json(DEVELOPMENT)
    choice = next((r for r in study["leading"] if r["settings"] == OPTION), None)
    if choice is None or not choice["outcomes"]["public-development"]["new_matches"]:
        raise ValueError("The prospectively selected branch needs an observed development gain.")
    load_cases(bundle)
    files = (Path("scripts/detection_mask_crowded_audit.py"), DEVELOPMENT,
             Path("scripts/detection_mask_retrieval.py"), Path("scripts/detection_mask_verification.py"))
    write_json(SELECTION, dict(settings=OPTION, code={str(p): digest(p) for p in files},
                               archive_sha256=digest(bundle / MANIFEST), auxiliary_weights={
                                   k: model_lock(k) for k in OPEN_MODELS},
                               criterion="At least one extra instance match and improved exact-count rate on "
                                         "the native cohort, no lost matches, no added false positives, and no "
                                         "per-scene count changes in correctly counted images. Earlier archives "
                                         "are regression controls. Skipping fewer than two accepted objects is "
                                         "part of the frozen architecture."))


def cohort(bundle):
    proof = read_json(SELECTION)
    hashes = dict(proof["code"])
    if EXECUTION.exists():
        change = read_json(EXECUTION)
        if change["selection_sha256"] != digest(SELECTION):
            raise ValueError("The execution record belongs to another selection.")
        hashes["scripts/detection_mask_crowded_audit.py"] = change["code_sha256"]
    if any(digest(p) != h for p, h in hashes.items()) or digest(bundle / MANIFEST) != proof["archive_sha256"]:
        raise ValueError("An archived data or study definition changed.")
    return load_cases(bundle), proof


def resume_execution(bundle):
    if any(p.exists() for p in (CACHE, RESULT, EXECUTION)):
        raise ValueError("The corrected execution must be recorded before collecting outputs.")
    freeze = read_json(SELECTION)
    if digest(bundle / MANIFEST) != freeze["archive_sha256"] or freeze["settings"] != OPTION:
        raise ValueError("The execution correction must use the original sealed data and decision.")
    write_json(EXECUTION, dict(selection_sha256=digest(SELECTION), code_sha256=digest(__file__),
                              reason="The first collector stopped during text encoder setup because the Intel "
                                     "Transformers environment returns a different embedding object. No candidate "
                                     "forecasts were stored. Collect the masks and witnesses on Intel, then run the "
                                     "same pinned CPU appearance encoder during scoring."))


def wide(detector, picture):
    whole = detector(picture)
    edge = min(picture.size)
    x, y = round((picture.width-edge)/2), round((picture.height-edge)/2)
    if not x and not y:
        return whole
    center = (x, y, x+edge, y+edge)
    focused = [{**p, "bbox": box} for p in detector(picture.crop(center))
               if (box := project(p["bbox"], center, picture.size))]
    return suppress([*whole, *focused], .43)


def collect(bundle):
    scopes, proof = cohort(bundle)
    configure_cpu()
    if proof["auxiliary_weights"] != {k: model_lock(k) for k in OPEN_MODELS}:
        raise ValueError("One of the independent witnesses has changed.")
    bound = digest(SELECTION)
    stored = read_json(CACHE) if CACHE.exists() else dict(seal_sha256=bound, groups={})
    if stored["seal_sha256"] != bound:
        raise ValueError("The new proposal cache is incompatible with the frozen inputs.")
    retriever = witnesses = None
    for name, (rows, before) in scopes.items():
        result = stored["groups"].setdefault(name, {})
        for row in rows:
            identifier = row["id"]
            if identifier in result:
                continue
            base = before[identifier]
            if len(base) < 2:
                result[identifier] = []
                continue
            origin = bundle / "data" / confined(row["image"])
            if digest(origin) != row.get("derived_sha256", row["sha256"]):
                raise ValueError(f"Archived source image changed: {identifier}.")
            if retriever is None:
                retriever = MaskRetriever()
                witnesses = [Detector(k, "xpu") for k in OPEN_MODELS]
            picture = read_image(origin)
            frame = dict(grids=retriever.predict_grid(picture))
            open_proposals = [wide(m, picture) for m in witnesses]
            available = [{**p, **support(p["bbox"], open_proposals, [])} for p in candidates(frame, base)]
            result[identifier] = [p for p in available if p["open_votes"] and p["stability"] >= .90]
            write_json(CACHE, stored)
            print("Crowded mask check", name, len(result), "/", len(rows), flush=True)
    write_json(CACHE, stored)


def text_scores(candidates_, picture, encoder):
    scored = []
    for offset in range(0, len(candidates_), 8):
        batch = candidates_[offset:offset+8]
        vectors = encoder.encode([crop(picture, expanded(p["bbox"], .08)) for p in batch])["prompt"]
        scored.extend({**p, "margin": float(v[0]-v[1])} for p, v in zip(batch, vectors, strict=True))
    return scored


def measure(bundle):
    if RESULT.exists():
        raise ValueError("This crowded-scene branch was already measured.")
    scopes, proof = cohort(bundle)
    stored = read_json(CACHE)
    if stored["seal_sha256"] != digest(SELECTION):
        raise ValueError("Frozen evidence changed.")
    configure_cpu()
    findings = {}
    reader = None
    for name, (rows, before) in scopes.items():
        matrices = stored["groups"][name]
        if set(matrices) != set(before):
            raise ValueError("The sealed cohort is incomplete.")
        after = {}
        for row in rows:
            key = row["id"]
            proposals = matrices[key]
            if proposals:
                if reader is None:
                    reader = TextMarginEncoder()
                origin = bundle / "data" / confined(row["image"])
                if digest(origin) != row.get("derived_sha256", row["sha256"]):
                    raise ValueError(f"Archived source image changed: {key}.")
                proposals = text_scores(proposals, read_image(origin), reader)
            after[key] = detect(before[key], proposals, proof["settings"])
        old, new = [evaluate(rows, ranks, 0.) for ranks in (before, after)]
        check = compare(old, new)
        spoiled = [r["id"] for r in rows if len(before[r["id"]]) == len(r["objects"])
                   and len(after[r["id"]]) != len(r["objects"])]
        findings[name] = dict(previous=short(old), proposed=short(new), guard=check,
                              spoiled_correct_counts=spoiled, passed=not spoiled and check["passed"]
                              and new["fp"] <= old["fp"])
    native = findings[NATIVE]
    qualifies = (all(v["passed"] for v in findings.values()) and native["guard"]["recovered_matches"] > 0
                 and native["proposed"]["exact_count"] > native["previous"]["exact_count"])
    write_json(RESULT, dict(passed=qualifies, freeze_sha256=digest(SELECTION),
                            collection_sha256=digest(CACHE), scopes=findings))
    print("Crowded mask recovery", qualifies, native["previous"], native["proposed"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "resume-execution", "collect", "measure"))
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    {"seal": seal, "resume-execution": resume_execution,
     "collect": collect, "measure": measure}[args.action](args.archive)
