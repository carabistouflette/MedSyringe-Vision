"""Check the full local pipeline before enabling fill measurement by default."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.runtime.expanded_runtime import RELEASE as EXPANDED
from medsyringe.analysis.inference import Engine, FILL_RELEASE
from scripts.experiments.expanded.expanded_release import EVIDENCE as EXPANDED_EVIDENCE, evidence, prepared_root
from scripts.experiments.measurement.fill_study import ART, LOCK, inputs, references

AUDIT = ART / "audit.json"
RUNTIME = ART / "runtime.json"


def smoke(root, source):
    engine = Engine(root)
    _, labels = references()
    cases = []
    controls = (("IMG_3203", 0), ("IMG_3203", 180), ("IMG_3245", 0), ("IMG_3316", 0),
                ("IMG_3316", 180), ("IMG_3322", 0), ("IMG_3322", 180), ("IMG_3185", 0),
                ("IMG_3185", 180), ("IMG_3229", 0))
    try:
        for identifier, orientation in controls:
            image = read_image(source / f"{identifier}.jpeg")
            if orientation:
                image = image.transpose(Image.Transpose.ROTATE_180)
            observed = engine.analyze(image, ocr=True)
            if len(observed["objects"]) != 1:
                raise ValueError(f"Expected one detected syringe on {identifier} at {orientation} degrees.")
            obj = observed["objects"][0]
            fill = obj["fill_level"]
            expected = labels.get(identifier)
            if fill["status"] == "visual estimate":
                low, high = expected or (float("inf"), float("-inf"))
                if not (low <= fill["ml"] <= high):
                    raise ValueError(f"Fill is outside the visible bracket on {identifier} at {orientation} degrees.")
            if orientation == 0 and identifier in {"IMG_3185", "IMG_3229"}:
                if (obj["inscriptions"]["capacity"] or {}).get("value") != 20.:
                    raise ValueError("The existing expanded marking control failed.")
            cases.append(dict(image=identifier, orientation=orientation, fill=fill, bbox=obj["bbox"],
                              nominal_capacity=obj["inscriptions"]["capacity"], expected_ml=expected,
                              elapsed_ms=observed["elapsed_ms"]))
            print(identifier, orientation, fill.get("ml", fill.get("reason")), flush=True)
    finally:
        engine.close()
    answered = [c for c in cases if c["fill"]["status"] == "visual estimate"]
    complete = {c["fill"]["capacity_ml"] for c in answered} == {20, 60} and any(c["orientation"] == 180 for c in answered)
    return dict(passed=complete, cases=cases, audit_sha256=digest(AUDIT),
                scope="Full JPEG inputs and detector crops, including physically reversed raster input.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    phase = read_json(LOCK)
    audit = read_json(AUDIT)
    if not audit["passed"] or audit["selection_sha256"] != digest(LOCK) or phase["inputs"] != inputs():
        raise ValueError("The fill estimate needs frozen passing evaluation evidence.")
    historical = read_json(EXPANDED_EVIDENCE)
    previous = read_json(EXPANDED)
    check, expanded = evidence()
    if (check["gates"] != historical["gates"] or check["inputs"] != historical["inputs"] or
            expanded["features"] != previous["features"] or digest(EXPANDED_EVIDENCE) != previous["evidence_sha256"]):
        raise ValueError("The existing expanded release inputs changed.")
    expanded["evidence_sha256"] = previous["evidence_sha256"]
    code_paths = ("medsyringe/measurement/fill.py", "medsyringe/analysis/inference.py", "medsyringe/measurement/fill_orientation.py",
                  "medsyringe/measurement/liquid_color.py", __file__)
    config = dict(id="graduated-visible-fill", settings=phase["selected"]["settings"],
                  code={str(Path(p).resolve().relative_to(Path.cwd())): digest(p) for p in code_paths})
    with TemporaryDirectory(prefix="medsyringe-fill-") as scratch:
        temp = Path(scratch)
        prepared_root(temp)
        write_json(temp / EXPANDED_EVIDENCE, historical)
        write_json(temp / EXPANDED, expanded)
        (temp / ART).unlink()
        (temp / ART).mkdir()
        write_json(temp / AUDIT, audit)
        candidate = {**config, "evidence": str(AUDIT), "evidence_sha256": digest(temp / AUDIT)}
        write_json(temp / FILL_RELEASE, candidate)
        result = smoke(temp, args.source)
    write_json(RUNTIME, result)
    if not result["passed"]:
        raise ValueError("Full-pipeline coverage or the raster rotation control did not pass.")
    write_json(EXPANDED, expanded)
    write_json(FILL_RELEASE, {**config, "evidence": str(RUNTIME), "evidence_sha256": digest(RUNTIME)})
    print("Activated the local fill estimate.", flush=True)


if __name__ == "__main__":
    main()
