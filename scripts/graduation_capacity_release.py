"""Release only the nominal-capacity field that passed the frozen OCR study."""
import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from PIL import Image

from medsyringe.data import digest, read_json, write_json
from medsyringe.grounding_backend import analyze, enable_intel
from medsyringe.production import ProductionEngine
from medsyringe.scale_runtime import ENTRY, RELEASE, ScaleReader
from scripts.expanded_proposals import image_for
from scripts.fill_study import ART, references
from scripts.graduation_ocr_study import AUDIT, LOCK, checksums
from scripts.sam3_study import configure_cpu

PROOF = ART / "graduation_capacity_runtime.json"
CONTROLS = (("IMG_3292", 0), ("IMG_3294", 0), ("IMG_3190", 0), ("IMG_3316", 0), ("IMG_3316", 180))


def capacity_gate():
    selection, audit = read_json(LOCK), read_json(AUDIT)
    if audit["selection_sha256"] != digest(LOCK) or selection["inputs"] != checksums():
        raise ValueError("The OCR audit does not match the frozen selection.")
    groups = [result["counts"] for result in audit["results"].values()]
    additions = sum(r["added_capacity"] for r in groups)
    errors = sum(r["wrong_capacity"] for r in groups)
    if not selection["selected"] or selection["selected"]["counts"]["added_capacity"] == 0 or not additions or errors:
        raise ValueError("Nominal capacity did not pass its per-field release gate.")
    return selection["selected"]["policy"], additions


def capacity(report):
    return [(obj["inscriptions"].get("capacity") or {}).get("value") for obj in report["objects"]]


def measurements(report):
    return [obj["fill_level"] for obj in report["objects"] if (obj.get("fill_level") or {}).get("ml") is not None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--intel", action="store_true")
    args = parser.parse_args()
    configure_cpu()
    policy, additional = capacity_gate()
    root = Path(".")
    engine = ProductionEngine(root)
    reader = ScaleReader(root, policy)
    baseline = dict(engine.identity)
    rows, _ = references()
    outputs = []
    try:
        if args.intel:
            enable_intel(engine, root)
        for identifier, turn in CONTROLS:
            source = image_for(rows[identifier], args.source)
            if turn:
                source = source.transpose(Image.Transpose.ROTATE_180)
            first = analyze(engine, source, True)
            started = perf_counter()
            second = reader.apply(source, deepcopy(first))
            extra_ms = 1000*(perf_counter()-started)
            previous, current = capacity(first), capacity(second)
            expected = int(rows[identifier]["group_scale"].split("-")[0])
            valid = len(previous) == len(current) == 1 and measurements(first) == measurements(second)
            valid &= all(a == b if a is not None else b in (expected, None) for a,b in zip(previous,current))
            if turn or identifier == "IMG_3316":
                valid &= current == [20] and measurements(second)[0]["ml"] == 16 if measurements(second) else False
            if args.intel:
                valid &= second["model"].get("primary_backend") == "xpu"
            gained = sum(a is None and b is not None for a,b in zip(previous,current))
            print(identifier, turn, previous, current, valid, round(extra_ms), flush=True)
            outputs.append(dict(id=identifier, rotation=turn, before=previous, after=current, gained=gained,
                                added_ms=extra_ms, expected=expected, passed=valid,
                                previous_fill=measurements(first), final_fill=measurements(second)))
    finally:
        try:
            reader.close()
        finally:
            engine.close()
    passed = all(item["passed"] for item in outputs) and bool(sum(r["gained"] for r in outputs))
    write_json(PROOF, dict(passed=passed, regulation="Nominal capacity and visible scale only",
                           policy=policy, cases=outputs, crop_regression_additions=additional,
                           crop_audit_sha256=digest(AUDIT)))
    if not passed:
        return
    code = ("medsyringe/graduation_ocr.py", "medsyringe/scale_runtime.py", "medsyringe/pipeline.py",
            "scripts/scale_worker.py", str(ENTRY.relative_to(Path.cwd())))
    models = {name: checksum for name, checksum in checksums().items() if name.startswith("data/")}
    write_json(RELEASE, dict(id="stroke-guided-nominal-capacity", baseline=baseline, policy=policy,
                            code={name: digest(name) for name in code}, models=models,
                            evidence=str(PROOF), evidence_sha256=digest(PROOF)))


if __name__ == "__main__":
    main()
