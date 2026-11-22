"""Score frozen Fil Rouge evidence and run the field-indeterminacy public control."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import match
from medsyringe.field_status import describe_fields
from medsyringe.hardware import configure_device
from medsyringe.pipeline import MedSyringeEngine
from scripts.detection_crowd_field_data import MANIFEST
from scripts.detection_hillclimb_experiment import ART, DATA
from scripts.evaluate_field_abstention import score
from scripts.precision_study import source
from scripts.sam3_study import configure_cpu

FIL_ROUGE = Path("artifacts/fill/audit.json")
CACHE = DATA / "proposals/field-abstention-live.json"
OUTPUT = ART / "field_abstention_results.json"


def non_estimable_fields(row):
    return dict(syringe_type=dict(status="indeterminate",
                                  scope="Supported 20 mL and 60 mL scale categories"),
                liquid_volume=dict(status="indeterminate", reason=row["scope"]))


def run():
    configure_cpu()
    root = Path.cwd()
    policy = dict(manifest=digest(MANIFEST), code=digest(__file__),
                  meaning="The three scenes have visible syringe bodies but no established supported graduated "
                          "volume reference or visible 20/60 mL type evidence. Freeze before any field outputs.")
    cache = read_json(CACHE) if CACHE.exists() else dict(binding=policy, frames={})
    if cache["binding"] != policy:
        raise ValueError("The abstention control differs from the saved source records.")
    model = MedSyringeEngine(root)
    try:
        configure_device(model, root, "xpu")
        if model.identity.get("primary_backend") != "xpu":
            raise ValueError("The field audit expects the pinned Intel runtime.")
        for row in read_json(MANIFEST)["rows"]:
            if row["id"] in cache["frames"]:
                continue
            result = model.analyze(source(row), ocr=True)
            if result["detection_checks"]["status"] != "available":
                raise ValueError("Field qualification needs every expected detector stage.")
            cache["frames"][row["id"]] = {key: result[key] for key in ("objects", "model", "elapsed_ms")}
            write_json(CACHE, cache)
            print("Real field abstention", row["id"], len(result["objects"]), flush=True)
    finally:
        model.close()


def measure():
    source_rows = read_json(MANIFEST)["rows"]
    cache = read_json(CACHE)
    if set(cache["frames"]) != {r["id"] for r in source_rows} or digest(MANIFEST) != cache["binding"]["manifest"]:
        raise ValueError("Some real source field controls have not been collected.")
    outputs = {key: f["objects"] for key, f in cache["frames"].items()}
    paired = [dict(scene=identifier, index=target, reference=non_estimable_fields(row),
                   prediction=describe_fields(proposal))
              for identifier, proposal, target, _ in match(source_rows, outputs, .50)[0] if target is not None
              for row in source_rows if row["id"] == identifier]
    regression = {}
    for name, item in read_json(FIL_ROUGE)["cohorts"].items():
        records = [dict(
            reference=dict(liquid_volume=dict(status="observable" if row["reference_ml"] is not None else
                                               "indeterminate", interval_ml=row["reference_ml"])),
            prediction=describe_fields(dict(fill_level=row["result"])))
            for row in item["cases"].values()]
        regression[name] = score(records, "liquid_volume")
    write_json(OUTPUT, dict(
        live=dict(total_instances=sum(len(r["objects"]) for r in source_rows), localized=len(paired),
                  rows=paired, **{k: score(paired, k) for k in ("syringe_type", "liquid_volume")}),
        fil_rouge_replay=regression, cache_sha256=digest(CACHE), fil_rouge_audit_sha256=digest(FIL_ROUGE),
        interpretation="The live public control scores fields only on correctly localized boxes. The Fil Rouge "
                       "regression uses saved visual reference brackets and clear controls. "
                       "No exact physical mL accuracy or interval calibration is inferred."))
    print("Field abstention measured on", len(paired), "localized public instances.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "measure"))
    (run if parser.parse_args().action == "run" else measure)()
