"""Lock only improvements that pass the expanded application and evaluation gates."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.expanded_runtime import RELEASE
from medsyringe.inference import Engine
from scripts.expanded_dataset import ART

EVIDENCE = ART / "release_assessment.json"


def detector_gate(test, public):
    checks = [test, public]
    safe = all(r.get("candidate") and r["guard"]["passed"] and r["candidate"]["fp"] <= r["baseline"]["fp"]
               for r in checks)
    improved = safe and any(r["candidate"]["f1"] > r["baseline"]["f1"] for r in checks)
    return dict(approved=bool(improved), recall_preserved_and_no_extra_fp=bool(safe), f1_improved=bool(improved))


def marking_gate(calibration, holdout, choice, guard):
    if choice == "baseline":
        return dict(approved=False, reason="The original reader won calibration.")
    fields = ("capacity", "scale")
    before = holdout["totals"]["baseline"]
    after = {**holdout["totals"][choice], "capacity": guard["groups"]["test"]["metrics"]}
    safe = all(after[f]["wrong"] <= before[f]["wrong"] and after[f]["correct"] >= before[f]["correct"] for f in fields)
    dev = calibration["totals"]
    improved = any(dev[choice][f]["correct"] > dev["baseline"][f]["correct"] for f in fields)
    return dict(approved=safe and improved and after["capacity"]["wrong"] == 0,
                holdout_preserved=safe, calibration_improved=improved, errors_after_guard=after["capacity"]["wrong"])


def appearance_gate(calibration, holdout):
    old, new = calibration["legacy"], calibration["selected"]
    safe = holdout["candidate"]["uncolored_answered"] == 0
    adequate = new["supported"] >= old["supported"] and new["median_lstar_error"] <= old["median_lstar_error"] + 1.
    return dict(approved=safe and adequate, heldout_false_color=holdout["candidate"]["uncolored_answered"],
                calibration_lightness_preserved=adequate,
                measured_color=new["median_delta_e76"] is not None)


def evidence():
    def report(name):
        return read_json(ART / (name + ".json"))

    det = report("detector_freeze")
    det_recall = report("detector_recall_freeze")
    marking = report("markings_freeze")
    liquid = report("liquid_freeze")
    context = report("color_context_audit")
    checks = {
        "detection": detector_gate(report("detector_guarded_test"), report("detector_guarded_reserve")),
        "markings": marking_gate(report("markings_calibration"), report("markings_test"), marking["selected"],
                                  report("markings_guard")),
        "liquid": appearance_gate(report("liquid_calibration"), report("liquid_test")),
    }
    checks["liquid"]["context_false_on_clear_regression"] = context["false_regression"]
    checks["liquid"]["context_false_on_public_controls"] = context["false_public"]
    models = {}
    features = {}
    if checks["detection"]["approved"]:
        policy = det_recall["selected"]["policy"]
        features["detection"] = dict(settings=policy["options"], heads=det["heads"], recall_guard=policy["guard"])
    if checks["markings"]["approved"]:
        features["markings"] = marking["selected"]
    if checks["liquid"]["approved"]:
        features["liquid"] = {k: liquid["winner"][k] for k in ("model", "cutoff")}
    paths = ("medsyringe/expanded_runtime.py", "medsyringe/hybrid_detection.py", "medsyringe/inference.py",
             "medsyringe/marking_views.py", "medsyringe/markings_guard.py", "medsyringe/liquid_color.py",
             "medsyringe/liquid_context.py", "scripts/expanded_worker_xpu.sh",
             "scripts/expanded_worker.py", "scripts/expanded_liquid.py", "scripts/expanded_detection_study.py",
             "scripts/expanded_recall_selection.py")
    inputs = ("detector_freeze", "detector_recall_freeze", "detector_guarded_test", "detector_guarded_reserve",
              "detector_test", "detector_reserve", "markings_freeze", "markings_calibration",
              "markings_test", "markings_guard", "liquid_freeze", "liquid_calibration", "liquid_test",
              "color_context_audit", "split_review")
    proof = dict(passed=True, gates=checks, inputs={p: digest(ART / (p + ".json")) for p in inputs})
    config = dict(id="expanded-local-hybrid", features=features, code={p: digest(p) for p in paths}, models=models,
                  evidence=str(EVIDENCE))
    return proof, config


def prepared_root(root):
    (root / "artifacts").mkdir()
    (root / ART).mkdir()
    (root / "data").mkdir()
    for path in Path("artifacts").iterdir():
        if path.name != "expanded":
            (root / path).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    for path in ART.iterdir():
        if path not in {RELEASE, EVIDENCE}:
            (root / path).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    for name in ("models", "rfdetr", "sam3", "ocr-v6"):
        (root / "data" / name).symlink_to(Path("data", name).resolve(), target_is_directory=True)


def smoke(source, proof, config):
    with TemporaryDirectory(prefix="medsyringe-expanded-check-") as scratch:
        temporary = Path(scratch)
        prepared_root(temporary)
        write_json(temporary / EVIDENCE, proof)
        draft = {**config, "evidence_sha256": digest(temporary / EVIDENCE)}
        if config["features"]:
            write_json(temporary / RELEASE, draft)
        engine = Engine(temporary)
        cases = []
        try:
            for identifier in ("IMG_3185", "IMG_3229"):
                result = engine.analyze(read_image(source / f"{identifier}.jpeg"), ocr=True)
                if len(result["objects"]) != 1:
                    raise ValueError(f"Combined runtime control did not find one syringe for {identifier}.")
                box = result["objects"][0]
                capacity = box["inscriptions"]["capacity"]
                measured = box["liquid_appearance"]
                if capacity and capacity["value"] != 20.:
                    raise ValueError(f"Wrong combined runtime capacity for {identifier}.")
                if config["features"].get("liquid"):
                    appearance = measured["apparent_color"]
                    if (appearance is None) != (identifier == "IMG_3185") or measured["opacity"] is not None:
                        raise ValueError(f"Incorrect color or opacity behavior for {identifier}.")
                cases.append(dict(frame=identifier, result=result))
                print(identifier, result["accepted_count"], capacity and capacity["value"],
                      measured and measured["status"], round(result["elapsed_ms"]), flush=True)
        finally:
            engine.close()
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    proof, config = evidence()
    print("Release gates", proof["gates"], flush=True)
    proof["combined_smoke"] = smoke(args.source, proof, config)
    proof["activated"] = list(config["features"])
    write_json(EVIDENCE, proof)
    if config["features"]:
        write_json(RELEASE, {**config, "evidence_sha256": digest(EVIDENCE)})
    print("Expanded release", proof["activated"], flush=True)


if __name__ == "__main__":
    main()
