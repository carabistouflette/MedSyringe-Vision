"""Produce a portable ledger of code, inputs, models and experiment outputs."""
from importlib.metadata import version
from pathlib import Path
import platform

from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import code_hashes

ARTIFACTS = Path("artifacts")
OUTPUT = ARTIFACTS / "run_manifest.json"


def main():
    freeze = read_json(ARTIFACTS / "freeze.json")
    test = read_json(ARTIFACTS / "test_results.json")
    checks = {
        "frozen_code_sha256": (freeze["code_sha256"], code_hashes()),
        "training_inventory_sha256": (freeze["training_inventory_sha256"], digest(
            ARTIFACTS / "source_inventory.json")),
        "draft_annotations_sha256": (freeze["draft_annotations_sha256"], digest(
            ARTIFACTS / "annotations.json")),
        "test_freeze_sha256": (test["freeze_sha256"], digest(ARTIFACTS / "freeze.json")),
        "test_manifest_sha256": (test["test_manifest_sha256"], digest(
            ARTIFACTS / "test_manifest.json")),
        "environment_lock_sha256": (freeze["environment_lock_sha256"], digest("uv.lock")),
    }
    if not all(expected == observed for expected, observed in checks.values()):
        raise ValueError("Experiment inputs do not match the frozen run.")
    stages = {}
    for file in sorted(Path("data/cache").glob("*.json")):
        artifact = read_json(file)
        if "fingerprint" in artifact:
            stages[file.stem] = dict(
                file_sha256=digest(file), configuration=artifact["fingerprint"],
                timed_images=len(artifact["predictions"]),
                total_variant_seconds=artifact["seconds_for_all_prompt_and_view_variants"],
            )
            continue
        if "entries" in artifact:
            entries = artifact["entries"]
            configuration = artifact["key"]
            timings = [view["seconds"] for record in entries.values() for view in record.values()]
            image_count = len(entries)
        else:
            predictions = artifact["predictions"]
            configuration = artifact["config"]
            timings = [p["seconds"] for p in predictions.values()]
            image_count = len(predictions)
        stages[file.stem] = dict(file_sha256=digest(file), configuration=configuration,
                                timed_images=image_count, timed_proposal_passes=len(timings),
                                sum_per_image_seconds=sum(timings))
    files = sorted(p for p in ARTIFACTS.rglob("*") if p.is_file() and p != OUTPUT)
    sources = sorted(
        p for repo_path in ("medsyringe", "scripts", "tests")
        for p in Path(repo_path).rglob("*")
        if p.is_file() and p.suffix in {".py", ".js", ".html", ".css", ".sh"}
    )
    write_json(OUTPUT, {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {pkg: version(pkg) for pkg in ("torch", "torchvision", "transformers",
                     "pillow", "pillow-heif", "scikit-learn", "opencv-python-headless", "fastapi")},
        "artifact_sha256": {str(p): digest(p) for p in files},
        "source_sha256": {str(p): digest(p) for p in sources},
        "passed_checks": list(checks),
        "proposal_stages": stages,
        "time_convention": ("Sum of per-image wall times, excluding load and cache pauses. "
                            "Validation cache timing includes concurrent workload. Use the serial "
                            "benchmark for model latency comparison."),
    })
    print(f"Provenance saved for {len(stages)} measured model stages and {len(files)} artifacts.")


if __name__ == "__main__":
    main()
