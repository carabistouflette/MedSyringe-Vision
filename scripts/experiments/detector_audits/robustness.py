"""Probe controlled degradations on development validation only."""
from pathlib import Path

from PIL import ImageEnhance, ImageFilter

from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.detection.detectors import Grounder, suppress
from medsyringe.evaluate import evaluate


def main():
    root = Path(".")
    freeze_path = root / "artifacts/freeze.json"
    freeze = read_json(freeze_path)
    paths = root / "artifacts/validation_manifest.json"
    if digest(paths) != freeze["validation_sha256"]:
        raise ValueError("Expected the original frozen validation reference.")
    rows = read_json(paths)["rows"]
    detector = Grounder(root, freeze["backbone"])
    transforms = {
        "half-brightness": lambda image: ImageEnhance.Brightness(image).enhance(.5),
        "blur-radius-4px": lambda image: image.filter(ImageFilter.GaussianBlur(4)),
    }
    results = {}
    for name, transform in transforms.items():
        predictions = {}
        for row in rows:
            path = root / "data" / row["image"]
            if digest(path) != row["sha256"]:
                raise ValueError(f"Unexpected data for {row['id']}")
            predictions[row["id"]] = suppress(detector.propose(transform(read_image(path))))
        result = evaluate(rows, predictions, freeze["threshold"])
        print(name, result["recall"], result["f1"], flush=True)
        results[name] = result
    write_json("artifacts/development_robustness.json", dict(results=results, domain="Wikimedia validation",
        freeze_sha256=digest(freeze_path), validation_sha256=digest(paths),
        design="Post-freeze perturbation diagnostics. Same scene and reference for all three conditions."))


if __name__ == "__main__":
    main()
