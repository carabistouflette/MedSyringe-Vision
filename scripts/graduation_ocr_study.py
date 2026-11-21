"""Compare standalone local OCR on stroke-guided cropped scale markings."""
import argparse
from itertools import product
from pathlib import Path

from PIL import Image

from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.fill import estimate
from medsyringe.graduation_ocr import collect, supplement_graduations
from scripts.expanded_proposals import image_for
from scripts.fill_study import ART, LEGACY_ROWS, prepared, references

CACHE = Path("data/ocr-v6/scale-strokes")
ROOT = Path("data/ocr-v6/models")
LOCK = ART / "graduation_ocr_selection.json"
AUDIT = ART / "graduation_ocr_regression.json"
COHORTS = ("development", "holdout", "holdout-60", "clear")
MODELS = {"tiny": "PP-OCRv6_tiny_rec", "medium": "PP-OCRv6_medium_rec"}


def checksums():
    paths = [Path("medsyringe/graduation_ocr.py"), Path("artifacts/fill/release.json"),
             Path("artifacts/expanded/annotations.json")]
    paths += [path for name in MODELS.values() for path in sorted((ROOT / name).glob("*")) if path.is_file()]
    return {str(path): digest(path) for path in paths}


def collect_split(source, split):
    from paddleocr import TextRecognition

    if split != "development" and not LOCK.exists():
        raise ValueError("Freeze the graduation OCR policy before collecting the regression groups.")
    key = checksums()
    path = CACHE / f"{split}.json"
    if path.exists():
        cached = read_json(path)
        if cached["inputs"] != key:
            raise ValueError("The OCR crop cache differs from the frozen inputs.")
        return
    models = {name: TextRecognition(model_name=version, model_dir=str(ROOT / version),
                                    device="cpu", cpu_threads=2, enable_mkldnn=False) for name, version in MODELS.items()}
    rows, _ = references()
    legacy = {r["id"]: r for r in read_json(LEGACY_ROWS)["files"]}
    frames = {}
    for identifier, _, readings, _, _ in prepared(source, split):
        row = rows[identifier]
        if split == "holdout-60" or row["group"] == "amber-20-low-fill-dark-wall":
            with Image.open(legacy[identifier]["image"]) as image:
                region = image.convert("RGB")
        else:
            region = crop(image_for(row, source), row["objects"][0]["bbox"])
        frames[identifier] = collect(region, readings, models)
        if len(frames) % 8 == 0:
            print(split, len(frames), "crops", flush=True)
    write_json(path, dict(inputs=key, frames=frames))


def trial(items, cache, expected_capacity, settings, policy):
    answers = {}
    counts = dict(rows=len(items), baseline_capacity=0, added_capacity=0, wrong_capacity=0,
                  baseline_fill=0, added_fill=0, within_bracket=0, false_clear=0)
    for identifier, image, original, liquid, bracket in items:
        before = estimate(image, original, liquid, settings)
        upgraded = supplement_graduations(original, cache[identifier], policy)
        revised = before if "ml" in before else estimate(image, upgraded, liquid, settings)
        capacity = (upgraded.get("capacity") or {}).get("value")
        had_capacity = bool(original.get("capacity"))
        counts["baseline_capacity"] += had_capacity
        counts["baseline_fill"] += "ml" in before
        added_capacity = capacity is not None and not had_capacity
        added = "ml" in revised and "ml" not in before
        counts["added_capacity"] += added_capacity
        counts["wrong_capacity"] += added_capacity and capacity != expected_capacity[identifier]
        counts["added_fill"] += added
        counts["within_bracket"] += added and bracket is not None and bracket[0] <= revised["ml"] <= bracket[1]
        counts["false_clear"] += added and bracket is None
        if added_capacity or added or "graduation_crop_evidence" in upgraded:
            answers[identifier] = dict(before_fill=before, candidate_fill=revised, reference_bracket=bracket,
                                       capacity=upgraded.get("capacity"), added_capacity=added_capacity,
                                       expected_capacity=expected_capacity[identifier],
                                       extra_tokens=upgraded.get("graduation_crop_evidence", []))
    return dict(counts=counts, answers=answers)


def cases(source, split):
    payload = read_json(CACHE / f"{split}.json")
    if payload["inputs"] != checksums():
        raise ValueError(f"OCR inputs changed for {split}.")
    rows, _ = references()
    capacity = {key: int(row["group_scale"].split("-")[0]) for key, row in rows.items()}
    settings = read_json(ART / "release.json")["settings"]
    return list(prepared(source, split)), payload["frames"], capacity, settings


def select(source):
    if LOCK.exists():
        raise ValueError("The graduation OCR selection is already frozen.")
    inputs = cases(source, "development")
    trials = []
    for readers, confidence, stroke in product((["tiny"], ["medium"], ["tiny", "medium"]),
                                               (.72, .84, .93), (0., 4., 8.)):
        policy = dict(readers=readers, confidence=confidence, stroke=stroke)
        outcome = trial(*inputs, policy)
        trials.append(dict(policy=policy, counts=outcome["counts"]))
    safe = [t for t in trials if t["counts"]["added_fill"] == t["counts"]["within_bracket"]
            and not t["counts"]["wrong_capacity"]]
    key = lambda t: (t["counts"]["within_bracket"], t["counts"]["added_capacity"],
                     -len(t["policy"]["readers"]), t["policy"]["confidence"], t["policy"]["stroke"])
    winner = max(safe, key=key, default=None)
    selected = winner if winner and any(winner["counts"][k] for k in ("added_fill", "added_capacity")) else None
    chosen = selected or max(trials, key=key)
    write_json(LOCK, dict(inputs=checksums(), selected=selected, shadow=chosen, trials=trials,
                          development=trial(*inputs, chosen["policy"])))
    print("OCR calibration", chosen, flush=True)


def audit(source):
    selection = read_json(LOCK)
    if selection["inputs"] != checksums():
        raise ValueError("The OCR policy inputs changed.")
    results = {split: trial(*cases(source, split), selection["shadow"]["policy"]) for split in COHORTS[1:]}
    totals = [r["counts"] for r in results.values()]
    gain = sum(c["added_capacity"] + c["within_bracket"] for c in totals)
    safe = all(c["added_fill"] == c["within_bracket"] and not c["wrong_capacity"] for c in totals)
    write_json(AUDIT, dict(passed=bool(selection["selected"] and safe and gain),
                           selection_sha256=digest(LOCK), results=results,
                           scope="Previously consulted regression groups. No selection or threshold changes here."))
    print("OCR regression", {name: value["counts"] for name, value in results.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "select", "audit"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", choices=COHORTS, default="development")
    args = parser.parse_args()
    if args.stage == "collect":
        collect_split(args.source, args.split)
    else:
        (select if args.stage == "select" else audit)(args.source)


if __name__ == "__main__":
    main()
