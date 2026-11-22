"""Score the fixed small-scale reader on untouched public views from one new acquisition."""
import argparse
from pathlib import Path
from time import perf_counter

from medsyringe.attributes import crop
from medsyringe.capacity_ocr import SceneOCR, resolve
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.small_capacity import SmallCapacity

ART = Path("artifacts/detection_hillclimb")
CATALOG = Path("data/public-annotated-collection/manifest.json")
FREEZE = ART / "small_field_public_transfer_reference.json"
OUTPUT = ART / "small_field_public_transfer.json"
READER = Path("medsyringe/small_capacity.py")
CASES = (
    ("commons-18691600", [.04, .09, .94, .95], 5, "empty", "The printed 5 and ml are visible. The seal is seated."),
    ("commons-18691601", [.03, .13, .94, .84], 5, None, "The printed 5 and ml are visible. Contents are unclear."),
    ("commons-18691603", [.025, .12, .94, .92], 5, None, "A faint terminal indication is visible. Contents are unclear."),
    ("commons-18691589", [.04, .22, .96, .67], 1, None, "Source gives 1 mL. The photo does not establish an explicit unit."),
    ("commons-108248929", [.38, .06, .62, .96], None, None, "Hard negative. A container with an unrelated mL marking."),
)


def inputs():
    inventory = {row["id"]: row for row in read_json(CATALOG)["rows"]}
    rows = []
    for identifier, box, capacity, presence, note in CASES:
        row = inventory[identifier]
        image = Path("data") / row["image"]
        if digest(image) != row["sha256"]:
            raise ValueError(f"Changed photo {identifier}.")
        rows.append({**{key: row[key] for key in ("id", "sha256", "source_url", "license", "author", "group")},
                     "image": str(image), "bbox": box, "source_capacity_ml": capacity, "visual_presence": presence,
                     "note": note, "unit_status": "visible" if capacity == 5 else "indeterminate"})
    return rows


def run():
    frozen = read_json(FREEZE)
    if frozen["cases"] != inputs() or frozen["reader"] != digest(READER):
        raise ValueError("Public transfer inputs or frozen reader changed.")
    baseline, candidate = SceneOCR(), SmallCapacity()
    outcomes = []
    for row in frozen["cases"]:
        photo = crop(read_image(row["image"]), row["bbox"])
        for angle in (0, 90, 180, 270) if row["unit_status"] == "visible" else (0,):
            variant = photo.rotate(angle, expand=True)
            start = perf_counter()
            baseline_result = resolve(baseline.read(variant))
            advanced = candidate.read(variant)
            print(row["id"], angle, (baseline_result["capacity"] or {}).get("value"),
                  (advanced["capacity"] or {}).get("value"), flush=True)
            outcomes.append(dict(id=row["id"], degrees=angle, baseline=baseline_result,
                                 candidate=advanced, seconds=perf_counter()-start))
    write_json(OUTPUT, dict(reference=digest(FREEZE), outcomes=outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("freeze", "run"))
    if parser.parse_args().job == "freeze":
        if FREEZE.exists():
            raise FileExistsError(FREEZE)
        write_json(FREEZE, dict(reader=digest(READER), cases=inputs(), scope="Manual syringe crops. Rotation stress. "
                                                                            "Four photos from one acquisition."))
    else:
        run()
