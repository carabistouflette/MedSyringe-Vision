"""Stage independent source photographs of several small syringe bodies in clutter."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

from medsyringe.data import digest, read_image, read_json, valid_box, write_json
from scripts.experiments.crowds.detection_crowd_source_data import MANIFEST as PREVIOUS, MODERN_POOL, used_sources
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART, DATA
from scripts.experiments.detector_audits.detection_precision_negative_audit import overlaps, protect
from scripts.experiments.public_data.detection_public_benchmark import API, get_asset, metadata, request, sheet, withheld

STAGED = DATA / "commons-crowds-new-screen.json"
REVIEW = ART / "crowd_field_reference_review.json"
MANIFEST = ART / "crowd_field_manifest.json"
TITLES = (
    "Syringes injectienaalden.jpg",
    "Needle exchange supplies.jpg",
    "Schlassgoart, bei der Krankekeess-101.jpg",
    "Schlassgoart, bei der Krankekeess-102.jpg",
    "Syringes and intravenous drug use paraphenalia at SFSTC IMG 6996.JPG",
    "Syringes and intravenous drug use paraphenalia at SFSTC IMG 6998.JPG",
    "Heroin paraphernalia.jpg",
    "Used Syringes Next to Childrens Toys on the Fort Peck Indian Reservation.jpg",
    "Who needs a virtual world?.jpg",
    "Table with medical supplies and personal items in a room at midday.jpg",
    "Comirnaty Omicron XBB.1.5 vial and influenza vaccine 2023.jpg",
    "Ampoule prefilled JPN.jpg",
)


def previous_groups():
    known = used_sources()
    for file in (PREVIOUS, MODERN_POOL):
        for row in read_json(file)["rows"]:
            protect(known, row)
    return known


def discover():
    if MANIFEST.exists() or STAGED.exists():
        raise ValueError("The screened field scenes are already registered.")
    query = dict(action="query", prop="imageinfo", iiprop="url|size|mime|extmetadata|timestamp",
                 iiurlwidth=1280, format="json", titles="|".join(f"File:{name}" for name in TITLES))
    catalog = json.loads(request(API + "?" + urlencode(query), 12 * 1024 * 1024))
    known = previous_groups()
    rows, excluded = [], []
    for page in sorted(catalog.get("query", {}).get("pages", {}).values(), key=lambda p: p["title"]):
        row = metadata(page, "needle exchange and multiple disposable devices", "natural crowd candidate")
        if row is None or withheld(row, known):
            excluded.append([page["title"], "metadata or lineage"])
            continue
        staged = get_asset(row)
        if staged is None:
            excluded.append([page["title"], "download"])
            continue
        row, picture = staged
        if overlaps(known, row, picture):
            excluded.append([page["title"], "pixel overlap"])
            continue
        protect(known, row)
        rows.append(row)
    write_json(STAGED, dict(rows=rows, excluded=excluded, titles=TITLES))
    sheet(rows, "commons-new-crowd", 4, 3)
    print("Field sources screened", len(rows), "out of", len(TITLES), flush=True)


def freeze():
    if MANIFEST.exists():
        raise ValueError("The field reference dataset has already been sealed.")
    review = read_json(REVIEW)
    if review["screen_sha256"] != digest(STAGED):
        raise ValueError("Review decisions do not match the screened sources.")
    known = previous_groups()
    records = []
    for row in read_json(STAGED)["rows"]:
        label = review["cases"][row["id"]]
        if "boxes_px" not in label:
            continue
        image = Path("data") / row["image"]
        if digest(image) != row["sha256"] or withheld(row, known) or overlaps(known, row, read_image(image)):
            raise ValueError(f"Staged field source changed or overlaps an earlier group: {row['id']}.")
        width, height = read_image(image).size
        if [width, height] != label["dimensions"]:
            raise ValueError(f"The field reference dimensions changed for {row['id']}.")
        boxes = [[round(x / width, 7), round(y / height, 7), round(r / width, 7), round(b / height, 7)]
                 for x, y, r, b in label["boxes_px"]]
        if not all(valid_box(box) for box in boxes):
            raise ValueError(f"Malformed field reference for {row['id']}.")
        protect(known, row)
        records.append({**row, "split": "crowd-field-reserve", "scope": label["scope"],
                        "objects": [dict(kind="syringe", bbox=b) for b in boxes]})
    if sum(len(row["objects"]) >= 2 for row in records) < 2:
        raise ValueError("The descriptive field reserve needs two countable crowded acquisitions.")
    write_json(MANIFEST, dict(rows=records, reference_review_sha256=digest(REVIEW),
                             method="Online public photo groups acquired separately. Tiny or clipped bodies are "
                                    "included only when the barrel and a device-specific feature remain visible. "
                                    "The references were entered before any detector outputs. "
                                    "These few scenes form a descriptive transfer check."))
    print("Field source references frozen", len(records), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("discover", "freeze"))
    (discover if parser.parse_args().action == "discover" else freeze)()
