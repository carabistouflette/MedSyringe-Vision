"""Source a reviewed few-shot dental dataset outside all existing evaluation groups."""
import argparse
import json
from urllib.parse import urlencode

from medsyringe.data import digest, leakage, read_image, read_json, valid_box, write_json
from scripts.experiments.detector_training.detection_hillclimb_experiment import ART
from scripts.experiments.detector_audits.detection_precision_negative_audit import CANDIDATES as NEGATIVE_POOL, exposure, overlaps, protect
from scripts.experiments.public_data.detection_public_benchmark import API, ROOT, WORK, get_asset, metadata, order, request, sheet, withheld

POOL = WORK / "dental-transfer-pool.json"
REFERENCES = ART / "dental_transfer_review.json"
MANIFEST = ART / "dental_transfer_manifest.json"
CATEGORIES = (
    ("Dental syringes", "positive"),
    ("Dental local anesthesia", "positive"),
    ("Dental hand instruments", "lookalike"),
    ("Mouth mirrors", "lookalike"),
    ("Dental calculus scalers", "lookalike"),
    ("Dental angle handpieces", "lookalike"),
)


def previous():
    known = exposure()
    for row in read_json(NEGATIVE_POOL)["rows"]:
        protect(known, row)
    return known


def find():
    if MANIFEST.exists():
        raise ValueError("The transfer dataset is already sealed.")
    known = previous()
    acquired = []
    for category, role in CATEGORIES:
        args = dict(action="query", generator="categorymembers", gcmtitle=f"Category:{category}", gcmtype="file",
                    gcmlimit="50", prop="imageinfo", iiprop="url|size|mime|extmetadata|timestamp", iiurlwidth="960",
                    format="json")
        try:
            pages = json.loads(request(API + "?" + urlencode(args), 8 * 1024 * 1024))
        except (OSError, ValueError) as exc:
            print(category, exc, flush=True)
            continue
        rows = [row for page in pages.get("query", {}).get("pages", {}).values()
                if (row := metadata(page, category, role))]
        for raw in sorted(rows, key=lambda row: order(row["id"])):
            if withheld(raw, known):
                continue
            received = get_asset(raw)
            if received is None:
                continue
            row, picture = received
            if overlaps(known, row, picture):
                continue
            protect(known, row)
            acquired.append(row)
        print(category, len(rows), "records", len(acquired), "eligible cumulatively", flush=True)
    write_json(POOL, dict(rows=acquired))
    sheet(acquired, "dental-transfer", 4, 3)


def freeze():
    if MANIFEST.exists():
        raise ValueError("The transfer dataset is already sealed.")
    references = read_json(REFERENCES)
    if references["pool_sha256"] != digest(POOL):
        raise ValueError("The annotation record does not match the downloaded rasters.")
    known = previous()
    positive, negative = [], []
    for row in read_json(POOL)["rows"]:
        annotation = references["images"].get(row["id"])
        if annotation is None:
            continue
        path = ROOT / row["image"]
        if digest(path) != row["sha256"] or overlaps(known, row, read_image(path)):
            raise ValueError(f"Changed or overlapping photograph {row['id']}.")
        labels = [*annotation["objects"], *annotation["lookalikes"]]
        if not labels or any(not valid_box(o["bbox"]) for o in labels):
            raise ValueError("A reviewed image needs valid target or lookalike regions.")
        protect(known, row)
        (positive if annotation["objects"] else negative).append({**row, **annotation})
    if len(positive) < 6 or len(negative) < 6:
        raise ValueError("Insufficient independent positive or negative dental instrument acquisitions.")
    result = []
    for stratum in (positive, negative):
        ranked = sorted(stratum, key=lambda r: order("dental-development:" + r["id"]))
        # A whole-source audit and development each need positives and lookalikes.
        # All remaining distinct groups are kept for the low-data fit.
        development_size = max(2, len(stratum)//4)
        audit_size = 3 if stratum is positive else 4
        for role, group in (("development", ranked[:development_size]),
                            ("audit", ranked[development_size:development_size+audit_size]),
                            ("fit", ranked[development_size+audit_size:])):
            result.extend({**r, "split": role} for r in group)
    if len(result) != len(references["images"]) or leakage(result):
        raise ValueError("Invalid annotation coverage or cross-role overlap in the transfer dataset.")
    write_json(MANIFEST, dict(rows=result, references_sha256=digest(REFERENCES),
                             fit_role="Separate dental Commons acquisitions. Never use the published challenge to fit."))
    print("Reviewed dental transfer", len(positive), "positive and", len(negative), "negative sources", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("find", "freeze"))
    (find if parser.parse_args().action == "find" else freeze)()
