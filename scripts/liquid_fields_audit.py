"""Measure field coverage with frozen public examples and separate Fil Rouge replays."""
import argparse
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json

import numpy as np
from PIL import Image, ImageOps

from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.detectors import box_iou
from medsyringe.field_status import describe_fields
from medsyringe.liquid_color import display_rgb, lab_pixels
from medsyringe.liquid_fields import color_from_region
from scripts.evaluate_field_abstention import score
from scripts.liquid_fit import distance

ART = Path("artifacts/detection_hillclimb")
REFERENCES = ART / "liquid_field_public_reference.json"
LIVE = ART / "liquid_field_live_evaluation.json"
REPLAY = ART / "liquid_field_capture_replay.json"
FIELDS = ("nominal_capacity", "liquid_presence", "liquid_colour", "liquid_volume", "liquid_opacity")


def body_reference(image, points):
    display, _ = display_rgb(image)
    rgb = np.asarray(image)
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    warm = (hue >= 5) & (hue <= 39) & (saturation >= 45) & (value >= 25)
    positions = distance(points, (image.height, image.width), image.width, image.height) < .027
    samples = lab_pixels(np.asarray(display))[positions & warm]
    return np.median(samples, axis=0).tolist() if len(samples) >= 35 else None


def replay():
    files = read_json("artifacts/marking_detected_manifest.json")["files"]
    measurements = read_json("artifacts/liquid_detected_results.json")["frames"]
    marks = read_json("artifacts/liquid_region_scribbles.json")["regions"]
    holdout = read_json("artifacts/liquid_region_study.json")["holdout_groups"]
    clear = read_json("artifacts/expanded/liquid_test.json")["legacy_frames"]
    records = []
    for row in files:
        key = row["id"]
        image = ImageOps.contain(read_image(row["image"]), (420, 420))
        appearance = {**measurements[key]}
        color = color_from_region(image, appearance)
        if color:
            appearance["apparent_color"] = color
        lab = body_reference(image, marks[key]["points"])
        ref = dict(liquid_presence=dict(status="observable", value=True))
        if lab:
            ref["liquid_colour"] = dict(status="observable", lab=lab)
        records.append(dict(id=key, heldout_block=row["burst"] in holdout, reference=ref,
                            prediction=describe_fields(dict(liquid_appearance=appearance))))
    for key, row in clear.items():
        appearance = {**row["measurement"]}
        if appearance.get("lightness"):
            with np.load(Path("data/expanded/liquid/test") / f"{key}.npz") as case:
                measured = color_from_region(Image.fromarray(case["raw"]), appearance)
            if measured:
                appearance["apparent_color"] = measured
        records.append(dict(id=key, clear_control=True, reference=dict(liquid_presence=dict(status="indeterminate")),
                            prediction=describe_fields(dict(liquid_appearance=appearance))))
    scopes = {name: local for name, local in (
        ("visible colored regions", [r for r in records if not r.get("clear_control")]),
        ("held-out capture blocks", [r for r in records if r.get("heldout_block")]),
        ("uncolored capture blocks", [r for r in records if r.get("clear_control")]))}
    write_json(REPLAY, dict(
        cohorts={name: {label: score(cohort, label) for label in ("liquid_presence", "liquid_colour")}
                 for name, cohort in scopes.items()}, rows=records,
        scope="One historical capture. The clear photos are unresolved references. "
              "Color reference is the displayed Lab of pixels near a manually marked centerline."))
    print({name: score(local, "liquid_colour")["scored_colours"] for name, local in scopes.items()}, flush=True)


def live():
    from fastapi.testclient import TestClient

    from medsyringe.app import create_app

    workspace = Path.cwd()
    inputs = read_json(REFERENCES)["rows"]
    frames = []
    with TemporaryDirectory(prefix="medsyringe-field-audit-", dir=workspace / "data") as temporary:
        root = Path(temporary)
        for name in ("artifacts", "scripts", "medsyringe", ".venv"):
            (root / name).symlink_to(workspace / name, target_is_directory=True)
        (root / "data").mkdir()
        for name in ("models", "sam3", "ocr-v6", "rfdetr", "expanded"):
            (root / "data" / name).symlink_to(workspace / "data" / name, target_is_directory=True)
        with TestClient(create_app(root)) as client:
            for row in inputs:
                image = Path(row["image"])
                if digest(image) != row["sha256"]:
                    raise ValueError(f"Changed input photo {row['id']}.")
                response = client.post("/api/observations",
                                       files={"file": (image.name, image.read_bytes())}, data={"ocr": "true"})
                response.raise_for_status()
                result = response.json()
                health = result["original"]["detection_checks"]
                if health["status"] != "available":
                    raise ValueError(f"A detector worker was unavailable on {row['id']}.")
                matched = max(result["objects"], key=lambda o: box_iou(row["bbox"], o["bbox"]), default=None)
                matched = matched if matched and box_iou(row["bbox"], matched["bbox"]) >= .50 else None
                record = dict(id=row["id"], localized=matched is not None, reference=row["fields"],
                              observed_count=len(result["objects"]), engine=health,
                              prediction=result["field_status"][matched["id"]] if matched else {})
                if matched:
                    record["localized_box"] = matched["bbox"]
                    record["appearance"] = matched.get("liquid_appearance")
                if matched and row.get("review"):
                    revisions = [dict(item, **row["review"]) if item["id"] == matched["id"] else item
                                 for item in result["objects"]]
                    base = f"/api/observations/{result['id']}"
                    corrected = client.post(base + "/review", json=dict(
                        revision=0, reviewer="Local visual audit", status="approved",
                        note=row["notes"], objects=revisions))
                    if corrected.status_code != 200:
                        raise ValueError(f"Live review was rejected on {row['id']}: {corrected.text}")
                    report = corrected.json()
                    archive = client.post(base + "/export", json={"revision": 1})
                    archive.raise_for_status()
                    with ZipFile(BytesIO(archive.content)) as zipped:
                        exported = json.loads(zipped.read("report.json"))
                    if exported["field_status"] != report["field_status"] or exported["original_field_status"] != result["field_status"]:
                        raise ValueError("The live field review failed the export round-trip.")
                    record["reviewed_fields"] = report["field_status"][matched["id"]]
                    record["review_export_verified"] = True
                frames.append(record)
                print(record["id"], record["observed_count"], record["localized"], record["prediction"], flush=True)
    write_json(LIVE, dict(reference_sha256=digest(REFERENCES), frames=frames,
                          code={path: digest(path) for path in ("medsyringe/app.py", "medsyringe/field_status.py",
                                                               "medsyringe/liquid_fields.py")},
                          metrics={field: score(frames, field) for field in FIELDS},
                          scope="Exact local API with OCR on. Public visual references and a named Fil Rouge control "
                                "are listed separately in the source manifest. Missing detections count as abstentions."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("replay", "live"))
    (replay if parser.parse_args().stage == "replay" else live)()
