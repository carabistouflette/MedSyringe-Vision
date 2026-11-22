"""Local MedSyringe Vision API and review workspace."""
from contextlib import asynccontextmanager
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from medsyringe.measurement.attributes import FIELDS, crop
from .data import coco, digest, read_image, read_json, train_rows, valid_box
from medsyringe.runtime.grounding_backend import analyze as analyze_local
from medsyringe.runtime.hardware import configure_device, device_request
from medsyringe.measurement.field_status import report_fields
from medsyringe.measurement.liquid_appearance import relative_opacity
from medsyringe.measurement.liquid_color import display_rgb
from medsyringe.measurement.liquid_fields import enrich_liquid_fields
from .pipeline import MedSyringeEngine
from medsyringe.review.corrections import InvalidReview, Review, UnitCoordinate, prepare_review_objects
from medsyringe.review.scene import Conflict, Store, compare, inventory

WEB = Path(__file__).with_name("web")
MAX_UPLOAD = 24 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 30_000_000
SOURCE_TYPES = {"JPEG":"jpg", "PNG":"png", "WEBP":"webp", "HEIF":"heic"}


class Expected(BaseModel):
    field: Literal["count", "scale", "color", "scale_and_color"]
    counts: dict[str, Annotated[int, Field(ge=0, le=250, strict=True)]]


class ExportRequest(BaseModel):
    revision: int = Field(ge=0)
    expected: Expected | None = None
    reference_id: str | None = None
    reference_revision: int | None = None


def preview(image):
    thumbnail = ImageOps.contain(image, (1900, 1900))
    output = io.BytesIO()
    profile = image.info.get("icc_profile")
    thumbnail.save(output, format="JPEG", quality=92, optimize=True, icc_profile=profile)
    return output.getvalue()


def decode_upload(file):
    payload = file.file.read(MAX_UPLOAD + 1)
    if len(payload) > MAX_UPLOAD:
        raise HTTPException(413, "Choose an image smaller than 24 MiB.")
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            if probe.format not in SOURCE_TYPES or probe.width * probe.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("Unsupported format or dimensions")
            extension = SOURCE_TYPES[probe.format]
        image = read_image(io.BytesIO(payload))
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Choose a readable JPEG, PNG, WebP or HEIC image up to 30 MP.") from exc
    return payload, image, extension


def create_app(root):
    root = Path(root)
    engine = None

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            if engine is not None:
                engine.close()

    app = FastAPI(title="MedSyringe Vision", version="0.1.0", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    app.mount("/static", StaticFiles(directory=WEB), name="static")
    store = Store(root / "data/workspace")
    condition = Lock()
    library = {r["id"]: r for r in train_rows(root)} if (root / "artifacts/annotations.json").exists() else {}
    drafts = read_json(root / "artifacts/annotations.json") if library else {}
    color_manifest = root / "artifacts/color_profiles.json"
    source_profiles = read_json(color_manifest) if color_manifest.exists() else {}

    def load(identifier):
        try:
            return report_fields(store.report(identifier))
        except KeyError as exc:
            raise HTTPException(404, "Observation was not found.") from exc

    @app.get("/")
    def home():
        return FileResponse(WEB / "index.html")

    @app.get("/api/library")
    def list_drafts():
        return [{"id": identifier, "flags": drafts[identifier]["flags"]}
                for identifier in sorted(library)]

    @app.post("/api/library/{identifier}")
    def open_draft(identifier: str):
        if identifier not in library:
            raise HTTPException(404, "Dataset image was not found.")
        record_id = "draft-" + identifier
        try:
            return load(record_id)
        except HTTPException as error:
            if error.status_code != 404:
                raise
            row = library[identifier]
            path = root / "data" / row["image"]
            if not path.exists():
                raise HTTPException(409, "Run source preparation to make this image available.")
            if digest(path) != row["derived_sha256"]:
                raise HTTPException(409, "The prepared image does not match its annotation manifest.")
            original = dict(name=row["source"], source_sha256=row["sha256"], kind="annotation",
                            review_status="needs review", flags=drafts[identifier]["flags"],
                            provenance=drafts[identifier]["provenance"],
                            dimensions=[row["width"], row["height"]],
                            objects=[dict(**obj, id=f"object-{i}", origin="draft")
                                     for i, obj in enumerate(row["objects"], 1)])
            image = read_image(path)
            if identifier in source_profiles:
                checksum = source_profiles[identifier]["icc_sha256"]
                saved_profile = root / "data/original-color" / f"{checksum}.icc"
                if saved_profile.is_file() and digest(saved_profile) == checksum:
                    image.info["icc_profile"] = saved_profile.read_bytes()
            return report_fields(store.create(original, preview(image), record_id))

    @app.get("/api/library-coco")
    def export_annotations():
        rows = []
        for identifier, row in library.items():
            try:
                report = store.report("draft-" + identifier)
            except KeyError:
                continue
            if report["review_status"] == "approved":
                revision = report["reviews"][-1]
                objects = [dict(**obj, reviewed_by=revision["reviewer"], review_revision=report["revision"])
                           for obj in report["objects"]]
                rows.append({**row, "objects": objects})
        return coco(rows)

    @app.get("/api/observations")
    def recent():
        return store.summaries()

    @app.get("/api/observations/{identifier}")
    def get_report(identifier: str):
        return load(identifier)

    @app.get("/api/observations/{identifier}/image")
    def get_image(identifier: str):
        try:
            return Response(store.image(identifier), media_type="image/jpeg",
                            headers={"Cache-Control": "private, no-store"})
        except KeyError as exc:
            raise HTTPException(404, "Observation was not found.") from exc

    def source_bytes(report):
        raw = store.source(report["id"])
        if raw is not None and sha256(raw).hexdigest() != report["original"]["source_sha256"]:
            raise HTTPException(409, "The stored source fails its integrity check.")
        return raw

    def crop_response(report, bbox):
        raw = source_bytes(report)
        if raw is None:
            raise HTTPException(404, "A native-resolution crop is unavailable for this observation.")
        image = read_image(io.BytesIO(raw))
        output = io.BytesIO()
        crop(image, bbox).save(output, format="PNG", icc_profile=image.info.get("icc_profile"))
        return Response(output.getvalue(), media_type="image/png", headers={"Cache-Control":"private, no-store"})

    @app.get("/api/observations/{identifier}/crop")
    def region_crop(identifier: str, bbox: Annotated[list[UnitCoordinate], Query(min_length=4, max_length=4)]):
        if not valid_box(bbox):
            raise HTTPException(422, "Choose a valid image region.")
        return crop_response(load(identifier), bbox)

    @app.get("/api/observations/{identifier}/objects/{object_id}/crop")
    def native_crop(identifier: str, object_id: str):
        report = load(identifier)
        obj = next((item for item in report["objects"] if item["id"] == object_id), None)
        if obj is None:
            raise HTTPException(404, "The object was not found in this revision.")
        return crop_response(report, obj["bbox"])

    @app.post("/api/observations")
    def analyze(file: UploadFile = File(...), ocr: bool = Form(False), count_only: bool = Form(False)):
        nonlocal engine
        payload, image, extension = decode_upload(file)
        checksum = sha256(payload).hexdigest()
        with condition:
            if engine is None:
                try:
                    requested = device_request()
                except ValueError as exc:
                    raise HTTPException(503, str(exc)) from exc
                pending = None
                try:
                    pending = MedSyringeEngine(root)
                    configure_device(pending, root, requested or "cpu")
                except (OSError, ValueError) as exc:
                    if pending is not None:
                        pending.close()
                    raise HTTPException(503, "The verified local checkpoint is unavailable. See setup instructions.") from exc
                engine = pending
            result = analyze_local(engine, image, ocr, count_only=count_only)
            if not count_only:
                enrich_liquid_fields(image, result["objects"])
        record = dict(**result, name=Path(file.filename or "image").name[:160], kind="inference",
                      source_sha256=checksum, source_extension=extension, review_status="unreviewed")
        return report_fields(store.create(record, preview(image), source=payload))

    @app.post("/api/observations/{identifier}/review")
    def review(identifier: str, edit: Review):
        current = load(identifier)
        def load_native_image():
            payload = source_bytes(current)
            return read_image(io.BytesIO(payload)) if payload is not None else None

        try:
            objects = prepare_review_objects(current, edit.objects, load_native_image)
        except InvalidReview as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            return report_fields(store.correct(identifier, edit.revision, objects, edit.reviewer, edit.note, edit.status))
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/liquid-opacity")
    def measure_opacity(observed: UploadFile = File(...), clear_reference: UploadFile = File(...),
                        matched_setup: bool = Form(False)):
        first, liquid, _ = decode_upload(observed)
        second, clear, _ = decode_upload(clear_reference)
        # The endpoint expects small, registered barrel regions, rather than full camera frames.
        if max(liquid.width * liquid.height, clear.width * clear.height) > 2_000_000:
            raise HTTPException(422, "Supply two registered barrel crops of at most two megapixels each.")
        observed_rgb, observed_profile = display_rgb(liquid)
        clear_rgb, clear_profile = display_rgb(clear)
        result = relative_opacity(np.asarray(observed_rgb), np.asarray(clear_rgb), matched_setup=matched_setup)
        return {**result, "status": "estimated" if result["value"] is not None else "indeterminate",
                "observed_sha256": sha256(first).hexdigest(), "clear_reference_sha256": sha256(second).hexdigest(),
                "profiles": [observed_profile, clear_profile]}

    def checked_inventory(report, expected):
        if expected.field == "count":
            keys = {"syringe"}
        elif expected.field == "scale_and_color":
            keys = {f"{scale} / {color}" for scale in FIELDS["scale"] for color in FIELDS["color"]}
        else:
            keys = set(FIELDS[expected.field])
        if set(expected.counts) != keys:
            raise HTTPException(422, "Supply a count for every label in the selected inventory mode.")
        return dict(observation_id=report["id"], revision=report["revision"],
                    **inventory(report["objects"], expected.counts, expected.field))

    @app.post("/api/observations/{identifier}/inventory")
    def check_inventory(identifier: str, expected: Expected):
        return checked_inventory(load(identifier), expected)

    @app.get("/api/observations/{identifier}/compare/{reference}")
    def compare_observations(identifier: str, reference: str):
        return compare(load(reference), load(identifier))

    @app.post("/api/observations/{identifier}/export")
    def export_report(identifier: str, export: ExportRequest):
        report = load(identifier)
        if report["revision"] != export.revision:
            raise HTTPException(409, "This observation changed. Open the current revision before exporting.")
        report["evidence_image"] = "observation.jpg"
        if export.expected:
            report["inventory_check"] = checked_inventory(report, export.expected)
        output = io.BytesIO()
        with ZipFile(output, "w", ZIP_DEFLATED) as archive:
            archive.writestr("observation.jpg", store.image(identifier))
            raw = source_bytes(report)
            if raw is not None:
                report["original_image"] = "source." + report["original"]["source_extension"]
                archive.writestr(report["original_image"],raw)
            if export.reference_id:
                reference = load(export.reference_id)
                if reference["revision"] != export.reference_revision:
                    raise HTTPException(409, "The reference changed. Run comparison again before exporting.")
                report["observation_comparison"] = compare(reference, report)
                report["reference"] = {**reference, "evidence_image": "reference.jpg"}
                archive.writestr("reference.jpg", store.image(reference["id"]))
                original_reference = source_bytes(reference)
                if original_reference is not None:
                    filename = "reference_source." + reference["original"]["source_extension"]
                    report["reference"]["original_image"] = filename
                    archive.writestr(filename, original_reference)
            archive.writestr("report.json", json.dumps(report, indent=2, ensure_ascii=False))
        return Response(output.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="medsyringe-{identifier}.zip"'})

    return app


app = create_app(Path(os.environ.get("MEDSYRINGE_ROOT", ".")))
