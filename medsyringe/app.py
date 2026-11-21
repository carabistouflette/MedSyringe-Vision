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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, StringConstraints
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .attributes import FIELDS, crop
from .data import coco, digest, read_image, read_json, train_rows, valid_box
from .grounding_backend import analyze as analyze_local, enable_intel
from .manual_fill import estimate as estimate_manual_fill
from .pipeline import MedSyringeEngine
from .scene import Conflict, Store, compare, inventory

WEB = Path(__file__).with_name("web")
MAX_UPLOAD = 24 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 30_000_000
SOURCE_TYPES = {"JPEG":"jpg", "PNG":"png", "WEBP":"webp", "HEIF":"heic"}
Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


UnitCoordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
LandmarkPoint = tuple[UnitCoordinate, UnitCoordinate]


class Graduation(BaseModel):
    point: LandmarkPoint
    ml: float = Field(ge=0, le=250, allow_inf_nan=False)


class ManualFill(BaseModel):
    capacity_ml: float = Field(gt=0, le=250, allow_inf_nan=False)
    low: Graduation
    high: Graduation
    piston: LandmarkPoint


class ReviewObject(BaseModel):
    id: Annotated[str, StringConstraints(pattern="^[a-zA-Z0-9_-]{1,80}$")]
    bbox: tuple[float, float, float, float]
    scale: Literal["20-scale", "60-scale"] | None = None
    color: Literal["amber-brown", "orange"] | None = None
    transcription: str = Field(default="", max_length=320)
    score: float | None = None
    origin: Literal["model", "draft", "human"] = "human"
    manual_fill: ManualFill | None = None
    candidate_index: Annotated[int, Field(ge=0, strict=True)] | None = None


class Review(BaseModel):
    reviewer: Nonempty
    note: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)]
    revision: int = Field(ge=0)
    status: Literal["approved", "needs review"]
    objects: list[ReviewObject] = Field(max_length=100)


class Expected(BaseModel):
    field: Literal["count", "scale", "color"]
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
            return store.report(identifier)
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
            return store.report(record_id)
        except KeyError:
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
            return store.create(original, preview(image), record_id)

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

    @app.get("/api/observations/{identifier}/objects/{object_id}/crop")
    def native_crop(identifier: str, object_id: str):
        report = load(identifier)
        obj = next((item for item in report["objects"] if item["id"] == object_id), None)
        if obj is None:
            raise HTTPException(404, "The object was not found in this revision.")
        raw = source_bytes(report)
        if raw is None:
            raise HTTPException(404, "A native-resolution crop is unavailable for this observation.")
        image = read_image(io.BytesIO(raw))
        output = io.BytesIO()
        crop(image, obj["bbox"]).save(output, format="PNG", icc_profile=image.info.get("icc_profile"))
        return Response(output.getvalue(), media_type="image/png", headers={"Cache-Control":"private, no-store"})

    @app.post("/api/observations")
    def analyze(file: UploadFile = File(...), ocr: bool = Form(False)):
        nonlocal engine
        payload = file.file.read(MAX_UPLOAD+1)
        if len(payload) > MAX_UPLOAD:
            raise HTTPException(413, "Choose an image smaller than 24 MiB.")
        try:
            with Image.open(io.BytesIO(payload)) as probe:
                if probe.format not in SOURCE_TYPES or (
                    probe.width * probe.height > Image.MAX_IMAGE_PIXELS
                ):
                    raise ValueError("Unsupported format or dimensions")
                extension = SOURCE_TYPES[probe.format]
            image = read_image(io.BytesIO(payload))
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, "Choose a readable JPEG, PNG, WebP or HEIC image up to 30 MP.") from exc
        checksum = sha256(payload).hexdigest()
        with condition:
            if engine is None:
                try:
                    engine = MedSyringeEngine(root)
                except (OSError, ValueError) as exc:
                    raise HTTPException(503, "The verified local checkpoint is unavailable. See setup instructions.") from exc
                if os.environ.get("MEDSYRINGE_PRIMARY_DEVICE") == "xpu":
                    enable_intel(engine, root)
            result = analyze_local(engine, image, ocr)
        record = dict(**result, name=Path(file.filename or "image").name[:160], kind="inference",
                      source_sha256=checksum, source_extension=extension, review_status="unreviewed")
        return store.create(record, preview(image), source=payload)

    @app.post("/api/observations/{identifier}/review")
    def review(identifier: str, edit: Review):
        current = load(identifier)
        objects = [obj.model_dump(mode="json", exclude={"manual_fill"}) for obj in edit.objects]
        if any(not valid_box(obj["bbox"]) for obj in objects) or (
            len({obj["id"] for obj in objects}) != len(objects)
        ):
            raise HTTPException(422, "Every object needs a unique ID and a valid box.")
        prior = {obj["id"]: obj for obj in current["objects"]}
        suggestions = current["original"].get("below_threshold_candidates", [])
        indexes = [obj["candidate_index"] for obj in objects if obj["candidate_index"] is not None]
        if len(set(indexes)) != len(indexes):
            raise HTTPException(422, "A model candidate can be included only once in the current review.")
        for obj, submitted in zip(objects, edit.objects, strict=True):
            old = prior.get(obj["id"], {})
            changed_box = old.get("bbox") != obj["bbox"]
            changed = changed_box or any(old.get(k) != obj[k] for k in ("scale", "color", "transcription")
                                         if k in old or obj[k])
            obj["origin"] = "human" if changed or not old else old["origin"]
            obj["score"] = None if changed_box else old.get("score")
            index = obj["candidate_index"]
            if index is None:
                del obj["candidate_index"]
            else:
                if index >= len(suggestions) or suggestions[index]["bbox"] != obj["bbox"]:
                    raise HTTPException(422, "The candidate reference must match its original box.")
                suggestion = suggestions[index]
                obj["origin"] = "human"
                obj["score"] = None
                obj["reviewed_candidate"] = {
                    "index": index, "score": suggestion.get("original_score", suggestion["score"]),
                    "reason": suggestion.get("review_reason", "Below the automatic operating point"),
                }
            for key in ("background", "burst", "inscriptions", "detection_evidence", "liquid_appearance"):
                if key in old and (key in ("background", "burst") or not changed_box):
                    obj[key] = old[key]
            if "manual_fill" in submitted.model_fields_set:
                obj["origin"] = "human"
                if submitted.manual_fill is not None:
                    try:
                        obj["fill_level"] = estimate_manual_fill(
                            submitted.manual_fill.model_dump(mode="json"), obj["bbox"],
                            current["original"].get("dimensions", (0, 0)), obj["scale"])
                    except ValueError as exc:
                        raise HTTPException(422, str(exc)) from exc
            elif "fill_level" in old and not changed_box and obj["scale"] == old.get("scale"):
                obj["fill_level"] = old["fill_level"]
            if not changed_box and old.get("evidence"):
                obj["evidence"] = {key: evidence for key, evidence in old["evidence"].items()
                                   if obj.get(key) == old.get(key)}
        try:
            return store.correct(identifier, edit.revision, objects, edit.reviewer, edit.note, edit.status)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc

    def checked_inventory(report, expected):
        keys = {"syringe"} if expected.field == "count" else set(FIELDS[expected.field])
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
            archive.writestr("report.json", json.dumps(report, indent=2, ensure_ascii=False))
        return Response(output.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="medsyringe-{identifier}.zip"'})

    return app


app = create_app(Path(os.environ.get("MEDSYRINGE_ROOT", ".")))
