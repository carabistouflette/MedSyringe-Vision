"""Build correction records and retain evidence that still applies to the edited boxes."""
from functools import cache
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from medsyringe.data import valid_box
from medsyringe.measurement.liquid_fields import reviewed_liquid_color
from medsyringe.measurement.manual_fill import estimate as estimate_manual_fill

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


UnitCoordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
LandmarkPoint = tuple[UnitCoordinate, UnitCoordinate]
CropReviewField = Literal["scale", "color", "transcription", "nominal_capacity_override_ml",
                          "liquid_presence_override"]
CROP_REVIEW_DEFAULTS = {
    "scale": None, "color": None, "transcription": "",
    "nominal_capacity_override_ml": None, "liquid_presence_override": None,
}


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
    nominal_capacity_override_ml: float | None = Field(default=None, gt=0, le=250, allow_inf_nan=False)
    liquid_presence_override: Literal["present", "empty", "indeterminate"] | None = None
    liquid_sample: tuple[UnitCoordinate, UnitCoordinate, UnitCoordinate, UnitCoordinate] | None = None
    candidate_index: Annotated[int, Field(ge=0, strict=True)] | None = None
    confirmed_on_box: set[CropReviewField] = Field(default_factory=set, exclude=True)


class Review(BaseModel):
    reviewer: Nonempty
    note: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)]
    revision: int = Field(ge=0)
    status: Literal["approved", "needs review"]
    objects: list[ReviewObject] = Field(max_length=100)


class InvalidReview(ValueError):
    """A submitted correction conflicts with its image or retained evidence."""


def _apply_candidate(obj, suggestions):
    index = obj["candidate_index"]
    if index is None:
        del obj["candidate_index"]
        return
    if index >= len(suggestions) or suggestions[index]["bbox"] != obj["bbox"]:
        raise InvalidReview("The candidate reference must match its original box.")
    suggestion = suggestions[index]
    obj["origin"] = "human"
    obj["score"] = None
    obj["reviewed_candidate"] = {
        "index": index,
        "score": suggestion.get("original_score", suggestion["score"]),
        "reason": suggestion.get("review_reason", "Below the automatic operating point"),
    }


def _apply_fill(obj, submitted, old, dimensions, changed_box, markings_revised):
    manual = "manual_fill" in submitted.model_fields_set
    if manual:
        obj["origin"] = "human"
        if submitted.manual_fill is not None:
            try:
                obj["fill_level"] = estimate_manual_fill(
                    submitted.manual_fill.model_dump(mode="json"),
                    obj["bbox"], dimensions, obj["scale"],
                )
            except ValueError as exc:
                raise InvalidReview(str(exc)) from exc
    elif "fill_level" in old and not changed_box and obj["scale"] == old.get("scale"):
        if not markings_revised or (old["fill_level"] or {}).get("status") == "manual estimate":
            obj["fill_level"] = old["fill_level"]

    fill = obj.get("fill_level") or {}
    presence = obj["liquid_presence_override"]
    capacity = obj["nominal_capacity_override_ml"]
    if manual and submitted.manual_fill is not None:
        if (presence == "empty" and fill["ml"] > 0) or (presence == "present" and fill["ml"] == 0):
            raise InvalidReview("The reviewed presence and fill estimate disagree.")
        if capacity is not None and fill["capacity_ml"] != capacity:
            raise InvalidReview("The reviewed nominal capacity and fill landmarks disagree.")
    elif (
        (presence == "empty" and fill.get("ml") != 0)
        or (presence == "present" and fill.get("ml") == 0)
        or (capacity is not None and fill.get("capacity_ml") not in (None, capacity))
    ):
        obj.pop("fill_level", None)


def _apply_liquid_sample(obj, old, changed_box, load_native_image):
    sample = obj["liquid_sample"]
    previous = old.get("liquid_sample")
    if obj["liquid_presence_override"] == "empty":
        obj.pop("liquid_appearance", None)
        obj["liquid_sample"] = None
    elif sample is None:
        if previous is not None:
            obj.pop("liquid_appearance", None)
    elif sample == previous:
        if changed_box:
            obj["liquid_sample"] = None
    else:
        native = load_native_image()
        if native is None:
            raise InvalidReview("The native image is required to sample its liquid color.")
        try:
            obj["liquid_appearance"] = reviewed_liquid_color(native, obj["bbox"], sample)
        except ValueError as exc:
            raise InvalidReview(str(exc)) from exc
        obj["origin"] = "human"


def prepare_review_objects(current, submitted_objects, load_native_image):
    """Validate edits against the current report and load native pixels only for new samples."""
    objects = [obj.model_dump(mode="json", exclude={"manual_fill"}) for obj in submitted_objects]
    if any(not valid_box(obj["bbox"]) for obj in objects) or len({obj["id"] for obj in objects}) != len(objects):
        raise InvalidReview("Every object needs a unique ID and a valid box.")
    indexes = [obj["candidate_index"] for obj in objects if obj["candidate_index"] is not None]
    if len(set(indexes)) != len(indexes):
        raise InvalidReview("A model candidate can be included only once in the current review.")

    prior = {obj["id"]: obj for obj in current["objects"]}
    original = current["original"]
    suggestions = original.get("below_threshold_candidates", [])
    dimensions = original.get("dimensions", (0, 0))
    load_native_image = cache(load_native_image)
    for obj, submitted in zip(objects, submitted_objects, strict=True):
        old = prior.get(obj["id"], {})
        changed_box = old.get("bbox") != obj["bbox"]
        if changed_box and old:
            for field, default in CROP_REVIEW_DEFAULTS.items():
                if field not in submitted.confirmed_on_box and obj[field] == old.get(field):
                    obj[field] = default
        elif not changed_box:
            for field in ("nominal_capacity_override_ml", "liquid_presence_override", "liquid_sample"):
                if field not in submitted.model_fields_set:
                    obj[field] = old.get(field)

        capacity = obj["nominal_capacity_override_ml"]
        if capacity is not None and obj["scale"] != f"{capacity:g}-scale":
            obj["scale"] = None
        markings_revised = not changed_box and obj["transcription"] != old.get("transcription", "")
        if (
            markings_revised and (old.get("inscriptions") or {}).get("scale")
            and obj["scale"] == old.get("scale") and "scale" not in submitted.confirmed_on_box
        ):
            obj["scale"] = None

        changed = changed_box or any(old.get(k) != obj[k] for k in CROP_REVIEW_DEFAULTS if k in old or obj[k])
        obj["origin"] = "human" if changed or not old else old["origin"]
        obj["score"] = None if changed_box else old.get("score")
        _apply_candidate(obj, suggestions)

        for key in ("background", "burst", "inscriptions", "detection_evidence", "liquid_appearance"):
            if key == "inscriptions" and markings_revised:
                continue
            if key in old and (key in ("background", "burst") or not changed_box):
                obj[key] = old[key]
        _apply_fill(obj, submitted, old, dimensions, changed_box, markings_revised)
        _apply_liquid_sample(obj, old, changed_box, load_native_image)

        if not changed_box and old.get("evidence"):
            obj["evidence"] = {
                key: evidence for key, evidence in old["evidence"].items()
                if obj.get(key) == old.get(key)
            }
    return objects
