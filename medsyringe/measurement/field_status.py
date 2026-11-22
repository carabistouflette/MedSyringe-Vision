"""Describe supported and indeterminate measurements per detected object."""
from math import isfinite


def number(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def capacity_status(obj, scale):
    reviewed = obj.get("nominal_capacity_override_ml")
    if number(reviewed) and reviewed > 0:
        return dict(status="supported", ml=reviewed, source="review", reason="Nominal capacity confirmed in review")
    fill = obj.get("fill_level") or {}
    value = fill.get("capacity_ml")
    if fill.get("status") == "manual estimate" and number(value) and value > 0:
        return dict(status="supported", ml=value, source="reviewed fill",
                    reason="Nominal capacity supplied with the reviewed graduations")
    printed = (obj.get("inscriptions") or {}).get("capacity") or {}
    value = printed.get("value")
    if (printed.get("unit") == "mL" and number(value) and value > 0 and
            (scale is None or scale == f"{value:g}-scale")):
        return dict(status="supported", ml=value, source="marking", reason="Nominal capacity read from the visible scale")
    return dict(status="indeterminate", ml=None, source=None, reason="Nominal-capacity markings are unavailable or ambiguous")


def presence_status(obj, fill, appearance):
    override = obj.get("liquid_presence_override")
    if override in ("present", "empty", "indeterminate"):
        return dict(status=override, value={"present": True, "empty": False}.get(override), source="review",
                    reason="Liquid presence assessed in review")
    ml = fill.get("ml")
    if fill.get("status") == "manual estimate" and number(ml):
        return dict(status="present" if ml else "empty", value=bool(ml), source="reviewed fill",
                    reason="Visible piston position reviewed against numbered graduations")
    if obj.get("liquid_sample") and appearance.get("apparent_color"):
        return dict(status="present", value=True, source="reviewed sample", reason="The reviewer selected visible liquid")
    if appearance.get("status") == "appearance measured" and appearance.get("apparent_color") and appearance.get("region"):
        return dict(status="present", value=True, source="appearance", reason="The localized barrel interior supports visible liquid")
    interval = fill.get("interval_ml")
    if fill.get("status") in ("visual estimate", "visual interval") and interval and number(interval[0]) and interval[0] > 0:
        return dict(status="present", value=True, source="fill estimate", reason="The visible fill range is above zero")
    return dict(status="indeterminate", value=None, source=None,
                reason="Visible evidence does not resolve liquid presence or an empty barrel")


def describe_fields(obj):
    evidence = (obj.get("evidence") or {}).get("scale") or {}
    scale = obj.get("scale")
    type_status = dict(status="supported" if scale is not None else "indeterminate", value=scale,
                       reason=evidence.get("reason") or (
                           "Visual scale entered during review." if scale is not None and obj.get("origin") == "human" else
                           "Visible syringe scale is supported." if scale is not None else
                           "No supported type or scale evidence for this box."))
    fill = obj.get("fill_level") or {}
    value = fill.get("ml")
    interval = fill.get("interval_ml")
    interval = list(interval) if (isinstance(interval, (list, tuple)) and len(interval) == 2
                                  and all(number(x) for x in interval) and interval[0] <= interval[1]) else None
    if fill.get("status") in ("visual estimate", "manual estimate") and number(value):
        volume = dict(status="estimated", ml=value, interval_ml=interval, reason=fill.get("method"))
    elif fill.get("status") == "visual interval" and interval is not None:
        volume = dict(status="interval only", ml=None, interval_ml=interval, reason=fill.get("reason"))
    else:
        volume = dict(status="indeterminate", ml=None, interval_ml=None, reason=fill.get("reason") or
                      "No supported graduation and piston measurement for this box.")
    appearance = obj.get("liquid_appearance") or {}
    color = appearance.get("apparent_color")
    lightness = (appearance.get("lightness") or {}).get("lstar")
    opacity = appearance.get("opacity")
    if isinstance(opacity, dict):
        opacity = opacity.get("value")
    presence = presence_status(obj, fill, appearance)
    if presence["status"] == "empty":
        color, lightness, opacity = None, None, None
    capacity = capacity_status(obj, scale)
    return dict(
        syringe_type=type_status, nominal_capacity=capacity, liquid_presence=presence, liquid_volume=volume,
        liquid_colour=dict(status="estimated" if color else "indeterminate", lab=color.get("lab") if color else None,
                           srgb=color.get("srgb") if color else None, family=color.get("family") if color else None,
                           lightness_lstar=(color["lab"][0] if color and color.get("lab") else
                                            lightness if presence["status"] == "present" else None),
                           basis=appearance.get("color_basis"), reason=(
                               "The barrel was reviewed as empty." if presence["status"] == "empty" else
                               "Apparent color measured in the supported region." if color else
                               appearance.get("color_reason") or
                               "The visible liquid region does not support a color measurement.")),
        liquid_opacity=dict(status="estimated" if number(opacity) and opacity <= 1 else "indeterminate",
                            value=opacity if number(opacity) and opacity <= 1 else None,
                            basis="relative luminance contrast attenuation",
                            reason="The barrel was reviewed as empty." if presence["status"] == "empty" else
                            appearance.get("opacity_reason") or "No matched clear-barrel reference."),
    )


def report_fields(report):
    return {**report,
            "field_status": {obj["id"]: describe_fields(obj) for obj in report["objects"]},
            "original_field_status": {obj["id"]: describe_fields(obj)
                                      for obj in report["original"]["objects"]}}
