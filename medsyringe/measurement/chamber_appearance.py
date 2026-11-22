"""Describe a colored column supported by a transverse seal in an upright barrel crop."""
import cv2
import numpy as np

from medsyringe.measurement.liquid_appearance import OPACITY_REASON
from medsyringe.measurement.liquid_color import describe_region, display_rgb
from medsyringe.measurement.liquid_fields import visible_hue
from medsyringe.measurement.small_fill import spans


def unknown(reason):
    return dict(presence="unknown", appearance=None, fill_ml=None, opacity=None,
                reason=reason, opacity_reason=OPACITY_REASON)


def inspect_chamber(image):
    """Find a colored chamber using pixels alone, with no markings or capacity input."""
    view, profile = display_rgb(image)
    rgb = np.asarray(view)
    horizontal = image.width > image.height
    if horizontal:
        rgb = np.ascontiguousarray(rgb.transpose(1, 0, 2))
    height, width = rgb.shape[:2]
    if width < 20 or height < 2.2 * width:
        return unknown("The syringe crop does not establish an upright or horizontal barrel.")
    if height > 768:
        rgb = cv2.resize(rgb, (round(width * 768 / height), 768), interpolation=cv2.INTER_AREA)
        height, width = rgb.shape[:2]
    left, right = round(.31 * width), round(.69 * width)
    if right - left < 8:
        return unknown("Too few pixels across the barrel center.")
    hue, saturation, value = np.moveaxis(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), -1, 0)
    color = (saturation >= 33) & (value >= 48)
    center = color[:, left:right]
    brightness = value[:, left:right]
    dark = (brightness <= .40 * np.quantile(brightness, .75)).mean(axis=1) >= .68
    row = (center.mean(axis=1) >= .57).astype(np.uint8).reshape(-1, 1)
    gap = max(3, round(.014 * height)) | 1
    linked = cv2.morphologyEx(row, cv2.MORPH_CLOSE, np.ones((gap, 1), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0).ravel()
    columns = []
    for start, stop in spans(linked):
        if not (.10 * height <= stop - start <= .48 * height and .055 * height < start < stop < .945 * height):
            continue
        margin = max(2, round(.028 * height))
        tip = max(2, round(.009 * height))
        nearby = [dark[max(0, start-margin):start], dark[stop:min(height, stop+margin)]]
        if not any(np.sum(side) >= tip for side in nearby):
            continue
        sample = center[start:stop]
        if not sample.any() or np.median(saturation[start:stop, left:right][sample]) < 64:
            continue
        phase = hue[start:stop, left:right][sample] * (2. * np.pi / 180.)
        if len(phase) < max(35, .05 * center.size) or abs(np.exp(1j * phase).mean()) < .90:
            continue
        family = np.angle(np.exp(1j * phase).mean()) % (2. * np.pi)
        relative = np.abs(np.angle(np.exp(1j * (hue * (2. * np.pi / 180.) - family)))) < .34
        supported = color & relative
        outer = np.c_[supported[start:stop, :max(1, round(.12 * width))],
                      supported[start:stop, round(.88 * width):]]
        if outer.mean() > .27:
            continue
        mask = np.zeros(color.shape, dtype=bool)
        mask[start:stop, left:right] = supported[start:stop, left:right]
        columns.append(mask)
    if len(columns) != 1:
        return unknown("A single colored column bounded by a transverse dark seal was not established.")
    rgb, support = (rgb.transpose(1, 0, 2), columns[0].T) if horizontal else (rgb, columns[0])
    appearance = describe_region(rgb, support, profile)
    observed = appearance["apparent_color"]
    observed["family"] = visible_hue(observed["srgb"], observed["lab"][0])
    return dict(presence="present", appearance=appearance, fill_ml=None, opacity=None,
                reason="A coherent colored interior column adjoins a transverse dark band.",
                opacity_reason=OPACITY_REASON)
