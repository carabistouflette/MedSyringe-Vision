"""Normalize exposure using an unchanged external patch in a registered pair."""
import numpy as np

from .data import valid_box
from .liquid_appearance import luminance, relative_opacity


def registered_attenuation(observed, reference, barrel, control, *, matched_setup=False):
    if not matched_setup:
        return dict(value=None, reason="A registered pair and unchanged exterior patch are required.")
    if (observed.shape != reference.shape or observed.ndim != 3 or observed.shape[2] != 3
            or not valid_box(barrel) or not valid_box(control)):
        return dict(value=None, reason="Use matching RGB images with two valid measurement regions.")
    if (max(barrel[0], control[0]) < min(barrel[2], control[2])
            and max(barrel[1], control[1]) < min(barrel[3], control[3])):
        return dict(value=None, reason="The exterior calibration and liquid regions must be disjoint.")
    height, width = observed.shape[:2]

    def area(image, bounds):
        l, t, r, b = (round(v * n) for v, n in zip(bounds, (width, height) * 2, strict=True))
        return image[t:b, l:r]

    x = luminance(area(reference, control)).ravel()
    y = luminance(area(observed, control)).ravel()
    usable = (x > .025) & (x < .97) & (y > .025) & (y < .97)
    x, y = x[usable], y[usable]
    if len(x) < 100 or np.ptp(x) < .18:
        return dict(value=None, reason="The exterior patch lacks usable texture or exposure range.")
    gain, offset = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
    residual = y - (gain * x + offset)
    quality = 1 - np.sum(residual**2) / max(np.sum((y - y.mean())**2), 1e-12)
    if not (.50 <= gain <= 1.75 and abs(offset) <= .18 and quality >= .94):
        return dict(value=None, reason="The exterior patch does not support a shared exposure transform.")
    sample = (luminance(area(observed, barrel)) - offset) / gain
    if sample.size < 100 or np.mean((sample < 0) | (sample > 1)) > .025:
        return dict(value=None, reason="Exposure correction would clip the liquid measurement region.")
    sample = sample.clip(0., 1.)
    srgb = np.round(255. * np.where(sample <= .0031308, 12.92 * sample,
                                     1.055 * np.power(sample, 1 / 2.4) - .055)).astype(np.uint8)
    corrected = np.repeat(srgb[..., None], 3, axis=2)
    result = relative_opacity(corrected, area(reference, barrel), matched_setup=True)
    return {**result, "exterior_gain": round(float(gain), 4), "exterior_offset": round(float(offset), 4),
            "exterior_fit_r2": round(float(quality), 4)}
