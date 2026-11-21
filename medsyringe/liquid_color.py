"""Profile-aware apparent color of a supported liquid region."""
import io

import cv2
import numpy as np
from PIL import ImageCms

from .liquid_appearance import OPACITY_REASON


def display_rgb(image):
    profile = image.info.get("icc_profile")
    if profile:
        try:
            source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
            return ImageCms.profileToProfile(image.convert("RGB"), source, ImageCms.createProfile("sRGB"),
                                             outputMode="RGB"), "Embedded profile converted to sRGB"
        except ImageCms.PyCMSError:
            return image.convert("RGB"), "Invalid embedded profile, sRGB assumed"
    return image.convert("RGB"), "sRGB assumed"


def lab_pixels(rgb):
    return cv2.cvtColor(rgb.astype(np.float32)/255., cv2.COLOR_RGB2Lab)


def describe_region(rgb, support, profile):
    if not support.any():
        return dict(status="insufficient evidence", lightness=None, apparent_color=None,
                    opacity=None, opacity_reason=OPACITY_REASON)
    values = lab_pixels(rgb)[support]
    lower, center, upper = np.quantile(values, [.25, .50, .75], axis=0)
    lightness, a, b = center
    category = "dark" if lightness < 35 else "light" if lightness >= 65 else "medium"
    y, x = support.nonzero()
    height, width = support.shape
    rgb_median = [int(v) for v in np.median(rgb[support], axis=0)]
    return dict(status="appearance measured", reason="Measured within the supported barrel interior",
                lightness=dict(lstar=round(float(lightness), 1), iqr=[round(float(lower[0]), 1),
                               round(float(upper[0]), 1)], category=category),
                apparent_color=dict(lab=[round(float(v), 1) for v in center],
                                    chroma=round(float(np.hypot(a, b)), 1),
                                    srgb=rgb_median, profile=profile),
                region=[float(x.min()/width), float(y.min()/height), float((x.max()+1)/width),
                        float((y.max()+1)/height)], opacity=None, opacity_reason=OPACITY_REASON)
