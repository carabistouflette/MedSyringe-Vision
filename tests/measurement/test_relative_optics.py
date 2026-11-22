import numpy as np

from medsyringe.measurement.relative_optics import registered_attenuation
from scripts.experiments.measurement.liquid_exposure_study import BARREL, EXTERIOR, srgb


def test_external_patch_removes_exposure_bias_and_rejects_bad_alignment():
    y, x = np.indices((100, 200))
    texture = .40 + .17 * np.sin(.21*x + .09*y) + .14 * np.cos(.23*y)
    scene = texture.copy()
    scene[:, :100] = .66 * scene[:, :100] + .34 * .43
    observed = np.repeat(srgb(.75*scene + .05)[..., None], 3, axis=2)
    reference = np.repeat(srgb(texture)[..., None], 3, axis=2)
    result = registered_attenuation(observed, reference, BARREL, EXTERIOR, matched_setup=True)
    assert abs(result["value"] - .34) <= .003
    assert result["exterior_fit_r2"] >= .99
    changed = observed.copy()
    changed[:, 100:] = np.roll(observed[:, 100:], 12, axis=0)
    assert registered_attenuation(changed, reference, BARREL, EXTERIOR, matched_setup=True)["value"] is None
    assert registered_attenuation(observed, reference, BARREL, EXTERIOR)["value"] is None
