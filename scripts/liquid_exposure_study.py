"""Compare paired relative attenuation with and without exterior exposure correction."""
from pathlib import Path
from statistics import mean

import numpy as np

from medsyringe.data import digest, write_json
from medsyringe.liquid_appearance import relative_opacity
from medsyringe.relative_optics import registered_attenuation
from scripts.small_field_study import srgb

OUTPUT = Path("artifacts/detection_hillclimb/liquid_exposure_study.json")
BARREL = [0., 0., .5, 1.]
EXTERIOR = [.5, 0., 1., 1.]


def run():
    y, x = np.indices((112, 240))
    texture = .45 + .19 * np.sin(.25 * x + .06 * y) + .13 * np.cos(.27 * y)
    reference = np.repeat(srgb(texture)[..., None], 3, axis=2)
    rows = []
    for gain, offset in ((1., 0.), (.75, .04), (1.20, -.04)):
        for expected in (0., .30, .60, .82):
            scene = texture.copy()
            scene[:, :120] = (1 - expected) * scene[:, :120] + expected * .46
            observed = np.repeat(srgb(gain * scene + offset)[..., None], 3, axis=2)
            rows.append(dict(expected=expected, gain=gain, offset=offset,
                             baseline=relative_opacity(observed[:, :120], reference[:, :120], matched_setup=True),
                             normalized=registered_attenuation(observed, reference, BARREL, EXTERIOR,
                                                               matched_setup=True)))
    altered = reference.copy()
    altered[:, 120:] = np.roll(altered[:, 120:], 13, axis=0)
    mismatched = registered_attenuation(altered, reference, BARREL, EXTERIOR, matched_setup=True)
    exposure = [r for r in rows if r["gain"] != 1.]
    score = {}
    for name in ("baseline", "normalized"):
        errors = [abs(r[name]["value"] - r["expected"]) for r in exposure if r[name]["value"] is not None]
        score[name] = dict(emitted=len(errors), eligible=len(exposure), mae=mean(errors) if errors else None,
                           maximum_error=max(errors) if errors else None)
    write_json(OUTPUT, dict(cases=rows, varying_exposure=score, mismatched_exterior=mismatched,
                            code=digest("medsyringe/relative_optics.py"), scope="Rendered matched texture only"))
    print(score, mismatched)


if __name__ == "__main__":
    run()
