from dataclasses import asdict, replace

import numpy as np
import pytest

from scripts import public_liquid_augmentation as aug


@pytest.fixture(scope="module")
def scene():
    return aug.base_scene()


@pytest.fixture(scope="module")
def parent(scene):
    return aug.render(scene, aug.REFERENCE)


@pytest.mark.parametrize("case", aug.CASES[1:], ids=lambda case: case.name)
def test_controls_change_one_setting_and_preserve_protected_pixels(scene, parent, case):
    markings, rail = map(aug.mask_of, scene[1:])
    settings = asdict(case.settings)
    changed_settings = [name for name, value in asdict(aug.REFERENCE).items() if settings[name] != value]
    assert len(changed_settings) == 1

    frame = aug.render(scene, case.settings)
    audit = aug.compare(parent, frame, markings, rail)
    assert audit["changed_rgb_pixels"] > 0
    assert audit["changed_outside_allowed_pixels"] == 0
    assert markings.any() and rail.any()
    np.testing.assert_array_equal(frame.rgb[markings | rail], parent.rgb[markings | rail])
    if changed_settings != ["fill_tick"]:
        np.testing.assert_array_equal(frame.plunger, parent.plunger)
        np.testing.assert_array_equal(frame.visible_liquid, parent.visible_liquid)
        np.testing.assert_array_equal(frame.rgb[~parent.visible_liquid], parent.rgb[~parent.visible_liquid])


@pytest.mark.parametrize("tick", range(6))
def test_fill_reaches_drawn_piston_face_without_crossing_it(scene, tick):
    frame = aug.render(scene, replace(aug.REFERENCE, fill_tick=tick))
    piston_x = 142 + 74 * tick
    assert frame.piston_x == piston_x
    # The scale and seal share the same column in the rendered raster.
    assert tuple(frame.rgb[103, piston_x]) == (36, 58, 68)
    assert tuple(frame.rgb[136, piston_x]) == (35, 57, 65)
    assert frame.plunger[:, 559:].any()  # The thumb pad stays beyond the rear flange.
    assert np.where(frame.plunger)[1].max() == piston_x + aug.THUMB_OFFSET + 10
    assert not (frame.fill & frame.plunger).any()
    assert frame.fill[:, piston_x:].sum() == 0
    assert frame.fill.sum() == (149 - 85) * (piston_x - 142)
    assert frame.visible_liquid.sum() == 3034 * tick
    if tick:
        assert frame.visible_liquid[136, piston_x - 1]
    else:
        assert aug.probe(frame) is None


def test_visual_probe_measures_tint_saturation_and_background_contrast(scene):
    probes = {case.name: aug.probe(aug.render(scene, case.settings)) for case in aug.CASES}
    clear_contrast = aug.probe(aug.render(scene, aug.CLEAR_REFERENCE))[2]
    parent_hue, parent_saturation, parent_contrast = probes["parent"]
    assert parent_hue == pytest.approx(37, abs=2)
    assert probes["blue"][0] == pytest.approx(203, abs=2)
    assert probes["muted"][1] < parent_saturation / 2
    for name in ("blue", "muted", "tick_1", "tick_5"):
        assert probes[name][2] == pytest.approx(parent_contrast, abs=1)

    assert clear_contrast == pytest.approx(44, abs=0.01)
    assert probes["higher_transmission"][2] / clear_contrast == pytest.approx(0.87, abs=0.02)
    assert probes["lower_transmission"][2] / clear_contrast == pytest.approx(0.29, abs=0.02)
    assert probes["diffused"][2] < probes["lower_transmission"][2]
    assert probes["diffused"][1] == pytest.approx(parent_saturation, abs=0.02)


@pytest.mark.parametrize("region", ("markings", "rail", "background"))
def test_audit_rejects_single_pixel_leak(scene, parent, region):
    markings, rail = map(aug.mask_of, scene[1:])
    masks = {"markings": markings, "rail": rail, "background": ~(markings | rail | parent.visible_liquid)}
    y, x = np.argwhere(masks[region])[0]
    edited = aug.render(scene, aug.REFERENCE)
    edited.rgb[y, x, 0] ^= 1
    with pytest.raises(RuntimeError, match="escaped the authorized pixel region"):
        aug.compare(parent, edited, markings, rail)


@pytest.mark.parametrize("settings", [
    {"fill_tick": -1}, {"fill_tick": 6}, {"fill_tick": 2.5}, {"fill_tick": True},
    {"hue_deg": 360}, {"hue_deg": float("nan")}, {"saturation": -0.1},
    {"texture_transmission": 1.1}, {"texture_diffusion": float("inf")},
])
def test_rejects_unsupported_controls(settings):
    with pytest.raises(ValueError):
        aug.Settings(**settings)


def test_physical_labels_abstain_even_with_known_geometry(scene, parent):
    markings, rail = map(aug.mask_of, scene[1:])
    clear_contrast = aug.probe(aug.render(scene, aug.CLEAR_REFERENCE))[2]
    for case in aug.CASES:
        row = aug.describe(case, aug.render(scene, case.settings), parent, markings, rail, clear_contrast, "test-group")
        assert set(row["physical_fields"]) == {"fill_ml", "nominal_capacity_ml", "liquid_identity", "concentration",
                                               "optical_transmittance", "turbidity_ntu"}
        assert all(field == {"status": "unknown", "value": None} for field in row["physical_fields"].values())
        if case.settings.fill_tick == 0:
            assert row["visual_probe"]["status"] == "unknown"
            assert "stripe_contrast_ratio_to_clear_reference" not in row["visual_probe"]
