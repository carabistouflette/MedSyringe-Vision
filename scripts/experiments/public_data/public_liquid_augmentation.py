"""Throwaway local liquid-augmentation prototype using one original rendered parent."""

import argparse
import colorsys
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont


OUTPUT = Path(__file__).resolve().parents[3] / "data" / "public-liquid-augmentation"
WIDTH, HEIGHT = 1024, 228
ZERO_X, TICK_SPACING, MAX_TICK = 142, 74, 5
BORE_TOP, BORE_BOTTOM = 85, 149
THUMB_OFFSET = 432
PROBE = (160, 119, 200, 139)  # Four alternating 10 px bars, below the scale band.
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])


@dataclass(frozen=True)
class Settings:
    hue_deg: float = 37
    saturation: float = 0.72
    texture_transmission: float = 0.66
    texture_diffusion: float = 0
    fill_tick: int = 3

    def __post_init__(self):
        if type(self.fill_tick) is not int or not 0 <= self.fill_tick <= MAX_TICK:
            raise ValueError("fill_tick must be an integer on the drawn scale (0–5)")
        if not 0 <= self.hue_deg < 360:
            raise ValueError("hue_deg must be in [0, 360)")
        for name in ("saturation", "texture_transmission", "texture_diffusion"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


REFERENCE = Settings()


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    caption: str
    settings: Settings


CASES = (
    Case("parent", "Reference", "Amber tint, piston seal at tick 3", REFERENCE),
    Case("blue", "Hue", "Blue tint with the same blend settings", replace(REFERENCE, hue_deg=203)),
    Case("muted", "Saturation", "Lower saturation of the amber tint", replace(REFERENCE, saturation=0.22)),
    Case("higher_transmission", "Higher visual transparency", "More of the background pattern shows through",
         replace(REFERENCE, texture_transmission=0.87)),
    Case("lower_transmission", "Lower visual transparency", "Less of the background pattern shows through",
         replace(REFERENCE, texture_transmission=0.29)),
    Case("diffused", "Cloudy appearance", "The same blend with diffused background detail",
         replace(REFERENCE, texture_diffusion=0.85)),
    Case("tick_0", "Empty control", "Piston seal at tick 0", replace(REFERENCE, fill_tick=0)),
    Case("tick_1", "Lower fill", "Piston seal at tick 1", replace(REFERENCE, fill_tick=1)),
    Case("tick_5", "Higher fill", "Piston seal at tick 5", replace(REFERENCE, fill_tick=5)),
)
CLEAR_REFERENCE = replace(REFERENCE, texture_transmission=1)


@dataclass
class Frame:
    rgb: np.ndarray
    fill: np.ndarray
    visible_liquid: np.ndarray
    plunger: np.ndarray
    piston_x: int


def base_scene():
    """Draw the whole scene and fixed protective overlays from primitives."""
    y, x = np.indices((HEIGHT, WIDTH))
    paper = 222 + 12 * (1 - y / HEIGHT) + 4 * np.cos(x / 135)
    rgb = np.rint(paper[..., None] + np.array([-5, 0, 3])).astype(np.uint8)
    base = Image.fromarray(rgb)
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle((37, 54, 987, 174), 14, fill="#c7d3d8", outline="#b8c7cc")
    draw.rounded_rectangle((113, 70, 552, 163), 9, fill="#dae5e8", outline="#7e9aa4", width=2)
    draw.rounded_rectangle((80, 105, 127, 133), 5, fill="#d8e4e7", outline="#7e9aa4", width=2)
    draw.rounded_rectangle((59, 111, 86, 127), 3, fill="#e3ecee", outline="#7e9aa4", width=2)
    draw.rounded_rectangle((545, 57, 558, 176), 4, fill="#d3e0e4", outline="#7e9aa4", width=2)

    # A neutral stripe target is visible through the bore. A calibrated crop samples it.
    tone = 210 + 22 * np.where((x // 10) % 2 == 0, 1, -1) + 3 * np.cos((y - 117) / 9)
    texture = np.rint(tone[..., None] + np.array([-1, 2, 4])).astype(np.uint8)
    bore = (120, BORE_TOP, 541, BORE_BOTTOM)
    base.paste(Image.fromarray(texture).crop(bore), bore)

    # The printed scale sits on an opaque band so exact preservation includes
    # antialiased text edges and never restores the parent's liquid behind the ink.
    markings = Image.new("RGBA", base.size)
    ink = ImageDraw.Draw(markings)
    ink.rounded_rectangle((121, 74, 541, 107), 4, fill="#ebf1f2")
    font = ImageFont.load_default(size=13)
    ink.text((134, 82), "REF", anchor="rm", font=ImageFont.load_default(size=9), fill="#435e68")
    for tick in range(MAX_TICK + 1):
        axis_x = ZERO_X + tick * TICK_SPACING
        ink.text((axis_x, 86), str(tick), font=font, anchor="mm", fill="#243a44")
        ink.line((axis_x, 95, axis_x, 107), fill="#243a44", width=2)
        if tick < MAX_TICK:
            middle_x = axis_x + TICK_SPACING // 2
            ink.line((middle_x, 101, middle_x, 107), fill="#657e87")

    protected_rail = Image.new("RGBA", base.size)
    rail = ImageDraw.Draw(protected_rail)
    rail.rounded_rectangle((123, 151, 540, 157), 3, fill="#edf5f6")
    rail.line((131, 155, 532, 155), fill="#cbdde2")
    return base, markings, protected_rail


def render(scene, settings):
    base, markings, protected_rail = scene
    piston_x = ZERO_X + TICK_SPACING * settings.fill_tick
    fill = np.zeros((HEIGHT, WIDTH), dtype=bool)
    fill[BORE_TOP:BORE_BOTTOM, ZERO_X:piston_x] = True
    if fill.any():
        texture = base
        if settings.texture_diffusion:
            texture = texture.filter(ImageFilter.GaussianBlur(6 * settings.texture_diffusion))
        tint = 255 * np.array(colorsys.hsv_to_rgb(settings.hue_deg / 360, settings.saturation, 0.93))
        transmission = settings.texture_transmission
        rgb = np.array(base)
        rgb[fill] = np.rint(transmission * np.asarray(texture)[fill] + (1 - transmission) * tint)
        liquid = Image.fromarray(rgb)
    else:
        liquid = base.copy()

    plunger = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(plunger)
    thumb_x = piston_x + THUMB_OFFSET
    draw.rounded_rectangle((piston_x + 10, 111, thumb_x + 5, 130), 4, fill="#a8bec6", outline="#6f8d97")
    draw.line((piston_x + 20, 115, thumb_x, 115), fill="#e4edef", width=2)
    draw.rounded_rectangle((thumb_x, 96, thumb_x + 10, 145), 3, fill="#c4d4da", outline="#6f8d97")
    draw.rounded_rectangle((piston_x, BORE_TOP, piston_x + 19, BORE_BOTTOM - 1), 3, fill="#36525d")
    draw.line((piston_x, BORE_TOP + 3, piston_x, BORE_BOTTOM - 4), fill="#233941", width=2)
    draw.line((piston_x + 14, BORE_TOP + 3, piston_x + 14, BORE_BOTTOM - 4), fill="#66808a", width=2)

    liquid.paste(plunger, mask=plunger.getchannel("A"))
    liquid.paste(markings, mask=markings.getchannel("A"))
    liquid.paste(protected_rail, mask=protected_rail.getchannel("A"))
    visible = fill & ~mask_of(markings) & ~mask_of(protected_rail)
    return Frame(np.array(liquid), fill, visible, mask_of(plunger), piston_x)


def mask_of(layer):
    return np.asarray(layer.getchannel("A")) > 0


def compare(parent, frame, markings, protected_rail):
    protected = markings | protected_rail
    changed = np.any(parent.rgb != frame.rgb, axis=2)
    allowed = parent.visible_liquid | frame.visible_liquid
    if parent.piston_x != frame.piston_x:
        allowed = allowed | parent.plunger | frame.plunger
    counts = {
        "changed_rgb_pixels": int(changed.sum()),
        "changed_marking_pixels": int((changed & markings).sum()),
        "changed_protected_rail_pixels": int((changed & protected_rail).sum()),
        "changed_outside_allowed_pixels": int((changed & ~allowed).sum()),
        "markings_mask_pixels": int(markings.sum()),
        "protected_rail_mask_pixels": int(protected_rail.sum()),
    }
    if (changed & protected).any() or counts["changed_outside_allowed_pixels"]:
        raise RuntimeError(f"Edits escaped the authorized pixel region: {counts}")
    return counts


def probe(frame):
    """Measure rendered color and stripe contrast where every probe pixel is liquid."""
    x0, y0, x1, y1 = PROBE
    region = np.s_[y0:y1, x0:x1]
    if not frame.visible_liquid[region].all():
        return None
    rgb = frame.rgb[region]
    hue, saturation, _ = colorsys.rgb_to_hsv(*(rgb.mean(axis=(0, 1)) / 255))
    luma = rgb @ LUMA_WEIGHTS
    light_stripe = (np.arange(x0, x1) // 10) % 2 == 0
    contrast = abs(luma[:, light_stripe].mean() - luma[:, ~light_stripe].mean())
    return hue * 360 if saturation > 0 else None, saturation, float(contrast)


def physical_fields():
    return {
        name: {"status": "unknown", "value": None}
        for name in ("fill_ml", "nominal_capacity_ml", "liquid_identity", "concentration",
                     "optical_transmittance", "turbidity_ntu")
    }


def describe(case, frame, parent, markings, rail, clear_contrast, parent_id):
    observed = probe(frame)
    appearance = {"status": "unknown", "reason": "No fully liquid-covered probe region."}
    if observed is not None:
        hue, saturation, contrast = observed
        appearance = {
            "status": "measured_on_render",
            "mean_rgb_hue_deg": None if hue is None else round(hue, 3),
            "mean_rgb_hsv_saturation": round(saturation, 4),
            "stripe_contrast_rgb_code_values": round(contrast, 3),
            "stripe_contrast_ratio_to_clear_reference": round(contrast / clear_contrast, 4),
        }
    return {
        "id": case.name,
        "parent_id": None if case.name == "parent" else "parent",
        "source_group": parent_id,
        "image": f"{case.name}.png",
        "visible_liquid_mask": f"liquid_tick_{case.settings.fill_tick}_mask.png",
        "plunger_mask": f"plunger_tick_{case.settings.fill_tick}_mask.png",
        "renderer_parameters": asdict(case.settings),
        "geometry": {
            "piston_liquid_face_x_px": frame.piston_x,
            "scale_tick_x_px": ZERO_X + case.settings.fill_tick * TICK_SPACING,
            "normalized_scale_position": case.settings.fill_tick / MAX_TICK,
            "geometric_fill_pixels": int(frame.fill.sum()),
            "visible_liquid_pixels": int(frame.visible_liquid.sum()),
        },
        "visual_probe": appearance,
        "physical_fields": physical_fields(),
        "pixel_audit_against_parent": compare(parent, frame, markings, rail),
        "rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
    }


def make_board(frames, rows):
    """Lay out a fixed English comparison sheet with measured values."""
    tile_width, tile_height = 464, 196
    padding, gap, top = 28, 16, 111
    board = Image.new("RGB", (2 * padding + 3 * tile_width + 2 * gap, 767), "#eaf0f1")
    draw = ImageDraw.Draw(board)
    title = ImageFont.load_default(size=33)
    heading = ImageFont.load_default(size=19)
    body = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=14)
    draw.text((padding, 23), "Liquid appearance and fill controls", font=title, fill="#163744")
    draw.text((padding, 70), "Original synthetic syringe  /  One setting changed per variant",
              font=body, fill="#435e68")
    for index, (case, frame, row) in enumerate(zip(CASES, frames, rows, strict=True)):
        x = padding + index % 3 * (tile_width + gap)
        y = top + index // 3 * (tile_height + gap)
        draw.rounded_rectangle((x, y, x + tile_width, y + tile_height), 10, fill="#ffffff")
        draw.text((x + 13, y + 9), case.title, font=heading, fill="#163744")
        thumbnail_width = tile_width - 26
        thumbnail_height = round(HEIGHT * thumbnail_width / WIDTH)
        thumbnail = Image.fromarray(frame.rgb).resize((thumbnail_width, thumbnail_height),
                                                     Image.Resampling.LANCZOS)
        board.paste(thumbnail, (x + 13, y + 38))
        draw.text((x + 13, y + 145), case.caption, font=body, fill="#243e48")
        observed = row["visual_probe"]
        evidence = f'Tick {case.settings.fill_tick}  /  {row["geometry"]["visible_liquid_pixels"]:,} visible liquid px'
        if observed["status"] == "measured_on_render":
            evidence += f'  /  {observed["stripe_contrast_ratio_to_clear_reference"]:.2f} of clear contrast'
        draw.text((x + 13, y + 171), evidence, font=small, fill="#435e68")
    draw.text((padding, 750), "Ticks are scale positions. Physical volume, transmittance and turbidity are unknown.",
              anchor="ls", font=small, fill="#435e68")
    return board


def run():
    scene = base_scene()
    markings, rail = map(mask_of, scene[1:])
    frames = [render(scene, case.settings) for case in CASES]
    parent = frames[0]
    clear = render(scene, CLEAR_REFERENCE)
    _, _, clear_contrast = probe(clear)
    parent_id = "original-rendered-syringe-v1"
    rows = [describe(case, frame, parent, markings, rail, clear_contrast, parent_id)
            for case, frame in zip(CASES, frames, strict=True)]
    board = make_board(frames, rows)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    saved_ticks = set()
    for case, row, frame in zip(CASES, rows, frames, strict=True):
        Image.fromarray(frame.rgb).save(OUTPUT / row["image"])
        tick = case.settings.fill_tick
        if tick not in saved_ticks:
            Image.fromarray(frame.visible_liquid).save(OUTPUT / row["visible_liquid_mask"])
            Image.fromarray(frame.plunger).save(OUTPUT / row["plunger_mask"])
            saved_ticks.add(tick)
    Image.fromarray(markings).save(OUTPUT / "protected_markings_mask.png")
    Image.fromarray(rail).save(OUTPUT / "protected_rail_mask.png")
    Image.fromarray(clear.rgb).save(OUTPUT / "clear_reference.png")
    board.save(OUTPUT / "contact_sheet_en.png")

    manifest = {
        "provenance": {
            "kind": "original_procedural_synthetic",
            "source_group": parent_id,
            "parent_image": "parent.png",
            "generator": "scripts/experiments/public_data/public_liquid_augmentation.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "external_input_images": [],
            "mask_source": "Procedural layer coverage in the original raster coordinates.",
            "intended_use": "Synthetic controls. Keep all siblings in one group for any dataset split.",
        },
        "environment": {"python": platform.python_version(), "pillow": PIL.__version__, "numpy": np.__version__},
        "protected_masks": ["protected_markings_mask.png", "protected_rail_mask.png"],
        "probe_xyxy_exclusive": PROBE,
        "clear_reference": {
            "image": "clear_reference.png",
            "renderer_parameters": asdict(CLEAR_REFERENCE),
            "stripe_contrast_rgb_code_values": round(clear_contrast, 3),
            "pixel_audit_against_parent": compare(parent, clear, markings, rail),
            "rgb_sha256": hashlib.sha256(clear.rgb.tobytes()).hexdigest(),
        },
        "contact_sheet": "contact_sheet_en.png",
        "cases": rows,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(rows)} controls and a clear reference in {OUTPUT}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
