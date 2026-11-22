"""Measure small-capacity OCR and apparent fill on public and rendered controls."""
import argparse
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from medsyringe.measurement.attributes import crop
from medsyringe.measurement.capacity_ocr import SceneOCR, resolve
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.measurement.liquid_appearance import relative_opacity
from medsyringe.measurement.small_capacity import SmallCapacity
from medsyringe.measurement.small_fill import inspect_barrel

ART = Path("artifacts/detection_hillclimb")
DATA = Path("data/small-field-study")
REFERENCE = ART / "small_field_study_reference.json"
ROOT = Path("data/public-web-collection/images")
ANNOTATED = Path("data/public-annotated-collection/images")
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf")


def public_references():
    cases = [
        ("development-5", ROOT / "commons-106925058.jpg", [.125, .373, .693, .626], 5, "empty", None),
        ("public-yellow", ANNOTATED / "oi-9b1a594e0cea3393.jpg", [.376, .067, .607, .89], None, "present", "yellow"),
        ("public-blue-plunger", ROOT / "commons-102070543.jpg", [.401, .153, .552, .804], None, "empty", None),
        ("public-clear-barrel", ANNOTATED / "commons-971336.jpg", [.27, .13, .99, .86], None, "present", "neutral"),
        ("public-1-marked", ROOT / "openverse-edfa0cf9-21ec-4c8f-8908-2dc5bade3d72.jpg",
         [.35, .18, .91, .93], 1, None, None),
        ("public-overlapping-small", ROOT / "commons-34870812.jpg", [.32, .29, .81, .70], None, None, None),
    ]
    return [dict(id=key, image=str(path), sha256=digest(path), bbox=box, capacity_ml=value,
                 presence=presence, apparent_color=family, volume_ml=0. if presence == "empty" else None,
                 cohort="development" if key.startswith("development") else "public challenge")
            for key, path, box, value, presence, family in cases]


def render(level, seal_color, liquid, *, unreadable=False, blur=0, brightness=1.):
    height, width = 825, 285
    y, x = np.indices((height, width))
    paper = (229 + 4 * np.sin(x / 24) + 3 * np.cos(y / 59)).astype(np.uint8)
    pixels = np.repeat(paper[:, :, None], 3, axis=2)
    zero, pitch, left, right = 135, 94, 68, 240
    barrel_end = 743
    radius = 7 + np.sin((x - left) / (right - left) * np.pi) * 18
    barrel = (x >= left) & (x <= right) & (y >= zero - 25) & (y <= barrel_end)
    pixels[barrel] = np.clip(pixels[barrel].astype(float)
                             - radius[barrel, None] + (x[barrel, None] % 6 - 3) * .60, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(pixels)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((left, zero - 25, right, barrel_end), radius=13, outline=(163, 173, 177), width=3)
    draw.rounded_rectangle((139, 37, 174, zero - 24), radius=8, outline=(170, 184, 189), width=4)
    draw.rectangle((57, barrel_end-8, 253, barrel_end+8), fill=(180, 189, 191))
    draw.line([(133, barrel_end+6), (133, height-26)], fill=(170, 179, 181), width=5)
    draw.line([(167, barrel_end+6), (167, height-26)], fill=(170, 179, 181), width=5)
    draw.rectangle((102, height-26, 197, height-17), fill=(180, 189, 191))
    base = np.array(canvas)
    pieces = base.copy()
    edge = round(zero + pitch * level)
    mask = (x >= left + 6) & (x < right - 6) & (y >= zero) & (y < edge)
    if liquid:
        tint, blend = liquid
        pieces[mask] = np.round((1-blend)*base[mask] + blend*np.array(tint)).astype(np.uint8)
    seal = (x >= left + 4) & (x < right - 4) & (y >= edge) & (y < edge + 26)
    pieces[seal] = np.asarray(seal_color)
    stroke = Image.new("L", canvas.size)
    mark = ImageDraw.Draw(stroke)
    numbers = ImageFont.truetype(str(FONT), 28)
    units = ImageFont.truetype(str(FONT), 21)
    for index in range(26):
        center = round(zero + pitch * index / 5)
        finish = 225 if index % 5 == 0 else 186
        mark.line(((150, center), (finish, center)), fill=255, width=4 if index % 5 == 0 else 2)
        if index and index % 5 == 0 and not (unreadable and index >= 20):
            mark.text((230, center), str(index // 5), fill=255, font=numbers, anchor="lm")
    if not unreadable:
        mark.text((225, zero + 5*pitch + 26), "ml", font=units, fill=255, anchor="lm")
    ink = np.asarray(stroke) > 0
    pieces[ink] = (37, 40, 40)
    changed_unprotected = np.count_nonzero(np.any((pieces != base), axis=2) & ~(ink | mask | seal))
    if changed_unprotected:
        raise ValueError("The simulated presentation modified an unauthorized region.")
    pieces = np.clip(pieces.astype(float) * brightness, 0, 255).astype(np.uint8)
    if blur:
        pieces = cv2.GaussianBlur(pieces, (0, 0), blur)
    return pieces, [(150, zero + pitch * i, 225) for i in range(6)], pitch


def control_cases():
    blue = ((25, 181, 230), .43)
    amber = ((255, 166, 38), .36)
    red = ((236, 53, 64), .42)
    neutral = ((213, 219, 219), .24)
    green_seal = (32, 105, 86)
    dark_seal = (49, 49, 51)
    return [
        ("seated-green", 0., green_seal, None, "empty", None, {}, 0),
        ("seated-black-upside-down", 0., dark_seal, None, "empty", None, {}, 180),
        ("blue-quarter-turn", 2.7, dark_seal, blue, "present", "cyan", {}, 90),
        ("amber-light", 2.2, green_seal, amber, "present", "yellow", {}, 0),
        ("red-slight-blur", 3.4, dark_seal, red, "present", "red", {"blur": 1.2, "brightness": .93}, 270),
        ("empty-gap-is-not-proof-of-liquid", 2.2, green_seal, None, "empty", None, {}, 0),
        ("transparent-air-lookalike", 2.2, green_seal, neutral, "present", "neutral", {}, 0),
        ("occluded-graduations", 2.2, green_seal, amber, "present", "yellow", {"unreadable": True}, 0),
    ]


def srgb(values):
    return np.round(255. * np.where(values <= .0031308, 12.92*values,
                                     1.055*np.power(values, 1/2.4) - .055)).astype(np.uint8)


def optical_cases():
    y, x = np.indices((96, 148))
    texture = .47 + .20*np.sin(.19*x+.07*y) + .13*np.cos(.26*y)
    clear = np.repeat(srgb(texture)[..., None], 3, axis=2)
    observations = []
    for alpha in (.10, .35, .65, .85):
        transformed = (1-alpha) * texture + alpha * .48
        affected = np.repeat(srgb(transformed)[..., None], 3, axis=2)
        observations.append(dict(expected=alpha, observed=relative_opacity(affected, clear, matched_setup=True)))
    changed = np.roll(clear, 13, axis=1)
    observations.append(dict(expected=None, scenario="misregistered crop",
                             observed=relative_opacity(changed, clear, matched_setup=True)))
    observations.append(dict(expected=None, scenario="registration not established",
                             observed=relative_opacity(clear, clear, matched_setup=False)))
    return observations


def summarize(rows, truth, predicted):
    known = [r for r in rows if r[truth] is not None]
    covered = [r for r in known if r[predicted] is not None]
    return dict(known=len(known), emitted=len(covered), correct=sum(r[truth] == r[predicted] for r in covered),
                indeterminate=sum(r[truth] is None for r in rows),
                emissions_on_indeterminate=sum(r[truth] is None and r[predicted] is not None for r in rows))


def run(output_path):
    if read_json(REFERENCE)["rows"] != public_references():
        raise ValueError("The public photos or human reference changed.")
    DATA.mkdir(parents=True, exist_ok=True)
    candidate = SmallCapacity()
    baseline = SceneOCR()
    rows = []
    controls = []
    todo = []
    nominal = read_json(REFERENCE)["rows"]
    for r in nominal:
        base = crop(read_image(r["image"]), r["bbox"])
        for angle in (0, 90, 180, 270) if r["id"] == "development-5" else (0,):
            todo.append(({**r, "angle": angle}, base.rotate(angle, expand=True)))
    for key, level, seal, liquid, presence, family, options, rotation in control_cases():
        pixels, track, pitch = render(level, seal, liquid, **options)
        geom = inspect_barrel(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), track, pitch, 1, 5)
        frame = Image.fromarray(pixels).rotate(rotation, expand=True)
        frame.save(DATA / f"{key}.png")
        info = dict(id=key, angle=rotation, cohort="rendered controls",
                    capacity_ml=None if options.get("unreadable") else 5, presence=presence,
                    apparent_color=family, volume_ml=0. if presence == "empty" else level)
        controls.append({**info, "known_geometry": geom})
        todo.append((info, frame))
    for r, crop_image in todo:
        result = candidate.read(crop_image)
        raw_baseline = resolve(baseline.read(crop_image))
        found = result["capacity"]
        volume = found["barrel"] if found else {}
        color = (volume.get("appearance") or {}).get("apparent_color", {})
        predicted = {"capacity": found["value"] if found else None,
                     "baseline_capacity": (raw_baseline.get("capacity") or {}).get("value"),
                     "predicted_presence": volume.get("presence") if volume.get("presence") in ("present", "empty") else None,
                     "predicted_color": color.get("family"),
                     "estimated_ml": volume.get("fill_ml"), "volume_interval_ml": volume.get("interval_ml")}
        rows.append({**r, **predicted, "candidate": result, "baseline": raw_baseline})
        write_json(DATA / "progress.json", rows)
        print(r["id"], r["angle"], predicted, flush=True)
    output = {}
    for cohort in dict.fromkeys(r["cohort"] for r in rows):
        scope = [r for r in rows if r["cohort"] == cohort]
        output[cohort] = dict(
            baseline_capacity=summarize(scope, "capacity_ml", "baseline_capacity"),
            capacity=summarize(scope, "capacity_ml", "capacity"),
            presence=summarize(scope, "presence", "predicted_presence"),
            color=summarize(scope, "apparent_color", "predicted_color"),
            known_volumes=sum(r["volume_ml"] is not None for r in scope),
            emitted_intervals=sum(r["volume_interval_ml"] is not None for r in scope),
            spurious_intervals=sum(r["volume_interval_ml"] is not None and r["volume_ml"] is None for r in scope),
            true_volumes_in_intervals=sum(r["volume_ml"] is not None and r["volume_interval_ml"] is not None
                                          and r["volume_interval_ml"][0] <= r["volume_ml"] <= r["volume_interval_ml"][1]
                                          for r in scope),
        )
    optics = optical_cases()
    errors = [abs(r["observed"]["value"] - r["expected"]) for r in optics
              if r["expected"] is not None and r["observed"]["value"] is not None]
    write_json(output_path, dict(
        reference_sha256=digest(REFERENCE), evaluation=output, cases=rows,
        implementation={name: digest(Path("medsyringe") / name) for name in ("small_capacity.py", "small_fill.py")},
        known_geometry_controls=controls, relative_opacity_controls=optics,
        relative_opacity_error_mae=mean(errors) if errors else None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path, default=ART / "small_field_study_results.json")
    options = parser.parse_args()
    if options.job == "freeze":
        if REFERENCE.exists():
            raise FileExistsError(REFERENCE)
        write_json(REFERENCE, dict(rows=public_references(), scope="Public evidence and rendered challenge controls",
                                   unresolved="No physical measurement or registered public real opacity pair."))
    else:
        run(options.output)
