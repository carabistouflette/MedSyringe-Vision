"""Bind manually reviewed syringe landmarks and source-disjoint challenge photos."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from medsyringe.measurement.attributes import crop
from medsyringe.data import digest, leakage, read_image, read_json, valid_box, write_json
from medsyringe.measurement.part_geometry import PARTS
from scripts.experiments.public_data.detection_public_benchmark import lexical_key, url_key
from scripts.experiments.public_data.instance_source_audit import fingerprint_collision, protection
from scripts.experiments.crowds.low_count_prospective import REFERENCE as OLD_TRAY
from scripts.experiments.crowds.low_count_recovery import DATASET, crowd, source_rows
from scripts.experiments.public_data.public_training_intake import raster_hashes

OUT = crowd.ART / "part_verifier_references.json"
WORK = Path("data/part-verifier")
EXTRA = crowd.ART / "public_additional_fit.json"

# Human-observed centers on the three archived object-crop contact sheets.
# Missing landmarks are intentionally unobserved, including fully occluded parts.
LANDMARKS = {
    1: ((263, 104), None, (137, 210), (228, 137)),
    2: ((698, 87), None, (585, 214), (656, 136)),
    3: ((1122, 96), None, (1056, 217), (1088, 149)),
    5: ((646, 390), (530, 353), (483, 336), (759, 422)),
    6: ((1042, 351), (1167, 357), (1230, 352), (1011, 352)),
    7: ((296, 692), None, None, (245, 695)),
    9: ((1052, 675), None, None, (1091, 685)),
    10: ((198, 94), None, None, (195, 116)),
    12: ((1038, 65), None, None, (1100, 78)),
    14: ((666, 353), (538, 366), (490, 381), (759, 347)),
    17: ((668, 736), None, None, None),
    18: ((1071, 725), (1141, 782), (1170, 811), None),
    19: ((185, 78), (298, 76), (373, 87), None),
    20: ((660, 132), (663, 219), (663, 242), None),
    21: ((1091, 155), None, None, (1070, 204)),
    23: ((639, 355), (727, 356), (784, 355), None),
    24: ((1140, 357), (1199, 360), (1245, 348), None),
}

FIT_IDS = {"150790461", "155856137", "17283750", "115697835", "59284146", "97257014", "18864470",
           "127790110", "17714440", "116616738", "1576673"}


def render_fit_sheets(destination):
    rows = [row for row in source_rows("fit") if row["objects"]]
    destination.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), 9):
        sheet = Image.new("RGB", (1320, 900), "#dce1e6")
        drawing = ImageDraw.Draw(sheet)
        layout = []
        for i, row in enumerate(rows[start:start+9]):
            photo = crop(read_image(row["image"]), row["objects"][0]["bbox"])
            tile = ImageOps.contain(photo, (425, 270))
            x = i % 3 * 440 + (440 - tile.width) // 2
            y = i // 3 * 300 + 5
            sheet.paste(tile, (x, y))
            drawing.text((i%3*440+9, i//3*300+278), f"{start+i+1}  {row['id']}", fill="black")
            layout.append(dict(id=row["id"], xywh=[x, y, *tile.size]))
        stem = destination / f"parts-{start//9}"
        sheet.save(stem.with_suffix(".jpg"), quality=95)
        write_json(stem.with_suffix(".json"), layout)


def fit_landmarks(rows):
    sheets = [a for i in range(3) for a in read_json(WORK / f"parts-{i}.json")]
    indexed = {r["id"]: r for r in rows}
    for i, coordinates in LANDMARKS.items():
        tile = sheets[i - 1]
        left, top, width, height = tile["xywh"]
        row = indexed[tile["id"]]
        l, t, r, b = row["objects"][0]["bbox"]
        parts = {}
        for name, point in zip(PARTS, coordinates, strict=True):
            if point is None:
                parts[name] = None
                continue
            x, y = point
            nx, ny = (x-left)/width, (y-top)/height
            if not (0 <= nx <= 1 and 0 <= ny <= 1):
                raise ValueError(f"Review point outside crop {i}: {name}")
            parts[name] = [round(l + nx*(r-l), 5), round(t + ny*(b-t), 5)]
        row["objects"][0]["parts"] = parts


def file_row(key, name, author, licence, boxes):
    path = WORK / f"{key}.jpg"
    w, h = read_image(path).size
    return dict(id=key, image=str(path), sha256=digest(path), author=author,
                group=f"commons:{author.casefold()}", license=licence,
                source_url=f"https://commons.wikimedia.org/wiki/File:{name}",
                objects=[dict(bbox=[l/w, t/h, r/w, b/h]) for l, t, r, b in boxes])


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    fit = source_rows("fit")
    fit_landmarks(fit)
    extra = [r for r in read_json(EXTRA)["rows"] if r["id"] in FIT_IDS]
    if len(extra) != len(FIT_IDS) or any("fit" not in r["split"] and "train" not in r["split"] for r in extra):
        raise ValueError("Missing or unqualified fitting-source photos.")
    fit.extend({**r, "image": str(Path("data")/r["image"])} for r in extra)
    fit.append(file_row("pasteur-fit", "Pasteur_pipettes.jpg", "Cjp24", "https://creativecommons.org/licenses/by-sa/3.0/", []))
    clip = file_row("clinic", "Syringes_rest_on_a_clean_table_in_a_clinic.jpg", "Shixart1985",
                    "https://creativecommons.org/licenses/by/2.0/",
                    [(135,117,286,621), (340,108,481,579), (510,108,633,543), (638,103,780,519)])
    # The foreground syringe is sharp. The three remaining bodies can be counted,
    # while their detailed landmarks remain unobservable at this resolution.
    clip["objects"][0]["parts"] = {"barrel": [.204, .56], "flange": [.214,.82], "rod": [.22,.885], "seal": [.20,.328]}
    transfer = [clip, file_row("pasteur-test", "Pasteurpipetten.jpg", "Karbohut",
                             "https://creativecommons.org/licenses/by-sa/4.0/", []),
                file_row("with-needle", "Syringe_with_needle_(2).JPG", "Intropin",
                         "https://creativecommons.org/licenses/by-sa/3.0/",
                         [(404,64,833,241), (336,131,765,331), (225,191,839,549), (119,292,733,690)])]
    fresh = []
    web = {r["id"]: r for r in read_json("data/public-web-collection/manifest.json")["rows"]}
    for key in ("commons-178551508", "openverse-81cd0139-fa6a-4cf1-839b-79e7e8af1c9c"):
        r = web[key]
        fresh.append({**r, "image": str(Path("data") / r["image"]), "objects": []})
    old = [*read_json(OLD_TRAY)["rows"], *source_rows("regression")]
    old = [{**r, "image": r["image"] if r["image"].startswith("data/") else str(Path("data")/r["image"])} for r in old]
    roles = {"fit": fit, "fresh": fresh, "transfer": transfer, "regression": old}
    history, private, past_hashes = protection()
    for row in fresh:
        if (row["sha256"] in private | history["sha256"] or lexical_key(row["author"]) in history["authors"]
              or url_key(row["source_url"]).rstrip("/") in history["urls"]
              or fingerprint_collision(raster_hashes(row["image"]), past_hashes | history["dhash"])):
            raise ValueError(f"Prior exposure in fresh set: {row['id']}")
    combined = [{**row, "split": role} for role, rows in roles.items() for row in rows]
    if leakage(combined):
        raise ValueError("A group, publisher, or raster appears in multiple roles.")
    for row in combined:
        if digest(row["image"]) != row["sha256"] or any(not valid_box(o["bbox"]) for o in row["objects"]):
            raise ValueError(f"Invalid frozen annotation or missing asset: {row['id']}")
    write_json(OUT, dict(roles=roles, reviewed_parts=list(PARTS), coordinate_system="Normalized EXIF-oriented source",
                         origin="Human-visible part centers. A null value is unobservable or unreviewed.",
                         old_pilot=digest(OLD_TRAY), fitting_dataset=digest(DATASET)))
    print({role: (len(rows), sum(len(r["objects"]) for r in rows)) for role, rows in roles.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheets", type=Path)
    args = parser.parse_args()
    if args.sheets:
        render_fit_sheets(args.sheets)
    else:
        main()
