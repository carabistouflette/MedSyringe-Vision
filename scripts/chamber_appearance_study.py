"""Compare OCR-independent chamber observations on rendered and known public crops."""
from pathlib import Path

from PIL import Image

from medsyringe.attributes import crop
from medsyringe.chamber_appearance import inspect_chamber
from medsyringe.data import digest, read_image, read_json, write_json
from scripts.small_field_study import ART, REFERENCE, control_cases, render

RESULT = ART / "chamber_appearance_results.json"


def run():
    rows = []
    for r in read_json(REFERENCE)["rows"]:
        if digest(r["image"]) != r["sha256"]:
            raise ValueError("A public reference image changed.")
        frame = crop(read_image(r["image"]), r["bbox"])
        output = inspect_chamber(frame)
        flipped = inspect_chamber(frame.transpose(Image.Transpose.ROTATE_180))
        rows.append(dict(id=r["id"], expected_presence=r["presence"], expected_color=r["apparent_color"],
                         cohort="observed public photos", output=output, rotation180=flipped))

    for key, level, seal, liquid, presence, color, options, rotation in control_cases():
        pixels, _, _ = render(level, seal, liquid, **options)
        frame = Image.fromarray(pixels).rotate(rotation, expand=True)
        rows.append(dict(id=key, expected_presence=presence, expected_color=color, cohort="rendered controls",
                         output=inspect_chamber(frame)))
    for row in rows:
        observed = row["output"]
        appearance = observed["appearance"] or {}
        print(row["id"], row["expected_presence"], observed["presence"],
              (appearance.get("apparent_color") or {}).get("family"))
    write_json(RESULT, dict(rows=rows, code=digest(Path("medsyringe/chamber_appearance.py")),
                           reference=digest(REFERENCE),
                           scope="Existing public references and controlled images. No physical volume or opacity."))


if __name__ == "__main__":
    run()
