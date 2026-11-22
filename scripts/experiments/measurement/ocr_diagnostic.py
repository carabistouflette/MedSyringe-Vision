"""Measure exact graduation-token recovery on visually checked development crops."""
from collections import Counter
from pathlib import Path
from time import perf_counter

from medsyringe.measurement.attributes import crop, read_inscriptions
from medsyringe.data import digest, read_image, train_rows, write_json

# Visually read on enlarged source views before running OCR.
VISIBLE_TOKENS = {
    "IMG_3198": "60", "IMG_3206": "60", "IMG_3215": "20", "IMG_3228": "20",
    "IMG_3245": "20", "IMG_3251": "60", "IMG_3292": "20", "IMG_3318": "20",
    "IMG_3321": "20",
}


def main():
    root = Path(".")
    samples = []
    for row in train_rows(root):
        if row["id"] not in VISIBLE_TOKENS:
            continue
        expected = VISIBLE_TOKENS[row["id"]]
        source = read_image(root / "data" / row["image"])
        started = perf_counter()
        result = read_inscriptions(crop(source, row["objects"][0]["bbox"]))
        samples.append(dict(id=row["id"], expected_visible_token=expected, **result,
                            matched=any(x["text"] == expected for x in result["candidates"]),
                            seconds=perf_counter()-started))
    write_json(root / "artifacts/ocr_diagnostic.json",
               dict(design="known-legible same-session development crops; one target token per image",
                    code_sha256=digest(root / "medsyringe/measurement/attributes.py"),
                    samples=samples, exact=sum(s["matched"] for s in samples),
                    status_counts=dict(Counter(s["status"] for s in samples))))
    print("Exact target token recovered in", sum(s["matched"] for s in samples), "of", len(samples))


if __name__ == "__main__":
    main()
