"""The two visual exemplars come exclusively from prepared development sources."""
from pathlib import Path

from medsyringe.data import read_json, train_rows


def references():
    root = Path(".")
    local = next(row for row in train_rows(root) if row["id"] == "IMG_3228")
    public = next(row for row in read_json(root / "artifacts/precision_study_manifest.json")["train"]
                  if row["id"] == "36814819")
    return {"fil_rouge": local, "public": public}
