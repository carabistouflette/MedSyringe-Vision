"""Recreate the fitting-only proposal map used by the manual part review."""
from medsyringe.data import write_json
from medsyringe.detectors import box_iou
from scripts.native_candidate_bank import archive, cohorts
from scripts.thin_syringe_tiles import WORK


def run():
    rows = cohorts("fit")
    bank = archive(rows, "fit")
    entries = []
    for row in rows:
        if len(row["objects"]) != 1 or not row["objects"][0].get("parts"):
            continue
        ref = row["objects"][0]["bbox"]
        candidates = [p for p in bank[row["id"]] if .05 <= box_iou(ref, p["bbox"]) < .47]
        for proposed in sorted(candidates, key=lambda item: -item["score"])[:2]:
            entries.append(dict(index=len(entries)+1, id=row["id"], bbox=proposed["bbox"],
                                reference=ref, score=proposed["score"]))
    WORK.mkdir(parents=True, exist_ok=True)
    write_json(WORK / "fragment-review-map.json", entries)
    print(len(entries), "review candidates", flush=True)


if __name__ == "__main__":
    run()
