"""Reproduce the visual audit of priority crowd and negative fitting candidates."""
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.experiments.crowds.low_count_recovery import DATASET, crowd

OUT = crowd.ART / "low_count_priority_review.json"
NOTES = {
    "commons-154897844": "No countable syringe is visible in this clinical registration scene.",
    "commons-197653421": "No syringe is visible in the vaccination portrait.",
    "commons-53168121": "Car lettering with no device. Easy background negative.",
    "commons-651990": "Detached needles and their cap. Hard part-only negative.",
    "commons-82792522": "A door sign refers to syringes. No device is visible.",
    "commons-8398651": "Multiple closed syringe transport cases. No device is visible.",
    "commons-8685048": "Detached needle tip without barrel or plunger. Hard part-only negative.",
    "QxPD1f0UcIGZAjThZz1k": "Two distinct clear barrels in contact next to a screwdriver.",
    "gOTjoBD38tL678nEntTa": "Three clear, closely spaced bodies with distinct plungers on cyan. Stock-style origin unresolved.",
    "1hGL1WkNQ7NT3PR2E2hh": "Three horizontally aligned orange-capped devices. Stock-style origin unresolved.",
    "3goHYbmmDXIhTLhWPXJR": "Twelve visually distinct vertical devices, already sampled in the blind geometry audit.",
    "3SyRXjcIsD87JRJlDYUL": "Three slender capped devices aligned on a yellow background.",
    "512SZZUSSBPBgmt2SxP6": "Two overlapping clear bodies with visible separate flanges.",
    "6Opf8kKlqZKLU8F4GvVJ": "Six narrow bodies with red plungers, already sampled in the blind geometry audit.",
}
RECEIPTS = {
    "tai-qompv/stage-1-klug2": dict(public=True, catalog_images=961, original_image_credit=False),
    "giancarlo-millan/gloves-pill-wrapper-and-syringe": dict(public=True, catalog_images=1571,
                                                            original_image_credit=False),
}


def run():
    reviews = []
    for row in read_json(DATASET)["rows"]:
        if row["split"] != "public-fit" or not (len(row["objects"]) > 1 or not row["objects"] and row["eligible_for_fit"]):
            continue
        if (row["id"] not in NOTES or digest(row["image"]) != row["sha256"]
                or any(not valid_box(o["bbox"]) for o in row["objects"])):
            raise ValueError(f"Priority image lacks valid review identity: {row['id']}")
        reviews.append(dict(id=row["id"], image=row["image"], sha256=row["sha256"], group=row["group"],
                            source_url=row["source_url"], individual_rights=row["rights_status"],
                            count=len(row["objects"]), note=NOTES[row["id"]],
                            decision="fit group negative retained" if row["eligible_for_fit"]
                            else "geometry retained, item creator or original source unresolved"))
    write_json(OUT, dict(source=digest(DATASET), rows=reviews, publisher_project_receipts=RECEIPTS,
                         method="Visual overlay on the 14 local files and read-only public project metadata. "
                                "Project metadata verifies publication, with no original-photo link for these cases.",
                         new_fit_admissions=0))
    print(len(reviews), "reviewed cases, zero new fit admissions")


if __name__ == "__main__":
    run()
