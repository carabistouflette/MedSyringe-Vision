"""Summarize source distributions, visual similarity and draft annotation QA."""
from collections import Counter
from pathlib import Path

from medsyringe.data import digest, read_json, train_rows, write_json
from medsyringe.detection.detectors import box_iou


def main():
    root = Path(".")
    rows = train_rows(root)
    inventory = read_json("artifacts/source_inventory.json")
    drafts = read_json("artifacts/annotations.json")
    proposals = read_json("data/cache/dino-train-original.json")["predictions"]
    comparisons = []
    for row in rows:
        top = proposals[row["id"]]["proposals"]
        overlap = box_iou(row["objects"][0]["bbox"], top[0]["bbox"]) if top else 0
        comparisons.append(dict(id=row["id"], top_proposal_iou=overlap))
    comparisons.sort(key=lambda item: item["top_proposal_iou"])
    visual_pairs = []
    for index, a in enumerate(rows):
        for b in rows[index+1:]:
            distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            if distance <= 4:
                visual_pairs.append(dict(images=[a["id"], b["id"]], hamming_distance=distance))

    def count(attribute):
        return dict(Counter(obj[attribute] or "unknown" for row in rows for obj in row["objects"]))

    stats = dict(files=len(inventory), source_bytes=sum(r["bytes"] for r in inventory),
                 unique_training_scenes=len(rows), draft_instances=sum(len(r["objects"]) for r in rows),
                 local_groups=dict(Counter(row["group"] for row in rows)),
                 exact_duplicate_files=[dict(source=r["source"], canonical=r["duplicate_of"])
                                        for r in inventory if r.get("duplicate_of")],
                 possible_near_duplicate_pairs=visual_pairs,
                 scale=count("scale"), color=count("color"), background=count("background"),
                 burst=count("burst"), draft_flags=dict(Counter(flag for row in drafts.values()
                                                               for flag in row["flags"])),
                 orientation_corrected_size=dict(Counter(f"{r['width']} × {r['height']}" for r in rows)),
                 annotation_sha256=digest("artifacts/annotations.json"),
                 draft_agreement_warning="Top proposal is a discrepancy signal, not an annotation reference.",
                 draft_top_proposal_agreement=comparisons)
    write_json("artifacts/dataset_summary.json", stats)
    print(len(visual_pairs), "near-hash pairs; development counts", stats["scale"], stats["color"])


if __name__ == "__main__":
    main()
