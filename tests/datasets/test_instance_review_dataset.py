import numpy as np
from PIL import Image

from medsyringe.data import digest, write_json
from scripts.instance_review_dataset import coco_export, join


def test_duplicate_carries_prior_acquisition_protection_across_sources(tmp_path):
    rng = np.random.default_rng(93)
    sources = []
    for i in range(7):
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(path)
        sources.append(dict(id=str(i), image=str(path), sha256=digest(path), width=40, height=40,
                            group=f"camera-{i}", source_url=f"https://example.org/page/{i}",
                            original_url=f"https://example.org/assets/{i}", author=f"Camera owner {i}",
                            license="CC BY 4.0", objects=[dict(bbox=[.1, .2, .8, .9])] if i < 4 else []))
    earlier = {**sources[0], "id": "previous", "group": "old-acquisition",
               "author": "Another source", "source_url": "https://example.net/previous",
               "previously_scored_acquisition": True}
    entry, exposure = tmp_path / "reviewed.json", tmp_path / "earlier.json"
    write_json(entry, dict(rows=sources))
    write_json(exposure, dict(rows=[earlier]))
    retained, excluded, roles, _ = join([entry, exposure])
    duplicate = next(row for row in retained if row["id"] == "0")
    assert duplicate["split"] == "historical-reference-only"
    assert duplicate["eligible_for_fit"] is False
    assert excluded[0]["representative"] == "0"
    assert len(roles) == len(retained) == 7
    coco = coco_export(retained)
    assert len(coco["images"]) == 7 and len(coco["annotations"]) == 4
    assert len(coco["licenses"]) == 1
