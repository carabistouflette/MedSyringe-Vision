"""Audit project-local image separation for the occlusion study."""
from medsyringe.data import digest, read_json, write_json
from scripts.instance_source_audit import fingerprint_collision, protection
from scripts.occlusion_detector import ART, FRESH, PLAN, partitions, path_for
from scripts.part_verifier_new_acquisition import row as oriez
from scripts.part_verifier_study import REFERENCES
from scripts.public_training_intake import raster_hashes


def run():
    roles = read_json(REFERENCES)["roles"]
    fit, development, _ = partitions()
    _, private, private_visual = protection()
    reserve = [*roles["regression"], *roles["fresh"], *roles["transfer"], oriez(), *read_json(FRESH)["rows"]]
    cohorts = {"fit": fit, "development": development, "public evaluation": reserve}
    checked = {}
    for name, rows in cohorts.items():
        incoming = set()
        for row in rows:
            path = path_for(row)
            hashes = raster_hashes(path)
            if digest(path) in private or fingerprint_collision(hashes, private_visual):
                raise ValueError(f"Private visual near-duplicate in {name}: {row['id']}")
            incoming.update(hashes)
        checked[name] = incoming
    for name in ("development", "public evaluation"):
        if fingerprint_collision(checked[name], checked["fit"]):
            raise ValueError(f"Transformed visual collision across fit and {name}")
    summary = dict(plan_sha256=digest(PLAN), reserve_sha256=digest(FRESH),
                   rows_checked={name: len(rows) for name, rows in cohorts.items()},
                   transformed_hashes_checked={name: len(values) for name, values in checked.items()}, radius=4,
                   private_sha256_entries=len(private), private_dhash_entries=len(private_visual),
                   blocked_collisions=0, scope="Direct and transformed 64-bit visual fingerprints across distinct "
                                               "archived public rows; upstream pretraining exposure remains unknown.")
    write_json(ART / "occlusion_leakage_audit.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    run()
