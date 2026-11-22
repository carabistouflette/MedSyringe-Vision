"""Seal fresh public references and collect action-verifier transfer evidence."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, valid_box, write_json
from scripts.detection_public_benchmark import lexical_key, url_key
from scripts.instance_source_audit import fingerprint_collision, protection
from scripts.native_candidate_bank import cohorts, views
from scripts.native_candidate_reserve import references as prior_native
from scripts.occlusion_detector import ART, INITIAL, path_for, raster
from scripts.public_training_intake import raster_hashes
from scripts.recrop_proposal_bank import crop_proposals
from scripts.recrop_reserve import references as prior_recrop
from scripts.relocalization_reserve import references as prior_selector
from scripts.instance_separation_reserve import references as prior_separation
from scripts.complete_syringe_reserve import references as prior_complete
from scripts.completeness_calibration_reserve import references as prior_calibration
from scripts.syringe_action_verifier import POLICY, WORK, sealed
from scripts.syringe_assembly_reserve import references as prior_assembly

REFERENCE = ART / "syringe_action_verifier_reserve.json"
RELEASE = WORK / "released.json"
PROPOSALS = WORK / "proposals.json"

# References were drawn from visible device extents on the downloaded raster.
PEXELS = (
    ("5234480", "a-syringe-with-liquid-inside", "Polina Tankilevitch", (620, 414),
     ((263, 143, 466, 237),), "One syringe with the distal needle visible and the grip partly hidden."),
    ("5998508", "close-up-photo-of-a-person-holding-a-syringe", "Pavel Danilyuk", (434, 650),
     ((101, 297, 371, 363),), "One syringe with a visible plunger rod and needle. Previously screened without inference."),
    ("7581089", "silver-tray-with-syringes", "cottonbro studio", (433, 650),
     ((96, 339, 182, 454), (166, 336, 222, 451)), "Two complete assemblies. A loose cap, ampoules, and reflections are excluded."),
    ("17112252", "table-with-prepared-syringes-and-other-medical-instruments", "Jimmy Liao", (620, 414),
     ((223, 89, 422, 138), (45, 138, 168, 311)), "One plastic syringe and one dental cartridge syringe with visible barrel, needle and grip."),
    ("15831834", "scientist-with-a-test-tube-and-glass-container-in-her-hands", "Jonathan Borba", (433, 650),
     (), "Glass transfer tool and graduated vessel. No syringe."),
    ("3825578", "unrecognizable-chemist-taking-liquid-with-pipette-from-test-tube-in-modern-laboratory", "RF._.studio _", (464, 650),
     (), "Handheld glassware and colored sample tubes. No syringe."),
    ("33658644", "laboratory-technician-organizing-test-tubes", "Andy Coffie", (620, 414),
     (), "Rack of capped laboratory tubes and a capped blue pen. No syringe."),
)
VISUAL_EXCLUSIONS = {
    "8577983": "Previously screened and too densely overlapped to assign a reliable count.",
    "8071796": "Previously scored public reference.",
    "8413326": "SHVETS production and Anna Shvets have a documented contributor relationship.",
    "8413433": "SHVETS production and Anna Shvets have a documented contributor relationship.",
    "5721385": "Same photographer as the more informative two-syringe tray.",
    "33394283": "Previously screened. Wrapped objects lack sufficiently clear extents.",
    "7230835": "Additional negative screened. Three diverse negatives were selected before inference.",
}


def sources():
    for key, slug, author, size, boxes, note in PEXELS:
        page = f"https://www.pexels.com/photo/{slug}-{key}/"
        asset = f"https://images.pexels.com/photos/{key}/pexels-photo-{key}.jpeg?auto=compress&cs=tinysrgb&w=1000"
        yield key, page, asset, author, f"Pexels {lexical_key(author)} {key}", size, boxes, note


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    history, private_hashes, private_pixels = protection()
    previous = [*(row for role in ("fit", "development", "evaluation") for row in cohorts(role)),
                *prior_native(), *prior_assembly(), *prior_recrop(), *prior_selector(),
                *prior_separation(), *prior_complete(), *prior_calibration()]
    hashes = history["sha256"] | private_hashes | {r["sha256"] for r in previous}
    urls = history["urls"] | {url_key(r["source_url"]).rstrip("/") for r in previous if r.get("source_url")}
    groups = {r["group"] for r in previous}
    authors = history["authors"] | {lexical_key(r["author"]) for r in previous if r.get("author")}
    pixels = history["dhash"] | private_pixels
    for row in previous:
        pixels.update(raster_hashes(path_for(row)))
    rows, rejects = [], dict(VISUAL_EXCLUSIONS)
    for key, page, asset, author, group, grid, boxes, note in sources():
        path = WORK / f"{key}.jpg"
        content, fingerprints = digest(path), raster_hashes(path)
        first, url = lexical_key(author), url_key(page).rstrip("/")
        if (key in history["ids"] or group in groups or content in hashes or url in urls
                or any(first in other for other in authors) or fingerprint_collision(fingerprints, pixels)):
            rejects[key] = "Identity, author, source URL, capture group or near-image exposure in project history."
            continue
        normalized = [[value / span for value, span in zip(box, grid * 2, strict=True)] for box in boxes]
        row = dict(id=f"action-{key}", source_url=page, original_url=asset, author=author, group=group,
                   image=str(path), sha256=content, objects=[dict(bbox=b) for b in normalized],
                   review=note, license="https://www.pexels.com/license/")
        image = raster(row)
        if abs(image.width / image.height - grid[0] / grid[1]) > .004 or any(not valid_box(b) for b in normalized):
            raise ValueError(f"Review grid and source image do not match for {key}.")
        rows.append(row)
        hashes.add(content)
        pixels.update(fingerprints)
        urls.add(url)
        groups.add(group)
        authors.add(first)
    if len(rows) < 4 or not any(len(r["objects"]) >= 2 for r in rows) or not any(not r["objects"] for r in rows):
        raise ValueError(f"Not enough eligible independent instances and groups: {rejects}.")
    write_json(REFERENCE, dict(code=digest(__file__), rows=rows, exclusions=rejects,
                               scope="One visual reviewer. Distinct public captures, no output inspected before freeze."))
    print("admitted", len(rows), "targets", sum(len(r["objects"]) for r in rows), "excluded", rejects, flush=True)


def references():
    record = read_json(REFERENCE)
    if record["code"] != digest(__file__):
        raise ValueError("The reference protocol changed.")
    return record["rows"]


def binding():
    sealed()
    return dict(policy=digest(POLICY), reference=digest(REFERENCE),
                diagnostic=digest(ART / "syringe_action_diagnostic_policy.json"),
                rf_checkpoint=read_json(INITIAL)["checkpoint_sha256"])


def state(path):
    key = binding()
    saved = read_json(path) if path.exists() else dict(binding=key, frames={})
    if saved["binding"] != key:
        raise ValueError(f"Stale evidence at {path}.")
    return saved


def baseline():
    from medsyringe.grounding_backend import analyze
    from medsyringe.hardware import configure_device
    from medsyringe.pipeline import MedSyringeEngine
    from scripts.sam3_study import configure_cpu

    cache = state(RELEASE)
    pending = [r for r in references() if r["id"] not in cache["frames"]]
    if not pending:
        return
    configure_cpu()
    engine = MedSyringeEngine(Path.cwd())
    try:
        configure_device(engine, Path.cwd(), "xpu")
        for row in pending:
            result = analyze(engine, raster(row), False)
            check = result["detection_checks"]
            if check["status"] != "available" or check["primary_backend"] != "xpu":
                raise ValueError(f"Released detector unavailable on {row['id']}.")
            cache["frames"][row["id"]] = result["objects"]
            write_json(RELEASE, cache)
            print(row["id"], "released count", len(result["objects"]), flush=True)
    finally:
        engine.close()


def collect():
    import torch
    from rfdetr import RFDETRNano

    cache = state(PROPOSALS)
    settings = read_json(INITIAL)
    if digest(settings["checkpoint"]) != settings["checkpoint_sha256"]:
        raise ValueError("Unexpected RF-DETR checkpoint.")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model = RFDETRNano.from_checkpoint(settings["checkpoint"], device="cpu")
    for row in references():
        key = row["id"]
        if key in cache["frames"]:
            continue
        photo = raster(row)
        cache["frames"][key] = crop_proposals(model, photo, views(model, photo))
        write_json(PROPOSALS, cache)
        print(key, "proposals cached", flush=True)


def evidence():
    rows = references()
    inputs = {str(p): digest(p) for p in (REFERENCE, RELEASE, PROPOSALS)}
    base, proposals = (state(p)["frames"] for p in (RELEASE, PROPOSALS))
    if set(base) != set(proposals) or set(base) != {r["id"] for r in rows}:
        raise ValueError("Reserve evidence is incomplete.")
    return rows, base, proposals, inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "baseline", "collect"))
    {"freeze": freeze, "baseline": baseline, "collect": collect}[parser.parse_args().stage]()
