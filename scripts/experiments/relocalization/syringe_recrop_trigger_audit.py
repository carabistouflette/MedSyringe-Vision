"""Audit whether existing proposals can seed a bounded second crop pass."""
import argparse

from medsyringe.detection.additive import project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.evaluate import match
from scripts.experiments.crowds.native_candidate_bank import WORK
from scripts.experiments.crowds.native_candidate_reserve import FOLDER as HISTORY
from scripts.experiments.crowds.native_candidate_reserve import REFERENCE as HISTORY_REFERENCE
from scripts.experiments.occlusion.occlusion_detector import ART, EDGE, INITIAL, raster
from scripts.experiments.occlusion.occlusion_proposals import oracle
from scripts.experiments.occlusion.occlusion_relocalization import prepare_reference
from scripts.experiments.identity.syringe_assembly_reserve import REFERENCE, RELEASE
from scripts.experiments.identity.thin_syringe_tiles import inputs

BUDGETS = (3, 6, 12, 28)


def area(box):
    x, y, right, bottom = box
    return max(0., right - x) * max(0., bottom - y)


def context(box):
    x, y, right, bottom = box
    dx, dy = (right - x) / 2, (bottom - y) / 2
    return [max(0., x - dx), max(0., y - dy), min(1., right + dx), min(1., bottom + dy)]


def covers(region, box):
    intersection = [max(region[0], box[0]), max(region[1], box[1]),
                    min(region[2], box[2]), min(region[3], box[3])]
    return area(region) <= .40 and area(intersection) >= .90 * area(box)


def audit(rows, proposals, baseline):
    _, matched = match(rows, baseline, .50)
    cases = []
    for row in rows:
        key = row["id"]
        ranked = sorted(proposals[key], key=lambda p: -p["score"])
        windows = [(p["bbox"], context(p["bbox"])) for p in ranked]
        for idx, obj in enumerate(row["objects"]):
            if idx in matched[key]:
                continue
            target = obj["bbox"]
            first_box = next((rank for rank, (box, _) in enumerate(windows, 1)
                              if box_iou(target, box) >= .50), None)
            first_crop = next((rank for rank, (_, region) in enumerate(windows, 1)
                               if covers(region, target)), None)
            cases.append(dict(id=key, target_index=idx, first_box_rank=first_box,
                              first_crop_rank=first_crop))
    budgets = {}
    for limit in BUDGETS:
        budgets[str(limit)] = {
            "box_hits": sum(0 < (row["first_box_rank"] or 0) <= limit for row in cases),
            "crop_hits": sum(0 < (row["first_crop_rank"] or 0) <= limit for row in cases),
            "examined_proposals": sum(min(len(proposals[row["id"]]), limit) for row in rows),
        }
    return dict(images=len(rows), missed=len(cases), cases=cases, budgets=budgets)


def main():
    dev, baseline, baseline_sources, _ = prepare_reference()
    _, pool = inputs("development")
    development = audit(dev, pool, baseline)
    reserve, pool = inputs("prospective")
    reserve_replay = audit(reserve, pool, read_json(RELEASE)["outputs"])
    history, pool = inputs("history")
    pump = [row for row in history if row["id"] == "crop-reserve-chemyx"]
    history_release = HISTORY / "released.json"
    pump_replay = audit(pump, pool, read_json(history_release)["outputs"])
    source_files = [WORK / f"rf-{role}.json" for role in ("development", "assembly-reserve", "prospective")]
    source_files.extend((REFERENCE, RELEASE, HISTORY_REFERENCE, history_release))
    write_json(ART / "syringe_recrop_trigger_audit.json", dict(
        development=development, reserve_replay=reserve_replay, pump_replay=pump_replay,
        inputs={**baseline_sources, **{str(path): digest(path) for path in source_files}},
        code_sha256=digest(__file__), scope="Geometric crop feasibility on development and exposed replays. "
        "No second-pass inference or acceptance decisions."))
    for role, data in (("development", development), ("reserve replay", reserve_replay), ("pump replay", pump_replay)):
        print(role, data["missed"], data["budgets"], flush=True)


def replay_rf():
    """Run the pinned detector on automatically chosen windows in exposed scenes."""
    import torch
    from rfdetr import RFDETRNano

    from scripts.experiments.transfer.rfdetr_compare import ranked_region

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    settings = read_json(INITIAL)
    if digest(settings["checkpoint"]) != settings["checkpoint_sha256"]:
        raise ValueError("RF-DETR checkpoint changed.")
    model = RFDETRNano.from_checkpoint(settings["checkpoint"], device="cpu")
    former_reserve, reserve_pool = inputs("prospective")
    history, history_pool = inputs("history")
    pump = next(row for row in history if row["id"] == "crop-reserve-chemyx")
    seeds_by_scene = {**reserve_pool, pump["id"]: history_pool[pump["id"]]}
    budgets = BUDGETS[:-1]
    reports = {}
    for row in [*former_reserve, pump]:
        rgb = raster(row)
        key = row["id"]
        ranked = sorted(seeds_by_scene[key], key=lambda p: -p["score"])
        visited = set()
        localized, views, results = [], [], {}
        for rank, seed in enumerate(ranked[:max(budgets)], 1):
            region = context(seed["bbox"])
            window = tuple(round(v * n) for v, n in zip(region, rgb.size * 2, strict=True))
            x, y, right, bottom = window
            if area(region) <= .40 and window not in visited and right > x and bottom > y:
                visited.add(window)
                views.append(dict(seed_rank=rank, window=window))
                for item in ranked_region(model, rgb.crop(window), EDGE):
                    if box := project(item["bbox"], window, rgb.size):
                        localized.append({**item, "bbox": box, "seed_rank": rank})
            if rank in budgets:
                results[str(rank)] = dict(crop_only=oracle(row["objects"], localized),
                                          combined=oracle(row["objects"], [*ranked, *localized]),
                                          candidate_count=len(localized), windows=list(views))
        reports[key] = dict(original=oracle(row["objects"], ranked), budgets=results)
        print(key, {k: v["crop_only"]["matched"] for k, v in results.items()}, flush=True)
    write_json(ART / "syringe_recrop_rf_replay.json", dict(
        code_sha256=digest(__file__), geometric_audit_sha256=digest(ART / "syringe_recrop_trigger_audit.json"),
        checkpoint_sha256=settings["checkpoint_sha256"], scenes=reports,
        scope="Automatic proposal generation on previously seen public images. "
              "No confidence selection, accuracy estimate, or application change."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-rf", action="store_true")
    replay_rf() if parser.parse_args().replay_rf else main()
