"""Generate bounded automatic second-pass crops for the syringe verifier."""
import argparse
from pathlib import Path
from time import perf_counter

from medsyringe.additive import overlap_fraction, project
from medsyringe.data import digest, read_json, write_json
from medsyringe.detectors import box_iou, suppress
from scripts.native_candidate_bank import archive, cohorts, views
from scripts.occlusion_detector import EDGE, INITIAL, path_for, raster
from scripts.syringe_recrop_trigger_audit import area, context

WORK = Path("data/recrop-verifier")
BUDGETS = (3, 6, 8)
EXPERIMENT_MINIMUM = .001
CAP = 32


def windows(proposals, size):
    """Keep six score-ranked seeds and up to two spatially novel fallback seeds."""
    ranked = sorted(proposals, key=lambda item: -item["score"])[:12]
    options = []
    seen = set()
    for rank, item in enumerate(ranked, 1):
        region = context(item["bbox"])
        window = tuple(round(value * span) for value, span in zip(region, size * 2, strict=True))
        x, y, right, bottom = window
        if window in seen or area(region) > .40 or right <= x or bottom <= y:
            continue
        seen.add(window)
        options.append(dict(rank=rank, seed=item, region=region, window=window))
    shortlist = [w for w in options if w["rank"] <= 6]
    remaining = [w for w in options if w["rank"] > 6]
    for slot in (7, 8):
        if not remaining:
            break
        selected = min(remaining, key=lambda w: (max((box_iou(w["region"], a["region"])
                                                     for a in shortlist), default=0.), w["rank"]))
        remaining.remove(selected)
        shortlist.append({**selected, "slot": slot})
    return [{**w, "slot": w.get("slot", w["rank"])} for w in shortlist]


def crop_proposals(model, image, seeds):
    from scripts.rfdetr_compare import ranked_region

    found = []
    views_run = windows(seeds, image.size)
    for item in views_run:
        seed, window = item["seed"], item["window"]
        mapped = []
        for predicted in ranked_region(model, image.crop(window), EDGE):
            if predicted["score"] < EXPERIMENT_MINIMUM:
                continue
            if box := project(predicted["bbox"], window, image.size):
                mapped.append(dict(bbox=box, score=predicted["score"], slot=item["slot"],
                                   seed_rank=item["rank"], seed_score=seed["score"],
                                   seed_iou=box_iou(seed["bbox"], box), window_area=area(item["region"])))
        by_score = sorted(mapped, key=lambda p: -p["score"])[:12]
        by_seed = sorted(mapped, key=lambda p: -p["score"] * (.10 + p["seed_iou"]))[:8]
        found.extend(by_score)
        found.extend(p for p in by_seed if p not in by_score)
    return dict(seeds=seeds, windows=views_run, proposals=found)


def candidates(frame, budget):
    """Keep strong raw boxes and well-corroborated boxes at a common pool budget."""
    active = [p for p in frame["proposals"] if p["slot"] <= budget]
    enriched = []
    for p in active:
        box = p["bbox"]
        peers = [q for q in frame["seeds"] if box_iou(box, q["bbox"]) >= .45]
        supporting = {q["slot"] for q in active if q["slot"] != p["slot"] and box_iou(box, q["bbox"]) >= .45}
        enriched.append({**p, "first_pass": max((q["score"] for q in peers), default=0.),
                         "repeats": len(supporting), "views": budget,
                         "rank_score": p["score"] * (1 + .18 * len(supporting))})
    raw = suppress(enriched, .74)[:CAP // 2]
    by_support = suppress([{**p, "score": p["rank_score"], "raw_score": p["score"]} for p in enriched], .74)
    for q in by_support:
        if len(raw) >= CAP:
            break
        if not any(box_iou(q["bbox"], p["bbox"]) >= .74 for p in raw):
            raw.append({**q, "score": q["raw_score"]})
    return [dict(p, containing_seed=max((overlap_fraction(p["bbox"], s["bbox"]) for s in frame["seeds"]),
                                        default=0.)) for p in raw]


def inputs(role):
    if role in ("fit", "development"):
        rows = cohorts(role)
        return rows, archive(rows, role)
    if role == "archive":
        from scripts.thin_syringe_tiles import inputs as older_inputs

        old, bank = older_inputs("history")
        reserve, prior = older_inputs("prospective")
        return [*old, *reserve], {**bank, **prior}
    from scripts.recrop_reserve import references

    return references(), None


def binding(role, rows, seed_bank):
    # Hash the on-disk raster used by inference, including generated occlusion derivatives.
    seed_signature = ({key: seed_bank[key] for key in sorted(r["id"] for r in rows)}
                      if seed_bank is not None else None)
    import hashlib
    import json

    return dict(role=role, code=digest(__file__), crop_context=digest("scripts/syringe_recrop_trigger_audit.py"),
                checkpoint=read_json(INITIAL)["checkpoint_sha256"],
                images={r["id"]: digest(path_for(r)) for r in rows},
                seeds=hashlib.sha256(json.dumps(seed_signature, sort_keys=True).encode()).hexdigest())


def read_bank(role):
    rows, original = inputs(role)
    packet = read_json(WORK / f"{role}.json")
    if packet["binding"] != binding(role, rows, original) or set(packet["frames"]) != {r["id"] for r in rows}:
        raise ValueError(f"Incomplete or altered recrop bank for {role}.")
    return rows, packet["frames"]


def collect(roles):
    import torch
    from rfdetr import RFDETRNano

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    settings = read_json(INITIAL)
    if digest(settings["checkpoint"]) != settings["checkpoint_sha256"]:
        raise ValueError("RF-DETR checkpoint changed.")
    model = RFDETRNano.from_checkpoint(settings["checkpoint"], device="cpu")
    for role in roles:
        if role == "reserve":
            from scripts.recrop_reserve import assert_frozen_policy

            assert_frozen_policy()
        rows, original = inputs(role)
        path = WORK / f"{role}.json"
        key = binding(role, rows, original)
        packet = read_json(path) if path.exists() else dict(binding=key, frames={}, times={})
        if packet["binding"] != key:
            raise ValueError(f"Incompatible cached recrops for {role}.")
        pending = [r for r in rows if r["id"] not in packet["frames"]]
        for i, row in enumerate(pending, 1):
            image = raster(row)
            identifier = row["id"]
            seeds = original[identifier] if original is not None else views(model, image)
            start = perf_counter()
            packet["frames"][identifier] = crop_proposals(model, image, seeds)
            packet["times"][identifier] = round(perf_counter() - start, 5)
            if i % 4 == 0 or i == len(pending):
                write_json(path, packet)
                print(role, len(packet["frames"]), "/", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roles", nargs="+", choices=("fit", "development", "archive", "reserve"))
    collect(parser.parse_args().roles)
