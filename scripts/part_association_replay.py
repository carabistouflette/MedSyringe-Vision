"""Replay the locked association policies on the previously exposed failure cases."""
import cv2

from medsyringe.additive import project
from medsyringe.data import digest, write_json
from medsyringe.detectors import suppress
from medsyringe.evaluate import evaluate, match
from medsyringe.part_geometry import PARTS, PROMPTS
from scripts.native_candidate_verifier import simple_metrics
from scripts.occlusion_detector import raster
from scripts.occlusion_proposals import oracle
from scripts.part_association import ART, LOCK, action_model, apply, audit, rank, sealed
from scripts.part_association_observations import EDGE, frame_binding, prompt_parts, windows
from scripts.relocalization_reserve import evidence as crowded_evidence
from scripts.sam3_study import Segmenter, configure_cpu
from scripts.syringe_action_reserve import evidence as tray_evidence

OUT = ART / "part_association_exposed_replay.json"


def replay():
    cv2.setNumThreads(1)
    policy = sealed()
    rows, base, frames, hashes = [], {}, {}, {}
    for keys, reader in (({"action-17112252", "action-33658644"}, tray_evidence),
                         ({"refine-8670507"}, crowded_evidence)):
        source, incumbent, proposals, refs = reader()
        rows.extend(r for r in source if r["id"] in keys)
        base.update({k: incumbent[k] for k in keys})
        frames.update({k: proposals[k] for k in keys})
        hashes.update(refs)
    configure_cpu()
    sam = Segmenter(EDGE, "xpu")
    tokens = {key: sam.processor(text=value, return_tensors="pt").to(sam.device)
              for key, value in PROMPTS.items()}
    observed = {}
    for row in rows:
        key, img = row["id"], raster(row)
        found = {k: [] for k in PARTS}
        for j, window in enumerate(windows(img, frames[key])):
            for label, local, score in prompt_parts(sam, img.crop(window), tokens):
                if box := project(local, window, img.size):
                    found[label].append(dict(bbox=box, score=score, view=j))
        observed[key] = {k: suppress(v, .55)[:18] for k, v in found.items()}
    ranked, original = rank(rows, base, frames, observed, policy["model"], action_model()["heads"])
    before, (_, strict) = simple_metrics(rows, base), match(rows, base, .75)
    comparison = {}
    for kind in ("finalists", "active"):
        for arm, spec in policy["selection"][kind].items():
            pred, events = apply(base, ranked, original, spec["rule"])
            comparison[f"{kind}/{arm}"] = dict(delta=audit(rows, before, strict, pred, events),
                                              metrics=evaluate(rows, pred, 0.), events=events)
    write_json(OUT, dict(policy=digest(LOCK), code=digest(__file__), inputs=hashes,
                        binding=frame_binding(rows, frames), observations=observed,
                        baseline=evaluate(rows, base, 0.), comparison=comparison,
                        available={r["id"]: oracle(r["objects"], ranked[r["id"]]) for r in rows},
                        part_counts={k: {label: len(vals) for label, vals in parts.items()}
                                     for k, parts in observed.items()}))
    print({k: (v["delta"]["gain"], v["delta"]["extra_fp"]) for k, v in comparison.items()}, flush=True)


if __name__ == "__main__":
    replay()
