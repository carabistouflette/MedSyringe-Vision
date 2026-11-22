"""Produce review-only low-count recovery proposals using sealed verifier weights."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_image, read_json, valid_box, write_json
from medsyringe.detection.detectors import box_iou
from medsyringe.detection.recall_guard import compare
from scripts.experiments.crowds.crowded_sam_hybrid import MINIMUM_IOU
from scripts.experiments.crowds.low_count_recovery import (
    CACHE, FREEZE, REPORT, CrowdedMasks, accept_additions, brief, cached, configure_cpu,
    crowd, disjoint, load_policy, source_rows, verification_score,
)

OUTPUT = crowd.ART / "low_count_alignment_diagnostic.json"


def propose(known, raw, head):
    """Reserve the SAM peer of the released box before offering new instances."""
    proposals = disjoint(raw)
    if len(known) == 1 and proposals:
        peer = max(range(len(proposals)), key=lambda i: box_iou(known[0]["bbox"], proposals[i]["bbox"]))
        if box_iou(known[0]["bbox"], proposals[peer]["bbox"]) >= MINIMUM_IOU:
            proposals.pop(peer)
    return accept_additions(known, proposals, lambda p: verification_score(head, p) >= head["floor"])[len(known):]


def diagnose():
    head = load_policy()
    rows = source_rows("regression")
    bank = cached("regression")
    baseline = {k: v["objects"] for k, v in read_json(crowd.CACHE / "released.json")["frames"].items()}
    recovery = {key: propose(value, bank[key], head) for key, value in baseline.items()}
    scope = {
        "all": rows,
        "crowded with 0 or 1 starting prediction": [r for r in rows if len(r["objects"]) >= 2
                                                    and len(baseline[r["id"]]) < 2],
        "empty": [r for r in rows if not r["objects"]],
    }
    data = {}
    for name, cases in scope.items():
        ids = {r["id"] for r in cases}
        old = brief(cases, {i: baseline[i] for i in ids})
        new = brief(cases, {i: [*baseline[i], *recovery[i]] for i in ids})
        data[name] = dict(baseline=old, proposal_added=new, retention=compare(old, new))
    write_json(OUTPUT, dict(source=digest(REPORT), verifier=digest(FREEZE), overlap=MINIMUM_IOU,
                            notes="Post-audit duplicate diagnostic. The association rule comes from the earlier crowded "
                                  "fusion. Observed scenes require fresh confirmation before any automatic release.",
                            counts={r["id"]: dict(expected=len(r["objects"]), accepted=len(baseline[r["id"]]),
                                                  extra=len(recovery[r["id"]])) for r in rows},
                            cohorts=data))
    for name, outcomes in data.items():
        print(name, {k: {f: v[f] for f in ("tp", "fp", "exact_count")}
                     for k, v in outcomes.items() if k != "retention"}, flush=True)


def review(photo, baseline_path, path, device):
    if path.exists():
        raise FileExistsError(path)
    report = read_json(baseline_path)
    sha = digest(photo)
    if report.get("source_sha256") != sha:
        raise ValueError("The observation must refer to the same photograph.")
    known = report["objects"]
    if any(not valid_box(p["bbox"]) for p in known):
        raise ValueError("Observation geometry is invalid.")
    suggestions = []
    if len(known) <= 1:
        configure_cpu()
        head = load_policy()
        raw = dict(CrowdedMasks(756, device).candidates(read_image(photo)))
        suggestions = propose(known, raw, head)
    write_json(path, dict(status="experimental review only", source_sha256=sha, verified_count=len(known),
                          suggestions=suggestions, verifier=digest(FREEZE), observation=digest(baseline_path)))
    print(len(known), "accepted and", len(suggestions), "proposals for review", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="job", required=True)
    command.add_parser("diagnose")
    item = command.add_parser("review")
    item.add_argument("image", type=Path)
    item.add_argument("--observation", required=True, type=Path)
    item.add_argument("--output", required=True, type=Path)
    item.add_argument("--device", choices=("cpu", "xpu"), default="xpu")
    args = parser.parse_args()
    if args.job == "diagnose":
        diagnose()
    else:
        review(args.image, args.observation, args.output, args.device)
