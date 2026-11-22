"""Freeze active alternatives on development and measure their transfer separately."""
import argparse
from pathlib import Path

from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from scripts.experiments.relocalization.pair_relocalization import ART, POLICY, check, frozen, predict, rank_pairs
from scripts.experiments.relocalization.relocalization_selector import snapshot
from scripts.experiments.ranking.syringe_box_quality import frozen as box_quality

PROBE = ART / "pair_relocalization_probe_policy.json"
RESULT = ART / "pair_relocalization_probe_results.json"


def priority(t):
    lost = t["lost"]
    gained = t["gained"]
    gain_50, gain_75 = (len(gained[key]) for key in ("0.5", "0.75"))
    loss_50, loss_75 = (len(lost[key]) for key in ("0.5", "0.75"))
    return (gain_50 - loss_50, gain_75 - loss_75, gain_50, gain_75, -len(t["harmful_changes"]),
            -t["changed"], t["rule"]["floor"], t["rule"]["link"] or 0.)


def freeze():
    if PROBE.exists():
        raise FileExistsError(PROBE)
    policy = frozen()
    selected = {}
    for arm in policy["selection"]["finalists"]:
        candidates = [t for t in policy["selection"]["trials"] if t["rule"]["arm"] == arm and t["changed"]]
        if candidates:
            selected[arm] = max(candidates, key=priority)
    write_json(PROBE, dict(policy=digest(POLICY), code=digest(Path(__file__)), selected=selected,
                            scope="Exploratory comparisons chosen only from active development trials."))
    for arm, t in selected.items():
        print(arm, t["rule"], "changed", t["changed"], "passes", t["passes"], flush=True)


def score():
    from scripts.experiments.relocalization.pair_relocalization_reserve import packets

    policy, probe = frozen(), read_json(PROBE)
    if probe["policy"] != digest(POLICY) or probe["code"] != digest(Path(__file__)):
        raise ValueError("The active probe policy changed.")
    quality_model = box_quality()["models"]
    cohorts = {}
    for role, (rows, base, _, bank) in packets().items():
        choices = rank_pairs(base, bank, policy["heads"], quality_model)
        before = snapshot(rows, base)
        result = {}
        for arm, t in probe["selected"].items():
            p = predict(base, choices, t["rule"])
            result[arm] = dict(metrics=evaluate(rows, p, 0.), guard=check(rows, base, before, p),
                               changes={k: [i for i in values if "prior_bbox" in i] for k,values in p.items()
                                        if any("prior_bbox" in i for i in values)})
        cohorts[role] = result
        print(role, {a:(v["metrics"]["tp"], v["guard"]["changed"], v["guard"]["passes"])
                     for a,v in result.items()}, flush=True)
    write_json(RESULT, dict(probe=digest(PROBE), cohorts=cohorts))


if __name__ == "__main__":
    stages = dict(freeze=freeze, score=score)
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=stages)
    stages[cli.parse_args().stage]()
