"""Find the proposal ceiling after RF-DETR NMS and before the acceptance floor."""
from medsyringe.data import digest, read_json, write_json
from scripts.occlusion_detector import ART, FREEZE
from scripts.occlusion_proposals import oracle
from scripts.occlusion_relocalization_reserve import DATA, REFERENCE, refs


def run():
    release = read_json(DATA / "released.json")["predictions"]
    challenger = DATA / "challenger.json"
    predictions = read_json(challenger)["predictions"]
    floors = read_json(FREEZE)["floors"]
    report = {}
    for row in refs():
        objects = row["objects"]
        if not objects:
            continue
        key = row["id"]
        arms = {}
        for name, cases in predictions.items():
            ranked = cases[key]
            kept = [p for p in ranked if p["score"] >= floors[name]]
            arms[name] = dict(ranked=oracle(objects, ranked), accepted=oracle(objects, kept))
        report[key] = dict(expected=len(objects), released=oracle(objects, release[key]), arms=arms)
    write_json(ART / "occlusion_relocalization_proposal_ceiling.json",
               dict(reference=digest(REFERENCE), cache=digest(challenger), cases=report,
                    scope="Retrospective diagnostic, RF-DETR proposals already suppressed at NMS IoU 0.45."))


if __name__ == "__main__":
    run()
