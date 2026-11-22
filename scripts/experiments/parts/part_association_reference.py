"""Bind visible syringe parts to instance identities on public fitting and development scenes."""
from pathlib import Path

from medsyringe.data import digest, read_json, write_json
from medsyringe.measurement.part_geometry import PARTS
from scripts.experiments.occlusion.occlusion_detector import ART, raster
from scripts.experiments.relocalization.recrop_proposal_bank import WORK, read_bank

REFERENCE = ART / "part_association_reference.json"

# Points in the public review images, ordered by the existing instance boxes and PARTS.
# Null denotes a component the reviewer could not place reliably.
MARKS = {
    "150790461": ((760, 602), [((310, 136), None, None, None),
                                ((395, 452), (583, 451), (624, 455), (278, 451))]),
    "191240619": ((700, 700), [((281, 371), (334, 496), (344, 515), (222, 264)),
                                ((382, 320), (449, 430), (458, 452), (320, 215))]),
    "72197330": ((760, 428), [((592, 194), None, None, None), ((574, 315), None, None, None)]),
    "dc64fc3bbba44127": ((525, 700), [((283, 362), (402, 468), (420, 489), (216, 283)),
                                       ((250, 427), (312, 506), (346, 539), (173, 324)),
                                       ((369, 350), (474, 432), (500, 467), (300, 284))]),
    "78442109a927a8fd": ((760, 570), [((229, 492), None, None, (272, 435)),
                                       ((411, 104), (294, 65), (263, 55), (527, 149)),
                                       ((428, 360), (520, 487), (533, 516), (380, 299))]),
    "896c467dacafb3af": ((760, 524), [((350, 294), None, None, None),
                                       ((396, 258), None, None, None),
                                       ((441, 250), None, None, None),
                                       ((503, 274), None, None, None)]),
    "6105fe2c402a1664": ((525, 700), [((149, 430), (98, 350), (69, 314), (124, 382)),
                                       ((217, 474), (176, 376), (156, 337), (202, 417)),
                                       ((241, 378), (213, 315), (206, 300), (265, 440)),
                                       ((309, 391), (295, 325), (290, 314), (321, 438))]),
    "9778179": ((760, 570), [((462, 60), (262, 55), (196, 37), (717, 37)),
                              ((330, 176), (26, 153), (20, 168), (81, 168)),
                              ((424, 218), (125, 220), (161, 220), (227, 216)),
                              ((481, 271), (187, 272), (104, 270), (270, 271))]),
    "2962966": ((760, 524), [((177, 274), (119, 184), (99, 154), None),
                              ((314, 269), (244, 198), (197, 155), None),
                              ((422, 246), (343, 186), (317, 171), None),
                              ((518, 233), (438, 186), (381, 153), None)]),
    "phl-66b1cafb7070d45e": ((640, 360), [(None,) * 4, (None,) * 4]),
    "phl-0eb9f729dd4a8876": ((640, 480), [(None,) * 4, ((339, 167), None, None, None)]),
}


def annotate(row, size, objects):
    if len(row["objects"]) != len(objects):
        raise ValueError(f"Instance assignments changed in {row['id']}.")
    image = raster(row)
    if abs(size[0] / size[1] - image.width / image.height) > .004:
        raise ValueError(f"Review preview has a different aspect in {row['id']}.")
    labeled = []
    for obj, parts in zip(row["objects"], objects, strict=True):
        l, t, r, b = obj["bbox"]
        targets = {}
        for name, mark in zip(PARTS, parts, strict=True):
            if mark is None:
                targets[name] = None
                continue
            x, y = mark[0] / size[0], mark[1] / size[1]
            if not (l - .03 * (r - l) <= x <= r + .03 * (r - l) and
                    t - .03 * (b - t) <= y <= b + .03 * (b - t)):
                raise ValueError(f"{name} is outside the reviewed instance of {row['id']}.")
            targets[name] = [x, y]
        labeled.append(dict(bbox=obj["bbox"], parts=targets))
    return labeled


def dependencies():
    return {str(p): digest(p) for p in (Path(__file__), WORK / "fit.json", WORK / "development.json")}


def freeze():
    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    roles = {}
    for role in ("fit", "development"):
        rows, _ = read_bank(role)
        grouped = []
        for row in rows:
            key = row["id"]
            if key in MARKS:
                grouped.append(dict(id=key, sha256=row["sha256"], group=row["group"],
                                    objects=annotate(row, *MARKS[key])))
        roles[role] = grouped
    if set(MARKS) != {r["id"] for rows in roles.values() for r in rows}:
        raise ValueError("Missing reviewed image.")
    if {r["group"] for r in roles["fit"]} & {r["group"] for r in roles["development"]}:
        raise ValueError("Shared review acquisition.")
    write_json(REFERENCE, dict(inputs=dependencies(), roles=roles,
                               notes="One visual reviewer. Part centers are approximate. Null indicates unobserved."))
    print({role: (len(rows), sum(sum(p is not None for p in o["parts"].values()) for row in rows
                               for o in row["objects"])) for role, rows in roles.items()}, flush=True)


def reviewed(role, rows):
    record = read_json(REFERENCE)
    if record["inputs"] != dependencies():
        raise ValueError("Part annotations or fixed proposal archives changed.")
    by_id = {r["id"]: r for r in record["roles"][role]}
    result = []
    for row in rows:
        if row["id"] in by_id:
            gold = by_id[row["id"]]
            if gold["sha256"] != row["sha256"] or gold["group"] != row["group"] or (
                [o["bbox"] for o in gold["objects"]] != [o["bbox"] for o in row["objects"]]
            ):
                raise ValueError("Review cannot be mapped to existing references.")
            row = {**row, "objects": gold["objects"]}
        result.append(row)
    return result


if __name__ == "__main__":
    freeze()
