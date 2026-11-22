"""Compare a supervised seal-front head with transverse edge contrast on public images."""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from medsyringe.data import digest, read_image, read_json, write_json
from scripts.experiments.occlusion.occlusion_detector import ART, DATA, FRESH
from scripts.experiments.parts.part_verifier_study import REFERENCES
from scripts.experiments.measurement.small_field_study import render

REF = ART / "occlusion_seal_references.json"
FROZEN = ART / "occlusion_seal_models.json"
RESULT = ART / "occlusion_seal_results.json"
# Reviewed display coordinates on the saved sheets, converted back to source pixels.
# The axes use independently visible barrel endpoints. The endpoint nearer the
# nozzle comes first. The annotation targets the liquid-facing edge of the seal.
MARKS = {
    "commons-111974556": (0, 0, (594, 505, 1230, 907), (422, 267), ((371, 154), (94, 61), (326, 140)), 18),
    "commons-113687662": (0, 1, (36, 355, 1920, 782), (880, 199), ((146, 426), (591, 425), (254, 428)), 36),
    "commons-157529342": (0, 2, (87, 186, 897, 417), (880, 251), ((747, 698), (198, 744), (661, 705)), 56),
    "commons-8200485": (0, 3, (477, 425, 558, 611), (116, 267), ((72, 1017), (26, 1138), (39, 1111)), 12),
    "clinic": (0, 4, (135, 117, 286, 621), (80, 267), ((41, 1258), (49, 1434), (42, 1263)), 16),
    "cdc-phil-11856": (1, 0, (284, 223, 433, 376), (260, 267), ((211, 63), (129, 158), (145, 138)), 7),
    "commons-154870931": (1, 1, (341, 491, 458, 539), (651, 267), ((296, 464), (544, 394), (355, 449)), 26),
    "goonan-back": (1, 3, (0, 841, 630, 1096), (660, 267), ((401, 996), (116, 1076), (373, 1006)), 13),
    "goonan-front": (1, 3, (0, 841, 630, 1096), (660, 267), ((544, 999), (205, 1106), (508, 1014)), 16),
}


def coordinates(record, point):
    _, index, (x, y, X, Y), (w, h), _, _ = record
    return [x+(point[0]-12)*(X-x)/w, y+(point[1]-23-index*300)*(Y-y)/h]


def reference():
    roles = read_json(REFERENCES)["roles"]
    known = {r["id"]: r for block in roles.values() for r in block}
    new = {r["id"]: r for r in read_json(FRESH)["rows"]}
    rows = []
    for key, mark in MARKS.items():
        source = new["dvids-8071796"] if key.startswith("goonan") else known[key]
        _, _, (x, _, X, _), (width, _), points, radius = mark
        shoulder, rear, rim = [coordinates(mark, point) for point in points]
        role = "new public" if key.startswith("goonan") else "known-author transfer" if key == "clinic" else "fit"
        rows.append(dict(id=key, image=source["image"], group=source["group"], sha256=source["sha256"],
                         role=role, shoulder=shoulder, rear=rear, rim=rim, radius=radius*(X-x)/width,
                         marked_scale=None))
    for row in rows:
        if row["id"] == "commons-157529342":
            record = MARKS[row["id"]]
            row["marked_scale"] = [
                dict(ml=amount, point=coordinates(record, point))
                for amount, point in ((10, (668, 671)), (20, (587, 678)), (30, (508, 685)),
                                      (40, (430, 692)), (50, (359, 699)), (60, (286, 706)))]
            row["scale_scope"] = "Reviewed visual graduations. No independent dispensed volume."
    low = new["dvids-9584272"]
    rows.append(dict(id=low["id"], group=low["group"], sha256=low["sha256"], image=low["image"],
                     role="new public", shoulder=None, rear=None, rim=None, radius=None, marked_scale=None,
                     note="Gloved anatomy and occlusion do not support enough visible barrel geometry."))
    write_json(REF, dict(implementation=digest(__file__), photos_sha256=digest(FRESH),
                         review="One visual reviewer. Pixel endpoints contain annotation uncertainty.", rows=rows))


def visual_support(image, row):
    if row["shoulder"] is None or row["rear"] is None or row["radius"] is None:
        return None
    front = np.asarray(row["shoulder"], dtype=float)
    rear = np.asarray(row["rear"], dtype=float)
    length = np.linalg.norm(rear-front)
    radius = row["radius"]
    if length < 16 or radius < 2.5:
        return None
    axis = (rear-front)/length
    normal = np.array([-axis[1], axis[0]])
    positions = np.linspace(-min(.025*length, radius*.18), length*.975, 241)
    centers = front + positions[:, None]*axis
    lab = cv2.cvtColor(np.asarray(image, dtype=np.float32)/255., cv2.COLOR_RGB2LAB)
    along = np.array([-1.05, -.62, -.35, -.16, .15, .32, .55, .9])
    across = np.linspace(-.62, .62, 11)
    xy = (centers[:, None, None, :] + along[None, :, None, None]*radius*axis
          + across[None, None, :, None]*radius*normal)
    map_x, map_y = np.moveaxis(xy.reshape(len(positions), -1, 2).astype(np.float32), -1, 0)
    samples = cv2.remap(lab, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    samples = samples.reshape(len(positions), len(along), len(across), 3)
    lum = samples[..., 0]
    channels = np.median(samples, axis=2)
    delta = lum[:, 2:4].mean(axis=1) - lum[:, 4:6].mean(axis=1)
    contrast = np.median(delta, axis=1)
    variation = np.std(delta, axis=1)
    cross_tracks = (delta > 5).mean(axis=1)
    change = (channels[:, 1:4].mean(axis=1) - channels[:, 4:7].mean(axis=1))[:, 1:]
    features = np.column_stack((channels[..., 0], contrast, variation, cross_tracks,
                                change, np.linalg.norm(change, axis=1), np.std(lum, axis=2)))
    target = float((np.asarray(row["rim"])-front)@axis) if row.get("rim") is not None else None
    return dict(x=features, contrast=contrast, positions=positions, centers=centers, target=target,
                axis=axis, length=length, radius=radius, front=front)


def synthetic():
    rng = np.random.default_rng(631)
    colors = ((38, 41, 46), (60, 76, 101), (42, 82, 61), (113, 111, 109), (153, 117, 99))
    liquids = (None, ((240, 180, 54), .21), ((46, 152, 231), .26), ((125, 102, 103), .34))
    for i, level in enumerate(np.linspace(0., 4.55, 36)):
        picture, _, _ = render(level, colors[i % len(colors)], liquids[i % len(liquids)],
                            unreadable=i % 6 == 0, blur=float(rng.uniform(.0, 2.0)),
                            brightness=float(rng.uniform(.65, 1.12)))
        row = dict(shoulder=[154, 135], rear=[154, 738], rim=[154, round(135+94*level)], radius=69)
        yield visual_support(picture, row)


def fit(features):
    x, y = [], []
    for frame in features:
        err = abs(frame["positions"]-frame["target"])
        yes = err <= max(1.5, .13*frame["radius"])
        no = err >= max(2.5, .44*frame["radius"])
        x.extend(frame["x"][yes | no])
        y.extend(yes[yes | no])
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=.28, class_weight="balanced", max_iter=500, random_state=631)
    clf.fit(scaler.transform(x), y)
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                weights=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]))


def scores(frame, model):
    if model is None:
        return frame["contrast"]
    logit = (frame["x"]-model["mean"])/model["scale"] @ model["weights"] + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(logit, -70., 70.)))


def pick(frame, model):
    score = scores(frame, model)
    i = int(np.argmax(score))
    location = float(frame["positions"][i])
    competitors = abs(frame["positions"]-location) > .72*frame["radius"]
    runner_up = float(score[competitors].max()) if competitors.any() else 0.
    return dict(location=location, coordinate=frame["centers"][i].tolist(), score=float(score[i]),
                margin=float(score[i])-runner_up,
                error_pixels=(abs(location-frame["target"]) if frame["target"] is not None else None))


def convert_to_volume(row, frame, decision, tolerance):
    if decision["rim_status"] != "estimated":
        return dict(status="unknown", reason="Seal front could not be localized with sufficient evidence.")
    if not row["marked_scale"]:
        return dict(status="unknown", reason="No reviewed, readable millilitre graduation scale for this crop.")
    values = np.asarray([m["ml"] for m in row["marked_scale"]])
    marks = (np.asarray([m["point"] for m in row["marked_scale"]])-frame["front"]) @ frame["axis"]
    slope, zero = np.polyfit(values, marks, 1)
    residual = float(np.max(np.abs(marks-values*slope-zero)))
    if slope < 2 or residual > .12*np.min(np.diff(values))*slope:
        return dict(status="unknown", reason="Reviewed scale markings do not support a consistent projection.")
    low, high = (decision["location"]-zero + np.array([-1, 1])*(tolerance+residual))/slope
    if low < min(values) or high > max(values):
        return dict(status="unknown", reason="The uncertainty reaches beyond readable calibrated graduations.")
    return dict(status="visual interval", low_ml=float(low), high_ml=float(high),
                calibration_residual_pixels=residual)


def supported(row):
    path = Path(row["image"])
    if digest(path) != row["sha256"]:
        raise ValueError(f"Changed landmark raster: {row['id']}.")
    return visual_support(read_image(path), row)


def learn():
    sealed = read_json(REF)
    rows = [row for row in sealed["rows"] if row["role"] == "fit"]
    support = {row["id"]: supported(row) for row in rows}
    sim = list(synthetic())
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row["id"])
    trials = {key: {} for key in ("contrast", "real", "real-plus-rendered")}
    for held in grouped.values():
        rest = [frame for key, frame in support.items() if key not in held]
        classifiers = dict(contrast=None, real=fit(rest), **{"real-plus-rendered": fit([*rest, *sim])})
        for key in held:
            frame = support[key]
            for arm, model in classifiers.items():
                data = pick(frame, model)
                tolerance = max(2., min(.22*frame["radius"], .045*frame["length"]))
                trials[arm][key] = dict(**data, tolerance_pixels=tolerance,
                                        within_tolerance=data["error_pixels"] <= tolerance,
                                        radius=frame["radius"], axis_length=frame["length"])
    trained = dict(contrast=None, real=fit(list(support.values())),
                   **{"real-plus-rendered": fit([*support.values(), *sim])})
    policies = {}
    for arm, results in trials.items():
        # A floor above every incorrect grouped-fit maximum is a conservative
        # operating-point diagnostic. Six groups cannot calibrate a guarantee.
        minimum = 8. if arm == "contrast" else .55
        wrong = [r["score"] for r in results.values() if not r["within_tolerance"]]
        floor = max(minimum, float(np.nextafter(max(wrong), np.inf)) if wrong else minimum)
        policies[arm] = dict(head=trained[arm], floor=floor, margin=2. if arm == "contrast" else .065)
    write_json(FROZEN, dict(implementation=digest(__file__), reference=digest(REF), policies=policies,
                            fit_groups=len(grouped), renders=len(sim), grouped_fit=trials))


def measure():
    locked = read_json(FROZEN)
    if locked["implementation"] != digest(__file__) or locked["reference"] != digest(REF):
        raise ValueError("The trained seal evidence changed.")
    rows = [row for row in read_json(REF)["rows"] if row["role"] != "fit"]
    outputs = {}
    for row in rows:
        frame = supported(row)
        summary = {}
        for arm, policy in locked["policies"].items():
            if frame is None:
                decision = dict(rim_status="unknown", reason="No reliable visible barrel axis or width.")
            else:
                result = pick(frame, policy["head"])
                tolerance = max(2., min(.22*frame["radius"], .045*frame["length"]))
                if result["score"] < policy["floor"] or result["margin"] < policy["margin"]:
                    decision = dict(rim_status="unknown", reason="Weak or competing transverse evidence.")
                else:
                    decision = dict(rim_status="estimated")
                decision.update(result, tolerance_pixels=tolerance,
                                within_tolerance=result["error_pixels"] <= tolerance)
            decision["volume"] = convert_to_volume(row, frame, decision, decision.get("tolerance_pixels"))
            summary[arm] = decision
        outputs[row["id"]] = dict(role=row["role"], reference_rim=row["rim"], arms=summary)
    write_json(RESULT, dict(freeze_sha256=digest(FROZEN), cases=outputs,
                            scope="Landmark diagnostic with reviewed axes. No public reserve scale was traceable in mL."))
    for key, row in outputs.items():
        print(key, [(arm, info["rim_status"], info.get("error_pixels")) for arm, info in row["arms"].items()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reference", "learn", "measure"))
    {"reference": reference, "learn": learn, "measure": measure}[parser.parse_args().stage]()
