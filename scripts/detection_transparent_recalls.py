"""Find evidence that separates hand-held transparent syringes from lookalikes."""
import argparse
from itertools import product
from time import perf_counter

import numpy as np
import torch
from PIL import ImageOps

from medsyringe.appearance_runtime import RELEASE
from medsyringe.attributes import crop
from medsyringe.data import digest, read_json, write_json
from medsyringe.evaluate import evaluate
from medsyringe.recall_guard import compare
from scripts.appearance_text_encoder import TextMarginEncoder
from scripts.detection_context_study import CACHE as CONTEXT, inputs as context
from scripts.detection_crop_semantics import PADDINGS, SYRINGES
from scripts.detection_hillclimb_experiment import ART, DATA, cache, inputs, loaded
from scripts.detection_real_box_transfer import benchmark_sources
from scripts.detection_semantics_audit import apply_rules
from scripts.expanded_detection_study import short
from scripts.precision_study import source
from scripts.rfdetr_verifier import expanded
from scripts.sam3_study import configure_cpu

CACHE = DATA / "proposals/transparent-verifier-development.json"
OUTPUT = ART / "transparent_verifier_development.json"
PUBLIC = "public-development"
POSITIVE = (
    "a clear plastic medical syringe with a transparent barrel and visible plunger",
    "a hand holding a small transparent injection syringe with graduation marks",
    "a close-up of a clear hypodermic syringe pinched between fingers",
    "a graduated medical syringe with a barrel, flange and thumb rest",
    "a transparent syringe with a capped or exposed needle",
)
NEGATIVE = (
    "a hand holding a laboratory pipette or a plastic dropper",
    "a clear tube or glass vial with no plunger",
    "a pen or an insulin injector pen without a syringe barrel",
    "a hand holding a non-syringe implement",
    "loose clear tubing, packaging or a loose needle without a syringe",
)


class ExplicitAppearance(TextMarginEncoder):
    def __init__(self):
        super().__init__()
        self.original_text = self.text
        texts = [*POSITIVE, *NEGATIVE]
        prepared = self.processor(text=texts, padding="max_length", truncation=True, return_tensors="pt")
        with torch.inference_mode():
            self.explicit = torch.nn.functional.normalize(self.model.get_text_features(**prepared))

    @torch.inference_mode()
    def margins(self, patches):
        resized = [ImageOps.pad(picture, (224, 224), color=(124, 116, 104)) for picture in patches]
        images = self.processor(images=resized, return_tensors="pt")
        features = torch.nn.functional.normalize(self.model.get_image_features(**images))
        original = features @ self.original_text.T
        explicit = features @ self.explicit.T
        target, distractor = original[:, :len(SYRINGES)], original[:, len(SYRINGES):]
        new_target, new_distractor = explicit[:, :len(POSITIVE)], explicit[:, len(POSITIVE):]
        old_best = target.max(dim=1).values - distractor.max(dim=1).values
        new_best = torch.cat((new_target, target), dim=1).max(dim=1).values - torch.cat(
            (new_distractor, distractor[:, 1:]), dim=1).max(dim=1).values
        explicitly_transparent = new_target.max(dim=1).values - new_distractor.max(dim=1).values
        original_summary = torch.stack((target.max(dim=1).values, distractor.max(dim=1).values,
                                        target.mean(dim=1), distractor.mean(dim=1)), dim=1)
        return original_summary.numpy(), torch.stack((old_best, new_best, explicitly_transparent), dim=1).numpy()


def on_image(encoder, image, detections):
    if not detections:
        return [], []
    patches = [crop(image, expanded(p["bbox"], padding)) for padding in PADDINGS for p in detections]
    old, evidence = encoder.margins(patches)
    evidence = evidence.reshape(len(PADDINGS), len(detections), -1)
    return old.tolist(), [{"bbox": p["bbox"], "margins": evidence[:, i, :].tolist()}
                          for i, p in enumerate(detections)]


def data():
    fit, _, _ = benchmark_sources()
    context_result = read_json(CONTEXT)["frames"]
    public = loaded(PUBLIC, "baseline")
    return fit, dict(public=(inputs(PUBLIC), public), transparent=(context(), context_result))


def key(fit, groups):
    boxes = ("objects", "lookalikes")
    return dict(release_sha256=digest(RELEASE), context_sha256=digest(CONTEXT),
                public_sha256=digest(cache(PUBLIC, "baseline")), code_sha256=digest(__file__),
                fit={r["id"]: [r["sha256"], [[o["bbox"] for o in r.get(k, [])] for k in boxes]] for r in fit},
                dev={k: {r["id"]: [r["sha256"], [o["bbox"] for o in r["objects"]]] for r in rs}
                     for k, (rs, _) in groups.items()})


def originals(frame, is_live):
    if is_live:
        return [dict(bbox=p["bbox"], score=p["original_score"]) for p in frame["ranked"] if "original_score" in p]
    return [p for p in frame["ranked"] if p["score"] >= frame["floor"]]


def collect():
    configure_cpu()
    fit, groups = data()
    binding = key(fit, groups)
    saved = read_json(CACHE) if CACHE.exists() else dict(binding=binding, fit={}, development={})
    if saved["binding"] != binding:
        raise ValueError("Transparent crop verification inputs changed.")
    model = ExplicitAppearance()
    for row in fit:
        name = row["id"]
        if name in saved["fit"]:
            continue
        pos = row["objects"]
        neg = row.get("lookalikes", [])
        proposals = [*pos, *neg]
        _, scores = on_image(model, source(row), proposals)
        saved["fit"][name] = [dict(record, positive=i < len(pos)) for i, record in enumerate(scores)]
        write_json(CACHE, saved)
    release = read_json(RELEASE)
    for scope, (rows, frames) in groups.items():
        stored = saved["development"].setdefault(scope, {})
        for row in rows:
            name = row["id"]
            if name in stored:
                continue
            frame = frames[name]
            live = scope == "transparent"
            previous = [p for p in frame["ranked"] if p["score"] >= frame["floor"]]
            proposals = originals(frame, live)
            mark = perf_counter()
            old, scored = on_image(model, source(row), proposals)
            if proposals and not live:
                reconstructed, _ = apply_rules(frame, dict(candidates=proposals, values={"prompt": old}),
                                               release["components"])
                previous = [p for p in reconstructed if p["score"] >= frame["floor"]]
            retained = {tuple(p["bbox"]) for p in previous}
            demoted = [{**p, **details} for p, details in zip(proposals, scored, strict=True)
                       if tuple(p["bbox"]) not in retained]
            stored[name] = dict(previous=previous, demoted=demoted, elapsed_seconds=perf_counter() - mark)
            write_json(CACHE, saved)
            print("Transparent prompt comparison", scope, len(stored), "/", len(rows), flush=True)


def evidence(record, formulation, reduction):
    values = np.asarray(record["margins"])[:, formulation]
    return float(values.min() if reduction == "both" else values.max())


def promote(frame, choice):
    formulation, reduction, margin, original_floor = choice
    previous = frame["previous"]
    demoted = frame["demoted"]
    if previous or len(demoted) != 1:
        return previous
    record = demoted[0]
    if record["score"] < original_floor or evidence(record, formulation, reduction) < margin:
        return previous
    return [dict(bbox=record["bbox"], score=record["score"])]


def measure():
    if OUTPUT.exists():
        raise ValueError("The transparent-verifier comparison has already been measured.")
    fit, context_groups = data()
    saved = read_json(CACHE)
    groups = saved["development"]
    if saved["binding"] != key(fit, context_groups):
        raise ValueError("Verification cache does not match the registered inputs.")
    if any(set(groups[k]) != {r["id"] for r in rows} for k, (rows, _) in context_groups.items()):
        raise ValueError("The comparison requires both complete development cohorts.")
    base = {k: evaluate(rows, {identifier: frame["previous"] for identifier, frame in groups[k].items()}, 0.)
            for k, (rows, _) in context_groups.items()}
    examples = [p for patches in saved["fit"].values() for p in patches]
    experiment = tuple(product((0, 1, 2), ("both", "one"), (.0, .005, .01, .02, .03, .04), (.70, .78, .86)))
    trials = []
    for option in experiment:
        formulation, reduction, floor, _ = option
        fit_passed = [p for p in examples if evidence(p, formulation, reduction) >= floor]
        fit_fp = sum(not p["positive"] for p in fit_passed)
        fit_tp = sum(p["positive"] for p in fit_passed)
        results = {}
        for k, (rows, _) in context_groups.items():
            prediction = {identifier: promote(f, option) for identifier, f in groups[k].items()}
            outcome = evaluate(rows, prediction, 0.)
            guard = compare(base[k], outcome)
            results[k] = dict(metrics=short(outcome), guard=guard,
                              safe=guard["passed"] and outcome["fp"] <= base[k]["fp"])
        if all(v["safe"] for v in results.values()):
            trials.append(dict(options=option, fit_fp=fit_fp, fit_tp=fit_tp, development=results))
    trials.sort(key=lambda r: (-r["fit_fp"],
                               sum(v["guard"]["recovered_matches"] for v in r["development"].values()),
                               r["fit_tp"]), reverse=True)
    write_json(OUTPUT, dict(fit_crops=len(examples), fit_target_crops=sum(p["positive"] for p in examples),
                            base={k: short(v) for k, v in base.items()}, trials=len(experiment), guarded=len(trials),
                            leading=trials[:15], evidence_sha256=digest(CACHE),
                            prompt_arm_meanings=["current phrases", "distinct contextual and existing object phrases",
                                                 "transparent object and lookalike phrases only"],
                            protocol="Only reconsider an empty result with exactly one vetoed detector box. "
                                     "Compare fixed prompt banks, amount of agreement across two paddings "
                                     "and the original detector score. Select on fit crops and development only."))
    print("Transparent verification", len(trials), "/", len(experiment), trials[:1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "measure"))
    (collect if parser.parse_args().action == "collect" else measure)()
