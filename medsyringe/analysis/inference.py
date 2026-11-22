"""Image-to-observation inference independent of the API and browser."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

from joblib import load
from PIL import ImageOps

from medsyringe.detection.additive import additions
from medsyringe.measurement.attributes import crop, describe, read_inscriptions
from medsyringe.detection.auxiliary_detector import MANIFEST as RECOVERY_MANIFEST, load_verified_recovery
from medsyringe.data import digest, read_image, read_json, write_json
from medsyringe.detection.detectors import Grounder, model_hashes, suppress
from medsyringe.measurement.fill import estimate as estimate_fill, liquid_box, unavailable as fill_unavailable
from medsyringe.measurement.fill_orientation import upright_liquid_for_fill
from medsyringe.detection import proposal_learning, refinement, verification
from medsyringe.detection.recall_guard import release_allowed
from medsyringe.runtime.local_models import WorkerError
from medsyringe.runtime.pipeline_extensions import (LIQUID_RELEASE, MARKING_RELEASE, SAM_RELEASE,
                                  verified_config, worker_for)
from medsyringe.runtime.expanded_runtime import ExpandedRuntime

FILL_RELEASE = Path("artifacts/fill/release.json")
CROP_ANALYSIS_LIMIT = 10
COUNT_ONLY_REASON = "Only detection and counting were requested."
DETECTION_EVIDENCE = (
    "support", "view_count", "verification_score", "source",
    "auxiliary_score", "support_channels", "branch_score",
)


def unread_inscriptions(status):
    return dict(status=status, candidates=[], capacity=None, apparent_volume=None)


def observation_object(index, proposal, attributes, texts, liquid, fill):
    return dict(
        id=f"object-{index}",
        bbox=proposal["bbox"],
        score=proposal["score"],
        detection_evidence={key: proposal[key] for key in DETECTION_EVIDENCE if key in proposal},
        scale=attributes["scale"]["value"],
        color=attributes["color"]["value"],
        evidence=attributes,
        inscriptions=texts,
        liquid_appearance=liquid,
        fill_level=fill,
        origin="model",
    )


class Engine:
    def __init__(self, root):
        self.overlap_crop_checks = True
        frozen_path = root / "artifacts/freeze.json"
        config_path = root / "artifacts/detector_release.json"
        frozen = read_json(frozen_path)
        self.config = read_json(config_path)
        if (self.config["original_freeze_sha256"] != digest(frozen_path) or
            self.config["refinement_sha256"] != digest(refinement.__file__) or
            self.config["proposal_learning_sha256"] != digest(proposal_learning.__file__)):
            raise ValueError("Detection pipeline differs from the validated configuration.")
        self.attributes_path = root / "artifacts/attribute_model.json"
        self.attributes = read_json(self.attributes_path)
        backbone = frozen["backbone"]
        if model_hashes(root, backbone) != frozen["bundle_sha256"]:
            raise ValueError("Local model files differ from the frozen experiment.")
        self.heads = []
        if self.config["heads"]:
            proposal_heads = root / "artifacts/proposal_heads.joblib"
            if digest(proposal_heads) != self.config["proposal_heads_sha256"]:
                raise ValueError("Proposal checkpoint differs from the validated configuration.")
            self.heads = load(proposal_heads)
        self.detector = Grounder(root, backbone)
        self.identity = dict(name=frozen["winner"], revision=frozen["revision"],
                             pipeline=self.config["id"], selection_sha256=digest(config_path),
                             attributes_sha256=digest(self.attributes_path))
        precision_path = root / "artifacts/precision_release.json"
        self.verifier = None
        if precision_path.exists():
            release = read_json(precision_path)
            if (release["base_sha256"] != digest(config_path) or
                release["code_sha256"] != digest(verification.__file__)):
                raise ValueError("Crop verifier differs from the validated release.")
            assessment = root / "artifacts/precision_holdout.json"
            if (release.get("assessment_sha256") != digest(assessment) or
                not release_allowed(read_json(assessment))):
                raise ValueError("Crop verifier lacks a passing instance-level recall assessment.")
            self.verifier = verification.CropVerifier(root, release)
            self.identity["verification_sha256"] = digest(precision_path)
        self.recovery = load_verified_recovery(root, config_path)
        if self.recovery is not None:
            if self.verifier is not None:
                raise ValueError("The two verification releases lack joint validation.")
            self.identity["pipeline"] += "+rfdetr-verified"
            self.identity["recovery_sha256"] = digest(root / RECOVERY_MANIFEST)
        self.sam3 = worker_for(root,SAM_RELEASE,"sam3")
        self.markings = worker_for(root,MARKING_RELEASE,"ocr")
        self.liquid = bool(verified_config(root,LIQUID_RELEASE))
        if self.sam3:
            self.identity["pipeline"] += "+sam3-fallback"
            self.identity["sam3_sha256"] = digest(root/SAM_RELEASE)
        if self.markings:
            self.identity["marking_sha256"] = digest(root/MARKING_RELEASE)
        if self.liquid:
            self.identity["liquid_sha256"] = digest(root/LIQUID_RELEASE)
        self.expanded = ExpandedRuntime(root, self)
        self.fill_settings = verified_config(root, FILL_RELEASE)
        if self.fill_settings:
            self.identity["fill_sha256"] = digest(root / FILL_RELEASE)

    def rank_detections(self, image):
        proposals = {view: refinement.view_proposals(self.detector, image, view)
                     for view in self.config["views"]}
        ranked = refinement.fuse(proposals, strategy=self.config["strategy"],
                                 overlap=self.config["overlap"])
        if self.heads:
            descriptors = proposal_learning.describe_proposals(image, ranked)
            ranked = suppress(proposal_learning.adjust_scores(
                ranked, descriptors, self.heads, self.config["weight"]
            ), self.config["overlap"])
        return ranked

    def close(self):
        for worker in (self.sam3, self.markings, self.expanded.worker):
            if worker is not None:
                worker.close()

    def baseline_detections(self, image, ranked):
        operating_score = self.config["cutoff"]
        if self.recovery is not None:
            ranked = self.recovery.apply(image, ranked, operating_score)
        supplemental = None
        if self.sam3 and not verification.eligible(ranked,operating_score):
            try:
                proposals = self.sam3.run(image)
                ranked = additions(ranked,proposals,operating_score,support=None,
                                   aux_cutoff=self.sam3.options["cutoff"],iou=.30)
                for proposal in ranked:
                    if proposal.get("source")=="auxiliary":
                        proposal["source"]="sam3"
                supplemental="completed"
            except WorkerError:
                supplemental="unavailable"
        return ranked, operating_score, supplemental

    def measure_fill(self, region, texts, liquid):
        settings = self.fill_settings["settings"]
        if not liquid_box(liquid) and self.liquid and self.sam3:
            try:
                upright = upright_liquid_for_fill(region, texts, settings, self.expanded.measure_liquid)
            except WorkerError:
                upright = None
            if upright:
                result = estimate_fill(region, texts, upright, settings)
                result["appearance_rotation"] = 180
                return result
        return estimate_fill(region, texts, liquid, settings)

    def crop_markings(self, region):
        try:
            return self.expanded.read_markings(ImageOps.contain(region, (1900, 1900)))
        except WorkerError:
            return unread_inscriptions("unavailable")

    def crop_liquid(self, region):
        try:
            return self.expanded.measure_liquid(ImageOps.contain(region, (420, 420)))
        except WorkerError:
            return dict(status="unavailable", reason="The liquid measurement could not run.",
                        lightness=None, opacity=None)

    def count_objects(self, accepted):
        objects = []
        for index, proposal in enumerate(accepted, 1):
            attributes = {
                field: dict(value=None, reason=COUNT_ONLY_REASON)
                for field in ("scale", "color")
            }
            fill = fill_unavailable(COUNT_ONLY_REASON) if self.fill_settings else None
            objects.append(observation_object(
                index, proposal, attributes, unread_inscriptions("not requested"), None, fill,
            ))
        return objects

    def crop_evidence(self, region, index, ocr, measure_liquid, executor):
        attributes = describe(region, self.attributes)
        texts = unread_inscriptions("per-image OCR limit" if ocr else "not requested")
        read_markings = ocr and index <= CROP_ANALYSIS_LIMIT
        marking_task = None
        if read_markings:
            if executor is not None:
                marking_task = executor.submit(self.crop_markings, region)
            elif self.markings:
                texts = self.crop_markings(region)
            else:
                texts = read_inscriptions(region)
        liquid = self.crop_liquid(region) if measure_liquid and index <= CROP_ANALYSIS_LIMIT else None

        if marking_task is not None:
            texts = marking_task.result()
        if read_markings and self.markings:
            scale = texts.get("scale")
            attributes["scale"] = (
                dict(**scale, reason="Supported by visible markings") if scale
                else dict(value=None, reason="Scale markings are unreadable or ambiguous")
            )

        fill = None
        if self.fill_settings:
            fill = (
                self.measure_fill(region, texts, liquid) if ocr
                else fill_unavailable("Enable visible marking extraction to estimate fill.")
            )
        return attributes, texts, liquid, fill

    def describe_objects(self, image, accepted, ocr, count_only=False):
        if count_only:
            return self.count_objects(accepted)

        measure_liquid = bool(self.liquid and self.sam3)
        overlap = (
            accepted and ocr and self.markings and measure_liquid
            and self.overlap_crop_checks and self.identity.get("primary_backend") == "xpu"
        )
        scheduler = ThreadPoolExecutor(max_workers=1) if overlap else nullcontext()
        with scheduler as executor:
            return [
                observation_object(
                    index, proposal, *self.crop_evidence(
                        crop(image, proposal["bbox"]), index, ocr, measure_liquid, executor,
                    ),
                )
                for index, proposal in enumerate(accepted, 1)
            ]

    def analyze(self, image, ocr=False, *, count_only=False):
        started = perf_counter()
        ranked, operating_score, supplemental = self.expanded.detect(image)
        accepted = verification.eligible(ranked, operating_score)
        rejected = []
        if self.verifier is not None:
            accepted, rejected = self.verifier(image, accepted)
        review = [{**p, "review_reason": "Crop verification"} for p in rejected]
        review.extend(p for p in ranked if p["score"] < operating_score)
        objects = self.describe_objects(image, accepted, ocr, count_only)
        report = dict(model=self.identity, threshold=operating_score, objects=objects,
                    dimensions=[image.width, image.height], accepted_count=len(objects),
                    below_threshold_candidates=review[:5],sam3_status=supplemental,
                    elapsed_ms=1000*(perf_counter()-started))
        if count_only:
            report["analysis_mode"] = "count only"
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--ocr", action="store_true")
    args = parser.parse_args()
    engine = Engine(args.root)
    try:
        record = engine.analyze(read_image(args.image), args.ocr)
    finally:
        engine.close()
    record["source_sha256"] = digest(args.image)
    write_json(args.output, record)
    print(f"Saved {len(record['objects'])} proposed objects to {args.output}")


if __name__ == "__main__":
    main()
