"""Activate independently verified expanded detector and attribute changes."""
from pathlib import Path

from PIL import ImageOps

from .data import digest, read_json
from .local_models import ImageWorker, WorkerError
from .marking_views import agreement, contrast_view, supplement
from .markings_guard import confirm_untyped_capacity
from .pipeline_extensions import verified_config

RELEASE = Path("artifacts/expanded/release.json")
WORKER_ENTRY = Path(__file__).resolve().parents[1] / "scripts/expanded_worker_xpu.sh"


class ExpandedRuntime:
    def __init__(self, root, engine):
        self.engine = engine
        config = verified_config(root, RELEASE)
        self.features = config["features"] if config else {}
        self.detector = self.features.get("detection")
        self.reader = self.features.get("markings")
        self.liquid = self.features.get("liquid")
        self.worker = None
        self.trees = None
        if self.detector or self.liquid:
            worker_options = dict(edge=engine.sam3.options["edge"], device=engine.sam3.options["device"],
                                  liquid=self.liquid)
            self.worker = ImageWorker(root, WORKER_ENTRY, "sam3", worker_options)
        if self.detector and self.detector.get("tree_file"):
            from joblib import load

            self.trees = {self.detector["settings"]["method"]: load(root / self.detector["tree_file"])}
        if config:
            engine.identity["pipeline"] += "+expanded"
            engine.identity["expanded_sha256"] = digest(root / RELEASE)

    def detect(self, image):
        engine = self.engine
        if self.detector is None:
            return engine.baseline_detections(image, engine.rank_detections(image))
        settings = self.detector["settings"]
        all_sources = settings["channels"] == "all"
        channels = ("primary", "rf", "sam") if all_sources else settings["channels"]
        guarded = settings["preserve"] or self.detector.get("recall_guard") is not None
        need_primary = guarded or "primary" in channels
        primary = engine.rank_detections(image) if need_primary else []
        baseline = engine.baseline_detections(image, primary) if guarded else ([], 0., None)
        try:
            rf = engine.recovery._region(image) if "rf" in channels else []
            sam = self.worker.run(ImageOps.contain(image, (1500, 1500)), operation="detect") if "sam" in channels else []
        except WorkerError:
            if guarded:
                return baseline
            return engine.baseline_detections(image, primary if need_primary else engine.rank_detections(image))
        from scripts.expanded_detection_study import apply
        from .hybrid_detection import candidates

        branches = dict(primary=primary, rf=rf, sam=sam)
        frame = dict(prior=baseline[0], branches=branches, proposals=candidates(branches))
        if self.detector.get("recall_guard") is not None:
            from scripts.expanded_recall_selection import hybrid

            policy = {"options": settings, "guard": self.detector["recall_guard"]}
            return hybrid(frame, policy, self.detector["heads"], baseline[1]), settings["cutoff"], "expanded hybrid"
        return apply(frame, settings, self.detector["heads"], self.trees), settings["cutoff"], "expanded hybrid"

    def read_markings(self, crop):
        base = self.engine.markings.run(crop)
        if self.reader != "both" or (base.get("capacity") and base.get("scale")):
            return base
        try:
            local = self.engine.markings.run(contrast_view(crop, "clahe"))
            sharpened = self.engine.markings.run(contrast_view(crop, "enhanced"))
        except WorkerError:
            return confirm_untyped_capacity(base, {})
        validation = agreement(local, sharpened)
        return confirm_untyped_capacity(supplement(base, validation), validation)

    def measure_liquid(self, crop):
        if self.liquid is not None:
            # PNG carries its ICC profile through the isolated local worker.
            try:
                return self.worker.run(crop, operation="liquid")
            except WorkerError:
                pass
        return self.engine.sam3.run(crop, operation="liquid")
