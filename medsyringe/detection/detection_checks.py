"""Make detector fallback and corroboration outages visible in observations."""
from functools import wraps
from threading import local


class DetectionChecks:
    def __init__(self, engine):
        self.state = local()
        appearance = getattr(engine, "appearance", None)
        if appearance is not None:
            appearance.worker.run = self.observe(appearance.worker.run, "Syringe appearance verification")
        if engine.crop_verifier is not None:
            worker = engine.crop_verifier.support_worker
            worker.run = self.observe(worker.run, "SAM corroboration")
            if engine.recovery is not None:
                engine.recovery._region = self.observe(engine.recovery._region, "RF-DETR corroboration")

    def begin(self):
        self.state.failures = set()

    def observe(self, inference, name):
        @wraps(inference)
        def recorded(*args, **kwargs):
            try:
                return inference(*args, **kwargs)
            except RuntimeError:
                if hasattr(self.state, "failures"):
                    self.state.failures.add(name)
                raise
        return recorded

    def attach(self, report):
        failures = sorted(self.state.failures)
        if report.get("sam3_status") == "unavailable":
            failures.append("SAM detection fallback")
        primary_backend = report["model"].get("primary_backend", "cpu")
        if primary_backend == "cpu fallback":
            failures.append("Intel acceleration")
        incomplete = any(name != "Intel acceleration" for name in failures)
        report["detection_checks"] = {
            "status": "degraded" if failures else "available",
            "primary_backend": primary_backend,
            "unavailable": failures,
            "review_needed": incomplete,
        }
