from types import SimpleNamespace

import pytest

from medsyringe.detection_checks import DetectionChecks
from medsyringe.local_models import WorkerError


def test_corroboration_outage_is_recorded_without_changing_the_count():
    def unavailable(_image):
        raise WorkerError("Local worker failed")

    worker = SimpleNamespace(run=unavailable)
    engine = SimpleNamespace(crop_verifier=SimpleNamespace(support_worker=worker),recovery=None)
    diagnostics = DetectionChecks(engine)
    diagnostics.begin()
    with pytest.raises(WorkerError):
        worker.run(object())
    report = dict(model={"primary_backend":"xpu"},sam3_status=None,accepted_count=1,objects=[dict(id="object-1")])
    diagnostics.attach(report)
    assert report["detection_checks"] == dict(status="degraded",primary_backend="xpu",
                                               unavailable=["SAM corroboration"],review_needed=True)
    assert report["accepted_count"] == 1 and len(report["objects"]) == 1
    diagnostics.begin()
    diagnostics.attach(report)
    assert report["detection_checks"]["status"] == "available"


def test_cpu_acceleration_fallback_does_not_mark_the_count_uncorroborated():
    diagnostics = DetectionChecks(SimpleNamespace(crop_verifier=None,recovery=None))
    diagnostics.begin()
    report = dict(model={"primary_backend":"cpu fallback"},sam3_status="completed")
    diagnostics.attach(report)
    assert report["detection_checks"]["unavailable"] == ["Intel acceleration"]
    assert report["detection_checks"]["review_needed"] is False
    report["sam3_status"] = "unavailable"
    diagnostics.attach(report)
    assert report["detection_checks"]["unavailable"] == ["SAM detection fallback","Intel acceleration"]
    assert report["detection_checks"]["review_needed"] is True
