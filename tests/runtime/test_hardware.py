from types import SimpleNamespace

import pytest

from medsyringe import hardware


def engine_workers():
    workers = [SimpleNamespace(options={"device": "xpu", "edge": 504}) for _ in range(3)]
    engine = SimpleNamespace(sam3=workers[0], expanded=SimpleNamespace(worker=workers[1]),
                             crop_verifier=SimpleNamespace(support_worker=workers[2]), identity={})
    return engine, workers


def test_device_request_preserves_legacy_intel_and_checks_explicit_choice(monkeypatch):
    monkeypatch.delenv("MEDSYRINGE_DEVICE", raising=False)
    monkeypatch.setenv("MEDSYRINGE_PRIMARY_DEVICE", "xpu")
    assert hardware.device_request() == "xpu"
    monkeypatch.setenv("MEDSYRINGE_DEVICE", "cpu")
    assert hardware.device_request() == "cpu"
    monkeypatch.setenv("MEDSYRINGE_DEVICE", "invalid")
    with pytest.raises(ValueError, match="must be cpu or xpu"):
        hardware.device_request()


def test_cpu_selection_and_unavailable_intel_share_the_worker_device(monkeypatch, tmp_path):
    engine, workers = engine_workers()
    hardware.configure_device(engine, tmp_path, "cpu")
    assert [w.options for w in workers] == [{"device": "cpu", "edge": 504}] * 3
    engine, workers = engine_workers()
    monkeypatch.setattr(hardware, "enable_intel",
                        lambda selected, root: selected.identity.update(primary_backend="cpu fallback"))
    hardware.configure_device(engine, tmp_path, "xpu")
    assert engine.identity["primary_backend"] == "cpu fallback"
    assert all(w.options["device"] == "cpu" for w in workers)


def test_verified_intel_stays_on_xpu_and_installs_qualified_workers(monkeypatch, tmp_path):
    engine, workers = engine_workers()
    monkeypatch.setattr(hardware, "enable_intel",
                        lambda selected, root: selected.identity.update(primary_backend="xpu"))
    activations = []
    monkeypatch.setattr(hardware, "activate", lambda selected, root: activations.append((selected, root)))
    hardware.configure_device(engine, tmp_path, "xpu")
    assert activations == [(engine, tmp_path)]
    assert all(w.options["device"] == "xpu" for w in workers)
