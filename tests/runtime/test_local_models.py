import subprocess
import sys

import pytest
from PIL import Image

from medsyringe.runtime.local_models import ImageWorker, WorkerError


def test_worker_restarts_after_a_malformed_reply(monkeypatch, tmp_path):
    popen = subprocess.Popen
    replies = iter(("{}", '{"result": "recovered"}', '{"result": "reopened"}'))
    children = []

    def start_child(_command, **options):
        reply = next(replies)
        code = f"import sys\nfor line in sys.stdin:\n    print({reply!r}, flush=True)"
        child = popen([sys.executable, "-u", "-c", code], **options)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", start_child)
    worker = ImageWorker(tmp_path, sys.executable, "test", {})
    image = Image.new("RGB", (1, 1))
    try:
        with pytest.raises(WorkerError, match="invalid response"):
            worker.run(image)
        assert worker.process is None
        assert children[0].poll() is not None
        assert children[0].stdin.closed and children[0].stdout.closed

        assert worker.run(image) == "recovered"
        worker.close()
        worker.close()
        assert worker.run(image) == "reopened"
    finally:
        worker.close()

    assert all(child.poll() is not None for child in children)
