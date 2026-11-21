"""Private, persistent local model processes for isolated inference environments."""
import base64
import io
import json
import os
from pathlib import Path
from select import select
import subprocess
from threading import Lock
from time import monotonic
import weakref

from PIL import Image


class WorkerError(RuntimeError):
    """An optional local model failed to return an observation."""


class ImageWorker:
    def __init__(self, root, python, mode, options):
        self.root=Path(root).resolve()
        self.python=self.root/python
        self.mode=mode
        self.options=options
        self.lock=Lock()
        self.process=None
        self._finalizer=None
        self.buffer=b""

    def _start(self):
        logs=self.root/"data/workspace"
        logs.mkdir(parents=True,exist_ok=True)
        package=Path(__file__).resolve().parent.parent
        env={**os.environ,"HF_HUB_OFFLINE":"1","PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK":"True",
             "OMP_NUM_THREADS":"1" if self.mode=="ocr" else "4",
             "MKL_NUM_THREADS":"1" if self.mode=="ocr" else "4","PYTHONPATH":str(package)}
        with (logs/f"{self.mode}-worker.log").open("ab") as errors:
            self.process=subprocess.Popen([str(self.python),"-u","-m","scripts.local_model_worker",self.mode],
                                          cwd=self.root,env=env,stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE,stderr=errors)
        self._finalizer=weakref.finalize(self,self.stop,self.process)
        self.buffer=b""

    @staticmethod
    def stop(child):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        child.stdin.close()
        child.stdout.close()

    def _reply(self):
        deadline=monotonic()+90
        output=self.process.stdout.fileno()
        while b"\n" not in self.buffer:
            remaining=deadline-monotonic()
            if remaining<=0 or not select([output],[],[],remaining)[0]:
                raise WorkerError("The local model timed out.")
            chunk=os.read(output,65536)
            if not chunk:
                raise WorkerError("The local model stopped.")
            self.buffer+=chunk
        data,self.buffer=self.buffer.split(b"\n",1)
        response=json.loads(data)
        if "error" in response:
            raise WorkerError(response["error"])
        return response["result"]

    def run(self,image,operation=None):
        data=io.BytesIO()
        image.save(data,format="PNG",compress_level=1)
        request=json.dumps(dict(image=base64.b64encode(data.getvalue()).decode(),options=self.options,
                                operation=operation)).encode()+b"\n"
        with self.lock:
            try:
                if self.process is None or self.process.poll() is not None:
                    if self._finalizer is not None:
                        self._finalizer()
                    self._start()
                self.process.stdin.write(request)
                self.process.stdin.flush()
                return self._reply()
            except (WorkerError,OSError,ValueError) as error:
                if self._finalizer is not None:
                    self._finalizer()
                self.process=None
                raise WorkerError(str(error)) from error


def decode_picture(value):
    with Image.open(io.BytesIO(base64.b64decode(value,validate=True))) as image:
        return image.convert("RGB")
