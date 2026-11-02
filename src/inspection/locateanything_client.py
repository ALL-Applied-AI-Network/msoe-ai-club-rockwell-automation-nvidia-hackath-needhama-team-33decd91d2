"""Asynchronous client for the isolated Isaac/LocateAnything GPU worker."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading


class LocateAnythingProcessClient:
    def __init__(self, python, script, *, cwd, cuda_device="1", hf_home=None):
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        if hf_home is not None:
            env["HF_HOME"] = str(hf_home)
        self.process = subprocess.Popen(
            [str(python), str(script)], cwd=str(cwd), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wrist-vlm")
        self._next_id = 0
        # Remote-code imports can print informational lines.  Read until the
        # server's explicit JSON readiness record rather than assuming line 1.
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("LocateAnything worker exited during startup")
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("ready") is True:
                break

    def _request(self, image, class_name, expected_pixel):
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            payload = {"id": request_id, "image": str(Path(image).resolve()),
                       "class_name": class_name,
                       "expected_pixel": list(map(float, expected_pixel))}
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("LocateAnything worker exited")
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if response.get("id") == request_id:
                    break
            if not response.get("ok"):
                raise RuntimeError(
                    f"LocateAnything {response.get('error_type')}: {response.get('error')}")
            return response

    def submit(self, image, class_name, expected_pixel):
        return self._pool.submit(self._request, image, class_name, expected_pixel)

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
