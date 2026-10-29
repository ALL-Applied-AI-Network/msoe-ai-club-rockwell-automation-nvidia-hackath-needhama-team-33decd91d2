#!/usr/bin/env python3
"""Persistent JSON-lines LocateAnything worker used by the Isaac adapter."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inspection.locateanything_runtime import LocateAnythingRuntime  # noqa: E402


def main():
    runtime = LocateAnythingRuntime(dtype="float16")
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            image = Image.open(request["image"]).convert("RGB")
            result = runtime.predict(
                image, request["class_name"], request["expected_pixel"])
            response = {"id": request.get("id"), "ok": True, **result}
        except BaseException as exc:
            response = {"id": request.get("id"), "ok": False,
                        "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
