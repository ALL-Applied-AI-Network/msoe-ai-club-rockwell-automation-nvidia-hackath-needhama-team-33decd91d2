#!/usr/bin/env python3
"""Initialize and exercise the physical deployment perception stack.

This program deliberately has no robot-client import. It keeps YOLO resident
while loading LocateAnything so the smoke test catches the one-T4 memory and
dependency failures that matter before a physical-cell run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPECTED_CLASSES = {0: "good_cap", 1: "crooked_cap", 2: "missing_cap", 3: "table_cap"}


def gpu_memory():
    free, total = torch.cuda.mem_get_info()
    return {
        "device_used_gib": (total - free) / 2**30,
        "device_free_gib": free / 2**30,
        "device_total_gib": total / 2**30,
        "pytorch_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "shared/yolo26x/best.pt")
    parser.add_argument("--reference-image", type=Path,
                        default=ROOT / "compliant_remediation_test/perception/current_scene_tablecam.png")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "deployment_smoke_report.json"
    report = {
        "passed": False,
        "robot_connection_attempted": False,
        "checkpoint": str(args.checkpoint),
        "reference_image": str(args.reference_image),
        "expected_classes": EXPECTED_CLASSES,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside the Slurm allocation")
        device_index = int(args.device)
        torch.cuda.set_device(device_index)
        device_name = torch.cuda.get_device_name(device_index)
        if "T4" not in device_name.upper():
            raise RuntimeError(f"expected a Tesla T4, received {device_name}")
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"YOLO checkpoint is missing: {args.checkpoint}")
        if not args.reference_image.is_file():
            raise FileNotFoundError(f"reference image is missing: {args.reference_image}")
        image = cv2.imread(str(args.reference_image))
        if image is None or image.shape[:2] != (1080, 1920):
            raise RuntimeError(
                f"reference image must be readable native 1920x1080; got "
                f"{None if image is None else image.shape[:2][::-1]}")

        from ultralytics import YOLO
        from src.inspection.locateanything_runtime import LocateAnythingRuntime

        started = time.perf_counter()
        yolo = YOLO(str(args.checkpoint))
        yolo_load_s = time.perf_counter() - started
        names = {int(key): str(value) for key, value in yolo.names.items()}
        if names != EXPECTED_CLASSES:
            raise RuntimeError(f"unexpected YOLO class map: {names}")

        started = time.perf_counter()
        first = yolo.predict(
            image, conf=0.25, imgsz=1280, device=args.device,
            verbose=False)[0]
        yolo_cold_s = time.perf_counter() - started
        detections = []
        for box, confidence, class_id in zip(
                first.boxes.xyxy.detach().cpu().numpy(),
                first.boxes.conf.detach().cpu().numpy(),
                first.boxes.cls.detach().cpu().numpy()):
            detections.append({
                "class_id": int(class_id),
                "class_name": names[int(class_id)],
                "confidence": float(confidence),
                "bbox_xyxy": np.asarray(box, dtype=float).tolist(),
            })
        cap_detections = [item for item in detections
                          if item["class_name"] in ("table_cap", "good_cap", "crooked_cap")]
        if not cap_detections:
            raise RuntimeError("YOLO produced no cap target on the checked-in reference image")
        target = max(cap_detections, key=lambda item: (
            item["class_name"] == "table_cap", item["confidence"]))
        x0, y0, x1, y1 = target["bbox_xyxy"]
        expected_pixel = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]

        before_vlm = gpu_memory()
        started = time.perf_counter()
        vlm = LocateAnythingRuntime(dtype="float16")
        vlm_load_s = time.perf_counter() - started
        after_vlm_load = gpu_memory()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vlm_result = vlm.predict(
            rgb, "table_cap", expected_pixel,
            prompt="the yellow vial cap nearest the center of this image")
        after_vlm_inference = gpu_memory()

        started = time.perf_counter()
        yolo.predict(image, conf=0.25, imgsz=1280, device=args.device,
                     verbose=False)
        yolo_warm_after_vlm_s = time.perf_counter() - started

        annotated = image.copy()
        bx = [int(round(value)) for value in target["bbox_xyxy"]]
        cv2.rectangle(annotated, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 0), 3)
        vx = [int(round(value)) for value in vlm_result["bbox_xyxy"]]
        cv2.rectangle(annotated, (vx[0], vx[1]), (vx[2], vx[3]), (0, 255, 0), 3)
        cv2.putText(annotated, "blue=YOLO anchor  green=LocateAnything",
                    (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 3, cv2.LINE_AA)
        annotated_path = args.out / "deployment_smoke_annotated.png"
        if not cv2.imwrite(str(annotated_path), annotated):
            raise RuntimeError(f"could not write {annotated_path}")

        report.update({
            "passed": True,
            "gpu": {"index": device_index, "name": device_name},
            "versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
            },
            "yolo": {
                "class_map": names,
                "load_s": yolo_load_s,
                "cold_inference_s": yolo_cold_s,
                "warm_inference_after_vlm_s": yolo_warm_after_vlm_s,
                "detection_count": len(detections),
                "anchor": target,
            },
            "locateanything": {
                "model": "nvidia/LocateAnything-3B",
                "dtype": "float16",
                "load_s": vlm_load_s,
                "inference_s": float(vlm_result["latency_s"]),
                "bbox_xyxy": vlm_result["bbox_xyxy"],
                "roi_xyxy": vlm_result["roi_xyxy"],
            },
            "memory": {
                "after_yolo_inference": before_vlm,
                "after_vlm_load": after_vlm_load,
                "after_combined_inference": after_vlm_inference,
            },
            "annotated_image": str(annotated_path),
        })
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("\nPASS: YOLO + LocateAnything initialized and inferred together on one T4.",
          flush=True)


if __name__ == "__main__":
    main()
