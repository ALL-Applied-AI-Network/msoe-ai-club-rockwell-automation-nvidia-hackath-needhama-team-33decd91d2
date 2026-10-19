#!/usr/bin/env python3
"""
Dataset 1: Final Vial Fixture Detector & Slot Assignment Inspector.

Full-frame YOLO object detection with spatial slot matching:
1. Detects objects across full tray:
     - 'pass' (flush seated cap)
     - 'crooked_cap' (tilted / cross-threaded / unseated cap)
     - 'missing_cap' (open vial mouth without cap)
     - 'damaged_vial' (fractured or skewed vial)
2. Uses spatial Hungarian/minimum-distance assignment to map detections to fixture slots G01-G30.
3. Automatically assigns 'no_vial' to any slot without a matched detection.
4. Generates actionable remediation flags (press_reseat vs scrap_and_backfill)
   and recipe refills for the Fanuc robot.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

LABEL_PASS = "pass"
LABEL_MISSING_CAP = "missing_cap"
LABEL_CROOKED_CAP = "crooked_cap"
LABEL_DAMAGED_VIAL = "damaged_vial"
LABEL_NO_VIAL = "no_vial"

ACTION_PASS = "pass"
ACTION_PRESS_RESEAT = "press_reseat"
ACTION_SCRAP = "scrap"
ACTION_NEW_CAP = "new_cap"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_MODEL = REPO_ROOT / "shared" / "inspection_models" / "vial_fullframe_det_best.pt"


class FixtureDetectorInspector:
    def __init__(self, model_weights=None, apertures_config_path=None, max_match_dist_px=38.0, conf_thresh=0.25):
        self.max_match_dist = max_match_dist_px
        self.conf_thresh = conf_thresh
        self.apertures = {}

        cfg_path = apertures_config_path or (CONFIG_DIR / "fixture_g_apertures.json")
        with open(cfg_path) as f:
            self.apertures = json.load(f)

        self.model = None
        weights_to_load = model_weights or DEFAULT_MODEL
        if weights_to_load and os.path.exists(str(weights_to_load)):
            from ultralytics import YOLO
            self.model = YOLO(str(weights_to_load))
            print(f"[FixtureDetectorInspector] Loaded YOLOv11 detector: {weights_to_load}")
        else:
            print(f"[FixtureDetectorInspector] Warning: Model weights not found at {weights_to_load}. Running in fallback heuristic mode.")

    def detect_and_assign(self, image_bgr, expected_placed_count=30):
        h, w = image_bgr.shape[:2]
        detections = []

        if self.model is not None:
            results = self.model.predict(image_bgr, conf=self.conf_thresh, imgsz=max(h, w), verbose=False)
            if results and len(results) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = results[0].names.get(cls_id, f"class_{cls_id}")

                    cx = float((xyxy[0] + xyxy[2]) / 2.0)
                    cy = float((xyxy[1] + xyxy[3]) / 2.0)
                    bw = float(xyxy[2] - xyxy[0])
                    bh = float(xyxy[3] - xyxy[1])

                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "box_xyxy": [float(c) for c in xyxy],
                        "center": (cx, cy),
                        "width": bw,
                        "height": bh
                    })
        else:
            # Fallback heuristic detector using HSV + edge detection when model weights aren't yet trained
            detections = self._heuristic_fallback_detect(image_bgr)

        # Spatial Hungarian Assignment of Detections -> Fixture Slots
        slot_list = sorted(self.apertures.keys(), key=lambda k: self.apertures[k]["slot"])
        num_slots = len(slot_list)
        num_dets = len(detections)

        # Build distance / cost matrix
        # Rows: slots, Columns: detections
        cost_matrix = np.full((num_slots, max(num_slots, num_dets)), 1e6)

        for s_idx, s_id in enumerate(slot_list):
            sx, sy = self.apertures[s_id]["x"], self.apertures[s_id]["y"]
            for d_idx, det in enumerate(detections):
                dcx, dcy = det["center"]
                dist = np.hypot(sx - dcx, sy - dcy)
                if dist <= self.max_match_dist:
                    cost_matrix[s_idx, d_idx] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_detections = {}
        for r, c in zip(row_ind, col_ind):
            if c < num_dets and cost_matrix[r, c] <= self.max_match_dist:
                assigned_detections[slot_list[r]] = detections[c]

        # Assemble final inspection report per slot
        results = []
        for s_id in slot_list:
            info = self.apertures[s_id]
            slot_num = info["slot"]
            sx, sy = int(info["x"]), int(info["y"])

            matched_det = assigned_detections.get(s_id)
            if matched_det is not None:
                det_cls = matched_det["class"]
                conf = matched_det["confidence"]
                det_center = matched_det["center"]
                offset = float(np.hypot(sx - det_center[0], sy - det_center[1]))
                bbox = matched_det["box_xyxy"]

                if det_cls == "crooked_cap":
                    label = LABEL_CROOKED_CAP
                    action = ACTION_PRESS_RESEAT
                    reason = f"Detected crooked/tilted cap (offset={offset:.1f}px, conf={conf:.2f})."
                elif det_cls == "missing_cap":
                    label = LABEL_MISSING_CAP
                    action = ACTION_NEW_CAP
                    reason = f"Detected open vial rim without cap (conf={conf:.2f})."
                elif det_cls == "damaged_vial":
                    label = LABEL_DAMAGED_VIAL
                    action = ACTION_SCRAP
                    reason = f"Detected fractured or misaligned vial body (conf={conf:.2f})."
                elif det_cls == "no_vial":
                    label = LABEL_NO_VIAL
                    action = "emergency_missed_placement" if slot_num <= expected_placed_count else "pending_fill"
                    reason = f"Detected empty hole marker (conf={conf:.2f})."
                else:
                    label = LABEL_PASS
                    action = ACTION_PASS
                    reason = f"Cap flush and seated level (conf={conf:.2f})."
            else:
                # No detection mapped within radius -> Naturally empty aperture hole
                label = LABEL_NO_VIAL
                action = "emergency_missed_placement" if slot_num <= expected_placed_count else "pending_fill"
                conf = 0.99
                reason = "No vial or cap detected at aperture location (empty slot)."
                bbox = None
                det_center = (sx, sy)
                offset = 0.0

            results.append({
                "slot_id": s_id,
                "slot_number": slot_num,
                "grid_pos": {"row": info["row"], "col": info["col"]},
                "coords": [sx, sy],
                "detected_center": [round(det_center[0], 1), round(det_center[1], 1)],
                "bounding_box": [round(b, 1) for b in bbox] if bbox else None,
                "label": label,
                "action": action,
                "confidence": round(float(conf), 3),
                "offset_px": round(offset, 1),
                "rationale": reason
            })

        return results, detections

    def _heuristic_fallback_detect(self, img):
        """Fallback detector when trained YOLO detector weights are not yet compiled."""
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]

        is_yellow = (R > 130) & (G > 130) & (B < 120) & ((R.astype(int) + G.astype(int))/2 - B > 25)
        cnts, _ = cv2.findContours(is_yellow.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx, cy = x + bw / 2.0, y + bh / 2.0
            aspect = min(bw, bh) / (max(bw, bh) + 1e-5)
            # Crooked check: knurling/shadow or elongated aspect
            sub_gray = gray[max(0, y):min(h, y+bh), max(0, x):min(w, x+bw)]
            dark_shadow = int((sub_gray < 70).sum()) if sub_gray.size > 0 else 0

            if aspect < 0.72 or dark_shadow > 50:
                cls_name = "crooked_cap"
            else:
                cls_name = "pass"

            dets.append({
                "class": cls_name,
                "confidence": 0.90,
                "box_xyxy": [float(x), float(y), float(x + bw), float(y + bh)],
                "center": (cx, cy),
                "width": float(bw),
                "height": float(bh)
            })

        return dets


def annotate_inspection_frame(img, inspection_results, raw_detections=None):
    annotated = img.copy()
    color_map = {
        LABEL_PASS: (0, 210, 0),
        LABEL_CROOKED_CAP: (0, 140, 255),
        LABEL_MISSING_CAP: (0, 0, 255),
        LABEL_DAMAGED_VIAL: (0, 0, 220),
        LABEL_NO_VIAL: (150, 150, 150)
    }

    # 1. Draw detected bounding boxes
    if raw_detections:
        for det in raw_detections:
            bx = [int(v) for v in det["box_xyxy"]]
            cls_name = det["class"]
            col = color_map.get(cls_name, (200, 200, 200))
            cv2.rectangle(annotated, (bx[0], bx[1]), (bx[2], bx[3]), col, 2)
            tag = f"{cls_name} {det['confidence']:.2f}"
            cv2.putText(annotated, tag, (bx[0], max(12, bx[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    # 2. Draw fixture slot anchors and assignment lines
    for r in inspection_results:
        sx, sy = r["coords"]
        label = r["label"]
        col = color_map.get(label, (255, 255, 255))
        dcx, dcy = int(r["detected_center"][0]), int(r["detected_center"][1])

        # Draw nominal fixture aperture circle
        cv2.circle(annotated, (sx, sy), 14, (80, 80, 80), 1)

        # Vector from nominal slot to detected center
        if label != LABEL_NO_VIAL and (abs(sx - dcx) > 1 or abs(sy - dcy) > 1):
            cv2.line(annotated, (sx, sy), (dcx, dcy), (0, 255, 255), 1, cv2.LINE_AA)

        # Slot ID label
        tag = f"{r['slot_id']}"
        cv2.putText(annotated, tag, (sx - 10, sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Full-Frame YOLO Vial Detector & Slot Inspector")
    parser.add_argument("--image", default="shared/inspection_models/test_defect_scene/render.png", help="Input tray image")
    parser.add_argument("--weights", default="shared/inspection_models/vial_fullframe_det_best.pt", help="Path to trained YOLOv11 detector weights")
    parser.add_argument("--expected-count", type=int, default=30, help="Expected filled vial count")
    parser.add_argument("--out-annotated", default="shared/inspection_models/test_defect_scene/annotated_detection.png", help="Path to save annotated inspection image")
    parser.add_argument("--out-json", default="shared/inspection_models/test_defect_scene/decision_detection.json", help="Path to save inspection decision JSON")
    args = parser.parse_args()

    inspector = FixtureDetectorInspector(model_weights=args.weights)
    query_img = cv2.imread(args.image)
    if query_img is None:
        print(f"Error loading image: {args.image}", file=sys.stderr)
        sys.exit(1)

    results, raw_dets = inspector.detect_and_assign(query_img, expected_placed_count=args.expected_count)

    summary = {
        LABEL_PASS: sum(1 for r in results if r["label"] == LABEL_PASS),
        LABEL_CROOKED_CAP: sum(1 for r in results if r["label"] == LABEL_CROOKED_CAP),
        LABEL_MISSING_CAP: sum(1 for r in results if r["label"] == LABEL_MISSING_CAP),
        LABEL_DAMAGED_VIAL: sum(1 for r in results if r["label"] == LABEL_DAMAGED_VIAL),
        LABEL_NO_VIAL: sum(1 for r in results if r["label"] == LABEL_NO_VIAL)
    }

    replacement_recipes = []
    for r in results:
        if r["action"] in [ACTION_SCRAP, ACTION_NEW_CAP]:
            replacement_recipes.append({
                "slot_id": r["slot_id"],
                "slot_number": r["slot_number"],
                "defect_reason": r["label"],
                "action_required": r["action"],
                "recipe_type": "Replacement-Backfill",
                "target_fill_ml": 30,
                "dispense_tanks": {"100": 10, "200": 10, "400": 10},
                "status": "QUEUED_FOR_REFILL"
            })

    payload = {
        "inspection_image": args.image,
        "detector_model": args.weights,
        "total_detections_found": len(raw_dets),
        "summary": summary,
        "action_breakdown": {
            ACTION_PASS: sum(1 for r in results if r["action"] == ACTION_PASS),
            ACTION_PRESS_RESEAT: sum(1 for r in results if r["action"] == ACTION_PRESS_RESEAT),
            ACTION_NEW_CAP: sum(1 for r in results if r["action"] == ACTION_NEW_CAP),
            ACTION_SCRAP: sum(1 for r in results if r["action"] == ACTION_SCRAP),
        },
        "replacement_recipes_queued": replacement_recipes,
        "apertures": results
    }

    print(json.dumps(payload, indent=2))

    if args.out_annotated:
        Path(args.out_annotated).parent.mkdir(parents=True, exist_ok=True)
        ann = annotate_inspection_frame(query_img, results, raw_dets)
        cv2.imwrite(args.out_annotated, ann)
        print(f"[Detector Inspector] Saved visualization to: {args.out_annotated}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as fp:
            json.dump(payload, fp, indent=2)
        print(f"[Detector Inspector] Saved decision JSON to: {args.out_json}")


if __name__ == "__main__":
    main()
