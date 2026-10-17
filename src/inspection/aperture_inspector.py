"""
Dataset 1: Final Vial Fixture Aperture Inspector.

Implements the official MSOE AI Club / Rockwell-NVIDIA competition specification:
- Classifies each numbered fixture position into 5 official classes:
    1. 'pass'
    2. 'missing_cap'
    3. 'crooked_cap'
    4. 'damaged_vial'
    5. 'no_vial'
- Supports golden reference change detection (comparing query aperture to known-good golden crop).
- Tracks robot fill sequence (right-to-left columns, top-to-bottom) to differentiate trailing empty slots from missed placements.
- Provides actionable remediation flags (press_reseat vs scrap_and_backfill).
"""

import argparse
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np

LABEL_PASS = "pass"
LABEL_MISSING_CAP = "missing_cap"
LABEL_CROOKED_CAP = "crooked_cap"
LABEL_DAMAGED_VIAL = "damaged_vial"
LABEL_NO_VIAL = "no_vial"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent / "config"


class FixtureApertureInspector:
    def __init__(self, golden_image_path, apertures_config_path=None, crop_radius=15, yolo_weights=None):
        self.crop_radius = crop_radius
        self.apertures = {}
        
        cfg_path = apertures_config_path or (CONFIG_DIR / "fixture_g_apertures.json")
        with open(cfg_path) as f:
            self.apertures = json.load(f)

        self.golden_image = cv2.imread(str(golden_image_path))
        if self.golden_image is None:
            raise FileNotFoundError(f"Golden reference image not found: {golden_image_path}")
            
        self.golden_crops = {}
        self._extract_golden_crops()

        self.yolo_model = None
        if yolo_weights and os.path.exists(yolo_weights):
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_weights)
            print(f"[FixtureApertureInspector] Loaded YOLOv11 classifier: {yolo_weights}")

    def _extract_crop(self, img, cx, cy):
        r = self.crop_radius
        x1, x2 = max(0, int(cx - r)), min(img.shape[1], int(cx + r))
        y1, y2 = max(0, int(cy - r)), min(img.shape[0], int(cy + r))
        return img[y1:y2, x1:x2]

    def _extract_golden_crops(self):
        for slot_id, info in self.apertures.items():
            crop = self._extract_crop(self.golden_image, info["x"], info["y"])
            self.golden_crops[slot_id] = crop

    def inspect_fixture(self, query_image, expected_placed_count=30):
        """
        Inspects all 30 numbered aperture positions against golden reference and physical signatures.
        """
        results = []
        filled_slots = 0

        for slot_id in sorted(self.apertures.keys(), key=lambda k: self.apertures[k]["slot"]):
            info = self.apertures[slot_id]
            slot_num = info["slot"]
            cx, cy = info["x"], info["y"]

            q_crop = self._extract_crop(query_image, cx, cy)
            g_crop = self.golden_crops[slot_id]

            # 1. Yellow Cap Signature (HSV mask)
            hsv = cv2.cvtColor(q_crop, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv, (18, 70, 70), (38, 255, 255))
            yellow_px = np.count_nonzero(yellow_mask)

            # Golden reference yellow count
            g_hsv = cv2.cvtColor(g_crop, cv2.COLOR_BGR2HSV)
            g_yellow_px = np.count_nonzero(cv2.inRange(g_hsv, (18, 70, 70), (38, 255, 255)))

            # 2. Structural dissimilarity vs Golden Crop (Normalized Cross-Correlation / Absolute Difference)
            q_gray = cv2.cvtColor(q_crop, cv2.COLOR_BGR2GRAY)
            g_gray = cv2.cvtColor(g_crop, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(q_gray, g_gray)
            mean_diff = float(np.mean(diff))

            # 3. Contour & Aspect Ratio of Yellow Cap (if cap present)
            cnts, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cap_aspect = 1.0
            cap_tilt = 0.0
            cap_offset = 0.0
            if cnts:
                largest_c = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_c)
                cap_aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                cap_tilt = float(np.degrees(np.arccos(max(0.1, min(1.0, cap_aspect)))))
                cap_cx = x + w / 2.0
                cap_cy = y + h / 2.0
                cap_offset = float(np.hypot(cap_cx - self.crop_radius, cap_cy - self.crop_radius))

            # 4. Transparent Glass Body & Rim Presence (High gradient rim vs empty aperture background)
            edges = cv2.Canny(q_gray, 40, 120)
            edge_count = np.count_nonzero(edges)

            # 5. YOLO Classifier Inference (if model loaded)
            yolo_label = None
            yolo_conf = 0.0
            if self.yolo_model is not None:
                # YOLOv11 classification on 32x32 crop
                res = self.yolo_model.predict(q_crop, imgsz=64, verbose=False)
                if res and res[0].probs is not None:
                    top1_idx = int(res[0].probs.top1)
                    yolo_label = res[0].names[top1_idx]
                    yolo_conf = float(res[0].probs.top1conf)

            # Classify Aperture into 5 Official Classes:
            if yellow_px < 60:
                # No yellow cap present in aperture
                # Differentiate missing_cap (transparent vial present) vs no_vial (empty aperture)
                if yolo_label in [LABEL_MISSING_CAP, LABEL_NO_VIAL]:
                    label = yolo_label
                    confidence = yolo_conf
                    action = "scrap_and_backfill" if label == LABEL_MISSING_CAP else (
                        "emergency_missed_placement" if slot_num <= expected_placed_count else "pending_fill"
                    )
                    reason = f"YOLOv11 classified as {label} ({confidence:.2f} conf)."
                elif edge_count > 50 or mean_diff > 25.0:
                    label = LABEL_MISSING_CAP
                    action = "scrap_and_backfill"
                    confidence = 0.98
                    reason = "Vial body present in aperture but yellow cap is missing (open mouth detected)."
                else:
                    label = LABEL_NO_VIAL
                    # Check fill pattern: if within expected count, missed placement!
                    if slot_num <= expected_placed_count:
                        action = "emergency_missed_placement"
                        confidence = 0.98
                        reason = f"Empty aperture within fill sequence (Slot #{slot_num} <= expected {expected_placed_count}). Robot missed placement."
                    else:
                        action = "pending_fill"
                        confidence = 1.0
                        reason = f"Aperture #{slot_num} empty, waiting for scheduled robot placement."
            else:
                filled_slots += 1
                # Cap is present; evaluate seating and damage
                # If YOLO model is confident in crooked_cap or damaged_vial, prioritize it:
                if yolo_label in [LABEL_CROOKED_CAP, LABEL_DAMAGED_VIAL] and yolo_conf > 0.70:
                    label = yolo_label
                    confidence = yolo_conf
                    action = "press_reseat" if label == LABEL_CROOKED_CAP and cap_aspect >= 0.70 else "scrap_and_backfill"
                    reason = f"YOLOv11 detected {label} (conf {confidence:.2f}, delta {mean_diff:.1f})."
                elif mean_diff >= 10.0 or cap_aspect < 0.82 or cap_offset > 4.5:
                    label = LABEL_CROOKED_CAP
                    action = "press_reseat" if mean_diff < 35.0 and cap_aspect >= 0.70 else "scrap_and_backfill"
                    confidence = 0.95
                    reason = f"Cap visibly tilted or unseated (aspect ratio {cap_aspect:.2f}, offset {cap_offset:.1f}px, diff {mean_diff:.1f})."
                elif mean_diff > 55.0 and cap_aspect > 0.90:
                    label = LABEL_DAMAGED_VIAL
                    action = "scrap_and_backfill"
                    confidence = 0.90
                    reason = "Significant structural deviation detected on vial body/lip indicating glass fracture."
                else:
                    label = LABEL_PASS
                    action = "pass"
                    confidence = yolo_conf if (yolo_label == LABEL_PASS) else 0.99
                    reason = "Cap seated flush, vial body intact within golden reference envelope."

            results.append({
                "slot_id": slot_id,
                "slot_number": slot_num,
                "grid_pos": {"row": info["row"], "col": info["col"]},
                "coords": [int(cx), int(cy)],
                "label": label,
                "action": action,
                "confidence": confidence,
                "metrics": {
                    "yellow_px": int(yellow_px),
                    "cap_aspect_ratio": round(float(cap_aspect), 2),
                    "golden_diff_score": round(mean_diff, 1),
                    "edge_density": int(edge_count)
                },
                "rationale": reason
            })

        return results


def annotate_fixture_image(img, inspection_results):
    annotated = img.copy()
    color_map = {
        LABEL_PASS: (0, 200, 0),
        LABEL_CROOKED_CAP: (0, 140, 255),
        LABEL_MISSING_CAP: (0, 0, 255),
        LABEL_DAMAGED_VIAL: (0, 0, 220),
        LABEL_NO_VIAL: (180, 180, 180)
    }

    for r in inspection_results:
        cx, cy = r["coords"]
        label = r["label"]
        color = color_map.get(label, (255, 255, 255))
        rad = 14

        cv2.circle(annotated, (cx, cy), rad, color, 2)
        text = f"{r['slot_id']}:{label[:4]}"
        cv2.putText(annotated, text, (cx - 16, cy - rad - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Dataset 1 Fixture Aperture Defect Inspector")
    parser.add_argument("--image", default="shared/inspection_models/test_defect_scene/render.png", help="Input inspection image")
    parser.add_argument("--golden", default="shared/inspection_models/golden_reference.png", help="Golden good image reference")
    parser.add_argument("--yolo-weights", default="shared/inspection_models/vial_aperture_cls_best.pt", help="Path to trained YOLOv11 classifier weights")
    parser.add_argument("--expected-count", type=int, default=30, help="Expected filled vial count for fill-order logic")
    parser.add_argument("--out-annotated", default="shared/inspection_models/test_defect_scene/annotated_ensemble.png", help="Save annotated visualization")
    parser.add_argument("--out-json", default="shared/inspection_models/test_defect_scene/decision_ensemble.json", help="Save structured inspection JSON")
    args = parser.parse_args()

    inspector = FixtureApertureInspector(golden_image_path=args.golden, yolo_weights=args.yolo_weights)
    query_img = cv2.imread(args.image)
    if query_img is None:
        print(f"Error loading image: {args.image}", file=sys.stderr)
        sys.exit(1)

    results = inspector.inspect_fixture(query_img, expected_placed_count=args.expected_count)

    summary = {
        LABEL_PASS: sum(1 for r in results if r["label"] == LABEL_PASS),
        LABEL_CROOKED_CAP: sum(1 for r in results if r["label"] == LABEL_CROOKED_CAP),
        LABEL_MISSING_CAP: sum(1 for r in results if r["label"] == LABEL_MISSING_CAP),
        LABEL_DAMAGED_VIAL: sum(1 for r in results if r["label"] == LABEL_DAMAGED_VIAL),
        LABEL_NO_VIAL: sum(1 for r in results if r["label"] == LABEL_NO_VIAL)
    }

    payload = {
        "inspection_image": args.image,
        "golden_reference": args.golden,
        "summary": summary,
        "apertures": results
    }

    print(json.dumps(payload, indent=2))

    if args.out_annotated:
        ann = annotate_fixture_image(query_img, results)
        cv2.imwrite(args.out_annotated, ann)
        print(f"[Aperture Inspector] Saved visualization to: {args.out_annotated}")

    if args.out_json:
        with open(args.out_json, "w") as fp:
            json.dump(payload, fp, indent=2)
        print(f"[Aperture Inspector] Saved decision JSON to: {args.out_json}")


if __name__ == "__main__":
    main()
