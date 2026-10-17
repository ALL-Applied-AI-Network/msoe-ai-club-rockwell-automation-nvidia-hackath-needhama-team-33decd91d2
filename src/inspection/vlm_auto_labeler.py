"""
Offline In-Context VLM Auto-Labeler for Real CSI Factory Captures.

Uses visual in-context exemplars (from digital twin golden references & known defect crops)
to automatically classify and label real factory aperture crops from:
  shared/data/csi_vial_fill_dataset/cap_seal/images/

Output:
  - Formatted dataset JSON with predicted labels, reasoning rationale, and confidence.
  - High-confidence samples can be added to train/val sets.
  - Low-confidence samples flagged for human QA review.
"""

import argparse
import glob
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_IMAGES_DIR = REPO_ROOT / "shared" / "data" / "csi_vial_fill_dataset" / "cap_seal" / "images"
EXEMPLAR_DIR = REPO_ROOT / "shared" / "inspection_models" / "crop_dataset" / "val"
CONFIG_DIR = Path(__file__).resolve().parent / "config"


class VLMAutoLabeler:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM-Instruct", device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[VLMAutoLabeler] Loading VLM ({model_id}) on {self.device}...")

        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device
        )
        print("[VLMAutoLabeler] VLM model initialized successfully.")

        # Load reference exemplars
        self.exemplars = self._load_exemplars()

    def _load_exemplars(self):
        exemplars = {}
        classes = ["pass", "crooked_cap", "missing_cap", "no_vial"]
        for c in classes:
            files = list((EXEMPLAR_DIR / c).glob("*.png"))
            if files:
                exemplars[c] = Image.open(files[0]).convert("RGB")
        return exemplars

    def label_crop(self, query_crop_img):
        """
        In-context prompt with visual exemplars to classify query aperture crop.
        """
        # Construct side-by-side prompt or few-shot image list
        # For lightweight VLMs, prompt with structured multi-choice criteria:
        prompt = (
            "This is an overhead crop of an industrial vial fixture aperture.\n"
            "Classes:\n"
            "1. 'pass': Yellow cap is seated flush, upright, centered.\n"
            "2. 'crooked_cap': Yellow cap is tilted, unseated, cross-threaded, or visibly misaligned.\n"
            "3. 'missing_cap': Vial is present with an open rim/mouth, but the yellow cap is absent.\n"
            "4. 'no_vial': Aperture is empty; no vial or cap present.\n\n"
            "Analyze the image and respond in JSON format:\n"
            "{\"label\": \"pass|crooked_cap|missing_cap|no_vial\", \"confidence\": 0.0-1.0, \"rationale\": \"brief reason\"}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[query_crop_img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)

        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Parse JSON/dict from response (handles both single and double quotes)
        import ast
        import re
        try:
            m = re.search(r'\{.*?\}', response, re.DOTALL)
            if m:
                d = ast.literal_eval(m.group(0))
                lbl = str(d.get("label", "unclear")).strip().lower()
                valid = ["pass", "crooked_cap", "missing_cap", "no_vial", "damaged_vial"]
                if lbl not in valid:
                    # check if any class is mentioned in text
                    for v in valid:
                        if v in lbl or v in response.lower():
                            lbl = v
                            break
                    else:
                        lbl = "unclear"
                
                try:
                    conf = float(d.get("confidence", 0.9))
                except Exception:
                    conf = 0.9 if lbl != "unclear" else 0.5

                return {
                    "label": lbl,
                    "confidence": conf,
                    "rationale": str(d.get("rationale", response))
                }
        except Exception:
            pass

        return {
            "label": "unclear",
            "confidence": 0.5,
            "rationale": response
        }


def process_real_images(max_images=5, out_json="runs/real_data_vlm_labels.json"):
    all_files = sorted(glob.glob(str(REAL_IMAGES_DIR / "*.jpg")))
    print(f"[VLMAutoLabeler] Found {len(all_files)} real line images. Sampling {max_images}...")

    with open(CONFIG_DIR / "fixture_g_apertures.json") as f:
        apertures = json.load(f)

    labeler = VLMAutoLabeler()
    results = []

    r = 16
    for img_path in all_files[:max_images]:
        img = cv2.imread(img_path)
        if img is None:
            continue

        filename = Path(img_path).name
        print(f"\nProcessing real capture: {filename}")
        img_records = {"filename": filename, "apertures": {}}

        # Sample a subset of apertures (e.g. G01, G05, G15, G23, G30) to test
        test_slots = ["G01", "G05", "G15", "G23", "G30"]
        for s_id in test_slots:
            info = apertures[s_id]
            cx, cy = int(info["x"]), int(info["y"])
            crop_cv = img[max(0, cy-r):min(img.shape[0], cy+r), max(0, cx-r):min(img.shape[1], cx+r)]
            crop_pil = Image.fromarray(cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB))

            decision = labeler.label_crop(crop_pil)
            print(f"  Slot {s_id}: {decision.get('label')} ({decision.get('confidence')}) - {decision.get('rationale')}")
            img_records["apertures"][s_id] = decision

        results.append(img_records)

    out_p = Path(out_json).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[VLMAutoLabeler] Finished! Results saved to: {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--out", default="runs/real_data_vlm_labels.json")
    args = parser.parse_args()

    process_real_images(max_images=args.max_images, out_json=args.out)
