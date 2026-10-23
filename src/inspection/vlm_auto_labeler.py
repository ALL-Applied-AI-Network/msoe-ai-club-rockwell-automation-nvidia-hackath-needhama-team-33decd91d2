"""
Offline In-Context VLM Auto-Labeler for Real CSI Factory Captures.

Uses visual in-context exemplars (from digital twin golden references & known defect crops)
to automatically classify and label real factory aperture crops from:
  shared/data/csi_vial_fill_dataset/cap_seal/images/

Output:
  - Formatted dataset JSON with predicted labels and confidence scores.
  - High-confidence samples can be added to train/val sets.
  - Low-confidence samples flagged for human QA review.
"""

import argparse
import base64
import glob
import io
import json
import os
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
import cv2
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_IMAGES_DIR = REPO_ROOT / "shared" / "data" / "csi_vial_fill_dataset" / "cap_seal" / "images"
CONFIG_DIR = Path(__file__).resolve().parent / "config"
CLASS_LABELS = ("pass", "crooked_cap", "missing_cap", "damaged_vial", "no_vial")
OUTPUT_LABELS = (*CLASS_LABELS, "occluded", "unsure")
EXEMPLAR_FILES = {
    "pass": (
        "VLM_examples/pass/scene001_G01_pass.png",
        "VLM_examples/pass/scene019_G30_pass.png",
        "VLM_examples/pass/scene013_G05_pass.png",
    ),
    "crooked_cap": (
        "VLM_examples/crooked_cap/scene009_G20_crooked_cap.png",
        "VLM_examples/crooked_cap/scene012_G07_crooked_cap.png",
        "VLM_examples/crooked_cap/scene018_G16_crooked_cap.png",
        "VLM_examples/crooked_cap/real_crooked_cap_20260922_171522_702906_G02.png",
        "VLM_examples/crooked_cap/scene001_G06_crooked_cap.png",
        "VLM_examples/crooked_cap/scene002_G18_crooked_cap.png",
        "VLM_examples/crooked_cap/scene004_G17_crooked_cap.png",
        "VLM_examples/crooked_cap/scene008_G24_crooked_cap.png",
        "VLM_examples/crooked_cap/scene014_G16_crooked_cap.png",
        "VLM_examples/crooked_cap/real_crooked_cap_20260922_175155_329384_G18.png",
    ),
    "missing_cap": (
        "VLM_examples/missing_cap/scene003_G08_missing_cap.png",
        "VLM_examples/missing_cap/scene012_G10_missing_cap.png",
        "VLM_examples/missing_cap/scene006_G21_missing_cap.png",
    ),
    "damaged_vial": (
        "VLM_examples/damaged_vial/scene001_G14_damaged_vial.png",
        "VLM_examples/damaged_vial/scene001_G26_damaged_vial.png",
        "VLM_examples/damaged_vial/scene002_G02_damaged_vial.png",
    ),
    "no_vial": (
        "VLM_examples/no_vial/real_empty_fixture_G03.png",
        "VLM_examples/no_vial/real_empty_fixture_G06.png",
        "VLM_examples/no_vial/real_empty_fixture_G17.png",
    ),
    "occluded": (
        "VLM_examples/occluded/robot_occludes_G07.png",
        "VLM_examples/occluded/robot_occludes_G08.png",
        "VLM_examples/occluded/robot_occludes_G09.png",
    ),
}


class VLMAutoLabeler:
    def __init__(self, api_base, model_id, timeout=300):
        self.api_base = api_base.rstrip("/")
        self.api_url = self.api_base + "/chat/completions"
        self.model_id = model_id
        self.timeout = timeout
        try:
            with urlopen(self.api_base + "/models", timeout=5) as response:
                models = json.loads(response.read().decode("utf-8")).get("data", [])
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"No OpenAI-compatible VLM server responded at {self.api_base}. "
                "The weights may be downloaded, but inference needs a running server. "
                "Start one with: ./scripts/run_vlm_server.sh "
                "(vLLM, model nvidia/Qwen3.8-27B-NVFP4)"
            ) from exc
        served_models = {entry.get("id") for entry in models}
        if served_models and self.model_id not in served_models:
            raise RuntimeError(
                f"The VLM server at {self.api_base} serves {sorted(served_models)}, "
                f"but the labeler requested {self.model_id!r}. Pass the served name with --model."
            )
        print(f"[VLMAutoLabeler] Using VLM API {self.api_url} (model: {model_id})")
        self.exemplars = self._load_exemplars()

    def _load_exemplars(self):
        exemplars = {}
        for label, relative_paths in EXEMPLAR_FILES.items():
            exemplars[label] = []
            for relative_path in relative_paths:
                path = REPO_ROOT / "shared" / "inspection_models" / "crop_dataset" / relative_path
                if not path.is_file():
                    raise FileNotFoundError(f"VLM example for {label!r} is missing: {path}")
                with Image.open(path) as image:
                    exemplars[label].append(image.convert("RGB"))
                print(f"[VLMAutoLabeler] Example for {label}: {path.name}")
        return exemplars

    @staticmethod
    def _image_data_url(image):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        encoded = base64.b64encode(image_bytes.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def label_crop(self, query_crop_img):
        """Ask the configured vision-language model to classify one aperture crop."""
        prompt = (
            "Classify the query crop using the labeled reference groups. Choose one label: "
            "pass = intact vial with its cap fully and normally seated; crooked_cap = the vial "
            "has a cap, but the cap is not fully seated or is incorrectly oriented on the vial; "
            "missing_cap = "
            "glass vial/neck visible without a cap; no_vial = empty "
            "fixture opening with no visible glass; damaged_vial = visibly cracked, chipped, or "
            "broken glass, not just a cap defect. These are tiny fixture crops: a dark circular "
            "hole or metal rim is not proof of a vial. For missing_cap, require a clear glass "
            "neck or vial wall with glass reflections/refraction around the opening. If that glass "
            "evidence is absent, choose no_vial, not missing_cap. If the opening is clearly empty "
            "with no visible glass, classify no_vial directly; save detailed comparison for crops "
            "with a visible vial or ambiguous cap condition. The yellow robot arm may cross "
            "or cover an aperture; its color is not evidence of a vial or a pass. If the arm or "
            "another object blocks the opening so the vial and cap condition cannot be judged, "
            "choose occluded, matching the occluded reference group, even if part of the vial is "
            "visible. For this dataset, a visible yellow cap whose outline is not clearly circular "
            "must be labeled crooked_cap: this includes oval, flattened, angular, rectangular, "
            "cocked, or visibly rotated cap shapes, even when low resolution makes seating hard "
            "to judge. A tilted vial alone is not crooked if its yellow cap is clearly circular "
            "and fully seated. Also mark crooked_cap when the cap is visibly raised, shifted, "
            "partly threaded, inverted, or otherwise not fully seated; look for an uneven gap or "
            "exposed neck caused by the cap position. Favor catching crooked caps: choose pass "
            "only when the yellow cap is clearly circular and fully seated. When pass versus "
            "crooked_cap is uncertain and the cap may be noncircular or improperly seated, choose "
            "crooked_cap. Do not apply this preference to clearly empty openings. If the robot "
            "arm or another object blocks the aperture so its condition cannot be judged, choose "
            "occluded. Use unsure when the image is visible but too low quality or ambiguous to "
            "reliably choose among the other labels. "
            "Think through the visible evidence internally, but return only JSON containing label "
            "and confidence from 0 to 1. Do not include reasoning in the response."
        )
        content = [{"type": "text", "text": prompt}]
        for label, images in self.exemplars.items():
            content.append({"type": "text", "text": f"Reference group: {label}."})
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": self._image_data_url(image)}})
        content.extend([
            {"type": "text", "text": "Classify this query crop:"},
            {"type": "image_url", "image_url": {"url": self._image_data_url(query_crop_img)}},
        ])
        payload = json.dumps({
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            # Reasoning and the final JSON share this budget; leave enough room for both.
            "max_tokens": 4096,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "vial_inspection_label",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "enum": list(OUTPUT_LABELS)},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["label", "confidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "extra_body": {"chat_template_kwargs": {
                "enable_thinking": True,
            }},
            # Qwen3.8's chat template only accepts xhigh, medium, or low.
            "reasoning_effort": "medium",
        }).encode("utf-8")
        request = Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                api_result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            server_detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(
                f"VLM request failed with HTTP {exc.code}: {server_detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not get a classification from the VLM at {self.api_url}: {exc}. "
                "Start the model's OpenAI-compatible server and try again."
            ) from exc

        response_text = api_result["choices"][0]["message"]["content"]
        try:
            start, end = response_text.index("{"), response_text.rindex("}") + 1
            result = json.loads(response_text[start:end])
            label = str(result.get("label", "unsure")).strip().lower()
            if label not in OUTPUT_LABELS:
                label = "unsure"
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
            return {"label": label, "confidence": confidence}
        except (ValueError, TypeError, json.JSONDecodeError):
            return {"label": "unsure", "confidence": 0.0}


def process_real_images(max_images=5, out_json="runs/real_data_vlm_labels.json", all_apertures=False, save_crops=True,
                        api_base="http://localhost:8000/v1", model_id="nvidia/Qwen3.8-27B-NVFP4",
                        crop_output_dir=None):
    all_files = sorted(glob.glob(str(REAL_IMAGES_DIR / "*.jpg")))
    print(f"[VLMAutoLabeler] Found {len(all_files)} real line images. Sampling {max_images}...")

    with open(CONFIG_DIR / "fixture_g_apertures.json") as f:
        apertures = json.load(f)

    labeler = VLMAutoLabeler(api_base=api_base, model_id=model_id)
    results = []

    r = 16
    for img_path in all_files[:max_images]:
        img = cv2.imread(img_path)
        if img is None:
            continue

        filename = Path(img_path).name
        stem = Path(img_path).stem
        print(f"\nProcessing real capture: {filename}")
        img_records = {"filename": filename, "apertures": {}}

        # Either inspect all 30 slots or a representative sample
        target_slots = list(apertures.keys()) if all_apertures else ["G01", "G05", "G15", "G23", "G30"]
        
        for s_id in target_slots:
            info = apertures[s_id]
            cx, cy = int(info["x"]), int(info["y"])
            crop_cv = img[max(0, cy-r):min(img.shape[0], cy+r), max(0, cx-r):min(img.shape[1], cx+r)]
            crop_pil = Image.fromarray(cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB))

            decision = labeler.label_crop(crop_pil)
            lbl = decision.get("label", "unsure")
            conf = decision.get("confidence", 0.0)
            print(f"  Slot {s_id}: {lbl} ({conf})")

            # Optionally export each recognized label under the configured crop output root.
            if save_crops and (lbl in CLASS_LABELS or (
                crop_output_dir is not None and lbl in {"occluded", "unsure"}
            )):
                out_root = Path(crop_output_dir) if crop_output_dir else (
                    REPO_ROOT / "shared" / "inspection_models" / "crop_dataset" / "train"
                )
                if not out_root.is_absolute():
                    out_root = REPO_ROOT / out_root
                out_dir = out_root / lbl
                out_dir.mkdir(parents=True, exist_ok=True)
                crop_out = out_dir / f"real_{stem}_{s_id}.png"
                cv2.imwrite(str(crop_out), crop_cv)
                decision["crop_file"] = str(crop_out.relative_to(REPO_ROOT))

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
    parser.add_argument("--out", "--output", dest="out", default="runs/vlm_labels/real_data_vlm_labels.json")
    parser.add_argument("--all-apertures", action="store_true", help="Label all 30 apertures per image (instead of sample)")
    parser.add_argument("--no-save-crops", dest="save_crops", action="store_false", help="Do not save crop images to dataset folders")
    parser.add_argument("--crop-output", default=None,
                        help="Root directory for labeled crop images (default: shared/inspection_models/crop_dataset/train)")
    parser.add_argument("--api-base", default=os.environ.get("VLM_API_BASE", "http://localhost:8000/v1"),
                        help="Base URL for an OpenAI-compatible VLM server (default: %(default)s)")
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", "nvidia/Qwen3.8-27B-NVFP4"),
                        help="Model name served by the VLM API (default: %(default)s)")
    parser.set_defaults(save_crops=True)
    args = parser.parse_args()

    process_real_images(
        max_images=args.max_images,
        out_json=args.out,
        all_apertures=args.all_apertures,
        save_crops=args.save_crops,
        api_base=args.api_base,
        model_id=args.model,
        crop_output_dir=args.crop_output,
    )
