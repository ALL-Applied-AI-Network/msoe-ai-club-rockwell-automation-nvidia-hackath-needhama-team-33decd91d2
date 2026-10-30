#!/usr/bin/env python3
"""Generate defect proposals with Qwen3-VL's visual-grounding ability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inspection.defect_training.config import load_project_config  # noqa: E402
from src.inspection.defect_training.locate_anything import generate_proposals  # noqa: E402
from src.inspection.defect_training.qwen_grounding import parse_qwen_grounding_boxes  # noqa: E402
from src.inspection.defect_training.seed_import import load_split_manifest, read_yolo_labels  # noqa: E402


REFERENCE_CLASSES = ("good_cap", "crooked_cap", "missing_cap", "table_cap")
REFERENCE_COLORS = ((50, 190, 80), (245, 145, 35), (225, 65, 70), (65, 140, 235))


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size) if path.is_file() else ImageFont.load_default()


def build_reference_overlays(images_dir: Path, labels_dir: Path, output: Path) -> list[Image.Image]:
    """Render human YOLO annotations into explicit visual prompt examples."""

    output.mkdir(parents=True, exist_ok=True)
    references = []
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing reference labels: {label_path}")
        with Image.open(image_path) as opened:
            rendered = opened.convert("RGB")
        draw = ImageDraw.Draw(rendered)
        label_font = _font(18)
        for class_id, box in read_yolo_labels(label_path, rendered.size, len(REFERENCE_CLASSES)):
            color = REFERENCE_COLORS[class_id]
            coordinates = tuple(round(value) for value in box.as_list())
            draw.rectangle(coordinates, outline=color, width=4)
            name = REFERENCE_CLASSES[class_id]
            bounds = draw.textbbox((coordinates[0], coordinates[1]), name, font=label_font)
            draw.rectangle(bounds, fill=(0, 0, 0))
            draw.text((coordinates[0], coordinates[1]), name, fill=color, font=label_font)
        rendered.save(output / image_path.name)
        references.append(rendered)
    if not references:
        raise FileNotFoundError(f"No reference images found in {images_dir}")
    return references


class QwenGroundingWorker:
    def __init__(self, model_name: str, max_new_tokens: int, references=None) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.references = list(references or [])
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def predict(self, image, prompt: str) -> str:
        instruction = (
            f"Locate every visible instance of: {prompt}. "
            "Return only a JSON array. Each element must be "
            '{"bbox_2d":[x1,y1,x2,y2],"label":"object"}. '
            "Use integer coordinates normalized from 0 to 1000. "
            "Do not return boxes for similar background structures. Return [] if none are visible."
        )
        content = []
        if self.references:
            content.append({"type": "text", "text": (
                "The following images are human-labeled reference examples. Green boxes are "
                "good_cap, orange boxes are crooked_cap, red boxes are missing_cap vial mouths, "
                "and blue boxes are table_cap. Use their box size and visual meaning as the "
                "standard for the unlabeled target image."
            )})
            content.extend({"type": "image", "image": reference} for reference in self.references)
        content.extend([
            {"type": "text", "text": "This is the unlabeled target image:"},
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ])
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--include-splits", default="train,val")
    parser.add_argument("--reference-images", type=Path)
    parser.add_argument("--reference-labels", type=Path)
    args = parser.parse_args()

    config = load_project_config(args.config)
    if bool(args.reference_images) != bool(args.reference_labels):
        parser.error("--reference-images and --reference-labels must be provided together")
    references = (
        build_reference_overlays(
            args.reference_images, args.reference_labels, args.output / "reference_context"
        ) if args.reference_images else []
    )
    worker = QwenGroundingWorker(args.model, args.max_new_tokens, references)
    report = generate_proposals(
        config, args.images, load_split_manifest(args.splits), args.output, worker.predict,
        model=args.model,
        include_splits={value.strip() for value in re.split(r"[,+]", args.include_splits) if value.strip()} or None,
        response_parser=parse_qwen_grounding_boxes,
        localization_source="qwen3vl_grounding",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
