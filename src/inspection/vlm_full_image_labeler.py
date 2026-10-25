"""Classify every fixture slot from each full image with one VLM request.

Unlike ``vlm_auto_labeler.py``, this labeler sends the complete fixture image
and asks the model for all 30 aperture decisions in one response. Multiple
OpenAI-compatible llama-server endpoints can be supplied so a Slurm job can
run one model replica per GPU.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = REPO_ROOT / "images" / "pruned_images"
DEFAULT_EXAMPLES_DIR = REPO_ROOT / "images" / "VLM_examples"
APERTURES_PATH = Path(__file__).resolve().parent / "config" / "fixture_g_apertures.json"
CLASS_LABELS = ("pass", "crooked_cap", "missing_cap", "damaged_vial", "no_vial")
OUTPUT_LABELS = (*CLASS_LABELS, "occluded", "unsure")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_data_url(image: Image.Image) -> str:
    """Encode an image as a compact JPEG data URL for the chat API."""

    image_bytes = io.BytesIO()
    image.convert("RGB").save(image_bytes, format="JPEG", quality=92, optimize=True)
    encoded = base64.b64encode(image_bytes.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def load_examples(examples_dir: Path, examples_per_label: int) -> Dict[str, List[Image.Image]]:
    """Load reference images from one directory per label."""

    # Accept either the VLM_examples directory itself or its crop_dataset
    # parent, which is convenient when the examples archive is copied over.
    if not any((examples_dir / label).is_dir() for label in OUTPUT_LABELS):
        nested_examples_dir = examples_dir / "VLM_examples"
        if nested_examples_dir.is_dir():
            examples_dir = nested_examples_dir
    if not examples_dir.is_dir():
        raise FileNotFoundError(
            f"VLM examples directory is missing: {examples_dir}. "
            "Copy the examples dataset there or pass --examples-dir."
        )

    examples: Dict[str, List[Image.Image]] = {}
    for label in OUTPUT_LABELS:
        label_dir = examples_dir / label
        if not label_dir.is_dir():
            continue
        paths = sorted(
            path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if examples_per_label > 0:
            paths = paths[:examples_per_label]
        if paths:
            examples[label] = []
            for path in paths:
                with Image.open(path) as image:
                    examples[label].append(ImageOps.exif_transpose(image).convert("RGB"))
            print(f"[FullImageLabeler] Loaded {len(paths)} {label} example(s)")

    if not examples:
        raise FileNotFoundError(
            f"No labeled example images found below {examples_dir}. "
            "Expected subdirectories such as pass/ and crooked_cap/."
        )
    return examples


def transform_slot_centers(
    apertures: Mapping[str, Mapping[str, Any]],
    image_size: Tuple[int, int],
    source_size: Tuple[int, int],
    crop_fraction: float,
) -> Dict[str, Tuple[int, int]]:
    """Map fixture coordinates onto a centered-cropped image.

    The aperture config uses the original 640x480 capture coordinates. The
    default input directory contains the centered 50% crops, so the original
    crop offset must be removed before scaling to the actual image dimensions.
    """

    source_width, source_height = source_size
    target_width, target_height = image_size
    crop_width = max(1, int(round(source_width * crop_fraction)))
    crop_height = max(1, int(round(source_height * crop_fraction)))
    left = (source_width - crop_width) // 2
    top = (source_height - crop_height) // 2

    centers = {}
    for slot, info in apertures.items():
        x = (float(info["x"]) - left) * target_width / crop_width
        y = (float(info["y"]) - top) * target_height / crop_height
        centers[slot] = (round(x), round(y))
    return centers


def slot_map_text(centers: Mapping[str, Tuple[int, int]], apertures: Mapping[str, Mapping[str, Any]]) -> str:
    lines = []
    for slot, (x, y) in centers.items():
        row = apertures[slot].get("row", "?")
        col = apertures[slot].get("col", "?")
        lines.append(f"{slot}: row {row}, column {col}, center pixel ({x}, {y})")
    return "\n".join(lines)


def inspection_prompt(slot_text: str) -> str:
    return (
        "Inspect the single full fixture image and classify every listed aperture. "
        "The slot map gives the center pixel for each slot in this exact image; use it "
        "to associate each vial or empty opening with the correct slot. Classify all 30 "
        "slots in one response, even when a slot is empty or partly occluded.\n\n"
        "Labels:\n"
        "- pass: intact vial with a yellow cap fully and normally seated.\n"
        "- crooked_cap: a cap is present but raised, shifted, tilted, noncircular, or not fully seated.\n"
        "- missing_cap: clear glass vial or neck is visible without a cap.\n"
        "- no_vial: the fixture opening is empty, with no visible glass.\n"
        "- damaged_vial: visibly cracked, chipped, or broken glass; do not use for a cap defect.\n"
        "- occluded: the vial/cap condition cannot be judged because the robot arm or another object blocks it.\n"
        "- unsure: visible but too ambiguous or low quality to choose reliably.\n\n"
        "The labeled reference images that follow this instruction are part of the "
        "classification instructions. Each reference group is explicitly named with "
        "its label; use those examples to calibrate the appearance of each class, and "
        "do not classify the reference images themselves.\n\n"
        "A dark circular hole or metal rim is not evidence of a vial. For missing_cap, "
        "require visible glass reflections or refraction. A yellow cap whose outline is "
        "not clearly circular is crooked_cap. Favor catching crooked caps: choose pass only "
        "when the yellow cap is clearly circular and fully seated. The yellow robot arm is "
        "not a cap or vial. Compare with the labeled reference groups, but judge each slot "
        "from the full image. Return only JSON with this exact shape: "
        '{"G01":{"label":"pass","confidence":0.0}, ... }\n\n'
        "Slot map:\n" + slot_text
    )


def response_schema(slots: Iterable[str]) -> Dict[str, Any]:
    decision_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(OUTPUT_LABELS)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {slot: decision_schema for slot in slots},
        "required": list(slots),
        "additionalProperties": False,
    }


def normalize_decisions(raw: Any, slots: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Accept the requested object shape and a few common model variants."""

    if isinstance(raw, dict) and isinstance(raw.get("apertures"), dict):
        raw = raw["apertures"]
    elif isinstance(raw, dict) and isinstance(raw.get("slots"), list):
        raw = {entry.get("slot"): entry for entry in raw["slots"] if isinstance(entry, dict)}
    if not isinstance(raw, dict):
        raw = {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for slot in slots:
        value = raw.get(slot, {})
        if not isinstance(value, dict):
            value = {}
        label = str(value.get("label", "unsure")).strip().lower()
        if label not in OUTPUT_LABELS:
            label = "unsure"
        try:
            confidence = float(value.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        normalized[slot] = {
            "label": label,
            "confidence": max(0.0, min(1.0, confidence)),
        }
    return normalized


class FullImageLabeler:
    def __init__(
        self,
        server_urls: Sequence[str],
        model_id: str,
        examples: Mapping[str, Sequence[Image.Image]],
        apertures: Mapping[str, Mapping[str, Any]],
        source_size: Tuple[int, int],
        crop_fraction: float,
        timeout: int,
        retries: int,
    ) -> None:
        self.server_urls = [url.rstrip("/") for url in server_urls]
        self.model_id = model_id
        self.examples = examples
        self.apertures = apertures
        self.slots = list(apertures.keys())
        self.source_size = source_size
        self.crop_fraction = crop_fraction
        self.timeout = timeout
        self.retries = retries
        self.example_content: List[Dict[str, Any]] = []
        for label, images in self.examples.items():
            self.example_content.append({"type": "text", "text": f"Reference group: {label}."})
            for example in images:
                self.example_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data_url(example)},
                })

    def label_image(self, image_path: Path, server_url: str) -> Dict[str, Any]:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        centers = transform_slot_centers(
            self.apertures, image.size, self.source_size, self.crop_fraction
        )
        prompt = inspection_prompt(slot_map_text(centers, self.apertures))
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(self.example_content)
        content.extend([
            {"type": "text", "text": "Now classify every slot in this full image."},
            {"type": "image_url", "image_url": {"url": image_data_url(image)}},
        ])
        payload = json.dumps({
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            # Reserve room for Qwen's hidden reasoning plus the 30-slot JSON.
            "max_tokens": 8192,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "full_fixture_vial_labels",
                    "schema": response_schema(self.slots),
                },
            },
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_effort": "medium",
        }).encode("utf-8")

        request = Request(
            server_url + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error = "unknown error"
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    api_result = json.loads(response.read().decode("utf-8"))
                response_content = api_result["choices"][0]["message"]["content"]
                if isinstance(response_content, list):
                    response_content = "".join(
                        part.get("text", "") for part in response_content if isinstance(part, dict)
                    )
                start, end = response_content.index("{"), response_content.rindex("}") + 1
                decisions = normalize_decisions(json.loads(response_content[start:end]), self.slots)
                return {
                    "filename": image_path.name,
                    "image_path": str(image_path.relative_to(REPO_ROOT)),
                    "apertures": decisions,
                    "server": server_url,
                }
            except (
                HTTPError,
                URLError,
                TimeoutError,
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
                json.JSONDecodeError,
            ) as exc:
                last_error = str(exc)
                if attempt < self.retries:
                    continue

        return {
            "filename": image_path.name,
            "image_path": str(image_path.relative_to(REPO_ROOT)),
            "apertures": normalize_decisions({}, self.slots),
            "server": server_url,
            "error": last_error,
        }


def parse_server_urls(value: str) -> List[str]:
    urls = [url.strip() for url in value.split(",") if url.strip()]
    if not urls:
        raise ValueError("At least one VLM server URL is required")
    return urls


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON checkpoint without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify all 30 vial slots from each full image in one VLM request."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES_DIR)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "images/VLM_samples/full_image_labels.json")
    parser.add_argument(
        "--per-image-dir",
        type=Path,
        default=None,
        help="Write each successful result to <per-image-dir>/<image filename>.json",
    )
    parser.add_argument(
        "--skip-list",
        type=Path,
        default=None,
        help="Text file of image filenames to leave for a later labeling pass",
    )
    parser.add_argument("--server-urls", default=os.environ.get("VLM_SERVER_URLS", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", "qwen3.8-27b-ud-iq3-s"))
    parser.add_argument("--max-images", type=int, default=0, help="Limit images; 0 means all")
    parser.add_argument(
        "--examples-per-label",
        type=int,
        default=0,
        help="Maximum examples per label; 0 means use all examples (default)",
    )
    parser.add_argument("--workers", type=int, default=0, help="Concurrent requests; 0 means one per server")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--source-width", type=int, default=640)
    parser.add_argument("--source-height", type=int, default=480)
    parser.add_argument("--crop-fraction", type=float, default=0.50)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed entries already in --out or --per-image-dir",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.crop_fraction <= 1:
        print("error: --crop-fraction must be greater than 0 and at most 1", file=sys.stderr)
        return 2
    if args.examples_per_label < 0 or args.workers < 0 or args.retries < 0:
        print("error: count options cannot be negative", file=sys.stderr)
        return 2

    image_dir = args.images.expanduser().resolve()
    examples_dir = args.examples_dir.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    per_image_dir = args.per_image_dir.expanduser().resolve() if args.per_image_dir else None
    if not image_dir.is_dir():
        print(f"error: image directory does not exist: {image_dir}", file=sys.stderr)
        return 2

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        print(f"error: no images found in {image_dir}", file=sys.stderr)
        return 2

    skip_names = set()
    if args.skip_list:
        skip_path = args.skip_list.expanduser().resolve()
        if not skip_path.is_file():
            print(f"error: skip list does not exist: {skip_path}", file=sys.stderr)
            return 2
        skip_names = {
            line.strip()
            for line in skip_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    try:
        server_urls = parse_server_urls(args.server_urls)
        examples = load_examples(examples_dir, args.examples_per_label)
        with APERTURES_PATH.open() as handle:
            apertures = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    workers = args.workers or len(server_urls)
    workers = max(1, min(workers, len(image_paths)))
    labeler = FullImageLabeler(
        server_urls=server_urls,
        model_id=args.model,
        examples=examples,
        apertures=apertures,
        source_size=(args.source_width, args.source_height),
        crop_fraction=args.crop_fraction,
        timeout=args.timeout,
        retries=args.retries,
    )

    results: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        if per_image_dir:
            for path in image_paths:
                result_path = per_image_dir / f"{path.name}.json"
                if not result_path.is_file():
                    continue
                try:
                    with result_path.open() as handle:
                        result = json.load(handle)
                    if (
                        isinstance(result, dict)
                        and result.get("filename") == path.name
                        and isinstance(result.get("apertures"), dict)
                        and not result.get("error")
                    ):
                        results[path.name] = result
                except (OSError, json.JSONDecodeError, AttributeError):
                    print(f"[FullImageLabeler] Ignoring invalid result file: {result_path}")
            print(f"[FullImageLabeler] Resuming {len(results)} existing per-image result(s)")
        elif out_path.is_file():
            try:
                with out_path.open() as handle:
                    previous = json.load(handle)
                results = {
                    entry["filename"]: entry
                    for entry in previous.get("images", [])
                    if isinstance(entry, dict) and entry.get("filename")
                }
                print(f"[FullImageLabeler] Resuming {len(results)} existing result(s)")
            except (OSError, json.JSONDecodeError, AttributeError):
                print(f"[FullImageLabeler] Could not read resume file; starting fresh: {out_path}")

    # Retry entries that were recorded with an error. A timeout result is not
    # a completed classification and should not prevent a later continuation
    # job from trying the image again.
    pending = [path for path in image_paths if path.name not in results and path.name not in skip_names]
    print(
        f"[FullImageLabeler] {len(image_paths)} image(s), {len(pending)} pending, "
        f"{len(skip_names)} skipped, {len(server_urls)} server(s), {workers} worker(s)"
    )

    def write_results() -> None:
        ordered = [results[path.name] for path in image_paths if path.name in results]
        write_json_atomic(out_path, {
            "model": args.model,
            "images_dir": str(image_dir),
            "crop_fraction": args.crop_fraction,
            "images": ordered,
        })

    def write_per_image_result(result: Mapping[str, Any]) -> None:
        if per_image_dir is None or result.get("error"):
            return
        filename = result.get("filename")
        if isinstance(filename, str) and filename:
            write_json_atomic(per_image_dir / f"{filename}.json", result)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(labeler.label_image, path, server_urls[index % len(server_urls)]): path
            for index, path in enumerate(pending)
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            path = futures[future]
            try:
                results[path.name] = future.result()
            except Exception as exc:  # Keep the batch moving if one image fails unexpectedly.
                results[path.name] = {
                    "filename": path.name,
                    "image_path": str(path.relative_to(REPO_ROOT)),
                    "apertures": normalize_decisions({}, list(apertures.keys())),
                    "error": str(exc),
                }
            if per_image_dir:
                write_per_image_result(results[path.name])
            else:
                write_results()
            print(f"[FullImageLabeler] {index}/{len(pending)} complete: {path.name}")

    if per_image_dir:
        print(f"[FullImageLabeler] Per-image results saved to {per_image_dir}")
    else:
        write_results()
    failed = sum(1 for result in results.values() if result.get("error"))
    print(f"[FullImageLabeler] Results saved to {out_path}")
    print(f"[FullImageLabeler] Completed: {len(results)}; failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
