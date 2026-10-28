#!/usr/bin/env python3
"""Run the downloaded Qwen3.8 VLM as a zero-shot grounding baseline."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import random
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from evaluate_locateanything_real5 import (
    CLASS_NAMES,
    PROMPTS,
    load_ground_truth,
    make_view,
    nms,
    render_comparison,
    render_report,
    summarize,
)


ANGLE_BOX_RE = re.compile(r"<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>", re.I)
QWEN_BOX_RE = re.compile(
    r"<box>\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*,?\s*"
    r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*</box>",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--servers", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3.8-27b-ud-iq3-s")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", default="raw,clahe")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def query(
    server: str,
    model: str,
    image: Image.Image,
    phrase: str,
    max_tokens: int,
    timeout: int,
) -> tuple[str, str]:
    # Keep the user question byte-for-byte equivalent to LocateAnything's
    # documented detection template. The system message only supplies the
    # output grammar that LocateAnything has built into its special tokens.
    question = f"Locate all the instances that match the following description: {phrase}."
    system = (
        "You are an object-grounding detector. Coordinates are normalized integers from 0 to 1000. "
        "Answer only with zero or more tight bounding boxes in this exact format: "
        "<box><x1><y1><x2><y2></box>. If no matching object exists, answer <box>None</box>. "
        "Do not describe the image and do not box objects that do not match the requested description."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url(image)}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_effort": "medium",
    }
    request = Request(
        server.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    message = result["choices"][0]["message"]
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    reasoning = message.get("reasoning_content", "") or message.get("reasoning", "") or ""
    return str(content), str(reasoning)


def parse_boxes(answer: str, width: int, height: int, class_id: int) -> list[dict]:
    matches = list(ANGLE_BOX_RE.finditer(answer))
    if not matches:
        matches = list(QWEN_BOX_RE.finditer(answer))
    boxes = []
    for match in matches:
        x1, y1, x2, y2 = (max(0, min(1000, int(value))) for value in match.groups())
        px = [x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height]
        px[0], px[2] = sorted((px[0], px[2]))
        px[1], px[3] = sorted((px[1], px[3]))
        if px[2] - px[0] >= 2 and px[3] - px[1] >= 2:
            boxes.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id], "box": px})
    return nms(boxes, 0.75)


def main() -> None:
    args = parse_args()
    views = [value.strip() for value in args.views.split(",") if value.strip()]
    servers = [value.strip() for value in args.servers.split(",") if value.strip()]
    if not servers:
        raise ValueError("At least one llama-server URL is required")
    workers = args.workers or len(servers)
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    image_dir = args.dataset / "images" / "test"
    label_dir = args.dataset / "labels" / "test"
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(images) != 5:
        raise RuntimeError(f"Expected exactly five reviewed images, found {len(images)}")

    records_by_view = {view: [] for view in views}
    sample_paths = {view: [] for view in views}
    originals = {}
    tasks = []
    started = time.time()
    for image_index, image_path in enumerate(images, 1):
        original = Image.open(image_path).convert("RGB")
        originals[image_path.name] = original
        width, height = original.size
        gt = load_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        for view in views:
            inference_image = make_view(original, view)
            view_path = args.output / "inference_views" / view / f"{image_path.stem}.jpg"
            view_path.parent.mkdir(parents=True, exist_ok=True)
            inference_image.save(view_path, format="JPEG", quality=95, optimize=True)
            record = {
                "image": str(image_path), "view": view, "size": [width, height],
                "ground_truth": gt, "predictions": [], "responses": {},
            }
            records_by_view[view].append(record)
            for class_id in range(len(CLASS_NAMES)):
                tasks.append(
                    (len(tasks), image_index, image_path.name, view, class_id, inference_image.copy(), record)
                )

    def run_task(task: tuple) -> tuple:
        task_index, image_index, image_name, view, class_id, inference_image, record = task
        call_started = time.time()
        server = servers[task_index % len(servers)]
        answer, reasoning = query(
            server, args.model, inference_image, PROMPTS[class_id], args.max_tokens, args.timeout
        )
        width, height = record["size"]
        boxes = parse_boxes(answer, width, height, class_id)
        return task, server, answer, reasoning, boxes, time.time() - call_started

    print(f"Dispatching {len(tasks)} grounding calls across {workers} workers and {len(servers)} servers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            task, server, answer, reasoning, boxes, elapsed = future.result()
            _, image_index, image_name, view, class_id, _, record = task
            record["predictions"].extend(boxes)
            record["responses"][CLASS_NAMES[class_id]] = {
                "prompt": PROMPTS[class_id], "answer": answer, "reasoning": reasoning,
                "box_count": len(boxes), "server": server,
            }
            print(
                f"[{image_index}/{len(images)}] {image_name} {view} {CLASS_NAMES[class_id]}: "
                f"{len(boxes)} boxes in {elapsed:.1f}s on {server}",
                flush=True,
            )

    for view, records in records_by_view.items():
        for record in records:
            image_path = Path(record["image"])
            original = originals[image_path.name]
            sample = args.output / "prediction_examples" / view / f"{image_path.stem}_gt_vs_prediction.png"
            render_comparison(
                original,
                record["ground_truth"],
                record["predictions"],
                f"{image_path.name} — {view}",
                sample,
                backend_name="Qwen3.8-27B IQ3",
            )
            sample_paths[view].append(sample)

    summaries = {
        view: {"iou_0.25": summarize(records, 0.25), "iou_0.50": summarize(records, 0.50)}
        for view, records in records_by_view.items()
    }
    metrics = {
        "schema_version": 1,
        "model": args.model,
        "runtime": "llama.cpp, UD-IQ3_S GGUF, reasoning enabled",
        "dataset": str(args.dataset),
        "image_count": len(images),
        "views": views,
        "prompts": {CLASS_NAMES[key]: value for key, value in PROMPTS.items()},
        "output_schema_wrapper": "normalized [0,1000] <box><x1><y1><x2><y2></box>",
        "notes": [
            "The class descriptions and user query template match the LocateAnything concise evaluation.",
            "Qwen requires an additional system instruction defining the coordinate output syntax.",
            "The model does not emit calibrated box confidences, so AP/mAP is not defined.",
            "Loose table caps are rendered but not scored because the reviewed annotations omit them.",
        ],
        "elapsed_seconds": time.time() - started,
        "results": summaries,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "predictions_and_raw_responses.json").write_text(json.dumps(records_by_view, indent=2) + "\n")
    render_report(
        args.output,
        summaries,
        sample_paths,
        title="Qwen3.8-27B IQ3 — five reviewed physical frames",
        filename="qwen3_8_27b_real5_report.png",
    )
    print(f"Finished in {metrics['elapsed_seconds'] / 60:.1f} minutes", flush=True)
    print(f"Report: {args.output / 'qwen3_8_27b_real5_report.png'}", flush=True)


if __name__ == "__main__":
    main()
