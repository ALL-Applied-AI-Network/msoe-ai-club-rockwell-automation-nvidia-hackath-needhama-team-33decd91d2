"""Small deterministic LocateAnything runtime for wrist-camera box refinement."""
from __future__ import annotations

import re
import time

import numpy as np
from PIL import Image


BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
PROMPTS = {
    "table_cap": "the single loose yellow vial cap near the center of this wrist camera image",
    "missing_cap": "the single circular open mouth of the uncapped clear plastic vial near the center of this wrist camera image",
    "crooked_cap": "the single tilted yellow cap on a clear plastic vial near the center of this wrist camera image",
}
WRIST_ROI_SIZE_PX = 640
WRIST_ROI_SIZE_BY_CLASS = {
    "table_cap": 640,
    # A centred 320 px crop contains the uncapped mouth while excluding the
    # adjacent capped vials at the calibrated working distance. The prior
    # 640 px crop allowed a neighbouring cap opening 51 mm away to win.
    "missing_cap": 320,
    "crooked_cap": 640,
}


def parse_boxes(answer, width, height):
    boxes = []
    for match in BOX_RE.finditer(str(answer)):
        x0, y0, x1, y1 = map(int, match.groups())
        box = [x0 * width / 1000.0, y0 * height / 1000.0,
               x1 * width / 1000.0, y1 * height / 1000.0]
        box[0], box[2] = sorted((max(0.0, box[0]), min(float(width), box[2])))
        box[1], box[3] = sorted((max(0.0, box[1]), min(float(height), box[3])))
        if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
            boxes.append(box)
    return boxes


def target_roi(image_size, expected_pixel, size_px=WRIST_ROI_SIZE_PX):
    """Return a clamped square ROI centred on the TableCam-projected target."""
    width, height = map(int, image_size)
    side = min(int(size_px), width, height)
    expected = np.asarray(expected_pixel, dtype=float).reshape(2)
    if not np.isfinite(expected).all() or not (0 <= expected[0] < width and
                                               0 <= expected[1] < height):
        raise ValueError("expected wrist target pixel is outside the image")
    x0 = int(round(expected[0] - side / 2))
    y0 = int(round(expected[1] - side / 2))
    x0 = min(max(0, x0), width - side)
    y0 = min(max(0, y0), height - side)
    return x0, y0, x0 + side, y0 + side


def box_from_roi(box, roi_xyxy):
    x0, y0, _x1, _y1 = map(float, roi_xyxy)
    return [float(box[0]) + x0, float(box[1]) + y0,
            float(box[2]) + x0, float(box[3]) + y0]


class LocateAnythingRuntime:
    def __init__(self, model="nvidia/LocateAnything-3B", device="cuda",
                 dtype="float16", max_new_tokens=128):
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("LocateAnything wrist refinement requires a CUDA GPU")
        self.torch = torch
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model, torch_dtype=self.dtype, trust_remote_code=True,
            low_cpu_mem_usage=True).to(device).eval()

    @staticmethod
    def _image(value):
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("wrist RGB must be HxWx3")
        return Image.fromarray(array[:, :, :3].astype(np.uint8), "RGB")

    @staticmethod
    def _nearest(boxes, expected_pixel):
        expected = np.asarray(expected_pixel, dtype=float)
        return min(boxes, key=lambda box: np.linalg.norm(
            np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) - expected))

    def predict(self, image, class_name, expected_pixel):
        if class_name not in PROMPTS:
            raise ValueError(f"unsupported wrist target class: {class_name}")
        full_image = self._image(image)
        roi = target_roi(full_image.size, expected_pixel,
                         size_px=WRIST_ROI_SIZE_BY_CLASS[class_name])
        image = full_image.crop(roi)
        local_expected = np.asarray(expected_pixel, dtype=float) - np.asarray(roi[:2])
        question = f"Locate all the instances that match the following description: {PROMPTS[class_name]}."
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos,
            return_tensors="pt").to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                generation_mode="hybrid",
                do_sample=False,
                repetition_penalty=1.05,
                verbose=False)
        answer = response[0] if isinstance(response, tuple) else response
        answer = answer if isinstance(answer, str) else str(answer)
        boxes = parse_boxes(answer, image.width, image.height)
        if not boxes:
            raise RuntimeError(f"LocateAnything returned no {class_name} box: {answer}")
        box = self._nearest(boxes, local_expected)
        full_boxes = [box_from_roi(item, roi) for item in boxes]
        full_box = box_from_roi(box, roi)
        return {
            "class_name": class_name,
            "bbox_xyxy": list(map(float, full_box)),
            "all_boxes_xyxy": [list(map(float, item)) for item in full_boxes],
            "roi_xyxy": list(map(int, roi)),
            "vlm_input_size": [image.width, image.height],
            "prompt": PROMPTS[class_name],
            "answer": answer,
            "latency_s": time.perf_counter() - started,
        }
