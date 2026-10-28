#!/usr/bin/env python3
"""LoRA-tune LocateAnything-3B on cap-state grounding queries.

Training targets use LocateAnything's native normalized coordinate tokens:
<ref>description</ref><box><x1><y1><x2><y2></box>.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoProcessor, AutoTokenizer, get_cosine_schedule_with_warmup


CLASS_NAMES = ["good_cap", "crooked_cap", "missing_cap", "table_cap"]
PROMPTS = {
    0: "yellow cap seated flat and centered on top of a clear plastic vial in the metal tray",
    1: "yellow cap tilted, diagonal, or off-center on top of a clear plastic vial in the metal tray",
    2: "open top of an uncapped clear plastic vial in the metal tray",
    3: "loose yellow cap lying on the white table outside the metal tray",
}


@dataclass(frozen=True)
class Example:
    image: Path
    label: Path
    class_id: int
    source: str
    repeat_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--projector-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--real-repeat", type=int, default=20)
    parser.add_argument("--missing-repeat", type=int, default=2)
    parser.add_argument("--clahe-prob", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def paired_images(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    pairs = []
    for image in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}):
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {image}: {label}")
        pairs.append((image, label))
    return pairs


def build_examples(args: argparse.Namespace) -> list[Example]:
    examples = []
    class_repeats = {0: 1, 1: 1, 2: args.missing_repeat, 3: 1}
    for image, label in paired_images(args.synthetic, "train"):
        for class_id, repeat in class_repeats.items():
            for repeat_index in range(repeat):
                examples.append(Example(image, label, class_id, "synthetic", repeat_index))
    for image, label in paired_images(args.real, "test"):
        for class_id, class_repeat in class_repeats.items():
            for repeat_index in range(args.real_repeat * class_repeat):
                examples.append(Example(image, label, class_id, "real", repeat_index))
    return examples


def clahe_view(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
    return Image.fromarray(cv2.addWeighted(enhanced, 1.25, blur, -0.25, 0))


def target_for(label_path: Path, class_id: int) -> str:
    boxes = []
    for line in label_path.read_text().splitlines():
        values = line.split()
        if not values or int(values[0]) != class_id:
            continue
        cx, cy, width, height = map(float, values[1:5])
        x1 = round(max(0.0, cx - width / 2) * 1000)
        y1 = round(max(0.0, cy - height / 2) * 1000)
        x2 = round(min(1.0, cx + width / 2) * 1000)
        y2 = round(min(1.0, cy + height / 2) * 1000)
        boxes.append((x1, y1, x2, y2))
    boxes.sort(key=lambda box: (box[1], box[0]))
    encoded = "".join(f"<box><{x1}><{y1}><{x2}><{y2}></box>" for x1, y1, x2, y2 in boxes)
    if not encoded:
        encoded = "<box>None</box>"
    return f"<ref>{PROMPTS[class_id]}</ref>{encoded}"


class GroundingDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        processor,
        tokenizer,
        clahe_prob: float,
        seed: int,
    ) -> None:
        self.examples = examples
        self.processor = processor
        self.tokenizer = tokenizer
        self.clahe_prob = clahe_prob
        self.seed = seed
        self.epoch = 0
        self.assistant_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.examples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _assistant_start(self, token_ids: list[int]) -> int:
        marker = self.assistant_marker
        for index in range(len(token_ids) - len(marker), -1, -1):
            if token_ids[index:index + len(marker)] == marker:
                return index + len(marker)
        raise RuntimeError("Assistant marker was not found in tokenized training conversation")

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        image = Image.open(example.image).convert("RGB")
        # Deterministic per epoch/rank-worker so a resumed epoch reproduces its view.
        rng = random.Random(self.seed + self.epoch * len(self.examples) + index)
        use_clahe = rng.random() < self.clahe_prob
        if use_clahe:
            image = clahe_view(image)
        question = f"Locate all the instances that match the following description: {PROMPTS[example.class_id]}."
        answer = target_for(example.label, example.class_id)
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": answer},
        ]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        answer_start = self._assistant_start(labels[0].tolist())
        labels[:, :answer_start] = -100
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_grid_hws": torch.as_tensor(inputs["image_grid_hws"]).squeeze(0),
            "labels": labels.squeeze(0),
            "metadata": {
                "image": str(example.image), "class": CLASS_NAMES[example.class_id],
                "source": example.source, "clahe": use_clahe, "target": answer,
            },
        }


def collate_single(items: list[dict]) -> dict:
    if len(items) != 1:
        raise RuntimeError("This trainer currently requires per-rank batch size 1")
    return items[0]


def save_checkpoint(model, output: Path, state: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    unwrapped.language_model.save_pretrained(output / "llm_lora_adapter")
    torch.save(unwrapped.mlp1.state_dict(), output / "vision_projector.pt")
    (output / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    if is_main:
        print(f"world_size={world_size} device={torch.cuda.get_device_name(device)}", flush=True)
        print("Loading processor/tokenizer/model", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.wrap_llm_lora(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    # PEFT enables input-embedding gradients when gradient checkpointing is on.
    # LocateAnything injects vision embeddings into those token embeddings with an
    # in-place indexed assignment, which PyTorch rejects when the embeddings are a
    # leaf requiring gradients. LoRA parameters still receive gradients without
    # this hook, so remove it before the first forward pass.
    if hasattr(model.language_model, "disable_input_require_grads"):
        model.language_model.disable_input_require_grads()
    model.language_model.get_input_embeddings().weight.requires_grad_(False)
    model.mlp1.requires_grad_(True)
    model.config.use_cache = False
    model.to(device)
    model.train()
    # The bundled Qwen training forward returns an MTP auxiliary tuple and has a
    # bug when the outer LocateAnything wrapper (correctly) computes the labels
    # and loss itself: ``pos_loss_list`` is referenced before assignment. Keep
    # the language-model module in eval mode to select its ordinary structured
    # return path. This does not disable autograd or LoRA updates; it only turns
    # off dropout and the broken auxiliary training return.
    model.language_model.eval()
    # Its inner transformer must remain in training mode so LocateAnything uses
    # the block-mask path that supports injected image embeddings. Only the
    # Qwen2ForCausalLM wrapper itself needs ``training == False``.
    model.language_model.get_base_model().model.train()
    model.mlp1.train()

    examples = build_examples(args)
    dataset = GroundingDataset(examples, processor, tokenizer, args.clahe_prob, args.seed)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_single,
        persistent_workers=args.num_workers > 0,
    )

    lora_params, projector_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (projector_params if name.startswith("mlp1.") else lora_params).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.learning_rate},
            {"params": projector_params, "lr": args.projector_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = args.max_steps or (args.epochs * steps_per_epoch)
    warmup_steps = max(1, round(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    if is_main:
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        config = {
            **vars(args),
            "synthetic": str(args.synthetic), "real": str(args.real), "output": str(args.output),
            "world_size": world_size, "examples": len(examples), "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps, "trainable_parameters": trainable, "total_parameters": total,
            "prompts": {CLASS_NAMES[key]: value for key, value in PROMPTS.items()},
        }
        (args.output / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"examples={len(examples)} trainable={trainable:,}/{total:,} optimizer_steps={total_steps}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    started = time.time()
    stop = False
    for epoch in range(args.epochs if not args.max_steps else 1000000):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            micro_step += 1
            should_step = micro_step % args.grad_accum == 0
            sync_context = nullcontext()
            if isinstance(model, DistributedDataParallel) and not should_step:
                sync_context = model.no_sync()
            with sync_context:
                kwargs = {
                    "input_ids": batch["input_ids"].unsqueeze(0).to(device, non_blocking=True),
                    "attention_mask": batch["attention_mask"].unsqueeze(0).to(device, non_blocking=True),
                    # LocateAnything's processor already flattens one image to
                    # [patches, 3, 14, 14]; the vision tower expects that 4-D form.
                    "pixel_values": batch["pixel_values"].to(device, dtype=torch.bfloat16, non_blocking=True),
                    "image_grid_hws": batch["image_grid_hws"].unsqueeze(0).to(device, non_blocking=True),
                    "labels": batch["labels"].unsqueeze(0).to(device, non_blocking=True),
                    "use_cache": False,
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(**kwargs).loss / args.grad_accum
                loss.backward()
                running_loss += float(loss.detach()) * args.grad_accum
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if is_main and (global_step == 1 or global_step % args.log_every == 0):
                elapsed = time.time() - started
                avg_loss = running_loss / args.log_every if global_step % args.log_every == 0 else running_loss
                print(
                    f"step={global_step}/{total_steps} epoch={epoch} loss={avg_loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} elapsed_min={elapsed / 60:.1f}",
                    flush=True,
                )
                running_loss = 0.0
            if global_step >= total_steps:
                stop = True
                break
        if stop:
            break

    if world_size > 1:
        dist.barrier()
    if is_main:
        state = {
            "global_step": global_step,
            "micro_step": micro_step,
            "elapsed_seconds": time.time() - started,
            "final_learning_rate": scheduler.get_last_lr(),
            "completed": global_step >= total_steps,
        }
        save_checkpoint(model, args.output / "final", state)
        print(f"Saved final adapter to {args.output / 'final'}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
