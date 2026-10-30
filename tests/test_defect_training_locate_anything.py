import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.inspection.defect_training.boxes import Box
from src.inspection.defect_training.config import project_config_from_mapping
from src.inspection.defect_training.locate_anything import (
    generate_proposals,
    merge_prompt_proposals,
    parse_locate_anything_boxes,
)
from src.inspection.defect_training.schemas import read_object_records
from tests.test_defect_training_schemas import valid_config


class LocateAnythingProposalTests(unittest.TestCase):
    def test_parser_scales_clamps_and_suppresses_duplicates(self):
        response = (
            "<box><100><200><300><400></box> "
            "<box><105><205><295><395></box> "
            "<box><900><900><1100><1200></box> "
            "<box><10><10><10><12></box>"
        )
        boxes, raw_count = parse_locate_anything_boxes(response, (200, 100), 0.7)
        self.assertEqual(raw_count, 3)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[-1].as_list(), [180.0, 90.0, 200.0, 100.0])

    def test_multi_prompt_merge_preserves_provenance(self):
        merged = merge_prompt_proposals([
            (Box(10, 10, 30, 30), "vial tops"),
            (Box(11, 11, 29, 29), "loose caps"),
            (Box(50, 50, 70, 70), "loose caps"),
        ], 0.7)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0][1], ("loose caps", "vial tops"))

    def test_generation_resumes_without_reinvoking_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (100, 80), "white").save(images / "a.png")
            Image.new("RGB", (100, 80), "white").save(images / "b.png")
            splits = {
                "a.png": {"split": "train", "group": "a"},
                "b.png": {"split": "test", "group": "b"},
            }
            calls = []

            def predictor(image, prompt):
                calls.append((image.size, prompt))
                return "<box><100><100><300><300></box>"

            config = project_config_from_mapping(valid_config())
            output = root / "output"
            first = generate_proposals(config, images, splits, output, predictor, model="test/model")
            self.assertEqual(first["objects"], 2)
            self.assertEqual(len(calls), 2)
            records = read_object_records(output / "manifest.jsonl")
            self.assertTrue(all(record.crop_box is None for record in records))
            second = generate_proposals(
                config,
                images,
                splits,
                output,
                lambda *_: self.fail("predictor called during resume"),
                model="test/model",
            )
            self.assertEqual(second["resumed_images"], 2)
            prompt = config.localization.prompt
            self.assertEqual(
                json.loads((output / "records" / "a.json").read_text())["responses"][prompt]["raw_response"],
                "<box><100><100><300><300></box>",
            )

    def test_resume_rejects_configuration_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (20, 20), "white").save(images / "a.png")
            splits = {"a.png": {"split": "train", "group": "a"}}
            config = project_config_from_mapping(valid_config())
            output = root / "output"
            generate_proposals(config, images, splits, output, lambda *_: "", model="model-a")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                generate_proposals(config, images, splits, output, lambda *_: "", model="model-b")

    def test_generation_queries_every_configured_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (100, 100), "white").save(images / "a.png")
            payload = valid_config()
            payload["localization"].pop("prompt")
            payload["localization"]["prompts"] = ["vial tops", "loose caps"]
            config = project_config_from_mapping(payload)
            calls = []

            def predictor(_image, prompt):
                calls.append(prompt)
                return "<box><100><100><300><300></box>"

            report = generate_proposals(
                config,
                images,
                {"a.png": {"split": "train", "group": "a"}},
                root / "output",
                predictor,
                model="test/model",
            )
            self.assertEqual(calls, ["vial tops", "loose caps"])
            self.assertEqual(report["objects"], 1)
            record = read_object_records(root / "output" / "manifest.jsonl")[0]
            self.assertEqual(record.metadata["source_prompts"], ["loose caps", "vial tops"])

    def test_over_limit_image_is_flagged_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (100, 100), "white").save(images / "a.png")
            payload = valid_config()
            payload["localization"]["maximum_objects_per_image"] = 1
            config = project_config_from_mapping(payload)
            response = "<box><100><100><200><200></box><box><600><600><700><700></box>"
            report = generate_proposals(
                config,
                images,
                {"a.png": {"split": "train", "group": "a"}},
                root / "output",
                lambda *_: response,
                model="test/model",
            )
            self.assertEqual(report["objects"], 2)
            self.assertEqual(report["over_limit_images"], ["a.png"])

    def test_split_filter_avoids_locked_test_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (50, 50), "white").save(images / "train.png")
            Image.new("RGB", (50, 50), "white").save(images / "test.png")
            calls = []
            report = generate_proposals(
                project_config_from_mapping(valid_config()),
                images,
                {
                    "train.png": {"split": "train", "group": "train"},
                    "test.png": {"split": "test", "group": "test"},
                },
                root / "output",
                lambda *_: calls.append(True) or "",
                model="test/model",
                include_splits={"train"},
            )
            self.assertEqual(report["images"], 1)
            self.assertEqual(report["included_splits"], ["train"])
            self.assertEqual(len(calls), 1)
            self.assertFalse((root / "output" / "records" / "test.json").exists())


if __name__ == "__main__":
    unittest.main()
