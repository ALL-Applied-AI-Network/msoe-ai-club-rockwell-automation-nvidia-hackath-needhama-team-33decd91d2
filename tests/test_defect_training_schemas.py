import tempfile
import unittest
from pathlib import Path

from src.inspection.defect_training.boxes import Box, expand_box
from src.inspection.defect_training.config import project_config_from_mapping
from src.inspection.defect_training.schemas import (
    ObjectRecord,
    read_object_records,
    stable_object_id,
    write_object_records,
)


def valid_config():
    return {
        "schema_version": 1,
        "project_name": "vial_caps",
        "localization": {
            "model": "nvidia/LocateAnything-3B",
            "prompt": "cap or open top of each vial",
            "nms_iou": 0.7,
        },
        "classes": [
            {"name": "good_cap", "description": "An intact cap."},
            {"name": "crooked_cap", "description": "A visibly crooked cap."},
            {"name": "missing_cap", "description": "An open vial top."},
        ],
        "crop": {"scale": 1.25, "square": False},
        "classification": {
            "uncertain_label": "uncertain",
            "automatic_acceptance": 0.9,
            "review_threshold": 0.7,
            "require_consensus": True,
        },
        "split": {"group_key": "capture", "train": 0.7, "val": 0.15, "test": 0.15},
    }


class DefectTrainingConfigTests(unittest.TestCase):
    def test_valid_configuration(self):
        config = project_config_from_mapping(valid_config())
        self.assertEqual(config.class_names, ("good_cap", "crooked_cap", "missing_cap"))
        self.assertEqual(config.crop.scale, 1.25)

    def test_duplicate_classes_are_rejected(self):
        payload = valid_config()
        payload["classes"][1]["name"] = "good_cap"
        with self.assertRaisesRegex(ValueError, "unique"):
            project_config_from_mapping(payload)

    def test_multiple_unique_localization_prompts_are_supported(self):
        payload = valid_config()
        payload["localization"].pop("prompt")
        payload["localization"]["prompts"] = ["vial tops", "loose caps"]
        config = project_config_from_mapping(payload)
        self.assertEqual(config.localization.prompts, ("vial tops", "loose caps"))

    def test_prompt_and_prompts_cannot_both_be_defined(self):
        payload = valid_config()
        payload["localization"]["prompts"] = ["loose caps"]
        with self.assertRaisesRegex(ValueError, "not both"):
            project_config_from_mapping(payload)

    def test_maximum_objects_must_be_a_positive_integer(self):
        payload = valid_config()
        payload["localization"]["maximum_objects_per_image"] = 35
        self.assertEqual(project_config_from_mapping(payload).localization.maximum_objects_per_image, 35)
        payload["localization"]["maximum_objects_per_image"] = 0
        with self.assertRaisesRegex(ValueError, "positive integer"):
            project_config_from_mapping(payload)

    def test_invalid_threshold_order_is_rejected(self):
        payload = valid_config()
        payload["classification"]["review_threshold"] = 0.95
        with self.assertRaisesRegex(ValueError, "threshold"):
            project_config_from_mapping(payload)

    def test_split_fractions_must_sum_to_one(self):
        payload = valid_config()
        payload["split"]["test"] = 0.2
        with self.assertRaisesRegex(ValueError, "sum"):
            project_config_from_mapping(payload)


class DefectTrainingObjectRecordTests(unittest.TestCase):
    def record(self):
        source = Box(40, 30, 80, 50)
        crop, clipped = expand_box(source, (200, 100), 1.25)
        return ObjectRecord(
            object_id=stable_object_id("images/frame 1.png", source, "human"),
            image="images/frame 1.png",
            image_size=(200, 100),
            split="train",
            split_group="capture_01",
            source_box=source,
            crop_box=crop,
            localization_source="human",
            appearance_label="good_cap",
            label_source="human",
            confidence=1.0,
            status="accepted",
            crop_clipped=clipped,
        )

    def test_stable_id_is_order_independent_and_geometry_sensitive(self):
        box = Box(1, 2, 3, 4)
        first = stable_object_id("a/frame.png", box, "locateanything")
        second = stable_object_id(Path("a/frame.png"), Box.from_sequence(box.as_list()), "locateanything")
        different = stable_object_id("a/frame.png", Box(1, 2, 3, 4.1), "locateanything")
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_jsonl_round_trip_is_atomic_and_lossless(self):
        record = self.record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "manifest.jsonl"
            write_object_records(path, [record])
            self.assertEqual(read_object_records(path), [record])

    def test_duplicate_ids_are_rejected(self):
        record = self.record()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                write_object_records(Path(directory) / "manifest.jsonl", [record, record])

    def test_accepted_record_requires_a_label(self):
        record = self.record()
        values = record.to_dict()
        values["appearance_label"] = None
        values["label_source"] = None
        with self.assertRaisesRegex(ValueError, "requires appearance_label"):
            ObjectRecord.from_dict(values)


if __name__ == "__main__":
    unittest.main()
