import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.inspection.defect_training.config import project_config_from_mapping
from src.inspection.defect_training.schemas import read_object_records
from src.inspection.defect_training.seed_import import import_seed_dataset, load_split_manifest
from tests.test_defect_training_schemas import valid_config


class DefectTrainingSeedImportTests(unittest.TestCase):
    def make_source(self, root: Path):
        images = root / "images"
        labels = root / "labels"
        images.mkdir()
        labels.mkdir()
        Image.new("RGB", (100, 80), "white").save(images / "train.png")
        Image.new("RGB", (100, 80), "white").save(images / "test.png")
        (labels / "train.txt").write_text("0 0.500000 0.500000 0.200000 0.250000\n")
        (labels / "test.txt").write_text("1 0.250000 0.250000 0.100000 0.125000\n")
        splits = root / "splits.json"
        splits.write_text(json.dumps({
            "images": {
                "train.png": {"split": "train", "group": "capture_a"},
                "test.png": {"split": "test", "group": "capture_b"},
            }
        }))
        return images, labels, splits

    def test_import_creates_only_training_reference_crops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels, splits = self.make_source(root)
            output = root / "output"
            report = import_seed_dataset(
                project_config_from_mapping(valid_config()),
                images,
                labels,
                load_split_manifest(splits),
                output,
            )
            records = read_object_records(output / "manifest.jsonl")
            self.assertEqual(len(records), 2)
            self.assertEqual(report["objects_by_class"], {"crooked_cap": 1, "good_cap": 1})
            self.assertEqual(len(list((output / "crops" / "good_cap").glob("*.png"))), 1)
            self.assertFalse((output / "crops" / "crooked_cap").exists())
            self.assertTrue((output / "overlays" / "test" / "test.png").is_file())
            self.assertTrue((output / "contact_sheets" / "good_cap.png").is_file())

    def test_group_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps({
                "a.png": {"split": "train", "group": "same"},
                "b.png": {"split": "test", "group": "same"},
            }))
            with self.assertRaisesRegex(ValueError, "cross"):
                load_split_manifest(path)

    def test_split_manifest_must_match_images_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels, splits = self.make_source(root)
            assignments = load_split_manifest(splits)
            assignments.pop("test.png")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                import_seed_dataset(
                    project_config_from_mapping(valid_config()),
                    images,
                    labels,
                    assignments,
                    root / "output",
                )


if __name__ == "__main__":
    unittest.main()
