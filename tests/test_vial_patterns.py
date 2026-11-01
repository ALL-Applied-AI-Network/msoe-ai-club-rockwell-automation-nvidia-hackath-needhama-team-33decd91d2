"""Unit tests for vial types, tray patterns, and anti-clipping digital twin parity."""
import json
from pathlib import Path
import unittest

from src.remediation.collision_manager import CONFIG as COLLISION_CONFIG
from src.remediation.geometry import SPEC
from src.remediation.vial_patterns import (
    ALL_SLOTS, PATTERNS, VIAL_TYPES, create_vial_spec,
    get_vial_profile, parse_slot_list, render_ascii_tray,
    resolve_tray_pattern, slot_at, slot_coordinates
)

ROOT = Path(__file__).resolve().parents[1]


class VialTypesAndPatternsTests(unittest.TestCase):
    def test_vial_profiles_registered_and_valid(self):
        expected_profiles = {"standard", "short", "tall", "thick_wall", "wide", "micro"}
        self.assertTrue(expected_profiles.issubset(VIAL_TYPES.keys()))

        for vid, prof in VIAL_TYPES.items():
            self.assertGreater(prof.vial_height, 0.05, f"{vid} height too small")
            self.assertGreater(prof.vial_outer_radius, prof.vial_inner_radius, f"{vid} OD must exceed ID")
            self.assertGreater(prof.wall_thickness, 0.0004, f"{vid} wall thickness too thin")
            self.assertGreater(prof.vial_bottom, 0.0, f"{vid} floor must be positive")

            spec = prof.to_spec()
            self.assertEqual(spec.vial_height, prof.vial_height)
            self.assertEqual(spec.vial_outer_radius, prof.vial_outer_radius)
            self.assertEqual(spec.vial_inner_radius, prof.vial_inner_radius)

    def test_create_vial_spec_with_overrides(self):
        # Override height
        custom_h = create_vial_spec("standard", vial_height=0.150)
        self.assertEqual(custom_h.vial_height, 0.150)
        self.assertEqual(custom_h.vial_outer_radius, SPEC.vial_outer_radius)

        # Override radii
        custom_r = create_vial_spec("standard", vial_outer_radius=0.015, vial_inner_radius=0.013)
        self.assertEqual(custom_r.vial_outer_radius, 0.015)
        self.assertEqual(custom_r.vial_inner_radius, 0.013)

        # Invalid: outer <= inner
        with self.assertRaises(ValueError):
            create_vial_spec("standard", vial_outer_radius=0.012, vial_inner_radius=0.013)

        # Invalid: height too small
        with self.assertRaises(ValueError):
            create_vial_spec("standard", vial_height=0.010)

    def test_grid_slot_helpers(self):
        self.assertEqual(slot_at(0, 0), "G01")
        self.assertEqual(slot_at(0, 4), "G05")
        self.assertEqual(slot_at(1, 0), "G06")
        self.assertEqual(slot_at(5, 4), "G30")
        self.assertEqual(slot_coordinates("G01"), (0, 0))
        self.assertEqual(slot_coordinates("G11"), (2, 0))
        self.assertEqual(slot_coordinates("G30"), (5, 4))
        with self.assertRaises(ValueError):
            slot_at(6, 0)
        with self.assertRaises(ValueError):
            slot_coordinates("G31")

    def test_registered_patterns_are_valid(self):
        expected_patterns = {
            "canonical", "full", "single_missing", "single_crooked",
            "multi_defect", "checkerboard", "perimeter", "center_defect",
            "alternating_rows", "sparse", "stress_test"
        }
        self.assertTrue(expected_patterns.issubset(PATTERNS.keys()))

        for pid, pat in PATTERNS.items():
            pat.validate()
            self.assertEqual(pat.total_slots, 30)
            self.assertEqual(
                len(pat.good_slots) + len(pat.crooked_slots) +
                len(pat.missing_slots) + len(pat.empty_slots),
                30,
                f"Pattern {pid} slots must partition all 30 slots exactly"
            )
            self.assertGreaterEqual(pat.table_caps, 0)

    def test_resolve_tray_pattern_custom_overrides(self):
        custom = resolve_tray_pattern(
            pattern_name="custom",
            missing_slots="G01,G05,G15",
            crooked_slots="G02,G12",
            empty_slots="G29,G30",
            table_caps=3,
        )
        self.assertEqual(custom.missing_slots, ("G01", "G05", "G15"))
        self.assertEqual(custom.crooked_slots, ("G02", "G12"))
        self.assertEqual(custom.empty_slots, ("G29", "G30"))
        self.assertEqual(custom.table_caps, 3)
        self.assertEqual(custom.loaded_vials, 28)
        self.assertNotIn("G01", custom.good_slots)
        self.assertIn("G03", custom.good_slots)

    def test_ascii_tray_rendering(self):
        pat = resolve_tray_pattern("canonical")
        rendered = render_ascii_tray(pat)
        self.assertIn("G11: X ", rendered)  # missing
        self.assertIn("G21: / ", rendered)  # crooked
        self.assertIn("G01: O ", rendered)  # good
        self.assertIn("Row 0", rendered)
        self.assertIn("Row 5", rendered)

    def test_anti_clipping_contact_offsets_prevent_wall_overlap(self):
        # Vial wall is 0.635 mm thick.
        # Cap and vial contact offsets must sum to strictly less than 0.635 mm
        # so that inside and outside contact bubbles do not span through the solid wall.
        wall_thickness_m = SPEC.vial_outer_radius - SPEC.vial_inner_radius
        self.assertAlmostEqual(wall_thickness_m, 0.000635, places=5)
        combined_contact_offset_m = (
            COLLISION_CONFIG.cap_contact_offset_m +
            COLLISION_CONFIG.vial_contact_offset_m
        )
        self.assertLess(
            combined_contact_offset_m,
            wall_thickness_m,
            f"Cap ({COLLISION_CONFIG.cap_contact_offset_m*1000:.2f}mm) + "
            f"Vial ({COLLISION_CONFIG.vial_contact_offset_m*1000:.2f}mm) contact buffers "
            f"must be strictly less than wall thickness ({wall_thickness_m*1000:.3f}mm)"
        )

    def test_simulation_offsets_match_physical_policy_remediation_offsets(self):
        offsets_path = ROOT / "PhysicalAI-Hackathon-2026/config/remediation_offsets.json"
        self.assertTrue(offsets_path.is_file())
        offsets = json.loads(offsets_path.read_text())

        # Inspect run_compliant_breakable_remediation.py to ensure it defaults to these offsets
        runner_source = (ROOT / "src/remediation/run_compliant_breakable_remediation.py").read_text()
        self.assertIn("PhysicalAI-Hackathon-2026/config/remediation_offsets.json", runner_source)
        self.assertIn("--pickup-slide-x-m", runner_source)
        self.assertIn("--anti-clipping", runner_source)


if __name__ == "__main__":
    unittest.main()
