#!/usr/bin/env python3
"""Scene Pattern and Vial Type Management Tool for Final Presentation & Testing.

Allows operators to easily:
1. Inspect registered vial profiles (standard, short, tall, thick_wall, wide, micro).
2. Visualize 6x5 tray patterns in ASCII (canonical, full, checkerboard, perimeter, etc.).
3. Generate customized USDA simulation scenes with one terminal command.
4. Test different vial types and defect patterns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.remediation.vial_patterns import (
    ALL_SLOTS, PATTERNS, VIAL_TYPES, create_vial_spec,
    get_vial_profile, render_ascii_tray, resolve_tray_pattern
)


def list_vials() -> None:
    print("=" * 78)
    print("  REGISTERED VIAL PROFILES (for test suites and digital twin simulation)")
    print("=" * 78)
    header = f"{'Type ID':<12} | {'Height (mm)':<11} | {'OD (mm)':<8} | {'ID (mm)':<8} | {'Wall (mm)':<9} | {'Description'}"
    print(header)
    print("-" * 78)
    for vid, prof in sorted(VIAL_TYPES.items()):
        print(f"{prof.type_id:<12} | {prof.height_mm:<11.1f} | {prof.outer_diameter_mm:<8.2f} | {prof.inner_diameter_mm:<8.2f} | {prof.wall_thickness*1000:<9.2f} | {prof.name}")
    print("=" * 78)


def list_patterns() -> None:
    print("=" * 78)
    print("  REGISTERED TRAY PATTERNS (6x5 Grid Layout)")
    print("=" * 78)
    for pid, pat in sorted(PATTERNS.items()):
        print(f"\n[{pid}] - {pat.name}")
        print(f"  Description: {pat.description}")
        print(f"  Good: {len(pat.good_slots)} | Missing: {len(pat.missing_slots)} {list(pat.missing_slots)} | Crooked: {len(pat.crooked_slots)} {list(pat.crooked_slots)} | Empty: {len(pat.empty_slots)} | Spares: {pat.table_caps}")
    print("=" * 78)


def show_pattern(pattern_id: str, missing: str = None, crooked: str = None, empty: str = None, table_caps: int = None) -> None:
    resolved = resolve_tray_pattern(pattern_id, missing, crooked, empty, table_caps)
    print("\n" + render_ascii_tray(resolved) + "\n")


def print_terminal_commands() -> None:
    print("=" * 78)
    print("  READY-TO-USE TERMINAL COMMANDS FOR THE FINAL PRESENTATION")
    print("=" * 78)
    print("""
1. QUALIFY A SCENE WITH A PRESET PATTERN:
   # Checkerboard pattern:
   ./scripts/run_isaac.sh src/remediation/qualify_scene.py \\
       --pattern checkerboard --out shared/runs/demo_checkerboard --scene-out shared/scenes/demo_checkerboard.usda

   # Perimeter / inner-defect pattern:
   ./scripts/run_isaac.sh src/remediation/qualify_scene.py \\
       --pattern perimeter --out shared/runs/demo_perimeter --scene-out shared/scenes/demo_perimeter.usda

   # Single missing cap at G11:
   ./scripts/run_isaac.sh src/remediation/qualify_scene.py \\
       --pattern single_missing --out shared/runs/demo_single_missing --scene-out shared/scenes/demo_single_missing.usda

   # Single crooked cap at G21:
   ./scripts/run_isaac.sh src/remediation/qualify_scene.py \\
       --pattern single_crooked --out shared/runs/demo_single_crooked --scene-out shared/scenes/demo_single_crooked.usda

2. TEST DIFFERENT VIAL TYPES:
   # Half-height 4-inch vials:
   ./scripts/run_isaac.sh src/remediation/qualify_scene.py \\
       --vial-type short --pattern canonical --out shared/runs/demo_short_vials --scene-out shared/scenes/demo_short.usda

   # Thick-walled cryo vials:
   ./scripts/run_isaac.sh src/remediation/qualify_scene.py \\
       --vial-type thick_wall --pattern canonical --out shared/runs/demo_thick --scene-out shared/scenes/demo_thick.usda

   # Custom height vial (180 mm = 0.180 m):
   ./scripts/run_isaac.sh src/remediation/qualify_scene.py \\
       --vial-height 0.180 --pattern canonical --out shared/runs/demo_custom_h --scene-out shared/scenes/demo_custom_h.usda

3. INSERT CUSTOM DEFECT PATTERNS:
   # Specific missing slots (G01, G05, G15) and crooked slots (G02, G12):
   ./scripts/run_isaac.sh src/remediation/qualify_scene.py \\
       --pattern custom --missing-slots G01,G05,G15 --crooked-slots G02,G12 --table-caps 3 \\
       --out shared/runs/demo_custom --scene-out shared/scenes/demo_custom.usda

4. RUN FULL SIMULATION REMEDIATION (PHYSICS ANTI-CLIPPING):
   sbatch scripts/run_compliant_breakable_pipeline.slurm \\
       shared/scenes/tablecam_right_pink_spares_qualified_v2.usda \\
       shared/yolo26x/best.pt \\
       shared/runs/remediation_physics_presentation

5. RUN FULL SIMULATION REMEDIATION (LOCK-INTO-PLACE ASSIST):
   sbatch scripts/run_compliant_breakable_pipeline.slurm \\
       shared/scenes/tablecam_right_pink_spares_qualified_v2.usda \\
       shared/yolo26x/best.pt \\
       shared/runs/remediation_lock_presentation \\
       --lock-seated
""")
    print("=" * 78)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-vials", action="store_true", help="List all registered vial types and physical dimensions")
    p.add_argument("--list-patterns", action="store_true", help="List all registered tray patterns")
    p.add_argument("--show-pattern", type=str, metavar="NAME", help="Display visual ASCII layout of a pattern")
    p.add_argument("--commands", action="store_true", help="Display ready-to-run terminal commands for presentations")
    p.add_argument("--missing-slots", help="Comma-separated missing slots for custom pattern preview")
    p.add_argument("--crooked-slots", help="Comma-separated crooked slots for custom pattern preview")
    p.add_argument("--empty-slots", help="Comma-separated empty slots for custom pattern preview")
    p.add_argument("--table-caps", type=int, help="Number of spare table caps")
    args = p.parse_args()

    if args.list_vials:
        list_vials()
    elif args.list_patterns:
        list_patterns()
    elif args.show_pattern:
        show_pattern(args.show_pattern, args.missing_slots, args.crooked_slots, args.empty_slots, args.table_caps)
    elif args.commands:
        print_terminal_commands()
    else:
        list_vials()
        print()
        list_patterns()
        print()
        print_terminal_commands()


if __name__ == "__main__":
    main()
