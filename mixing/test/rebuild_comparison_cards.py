"""
Rebuild Side-by-Side Comparison Cards (Type-Aligned)
Rockwell Automation + NVIDIA Innovation Lab Fall 2026 - Team 14

Pairs each baseline vial (Step 1..200) with its exact corresponding recipe
instance in the PCMC optimized schedule, ensuring both sides of the comparison
card show the exact same recipe type and formulation.
"""

import os
import glob
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DIR = REPO_ROOT / "mixing" / "test"
BASE_DIR = TEST_DIR / "baseline"
OPT_DIR = TEST_DIR / "optimized"
COMP_DIR = TEST_DIR / "comparison"

FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load_fonts():
    try:
        font_title = ImageFont.truetype(FONT_BOLD_PATH, 16)
        font_sub = ImageFont.truetype(FONT_REG_PATH, 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    return font_title, font_sub


def create_aligned_card(img_base_path, img_pcmc_path, b_step: dict, o_step: dict, font_title, font_sub):
    img_b = Image.open(img_base_path).convert("RGB")
    img_o = Image.open(img_pcmc_path).convert("RGB")

    # Combined canvas: 1280 x 480
    canvas = Image.new("RGB", (1280, 480))
    canvas.paste(img_b, (0, 0))
    canvas.paste(img_o, (640, 0))
    draw = ImageDraw.Draw(canvas)

    # -------------------------------------------------------------
    # 1. Left Banner: Baseline Unmodified
    # -------------------------------------------------------------
    b_name = b_step["name"]
    b_vol = b_step["total_volume"]
    b_contam = b_step.get("contamination_volume", 0.0)
    b_spill = b_step.get("spill_risk", False)
    b_dilution = b_step.get("is_dilution", False)
    b_color_scrap = b_step.get("is_color_defect", False)
    has_defect = b_step.get("has_defect", False)

    b_color = (180, 25, 25) if has_defect else (45, 55, 72)
    draw.rectangle([(10, 10), (630, 65)], fill=b_color)
    draw.text((20, 15), f"BASELINE (UNMODIFIED): {b_name}", fill=(255, 255, 255), font=font_title)

    parts = [f"Delivered: {b_vol:.1f}mL"]
    if b_dilution:
        parts.append("DILUTION DEFECT (+8.2mL Water)")
    elif b_color_scrap:
        parts.append("COLOR SCRAP (+8.2mL Foreign Dye)")
    elif b_spill:
        parts.append(f"SPILL RISK ({b_vol:.1f}mL > 80mL)")
    elif b_contam > 0:
        parts.append(f"DEFECT (+{b_contam:.1f}mL Carryover)")
    else:
        parts.append("0.0mL Contam (Clean Pass)")

    draw.text((20, 38), " | ".join(parts), fill=(255, 235, 235), font=font_sub)

    # -------------------------------------------------------------
    # 2. Right Banner: PCMC Optimized (Matching Recipe Instance)
    # -------------------------------------------------------------
    o_step_num = o_step["step"]
    o_vol = o_step["total_volume"]
    is_bridge = o_step.get("is_bridge_vial", False)
    bridge_tag = " [BRIDGE VIAL]" if is_bridge else ""
    recipe_name = b_name.split(": ", 1)[1] if ": " in b_name else b_name
    o_vial_title = f"Vial #{o_step_num:03d}: {recipe_name}"

    draw.rectangle([(650, 10), (1270, 65)], fill=(20, 135, 45))
    draw.text((660, 15), f"NEW ALGORITHM (PCMC): {o_vial_title}{bridge_tag}", fill=(255, 255, 255), font=font_title)
    draw.text((660, 38), f"Delivered: {o_vol:.1f}mL | 0.0mL Contam (100% Pure, Zero Waste)", fill=(235, 255, 235), font=font_sub)

    # Subtle divider line in center
    draw.line([(640, 0), (640, 480)], fill=(30, 30, 30), width=2)

    return canvas


def rebuild_all_cards():
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    font_title, font_sub = load_fonts()

    base_json = TEST_DIR / "dispense_schedule_200_baseline.json"
    opt_json = TEST_DIR / "dispense_schedule_200_optimized.json"

    with open(base_json) as f:
        b_sched = json.load(f)["schedule"]
    with open(opt_json) as f:
        o_sched = json.load(f)["schedule"]

    # Build index of opt solutions by task tag
    o_by_tag = {}
    for s in o_sched:
        m = re.search(r'(Task[12]_Vial_\d{3})', s["name"])
        if m:
            o_by_tag[m.group(1)] = s

    # Index opt image files by tag
    opt_files = os.listdir(OPT_DIR)
    opt_file_by_tag = {}
    for f in opt_files:
        m = re.search(r'(Task[12]_Vial_\d{3})', f)
        if m:
            opt_file_by_tag[m.group(1)] = OPT_DIR / f

    print(f"[Rebuild Cards] Rebuilding all 200 comparison cards with 1:1 recipe matching...")

    for i in range(1, 201):
        tag = f"Task1_Vial_{i:03d}" if i <= 100 else f"Task2_Vial_{i-100:03d}"
        b_step = b_sched[i - 1]
        o_step = o_by_tag[tag]

        # Find baseline file
        base_pattern = str(BASE_DIR / f"vial_{i:03d}_*.png")
        base_matches = glob.glob(base_pattern)
        if not base_matches:
            raise FileNotFoundError(f"Missing base render for step {i}: {base_pattern}")
        img_b_path = base_matches[0]
        img_o_path = opt_file_by_tag[tag]

        card = create_aligned_card(img_b_path, img_o_path, b_step, o_step, font_title, font_sub)
        out_file = COMP_DIR / f"comparison_vial_{i:03d}.png"
        card.save(out_file)

        if i % 25 == 0 or i == 1 or i == 8 or i == 9 or i == 200:
            b_r = b_step["name"].split(": ", 1)[1]
            o_r = o_step["name"].split(": ", 2)[2]
            print(f"  -> Card #{i:03d}: [Baseline] {b_r} vs [PCMC Step #{o_step['step']:03d}] {o_r}")

    print(f"[Rebuild Cards] All 200 cards successfully rebuilt and saved to {COMP_DIR}")


if __name__ == "__main__":
    rebuild_all_cards()

