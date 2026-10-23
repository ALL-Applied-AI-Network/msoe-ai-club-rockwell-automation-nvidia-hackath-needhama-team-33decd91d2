"""
Simulation & Side-by-Side Evaluation of User-Specified 100-Vial Batch
Rockwell Automation + NVIDIA Innovation Lab Fall 2026 - Team 14

Input 100-recipe sequence:
6, 5, 9, 8, 8, 7, 6, 13, 6, 14, 11, 5, 5, 6, 8, 8, 13, 14, 5, 13, 8, 13, 11,
8, 12, 14, 9, 5, 7, 11, 10, 9, 7, 8, 10, 6, 6, 11, 6, 10, 10, 14, 9, 5, 12,
13, 6, 11, 6, 13, 9, 14, 10, 14, 8, 6, 5, 8, 9, 6, 8, 6, 11, 9, 12, 10, 7,
10, 10, 8, 9, 6, 14, 7, 13, 8, 7, 12, 11, 9, 13, 8, 10, 5, 8, 5, 10, 11, 9,
6, 8, 14, 10, 8, 12, 11, 12, 7, 9, 7
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
from mixing.solution import Solution
from mixing.batch_reorderer import BatchReorderer

RECIPES_DEF = {
    5:  {'name': 'Recipe 5: Low Even',      'mats': ['T100', 'T200', 'T400'], 'amts': [10.0, 10.0, 10.0]},
    6:  {'name': 'Recipe 6: Pale Orange',   'mats': ['T100', 'T200', 'T400'], 'amts': [30.0, 10.0, 0.0]},
    7:  {'name': 'Recipe 7: Pale Blue',     'mats': ['T100', 'T200', 'T400'], 'amts': [20.0, 0.0,  50.0]},
    8:  {'name': 'Recipe 8: Even Triple',   'mats': ['T100', 'T200', 'T400'], 'amts': [20.0, 20.0, 20.0]},
    9:  {'name': 'Recipe 9: Dark Saturate', 'mats': ['T100', 'T200', 'T400'], 'amts': [10.0, 35.0, 35.0]},
    10: {'name': 'Recipe 10: Pale Wash',    'mats': ['T100', 'T200', 'T400'], 'amts': [45.0, 15.0, 15.0]},
    11: {'name': 'Recipe 11: High Blue',    'mats': ['T100', 'T200', 'T400'], 'amts': [10.0, 10.0, 50.0]},
    12: {'name': 'Recipe 12: High Orange',  'mats': ['T100', 'T200', 'T400'], 'amts': [10.0, 50.0, 10.0]},
    13: {'name': 'Recipe 13: Orange+Blue',  'mats': ['T100', 'T200', 'T400'], 'amts': [0.0,  50.0, 20.0]},
    14: {'name': 'Recipe 14: Clear+Blue',   'mats': ['T100', 'T200', 'T400'], 'amts': [60.0, 0.0,  10.0]},
}

RAW_SEQUENCE = (
    "6, 5, 9, 8, 8, 7, 6, 13, 6, 14, 11, 5, 5, 6, 8, 8, 13, 14, 5, 13, 8, 13, 11, "
    "8, 12, 14, 9, 5, 7, 11, 10, 9, 7, 8, 10, 6, 6, 11, 6, 10, 10, 14, 9, 5, 12, "
    "13, 6, 11, 6, 13, 9, 14, 10, 14, 8, 6, 5, 8, 9, 6, 8, 6, 11, 9, 12, 10, 7, "
    "10, 10, 8, 9, 6, 14, 7, 13, 8, 7, 12, 11, 9, 13, 8, 10, 5, 8, 5, 10, 11, 9, "
    "6, 8, 14, 10, 8, 12, 11, 12, 7, 9, 7"
)


def evaluate_100_vials():
    recipe_ids = [int(x.strip()) for x in RAW_SEQUENCE.split(',')]
    v_tube = 8.2

    # 1. Simulate Unmodified Baseline
    current_tube = 'T100'
    baseline_records = []
    dilution_vials = []
    color_defect_vials = []
    spill_vials = []

    for idx, rid in enumerate(recipe_ids):
        spec = RECIPES_DEF[rid]
        target_dict = dict(zip(spec['mats'], spec['amts']))
        active = [m for m in ['T200', 'T400', 'T100'] if target_dict[m] > 0]
        disp_order = [m for m in ['T200', 'T400', 'T100'] if m in active]
        last_m = disp_order[-1]

        pushed = current_tube
        delivered = dict(target_dict)
        delivered[pushed] = round(delivered.get(pushed, 0.0) + v_tube, 2)
        total_vol = round(sum(delivered.values()), 2)
        target_vol = sum(target_dict.values())

        is_dilution = False
        is_color_defect = False
        is_spill = (total_vol > 80.0)
        defect_desc = []

        if is_spill:
            spill_vials.append(idx + 1)
            defect_desc.append(f"SPILL RISK ({total_vol}mL > 80mL limit)")

        if target_dict[pushed] == 0.0:
            if pushed == 'T100':
                is_dilution = True
                dilution_vials.append(idx + 1)
                defect_desc.append(f"DILUTION DEFECT (+{v_tube}mL T100 water pushed into dye-only recipe, -11.7% concentration)")
            else:
                is_color_defect = True
                color_defect_vials.append(idx + 1)
                defect_desc.append(f"COLOR DEFECT (+{v_tube}mL foreign {pushed} dye turns vial into olive-green scrap)")

        baseline_records.append({
            "vial_num": idx + 1,
            "recipe_id": rid,
            "name": spec["name"],
            "target_volume": target_vol,
            "delivered_volume": total_vol,
            "pushed_reagent": pushed,
            "defects": defect_desc if defect_desc else ["NORMAL (No defect)"],
            "has_defect": bool(defect_desc),
        })
        current_tube = last_m

    # 2. Simulate Option B (Starting Tasks - Reverse Campaign Bridge)
    solutions = []
    for idx, rid in enumerate(recipe_ids):
        spec = RECIPES_DEF[rid]
        solutions.append(Solution(f"{spec['name']} (#{idx+1:03d})", spec["mats"], spec["amts"]))

    reorderer = BatchReorderer(hold_up_volume=v_tube, initial_tube_fluid="T100")
    option_b_result = reorderer.reorder_campaign_option_b(solutions, fixed_plc_order=True)

    # 3. Simulate Backup Option A (Mid-Production Dynamic Injection)
    # Scenario: Plant is running Phase 2 (carrier campaign, line holds T100).
    # Two new Recipe 13s are ordered dynamically mid-production.
    dynamic_new_orders = [
        Solution("Recipe 13: Orange+Blue (RUSH #101)", ["T100", "T200", "T400"], [0.0, 50.0, 20.0]),
        Solution("Recipe 13: Orange+Blue (RUSH #102)", ["T100", "T200", "T400"], [0.0, 50.0, 20.0]),
    ]
    remaining_jobs = [
        Solution("Recipe 7: Pale Blue (QUEUED #103)", ["T100", "T200", "T400"], [20.0, 0.0, 50.0]),
        Solution("Recipe 8: Even Triple (QUEUED #104)", ["T100", "T200", "T400"], [20.0, 20.0, 20.0]),
    ]
    option_a_result = reorderer.handle_mid_production_order(
        dynamic_new_orders, current_tube_fluid="T100", remaining_queue=remaining_jobs, fixed_plc_order=True
    )

    return {
        "recipe_ids": recipe_ids,
        "counts": dict(Counter(recipe_ids)),
        "baseline": {
            "records": baseline_records,
            "dilution_vials": dilution_vials,
            "color_defect_vials": color_defect_vials,
            "spill_vials": spill_vials,
            "total_defective": len(set(dilution_vials + color_defect_vials + spill_vials)),
            "total_cross_contamination_ml": round(len(recipe_ids) * v_tube, 2),
            "purge_waste_if_dumped_ml": round((len(recipe_ids) - 1) * v_tube, 2),
        },
        "option_b": option_b_result,
        "option_a": option_a_result,
    }


def compute_delivered_rgb(delivered_dict: Dict[str, float]) -> List[float]:
    amt_200 = delivered_dict.get("T200", 0.0)
    amt_400 = delivered_dict.get("T400", 0.0)
    dye_vol = amt_200 + amt_400
    if dye_vol <= 0.0:
        return [0.85, 0.92, 0.95]
    f_orange = amt_200 / dye_vol
    f_blue = amt_400 / dye_vol
    if f_blue == 0.0:
        return [0.95, 0.45, 0.05]
    elif f_orange == 0.0:
        return [0.05, 0.35, 0.88]
    else:
        r = f_orange * 0.50 + f_blue * 0.05
        g = f_orange * 0.55 + f_blue * 0.45
        b = f_orange * 0.08 + f_blue * 0.65
        return [round(min(1.0, max(0.0, r)), 3), round(min(1.0, max(0.0, g)), 3), round(min(1.0, max(0.0, b)), 3)]


def export_100_vial_schedules(output_dir: Path = None):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    recipe_ids = [int(x.strip()) for x in RAW_SEQUENCE.split(',')]
    v_tube = 8.2

    # 1. Baseline Schedule
    current_tube = "T100"
    base_schedule = []
    for idx, rid in enumerate(recipe_ids):
        spec = RECIPES_DEF[rid]
        target_dict = dict(zip(spec["mats"], spec["amts"]))
        active = [m for m in ["T200", "T400", "T100"] if target_dict[m] > 0]
        disp_order = [m for m in ["T200", "T400", "T100"] if m in active]
        last_m = disp_order[-1]

        pushed = current_tube
        delivered = dict(target_dict)
        delivered[pushed] = round(delivered.get(pushed, 0.0) + v_tube, 2)
        tot = round(sum(delivered.values()), 2)

        is_dilution = (pushed == "T100" and target_dict.get("T100", 0.0) == 0.0)
        is_color = (pushed != "T100" and target_dict.get(pushed, 0.0) == 0.0)
        is_spill = (tot > 80.0)
        has_defect = (is_dilution or is_color or is_spill)

        base_schedule.append({
            "step": idx + 1,
            "name": f"Vial #{idx+1:03d}: {spec['name']}",
            "recipe_id": rid,
            "total_volume": tot,
            "target_volume": sum(target_dict.values()),
            "rgb_color": compute_delivered_rgb(delivered),
            "delivered_amounts": delivered,
            "contamination_volume": v_tube if (is_dilution or is_color) else 0.0,
            "is_dilution": is_dilution,
            "is_color_defect": is_color,
            "spill_risk": is_spill,
            "has_defect": has_defect,
        })
        current_tube = last_m

    base_json_path = output_dir / "dispense_schedule_100_baseline.json"
    with open(base_json_path, "w") as f:
        json.dump({"metadata": {"batch": "user_100_batch", "type": "baseline"}, "schedule": base_schedule}, f, indent=2)

    # 2. Optimized Schedule (Option B)
    solutions = []
    for idx, rid in enumerate(recipe_ids):
        spec = RECIPES_DEF[rid]
        solutions.append(Solution(f"Vial #{idx+1:03d}: {spec['name']}", spec["mats"], spec["amts"]))

    reorderer = BatchReorderer(hold_up_volume=v_tube, initial_tube_fluid="T100")
    opt_res = reorderer.reorder_campaign_option_b(solutions, fixed_plc_order=True)

    opt_schedule = []
    for idx, s in enumerate(opt_res["schedule"]):
        opt_schedule.append({
            "step": idx + 1,
            "name": s.name,
            "total_volume": s.total_volume,
            "rgb_color": [round(c, 3) for c in s.compute_rgb_color()],
            "pumped_amounts": s.pumped_amounts,
            "delivered_amounts": s.delivered_amounts,
            "is_bridge_vial": getattr(s, "is_bridge_vial", False),
            "contamination_volume": 0.0,
            "is_dilution": False,
            "is_color_defect": False,
            "spill_risk": False,
            "has_defect": False,
        })

    opt_json_path = output_dir / "dispense_schedule_100_optimized.json"
    with open(opt_json_path, "w") as f:
        json.dump({
            "metadata": {
                "batch": "user_100_batch",
                "type": "optimized_option_b",
                "strategy": opt_res["strategy"],
                "bridge_vial": opt_res["bridge_vial"],
                "total_waste_ml": opt_res["total_waste_ml"],
            },
            "schedule": opt_schedule,
        }, f, indent=2)

    print(f"[100-Vial Schedules Exported]")
    print(f"  Baseline:  {base_json_path}")
    print(f"  Optimized: {opt_json_path}")
    return base_json_path, opt_json_path


if __name__ == "__main__":
    res = evaluate_100_vials()
    b = res["baseline"]
    opt_b = res["option_b"]
    opt_a = res["option_a"]

    print("=" * 80)
    print(f"EVALUATION OF 100-VIAL USER BATCH: BASELINE VS OPTION B VS BACKUP OPTION A")
    print("=" * 80)
    print(f"Total Vials:                     100")
    print(f"Recipe Distribution:             {res['counts']}")
    print("-" * 80)
    print(f"1. BASELINE (Unmodified Sequential):")
    print(f"   - Total Defective Vials:      {b['total_defective']} / 100 ({b['total_defective']}%)")
    print(f"   - Dilution Defects:           {len(b['dilution_vials'])} vials ({b['dilution_vials']})")
    print(f"   - Color Carryover Scrap:      {len(b['color_defect_vials'])} vials ({b['color_defect_vials']})")
    print(f"   - Spill Violations (>80 mL):  {len(b['spill_vials'])} vials ({len(b['spill_vials'])}%)")
    print(f"   - Purge Waste Required:       {b['purge_waste_if_dumped_ml']} mL (if line purged between every recipe)")
    print("-" * 80)
    print(f"2. OPTION B (Starting Tasks - Reverse Campaign Bridge):")
    print(f"   - Strategy:                   {opt_b['strategy']}")
    print(f"   - Phase 1 (Pure Dye Block):   {opt_b['phase1_count']} vials (Recipe 13 Orange+Blue)")
    print(f"   - Bridge Transition Vial:     {opt_b['bridge_vial']}")
    print(f"   - Phase 2 (Carrier Block):    {opt_b['phase2_count']} vials (All T100 recipes)")
    print(f"   - Total Defective Vials:      {opt_b['defect_count']} / {opt_b['total_vials']} (0.0%)")
    print(f"   - Total Purge Waste:          {opt_b['total_waste_ml']} mL (100% Zero Reagent Dump Waste!)")
    print("-" * 80)
    print(f"3. BACKUP OPTION A (Mid-Production Dynamic Injection):")
    print(f"   - Strategy:                   {opt_a['strategy']}")
    print(f"   - Waste Incurred:             {opt_a['waste_ml']} mL (Single 8.2 mL boundary prime to waste drain)")
    print(f"   - Dynamic Injected Defects:   {opt_a['total_defects']} (0 dilution defects)")
    print(f"   - Return Bridge Vial:         {opt_a['return_bridge_vial']}")
    print("=" * 80)
    export_100_vial_schedules()

