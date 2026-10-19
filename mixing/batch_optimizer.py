"""
Batch Sequence Optimizer & Tube Holdup Compensation
Rockwell Automation + NVIDIA Innovation Lab Fall 2026 - Team 14
"""

import itertools
import numpy as np

RECIPES = {
    5:  {'name': 'Mixed low even',    't100': 10, 't200': 10, 't400': 10, 'total': 30},
    6:  {'name': 'Pale orange',       't100': 30, 't200': 10, 't400': 0,  'total': 40},
    7:  {'name': 'Pale blue',         't100': 20, 't200': 0,  't400': 50, 'total': 70},
    8:  {'name': 'Even',              't100': 20, 't200': 20, 't400': 20, 'total': 60},
    9:  {'name': 'Dark saturated',     't100': 10, 't200': 35, 't400': 35, 'total': 80},
    10: {'name': 'Pale wash',         't100': 45, 't200': 15, 't400': 15, 'total': 75},
    11: {'name': 'Recipe 11',         't100': 10, 't200': 10, 't400': 50, 'total': 70},
    12: {'name': 'Recipe 12',         't100': 10, 't200': 50, 't400': 10, 'total': 70},
    13: {'name': 'Recipe 13',         't100': 0,  't200': 50, 't400': 20, 'total': 70},
    14: {'name': 'Recipe 14',         't100': 60, 't200': 0,  't400': 10, 'total': 70},
}

V_TUBE = 8.0  # mL dead volume in common tube / nozzle

def simulate_schedule(recipe_ids, adaptive_order=False, compensate_volume=False):
    """
    Simulates dispensing a sequence of recipes.
    Returns:
        total_purge_waste (mL)
        total_contamination (mL of unwanted cross-mixed fluid)
        vial_log: detailed per-vial metrics
    """
    tube_fluid = 't100'  # initially flushed with clear diluent
    total_waste = 0.0
    total_contamination = 0.0
    log = []

    for r_id in recipe_ids:
        recipe = RECIPES[r_id]
        needed_ingredients = [k for k in ['t200', 't400', 't100'] if recipe[k] > 0]
        
        if adaptive_order:
            # Reorder: if tube_fluid is needed in recipe, put it first!
            if tube_fluid in needed_ingredients:
                needed_ingredients.remove(tube_fluid)
                needed_ingredients = [tube_fluid] + needed_ingredients
        
        vial_delivered = {'t100': 0.0, 't200': 0.0, 't400': 0.0}
        
        for idx, ing in enumerate(needed_ingredients):
            vol_target = recipe[ing]
            
            # What comes out of the tube first? The resident fluid!
            resident = tube_fluid
            
            if compensate_volume and (resident == ing):
                # Tube already has this fluid!
                # We can credit the tube fluid toward recipe!
                vol_to_pump = max(0.0, vol_target - V_TUBE)
                vial_delivered[resident] += V_TUBE
                vial_delivered[ing] += vol_to_pump
                tube_fluid = ing
            else:
                # Fluid in tube gets pushed into vial
                vol_pushed = V_TUBE
                vial_delivered[resident] += vol_pushed
                
                # Check if resident fluid was unwanted in this recipe
                if recipe.get(resident, 0) == 0:
                    total_contamination += vol_pushed
                
                vol_to_pump = vol_target
                vol_delivered_new = max(0.0, vol_to_pump - V_TUBE)
                vial_delivered[ing] += vol_delivered_new
                tube_fluid = ing
        
        log.append({
            'recipe_id': r_id,
            'delivered': vial_delivered,
            'target': {k: recipe[k] for k in ['t100', 't200', 't400']}
        })

    return total_waste, total_contamination, log

if __name__ == '__main__':
    baseline_seq = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # Baseline: fixed order, no compensation
    waste_base, contam_base, _ = simulate_schedule(baseline_seq, adaptive_order=False, compensate_volume=False)
    print(f"BASELINE: Contamination = {contam_base:.1f} mL, Purge Waste = {waste_base:.1f} mL")
    
    # Adaptive ordering + volume compensation
    waste_opt, contam_opt, _ = simulate_schedule(baseline_seq, adaptive_order=True, compensate_volume=True)
    print(f"SAME SEQUENCE + ADAPTIVE ORDER & COMPENSATION: Contamination = {contam_opt:.1f} mL, Purge Waste = {waste_opt:.1f} mL")
    
    # Full batch optimization: find optimal recipe permutation (ATSP)
    best_seq = None
    min_contam = float('inf')
    
    # Search for optimal permutation among the 10 recipes
    # Since 10! is 3.6M, we can do a quick greedy / TSP search
    current = 5
    unvisited = set(baseline_seq) - {current}
    tsp_seq = [current]
    
    while unvisited:
        best_next = None
        best_cost = float('inf')
        for nxt in unvisited:
            _, c, _ = simulate_schedule([tsp_seq[-1], nxt], adaptive_order=True, compensate_volume=True)
            if c < best_cost:
                best_cost = c
                best_next = nxt
        tsp_seq.append(best_next)
        unvisited.remove(best_next)
    
    waste_tsp, contam_tsp, _ = simulate_schedule(tsp_seq, adaptive_order=True, compensate_volume=True)
    print(f"OPTIMAL BATCH SEQUENCE: {tsp_seq}")
    print(f"OPTIMIZED BATCH: Contamination = {contam_tsp:.1f} mL, Purge Waste = {waste_tsp:.1f} mL")

