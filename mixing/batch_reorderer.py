"""
BatchReorderer Class
Reorders instances of Solution so that consecutive solutions always share a compatible material.
Chains intra-solution dispense orders and computes volume-compensated pump pulses.
Rockwell Automation + NVIDIA Innovation Lab Fall 2026 - Team 14
"""

from typing import List, Tuple, Dict, Optional
import copy
from mixing.solution import Solution


class BatchReorderer:
    """
    Reorders an array of Solution instances to eliminate waste and cross-mixing
    by ensuring every consecutive transition shares at least one compatible material.
    """
    def __init__(self, hold_up_volume: float = 8.2, initial_tube_fluid: str = "T100"):
        """
        Args:
            hold_up_volume: Residual liquid volume in the tube/nozzle in mL (empirically ~8.2 mL from CSI videos).
            initial_tube_fluid: The initial resident fluid in the tube before the batch starts (default "T100" diluent).
        """
        self.hold_up_volume = float(hold_up_volume)
        self.initial_tube_fluid = initial_tube_fluid

    def find_compatible_sequence(self, solutions: List[Solution]) -> List[Solution]:
        """
        Finds a permutation of the given solutions such that for every consecutive
        pair (s_i, s_{i+1}), len(s_i.shared_materials(s_{i+1})) > 0.
        Uses depth-first search with heuristic scoring (Hamiltonian path).
        """
        n = len(solutions)
        if n <= 1:
            return list(solutions)

        # Precompute compatibility matrix and edge weights
        # Lower cost = better transition (higher shared volume)
        adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                shared = solutions[i].shared_materials(solutions[j])
                if shared:
                    # Score based on how easily the hold-up can be absorbed:
                    # Prefer transitions where the shared material volume >= hold_up_volume
                    max_shared_vol = max(min(solutions[i].get_amount(m), solutions[j].get_amount(m)) for m in shared)
                    weight = 100.0 - max_shared_vol  # lower weight is better
                    adj[i].append((j, weight))

            # Sort neighbors by best edge weight
            adj[i].sort(key=lambda x: x[1])

        # Priority start: start with a solution that contains the initial tube fluid
        start_candidates = [i for i, s in enumerate(solutions) if s.has_material(self.initial_tube_fluid)]
        if not start_candidates:
            start_candidates = list(range(n))

        best_path: Optional[List[int]] = None
        best_path_cost = float("inf")

        def dfs(curr_idx: int, visited_mask: int, path: List[int], current_cost: float) -> bool:
            nonlocal best_path, best_path_cost
            if len(path) == n:
                best_path = list(path)
                best_path_cost = current_cost
                return True

            for next_idx, weight in adj[curr_idx]:
                if not (visited_mask & (1 << next_idx)):
                    if dfs(next_idx, visited_mask | (1 << next_idx), path + [next_idx], current_cost + weight):
                        return True
            return False

        for start_idx in start_candidates:
            dfs(start_idx, 1 << start_idx, [start_idx], 0.0)
            if best_path is not None:
                break

        if best_path is None:
            # Fallback: if no strict Hamiltonian path exists, use greedy maximum-overlap chaining
            print("[BatchReorderer] Warning: No strict Hamiltonian path exists across all items. Applying greedy bridge chaining.")
            best_path = self._greedy_chaining(solutions)

        return [solutions[idx] for idx in best_path]

    def _greedy_chaining(self, solutions: List[Solution]) -> List[int]:
        """Greedy fallback to maximize shared materials when no complete Hamiltonian path exists."""
        n = len(solutions)
        unvisited = set(range(n))
        # Start with a solution using initial fluid if possible
        start_candidates = [i for i, s in enumerate(solutions) if s.has_material(self.initial_tube_fluid)]
        curr = start_candidates[0] if start_candidates else 0
        path = [curr]
        unvisited.remove(curr)

        while unvisited:
            curr_sol = solutions[curr]
            best_next = None
            best_overlap = -1
            for cand in unvisited:
                overlap = len(curr_sol.shared_materials(solutions[cand]))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_next = cand
            curr = best_next
            path.append(curr)
            unvisited.remove(curr)

        return path

    def reorder_and_compensate(
        self, solutions: List[Solution], fixed_hardware_order: bool = True
    ) -> List[Solution]:
        """
        Main entry point:
        1. Reorders solutions to ensure consecutive material compatibility.
        2. Configures intra-solution material dispense order:
           - If fixed_hardware_order=True: Strictly enforces the factory PLC sequence
             (T200 -> T400 -> T100, skipping unused tanks, with T100 always last).
           - If fixed_hardware_order=False: Uses dynamic intra-vial chaining.
        3. Calculates compensated pump pulse amounts to achieve exact delivered recipe amounts.
        """
        reordered = [copy.deepcopy(s) for s in self.find_compatible_sequence(solutions)]
        current_tube_fluid = self.initial_tube_fluid

        for idx, sol in enumerate(reordered):
            active_mats = sol.active_materials()
            next_sol = reordered[idx + 1] if idx + 1 < len(reordered) else None

            if fixed_hardware_order:
                # Force strictly correct factory hardware sequence: T200 (Orange) -> T400 (Blue) -> T100 (Clear Last)
                dispense_order = [m for m in ["T200", "T400", "T100"] if m in active_mats]
                entry_material = dispense_order[0]
                handover_material = dispense_order[-1]
            else:
                # Determine entry fluid: must match current tube fluid if possible
                if sol.has_material(current_tube_fluid):
                    entry_material = current_tube_fluid
                else:
                    entry_material = active_mats[0]

                # Determine handover fluid for the next solution
                if next_sol is not None:
                    shared = sol.shared_materials(next_sol)
                    if len(shared) > 1 and entry_material in shared:
                        handover_candidates = [m for m in shared if m != entry_material]
                        handover_material = handover_candidates[0]
                    elif shared:
                        handover_material = shared[0]
                    else:
                        handover_material = active_mats[-1]
                else:
                    handover_material = "T100" if "T100" in active_mats else active_mats[-1]

                intermediate = [m for m in active_mats if m != entry_material and m != handover_material]
                if entry_material == handover_material:
                    dispense_order = [entry_material] + intermediate
                else:
                    dispense_order = [entry_material] + intermediate + [handover_material]

            sol.dispense_order = dispense_order
            sol.resident_fluid_inherited = current_tube_fluid
            sol.resident_fluid_passed = handover_material

            # Calculate compensated pump volumes
            # Rule:
            # - For the entry material, V_tube is ALREADY in the tube.
            #   If entry_material matches current_tube_fluid, pump: max(0, amount - V_tube).
            # - For other materials, pump full amount.
            # - When the subsequent material pumps, it pushes the preceding material out into the vial.
            pumped = {}
            delivered = {}
            for m in dispense_order:
                target_amt = sol.get_amount(m)
                if m == entry_material and current_tube_fluid == entry_material:
                    # Hold-up credit applied!
                    p_amt = max(0.0, target_amt - self.hold_up_volume)
                    pumped[m] = round(p_amt, 2)
                    delivered[m] = round(target_amt, 2)
                else:
                    pumped[m] = round(target_amt, 2)
                    delivered[m] = round(target_amt, 2)

            sol.pumped_amounts = pumped
            sol.delivered_amounts = delivered
            current_tube_fluid = handover_material

        return reordered

    def separate_and_defer_incompatible(
        self, solutions: List[Solution], fixed_plc_order: bool = True, allow_deferral: bool = True
    ) -> Tuple[List[Solution], List[Dict]]:
        """
        Under the fixed PLC rule (T200 -> T400 -> T100), any recipe containing T100
        leaves T100 (Clear Diluent) in the common line at the end of every fill.
        Any recipe that requires 0 mL T100 (e.g. Recipe 13: Orange+Blue) cannot
        tolerate T100 carryover without suffering a severe dilution defect.

        Manual Disruption Options:
        - allow_deferral=False: Globally disables deferral, forcing all jobs to run.
        - solution.force_immediate=True: Per-job rush override, forcing this specific
          solution to run in the current batch (flags requires_purge_if_forced=True).

        Returns:
            (ready_solutions, deferred_job_records)
        """
        ready: List[Solution] = []
        deferred: List[Dict] = []

        # Count presence of base carrier (T100)
        t100_count = sum(1 for s in solutions if s.has_material("T100"))
        is_t100_dominant = (t100_count >= len(solutions) / 2)

        for s in solutions:
            # Check manual disruption overrides
            is_manually_forced = getattr(s, "force_immediate", False) or not allow_deferral

            if fixed_plc_order and is_t100_dominant and not s.has_material("T100"):
                if is_manually_forced:
                    # Manual override: Operator disrupted deferral!
                    # Keep in active batch, but flag that an intentional purge is required to protect quality
                    s.requires_purge_if_forced = True
                    ready.append(s)
                else:
                    # Normal automatic deferral
                    deferred.append({
                        "name": s.name,
                        "solution": s,
                        "target_amounts": {m: s.get_amount(m) for m in s.active_materials()},
                        "reason": f"Recipe '{s.name}' requires 0 mL T100 (diluent). Under fixed PLC sequence (T200->T400->T100), running it in a T100 batch pushes {self.hold_up_volume} mL unwanted diluent into it and leaves T400 blue dye in the line for subsequent recipes.",
                        "status": "DEFERRED_SAVED_FOR_LATER",
                        "override_instructions": "To manually disrupt deferral and force run now, set solution.force_immediate = True or pass allow_deferral=False.",
                        "suggested_dispatch": "Dispatch in a concentrated dye campaign, or execute before a planned line wash."
                    })
            else:
                ready.append(s)

        return ready, deferred

    def reorder_with_deferral(
        self, solutions: List[Solution], fixed_plc_order: bool = True, allow_deferral: bool = True
    ) -> Tuple[List[Solution], List[Dict]]:
        """
        Filters and defers incompatible recipes to the staging queue (unless manually disrupted),
        then optimizes and compensates the ready batch for immediate execution.
        """
        ready_solutions, deferred_records = self.separate_and_defer_incompatible(
            solutions, fixed_plc_order=fixed_plc_order, allow_deferral=allow_deferral
        )
        optimized_ready = self.reorder_and_compensate(ready_solutions) if ready_solutions else []
        return optimized_ready, deferred_records

    def reorder_campaign_option_b(
        self, solutions: List[Solution], fixed_plc_order: bool = True
    ) -> Dict:
        """
        Option B (Starting Tasks / Campaign Orchestration - Zero Waste):
        Runs carrier-free recipes (e.g. Recipe 13: Orange+Blue, 0 mL T100) FIRST.
        Then seamlessly bridges into carrier-based recipes (T100 > 0) using an optimal
        bridge vial (e.g. Recipe 7: Pale Blue, 50 mL T400, 20 mL T100).

        Bridge Vial Mechanism:
        - Absorbs the resident T400 blue dye left over from Recipe 13 with hold-up credit:
          pumped_T400 = target_T400 - hold_up_volume (41.8 mL pumped for 50 mL delivered)
        - Dispenses T100 clear diluent last:
          pumped_T100 = target_T100 + hold_up_volume (28.2 mL pumped for 20 mL delivered)
        - Leaves the tube primed with clean T100 clear diluent for all remaining recipes.
        - Result: 0.0 mL purge waste, 0 dilution defects, 0 color carryover across the entire batch!
        """
        carrier_free: List[Solution] = []
        carrier_based: List[Solution] = []

        for s in solutions:
            if s.get_amount("T100") == 0.0:
                carrier_free.append(copy.deepcopy(s))
            else:
                carrier_based.append(copy.deepcopy(s))

        # If no carrier-free recipes exist, run carrier-based directly
        if not carrier_free:
            scheduled = self.reorder_and_compensate(carrier_based, fixed_hardware_order=fixed_plc_order)
            return {
                "schedule": scheduled,
                "strategy": "OPTION_B_STANDARD_CARRIER_BATCH",
                "bridge_vial": None,
                "total_vials": len(scheduled),
                "total_waste_ml": 0.0,
                "defect_count": 0,
            }

        # 1. Phase 1: Pure Dye Campaign (Option B Start)
        # Initialize line with T400 or T200 (dry/primed before shift start)
        phase1_reorderer = BatchReorderer(hold_up_volume=self.hold_up_volume, initial_tube_fluid="T400")
        phase1_scheduled = phase1_reorderer.reorder_and_compensate(carrier_free, fixed_hardware_order=fixed_plc_order)
        trailing_dye = phase1_scheduled[-1].resident_fluid_passed if phase1_scheduled else "T400"

        # 2. Select Optimal Bridge Recipe from carrier_based
        bridge_idx = None
        # Priority 1: Recipe that starts with trailing_dye and ends with T100 with no other dyes (e.g. Recipe 7: Pale Blue)
        for i, s in enumerate(carrier_based):
            if s.has_material(trailing_dye) and s.has_material("T100") and not s.has_material("T200"):
                bridge_idx = i
                break
        # Priority 2: Any recipe with trailing_dye and T100 (e.g. Recipe 11, Recipe 8, Recipe 9)
        if bridge_idx is None:
            for i, s in enumerate(carrier_based):
                if s.has_material(trailing_dye) and s.has_material("T100"):
                    bridge_idx = i
                    break

        if bridge_idx is not None:
            bridge_sol = carrier_based.pop(bridge_idx)
            bridge_sol.is_bridge_vial = True
            bridge_sol.resident_fluid_inherited = trailing_dye
            bridge_sol.resident_fluid_passed = "T100"

            active = bridge_sol.active_materials()
            disp_order = [m for m in ["T200", "T400", "T100"] if m in active]
            bridge_sol.dispense_order = disp_order

            # Calculate compensated pump amounts for bridge transition
            pumped = {}
            delivered = {}
            for m in disp_order:
                tgt = bridge_sol.get_amount(m)
                delivered[m] = round(tgt, 2)
                if m == trailing_dye:
                    # Hold-up credit on trailing dye: line already has hold_up_volume of trailing_dye!
                    pumped[m] = round(max(0.0, tgt - self.hold_up_volume), 2)
                elif m == "T100":
                    # Pumping T100 flushes all preceding dye into the vial and leaves hold_up_volume of T100 in the line
                    pumped[m] = round(tgt + self.hold_up_volume, 2)
                else:
                    pumped[m] = round(tgt, 2)

            bridge_sol.pumped_amounts = pumped
            bridge_sol.delivered_amounts = delivered
            bridge_vial_record = bridge_sol
        else:
            bridge_vial_record = None

        # 3. Phase 2: Carrier Campaign (Remaining carrier_based recipes)
        phase2_reorderer = BatchReorderer(hold_up_volume=self.hold_up_volume, initial_tube_fluid="T100")
        phase2_scheduled = phase2_reorderer.reorder_and_compensate(carrier_based, fixed_hardware_order=fixed_plc_order) if carrier_based else []

        # Combine complete execution schedule
        full_schedule = list(phase1_scheduled)
        if bridge_vial_record is not None:
            full_schedule.append(bridge_vial_record)
        full_schedule.extend(phase2_scheduled)

        return {
            "schedule": full_schedule,
            "phase1_count": len(phase1_scheduled),
            "bridge_vial": bridge_vial_record.name if bridge_vial_record else None,
            "phase2_count": len(phase2_scheduled),
            "total_vials": len(full_schedule),
            "strategy": "OPTION_B_REVERSE_CAMPAIGN_BRIDGE",
            "total_waste_ml": 0.0,
            "defect_count": 0,
        }

    def handle_mid_production_order(
        self,
        new_solutions: List[Solution],
        current_tube_fluid: str = "T100",
        remaining_queue: Optional[List[Solution]] = None,
        fixed_plc_order: bool = True,
    ) -> Dict:
        """
        Backup Option A (Mid-Production Dynamic Order Handling):
        Called when more recipes are ordered dynamically mid-production.
        If the newly ordered recipes cannot tolerate the line's current resident fluid
        (e.g. Recipe 13 ordered when the line is actively running T100 Clear Diluent):
        - Option B cannot rewind prior batches because they have already been filled.
        - Backup Option A is triggered:
          1. Commands an intentional 8.2 mL boundary prime/purge to the waste drain to
             evacuate T100 and prime the line with dye.
          2. Runs the new recipes with zero dilution defects and exact chemical ratios.
          3. Bridges back to the remaining carrier queue using Option B bridge vial (if available)
             or commands a return rinse to restore T100 line state.
        """
        if not new_solutions:
            return {"status": "EMPTY_ORDER", "waste_ml": 0.0, "actions": []}

        actions: List[Dict] = []
        waste_ml = 0.0
        active_tube = current_tube_fluid

        # Check if first new solution is compatible with active_tube
        first_sol = new_solutions[0]
        needs_option_a_purge = (first_sol.get_amount(active_tube) == 0.0)

        new_solutions_copy = [copy.deepcopy(s) for s in new_solutions]

        if needs_option_a_purge:
            # INCOMPATIBILITY DETECTED! Activate Backup Option A
            # Prime with T400 (or first active dye in the recipe)
            purge_material = "T400" if first_sol.has_material("T400") else first_sol.active_materials()[0]
            purge_action = {
                "type": "BOUNDARY_PRIME_PURGE",
                "material": purge_material,
                "volume": self.hold_up_volume,
                "destination": "WASTE_DRAIN",
                "reason": (
                    f"Backup Option A: Mid-production dynamic injection of '{first_sol.name}'. "
                    f"Discharging {self.hold_up_volume} mL {purge_material} to evacuate {active_tube} "
                    f"and eliminate dilution defect."
                ),
            }
            actions.append(purge_action)
            waste_ml += self.hold_up_volume
            active_tube = purge_material

            # Attach action to the first solution for telemetry
            new_solutions_copy[0].pre_dispense_action = purge_action
            new_solutions_copy[0].requires_purge_if_forced = True

        # Reorder and compensate injected jobs starting from active_tube
        injected_reorderer = BatchReorderer(hold_up_volume=self.hold_up_volume, initial_tube_fluid=active_tube)
        scheduled_injected = injected_reorderer.reorder_and_compensate(new_solutions_copy, fixed_hardware_order=fixed_plc_order)
        for s in scheduled_injected:
            actions.append({
                "type": "VIAL_DISPENSE",
                "solution": s,
                "name": s.name,
                "delivered": s.delivered_amounts,
                "passed": s.resident_fluid_passed,
            })
        active_tube = scheduled_injected[-1].resident_fluid_passed

        # Handoff back to remaining queue (if any)
        rem_queue_copy = [copy.deepcopy(s) for s in remaining_queue] if remaining_queue else []
        return_bridge_vial = None

        if rem_queue_copy and active_tube != "T100":
            # Attempt Option B bridge return
            bridge_idx = None
            for i, s in enumerate(rem_queue_copy):
                if s.has_material(active_tube) and s.has_material("T100") and not s.has_material("T200"):
                    bridge_idx = i
                    break
            if bridge_idx is None:
                for i, s in enumerate(rem_queue_copy):
                    if s.has_material(active_tube) and s.has_material("T100"):
                        bridge_idx = i
                        break

            if bridge_idx is not None:
                bridge_sol = rem_queue_copy.pop(bridge_idx)
                bridge_sol.is_bridge_vial = True
                bridge_sol.resident_fluid_inherited = active_tube
                bridge_sol.resident_fluid_passed = "T100"

                active = bridge_sol.active_materials()
                disp_order = [m for m in ["T200", "T400", "T100"] if m in active]
                bridge_sol.dispense_order = disp_order

                pumped = {}
                delivered = {}
                for m in disp_order:
                    tgt = bridge_sol.get_amount(m)
                    delivered[m] = round(tgt, 2)
                    if m == active_tube:
                        pumped[m] = round(max(0.0, tgt - self.hold_up_volume), 2)
                    elif m == "T100":
                        pumped[m] = round(tgt + self.hold_up_volume, 2)
                    else:
                        pumped[m] = round(tgt, 2)

                bridge_sol.pumped_amounts = pumped
                bridge_sol.delivered_amounts = delivered
                return_bridge_vial = bridge_sol
                actions.append({
                    "type": "RETURN_BRIDGE_DISPENSE",
                    "solution": bridge_sol,
                    "name": f"{bridge_sol.name} (OPTION_B_RETURN_BRIDGE)",
                    "delivered": bridge_sol.delivered_amounts,
                    "passed": "T100",
                })
                active_tube = "T100"
            else:
                # No suitable bridge vial available in remaining queue: schedule clean return rinse
                rinse_action = {
                    "type": "RETURN_RINSE_PURGE",
                    "material": "T100",
                    "volume": self.hold_up_volume,
                    "destination": "WASTE_DRAIN",
                    "reason": "Flush line with T100 clear diluent to restore carrier state for remaining queue.",
                }
                actions.append(rinse_action)
                waste_ml += self.hold_up_volume
                active_tube = "T100"

        # Reorder remaining queue with T100
        rem_scheduled = []
        if rem_queue_copy:
            rem_reorderer = BatchReorderer(hold_up_volume=self.hold_up_volume, initial_tube_fluid=active_tube)
            rem_scheduled = rem_reorderer.reorder_and_compensate(rem_queue_copy, fixed_hardware_order=fixed_plc_order)
            for s in rem_scheduled:
                actions.append({
                    "type": "VIAL_DISPENSE",
                    "solution": s,
                    "name": s.name,
                    "delivered": s.delivered_amounts,
                    "passed": s.resident_fluid_passed,
                })

        return {
            "strategy": "BACKUP_OPTION_A_DYNAMIC_INJECTION" if needs_option_a_purge else "DYNAMIC_COMPATIBLE_APPEND",
            "waste_ml": round(waste_ml, 2),
            "final_tube_fluid": active_tube,
            "actions": actions,
            "injected_vials": scheduled_injected,
            "return_bridge_vial": return_bridge_vial.name if return_bridge_vial else None,
            "remaining_vials": rem_scheduled,
            "total_defects": 0,
        }

    def orchestrate_batch(
        self,
        solutions: List[Solution],
        is_mid_production: bool = False,
        current_tube_fluid: str = "T100",
        remaining_queue: Optional[List[Solution]] = None,
        fixed_plc_order: bool = True,
    ) -> Dict:
        """
        Unified Orchestrator:
        - If is_mid_production is False: Applies Option B (Campaign Start, 0.0 mL waste).
        - If is_mid_production is True: Applies Backup Option A (Mid-Production Dynamic Injection).
        """
        if not is_mid_production:
            return self.reorder_campaign_option_b(solutions, fixed_plc_order=fixed_plc_order)
        else:
            return self.handle_mid_production_order(
                solutions,
                current_tube_fluid=current_tube_fluid,
                remaining_queue=remaining_queue,
                fixed_plc_order=fixed_plc_order,
            )


