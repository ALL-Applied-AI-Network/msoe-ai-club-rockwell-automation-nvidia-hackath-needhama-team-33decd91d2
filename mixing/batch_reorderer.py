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

    def reorder_and_compensate(self, solutions: List[Solution]) -> List[Solution]:
        """
        Main entry point:
        1. Reorders solutions to ensure consecutive material compatibility.
        2. Configures intra-solution material dispense order (resident first, handover last).
        3. Calculates compensated pump pulse amounts to achieve exact delivered recipe amounts.
        """
        reordered = [copy.deepcopy(s) for s in self.find_compatible_sequence(solutions)]
        current_tube_fluid = self.initial_tube_fluid

        for idx, sol in enumerate(reordered):
            active_mats = sol.active_materials()
            next_sol = reordered[idx + 1] if idx + 1 < len(reordered) else None

            # Determine entry fluid: must match current tube fluid if possible
            if sol.has_material(current_tube_fluid):
                entry_material = current_tube_fluid
            else:
                # Disjoint transition (only happens if batch has disconnected clusters)
                entry_material = active_mats[0]

            # Determine handover fluid for the next solution
            if next_sol is not None:
                shared = sol.shared_materials(next_sol)
                # Pick shared material (prefer one that isn't the entry material if multiple exist)
                if len(shared) > 1 and entry_material in shared:
                    handover_candidates = [m for m in shared if m != entry_material]
                    handover_material = handover_candidates[0]
                elif shared:
                    handover_material = shared[0]
                else:
                    handover_material = active_mats[-1]
            else:
                # Last solution in batch: prefer clear T100 to leave line in clean state
                handover_material = "T100" if "T100" in active_mats else active_mats[-1]

            # Build intra-solution dispense order: [entry_material, intermediate..., handover_material]
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

