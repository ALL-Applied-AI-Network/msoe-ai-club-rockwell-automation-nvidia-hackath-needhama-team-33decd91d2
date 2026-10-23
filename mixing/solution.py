"""
Solution Class
Represents a liquid formulation/recipe with materials and corresponding volume amounts.
Part of the Predictive Carryover Minimization & Compensation (PCMC) framework.
Rockwell Automation + NVIDIA Innovation Lab Fall 2026 - Team 14
"""

from typing import List, Dict, Set, Optional


class Solution:
    """
    Represents a vial recipe/solution containing an array of materials and amounts.
    """
    def __init__(self, name: str, materials: List[str], amounts: List[float]):
        """
        Args:
            name: Human-readable name or identifier for the solution (e.g. "Recipe 6: Pale Orange").
            materials: Array of material names (e.g. ["T100", "T200", "T400"]).
            amounts: Array of volume amounts in mL corresponding to each material.
        """
        if len(materials) != len(amounts):
            raise ValueError(f"materials length ({len(materials)}) must match amounts length ({len(amounts)})")

        self.name = name
        self.materials = list(materials)
        self.amounts = [float(a) for a in amounts]

        # Lookup mapping: material -> amount
        self._material_dict: Dict[str, float] = {}
        for mat, amt in zip(self.materials, self.amounts):
            self._material_dict[mat] = self._material_dict.get(mat, 0.0) + amt

        # Fields populated after batch scheduling and compensation
        self.dispense_order: List[str] = []
        self.pumped_amounts: Dict[str, float] = {}
        self.delivered_amounts: Dict[str, float] = {}
        self.resident_fluid_inherited: Optional[str] = None
        self.resident_fluid_passed: Optional[str] = None
        self.force_immediate: bool = False  # Manual override to disrupt deferral
        self.requires_purge_if_forced: bool = False
        self.is_bridge_vial: bool = False  # True if this recipe bridges across incompatible campaigns (Option B)
        self.pre_dispense_action: Optional[Dict] = None  # Boundary prime/purge action if triggered (Option A)

        # Validate operating constraints from CSI / MAIC specification
        violations = self.validate_operating_constraints()
        if violations:
            raise ValueError(f"Solution '{name}' violates operating constraints: {'; '.join(violations)}")

    def validate_operating_constraints(self) -> List[str]:
        """
        Validates against the CSI Fluid Mixing Operating Constraints:
        - A vial should contain no more than 80 mL to avoid spilling.
        - A component may be set from 0 to 80 mL, but station should not use less than 10 mL in practice.
        - Do not put more than 50 mL from T-200 in a recipe (causes system fault / reset).
        - Do not use T-300 in a recipe (configuration issue / requires reset).
        """
        violations = []
        if self.total_volume > 80.0:
            violations.append(f"Total volume ({self.total_volume:.1f} mL) > 80 mL maximum")
        for mat, amt in self._material_dict.items():
            if 0.0 < amt < 10.0:
                violations.append(f"Component {mat} ({amt:.1f} mL) < 10 mL minimum practical limit")
            if amt > 80.0:
                violations.append(f"Component {mat} ({amt:.1f} mL) > 80 mL maximum")
        if self.get_amount("T200") > 50.0:
            violations.append(f"T-200 ({self.get_amount('T200'):.1f} mL) > 50 mL limit")
        if self.get_amount("T300") > 0.0:
            violations.append(f"T-300 ({self.get_amount('T300'):.1f} mL) is out of service")
        return violations

    @property
    def total_volume(self) -> float:
        """Total solution volume in mL."""
        return sum(self.amounts)

    def get_amount(self, material: str) -> float:
        """Returns the volume in mL of a specific material required by this solution."""
        return self._material_dict.get(material, 0.0)

    def has_material(self, material: str) -> bool:
        """Returns True if the material is part of this solution with amount > 0."""
        return self.get_amount(material) > 0.0

    def active_materials(self) -> List[str]:
        """Returns the list of materials that have a positive volume amount."""
        return [mat for mat, amt in self._material_dict.items() if amt > 0.0]

    def shared_materials(self, other: "Solution") -> List[str]:
        """
        Returns list of materials present in both this solution and another solution.
        """
        set_self = set(self.active_materials())
        set_other = set(other.active_materials())
        return list(set_self.intersection(set_other))

    def is_compatible_with(self, other: "Solution") -> bool:
        """
        Checks if there is at least one shared active material between the two solutions.
        """
        return len(self.shared_materials(other)) > 0

    def get_material_ratios(self) -> Dict[str, float]:
        """Returns normalized fraction (0.0 to 1.0) of each material."""
        tot = self.total_volume
        if tot <= 0.0:
            return {mat: 0.0 for mat in self.materials}
        return {mat: amt / tot for mat, amt in self._material_dict.items() if amt > 0.0}

    def compute_rgb_color(self) -> tuple:
        """
        Computes the theoretical RGB color tuple (0.0 to 1.0) of this solution.
        Reagents:
          T100: Clear diluent (adds volume, dilutes dye concentration)
          T200: Orange dye (0.95, 0.45, 0.05)
          T400: Blue dye   (0.05, 0.35, 0.88)
        """
        amt_200 = self.get_amount("T200")
        amt_400 = self.get_amount("T400")
        dye_vol = amt_200 + amt_400

        if dye_vol <= 0.0:
            # Pure water / clear
            return (0.85, 0.92, 0.95)

        # Dye relative ratio
        f_orange = amt_200 / dye_vol
        f_blue = amt_400 / dye_vol

        if f_blue == 0.0:
            # Pure orange dye (diluted with water)
            return (0.95, 0.45, 0.05)
        elif f_orange == 0.0:
            # Pure blue dye (diluted with water)
            return (0.05, 0.35, 0.88)
        else:
            # Mixed orange + blue (subtractive mixing produces olive green / dark teal)
            r = f_orange * 0.50 + f_blue * 0.05
            g = f_orange * 0.55 + f_blue * 0.45
            b = f_orange * 0.08 + f_blue * 0.65
            return (min(1.0, max(0.0, r)), min(1.0, max(0.0, g)), min(1.0, max(0.0, b)))

    def to_dict(self) -> dict:
        """Returns a serializable dictionary representation."""
        return {
            "name": self.name,
            "materials": self.materials,
            "amounts": self.amounts,
            "total_volume": self.total_volume,
            "dispense_order": self.dispense_order,
            "pumped_amounts": self.pumped_amounts,
            "delivered_amounts": self.delivered_amounts,
            "resident_fluid_inherited": self.resident_fluid_inherited,
            "resident_fluid_passed": self.resident_fluid_passed,
            "is_bridge_vial": self.is_bridge_vial,
            "pre_dispense_action": self.pre_dispense_action,
            "rgb_color": [round(c, 3) for c in self.compute_rgb_color()],
        }

    def __repr__(self) -> str:
        recipe_str = ", ".join(f"{m}: {self.get_amount(m)}mL" for m in self.active_materials())
        return f"Solution('{self.name}', [{recipe_str}], Total={self.total_volume}mL)"

