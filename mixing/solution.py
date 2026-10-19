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
        Computes the theoretical RGB color tuple (0.0 to 1.0) of this solution
        based on the dye mixing ratio of CSI reagents:
          T100: Clear / diluent (0.95, 0.95, 0.98)
          T200: Orange dye       (0.95, 0.45, 0.05)
          T400: Blue dye         (0.05, 0.40, 0.85)
        """
        ratios = self.get_material_ratios()
        c_100 = ratios.get("T100", 0.0)
        c_200 = ratios.get("T200", 0.0)
        c_400 = ratios.get("T400", 0.0)

        # Baseline color weighting
        r = c_100 * 0.95 + c_200 * 0.95 + c_400 * 0.05
        g = c_100 * 0.95 + c_200 * 0.45 + c_400 * 0.40
        b = c_100 * 0.98 + c_200 * 0.05 + c_400 * 0.85

        # Subtractive darkening effect when dyes are combined
        dye_fraction = c_200 + c_400
        if c_200 > 0.05 and c_400 > 0.05:
            # Orange + Blue produce dark greenish brown
            r *= 0.55
            g *= 0.70
            b *= 0.50

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
            "rgb_color": [round(c, 3) for c in self.compute_rgb_color()],
        }

    def __repr__(self) -> str:
        recipe_str = ", ".join(f"{m}: {self.get_amount(m)}mL" for m in self.active_materials())
        return f"Solution('{self.name}', [{recipe_str}], Total={self.total_volume}mL)"

