"""Vial types and tray layout patterns for remediation simulation and testing.

Provides a centralized registry of vial dimensions, tray slot defect patterns,
and terminal helpers to easily generate and validate diverse digital twin scenes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.remediation.geometry import PhysicalSpec, SPEC, load_layout

ALL_SLOTS = [f"G{i:02d}" for i in range(1, 31)]
GRID_ROWS = 6
GRID_COLS = 5


def slot_at(row: int, col: int) -> str:
    """Return slot ID for 0-indexed row (0..5) and col (0..4)."""
    if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
        raise ValueError(f"row must be in [0..5] and col in [0..4], got ({row}, {col})")
    return f"G{row * GRID_COLS + col + 1:02d}"


def slot_coordinates(slot_id: str) -> Tuple[int, int]:
    """Return (row, col) 0-indexed for slot like 'G01'..'G30'."""
    idx = int(slot_id.replace("G", "")) - 1
    if not (0 <= idx < 30):
        raise ValueError(f"Invalid slot identifier: {slot_id}")
    return divmod(idx, GRID_COLS)


@dataclass(frozen=True)
class VialTypeProfile:
    """Dimensional profile defining a vial geometry."""
    type_id: str
    name: str
    description: str
    vial_outer_radius: float  # metres
    vial_inner_radius: float  # metres
    vial_height: float        # metres
    vial_bottom: float = 0.003327  # metres

    @property
    def wall_thickness(self) -> float:
        return self.vial_outer_radius - self.vial_inner_radius

    @property
    def outer_diameter_mm(self) -> float:
        return self.vial_outer_radius * 2000.0

    @property
    def inner_diameter_mm(self) -> float:
        return self.vial_inner_radius * 2000.0

    @property
    def height_mm(self) -> float:
        return self.vial_height * 1000.0

    def to_spec(self, base_spec: PhysicalSpec = SPEC) -> PhysicalSpec:
        """Derive an authoritative PhysicalSpec with this vial profile's dimensions."""
        return replace(
            base_spec,
            vial_outer_radius=self.vial_outer_radius,
            vial_inner_radius=self.vial_inner_radius,
            vial_height=self.vial_height,
            vial_bottom=self.vial_bottom,
        )


VIAL_TYPES: Dict[str, VialTypeProfile] = {
    "standard": VialTypeProfile(
        type_id="standard",
        name="Standard 8-Inch Clear Plastic Vial",
        description="Official competition specification (1.00\" ID, 1.05\" OD, 8.00\" height).",
        vial_outer_radius=0.02667 / 2,
        vial_inner_radius=0.02540 / 2,
        vial_height=0.20320,
        vial_bottom=0.003327,
    ),
    "short": VialTypeProfile(
        type_id="short",
        name="Short 4-Inch Vial",
        description="Half-height vial (4.00\" height) for testing low-clearance transit.",
        vial_outer_radius=0.02667 / 2,
        vial_inner_radius=0.02540 / 2,
        vial_height=0.10160,
        vial_bottom=0.003327,
    ),
    "tall": VialTypeProfile(
        type_id="tall",
        name="Tall 10-Inch Vial",
        description="Extended height vial (10.00\" height) for testing maximum robot arm reach.",
        vial_outer_radius=0.02667 / 2,
        vial_inner_radius=0.02540 / 2,
        vial_height=0.25400,
        vial_bottom=0.003327,
    ),
    "thick_wall": VialTypeProfile(
        type_id="thick_wall",
        name="Thick-Walled Cryo Vial",
        description="Reinforced thick-walled vial (0.95\" ID, 1.10\" OD, 1.9 mm wall).",
        vial_outer_radius=0.02794 / 2,
        vial_inner_radius=0.02413 / 2,
        vial_height=0.20320,
        vial_bottom=0.005000,
    ),
    "wide": VialTypeProfile(
        type_id="wide",
        name="Wide-Mouth 1.15-Inch Vial",
        description="Wide-aperture vial (1.08\" ID, 1.15\" OD) for relaxed cap insertion.",
        vial_outer_radius=0.02921 / 2,
        vial_inner_radius=0.02743 / 2,
        vial_height=0.20320,
        vial_bottom=0.003327,
    ),
    "micro": VialTypeProfile(
        type_id="micro",
        name="Micro 3-Inch Vial",
        description="Compact vial (0.80\" ID, 0.90\" OD, 3.00\" height).",
        vial_outer_radius=0.02286 / 2,
        vial_inner_radius=0.02032 / 2,
        vial_height=0.07620,
        vial_bottom=0.002500,
    ),
}


def get_vial_profile(profile_or_id: str | VialTypeProfile) -> VialTypeProfile:
    """Retrieve a known vial profile or return the profile instance."""
    if isinstance(profile_or_id, VialTypeProfile):
        return profile_or_id
    key = profile_or_id.lower().strip()
    if key not in VIAL_TYPES:
        available = ", ".join(sorted(VIAL_TYPES.keys()))
        raise ValueError(f"Unknown vial type: '{profile_or_id}'. Available: {available}")
    return VIAL_TYPES[key]


def create_vial_spec(
    vial_type: str = "standard",
    vial_height: Optional[float] = None,
    vial_outer_radius: Optional[float] = None,
    vial_inner_radius: Optional[float] = None,
    base_spec: PhysicalSpec = SPEC,
) -> PhysicalSpec:
    """Create a customized PhysicalSpec combining a profile with optional overrides."""
    profile = get_vial_profile(vial_type)
    height = float(vial_height) if vial_height is not None else profile.vial_height
    outer_r = float(vial_outer_radius) if vial_outer_radius is not None else profile.vial_outer_radius
    inner_r = float(vial_inner_radius) if vial_inner_radius is not None else profile.vial_inner_radius
    if outer_r <= inner_r:
        raise ValueError(f"Outer radius ({outer_r:.5f} m) must be strictly greater than inner radius ({inner_r:.5f} m)")
    if height <= 0.020:
        raise ValueError(f"Vial height must be > 20 mm, got {height:.4f} m")
    return replace(
        base_spec,
        vial_outer_radius=outer_r,
        vial_inner_radius=inner_r,
        vial_height=height,
        vial_bottom=profile.vial_bottom,
    )


@dataclass(frozen=True)
class TrayPatternSpec:
    """Pattern configuration for tray slots and spare table caps."""
    pattern_id: str
    name: str
    description: str
    missing_slots: Tuple[str, ...]
    crooked_slots: Tuple[str, ...]
    empty_slots: Tuple[str, ...]
    table_caps: int

    @property
    def total_slots(self) -> int:
        return 30

    @property
    def loaded_vials(self) -> int:
        return 30 - len(self.empty_slots)

    @property
    def good_slots(self) -> Tuple[str, ...]:
        defective_or_empty = set(self.missing_slots) | set(self.crooked_slots) | set(self.empty_slots)
        return tuple(slot for slot in ALL_SLOTS if slot not in defective_or_empty)

    def validate(self) -> None:
        """Verify slot assignments are distinct and valid."""
        m_set = set(self.missing_slots)
        c_set = set(self.crooked_slots)
        e_set = set(self.empty_slots)
        for name, s in (("missing", m_set), ("crooked", c_set), ("empty", e_set)):
            invalid = [slot for slot in s if slot not in ALL_SLOTS]
            if invalid:
                raise ValueError(f"Unknown slot IDs in {name}: {invalid}")
        if m_set & c_set:
            raise ValueError(f"Slots marked as both missing and crooked: {m_set & c_set}")
        if m_set & e_set:
            raise ValueError(f"Slots marked as both missing and empty: {m_set & e_set}")
        if c_set & e_set:
            raise ValueError(f"Slots marked as both crooked and empty: {c_set & e_set}")
        if self.table_caps < 0 or self.table_caps > 10:
            raise ValueError(f"Table caps must be between 0 and 10, got {self.table_caps}")

    def to_dict(self) -> Dict[str, any]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "missing_slots": list(self.missing_slots),
            "crooked_slots": list(self.crooked_slots),
            "empty_slots": list(self.empty_slots),
            "good_slots": list(self.good_slots),
            "table_caps": self.table_caps,
            "loaded_vials": self.loaded_vials,
        }


def _make_checkerboard() -> Tuple[List[str], List[str]]:
    """Generate alternating checkerboard pattern."""
    missing = []
    crooked = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            slot = slot_at(r, c)
            if (r + c) % 3 == 1:
                missing.append(slot)
            elif (r + c) % 3 == 2:
                crooked.append(slot)
    return missing, crooked


def _make_perimeter() -> Tuple[List[str], List[str]]:
    """Perimeter slots good, interior slots defective."""
    missing = []
    crooked = []
    interior = []
    for r in range(1, GRID_ROWS - 1):
        for c in range(1, GRID_COLS - 1):
            interior.append(slot_at(r, c))
    missing = interior[:len(interior)//2]
    crooked = interior[len(interior)//2:]
    return missing, crooked


PATTERNS: Dict[str, TrayPatternSpec] = {
    "canonical": TrayPatternSpec(
        pattern_id="canonical",
        name="Canonical Benchmark",
        description="Official qualification layout: 1 missing (G11), 1 crooked (G21), 1 spare table cap.",
        missing_slots=("G11",),
        crooked_slots=("G21",),
        empty_slots=(),
        table_caps=1,
    ),
    "full": TrayPatternSpec(
        pattern_id="full",
        name="Fully Capped Clean Tray",
        description="All 30 vials loaded, capped and seated flush. Zero defects.",
        missing_slots=(),
        crooked_slots=(),
        empty_slots=(),
        table_caps=0,
    ),
    "single_missing": TrayPatternSpec(
        pattern_id="single_missing",
        name="Single Missing Cap",
        description="Exactly 1 missing cap at slot G11; tests pure pickup & placement.",
        missing_slots=("G11",),
        crooked_slots=(),
        empty_slots=(),
        table_caps=1,
    ),
    "single_crooked": TrayPatternSpec(
        pattern_id="single_crooked",
        name="Single Crooked Cap",
        description="Exactly 1 crooked cap at slot G21; tests shoulder / cone press remediation.",
        missing_slots=(),
        crooked_slots=("G21",),
        empty_slots=(),
        table_caps=0,
    ),
    "multi_defect": TrayPatternSpec(
        pattern_id="multi_defect",
        name="Multi-Defect Workload",
        description="2 missing caps (G08, G15) and 2 crooked caps (G12, G24) across the tray.",
        missing_slots=("G08", "G15"),
        crooked_slots=("G12", "G24"),
        empty_slots=(),
        table_caps=2,
    ),
    "checkerboard": TrayPatternSpec(
        pattern_id="checkerboard",
        name="Checkerboard Grid Pattern",
        description="Alternating pattern of missing, crooked, and good caps across 6x5 grid.",
        missing_slots=tuple(_make_checkerboard()[0]),
        crooked_slots=tuple(_make_checkerboard()[1]),
        empty_slots=(),
        table_caps=len(_make_checkerboard()[0]),
    ),
    "perimeter": TrayPatternSpec(
        pattern_id="perimeter",
        name="Perimeter Enclosed Pattern",
        description="Good vials on outer perimeter, defects concentrated in the central aperture core.",
        missing_slots=tuple(_make_perimeter()[0]),
        crooked_slots=tuple(_make_perimeter()[1]),
        empty_slots=(),
        table_caps=len(_make_perimeter()[0]),
    ),
    "center_defect": TrayPatternSpec(
        pattern_id="center_defect",
        name="Central Aperture Defects",
        description="The 6 central apertures (G12, G13, G14, G17, G18, G19) defective.",
        missing_slots=("G12", "G14", "G18"),
        crooked_slots=("G13", "G17", "G19"),
        empty_slots=(),
        table_caps=3,
    ),
    "alternating_rows": TrayPatternSpec(
        pattern_id="alternating_rows",
        name="Row-Wise Alternating Pattern",
        description="Rows 0, 2, 4 fully good; Row 1 missing caps; Row 3 crooked caps.",
        missing_slots=("G06", "G07", "G08", "G09", "G10"),
        crooked_slots=("G16", "G17", "G18", "G19", "G20"),
        empty_slots=(),
        table_caps=5,
    ),
    "sparse": TrayPatternSpec(
        pattern_id="sparse",
        name="Sparse Partially Populated Tray",
        description="Only 12 vials present; 18 apertures empty.",
        missing_slots=("G03", "G13"),
        crooked_slots=("G08", "G14"),
        empty_slots=tuple(f"G{i:02d}" for i in range(16, 31)),
        table_caps=2,
    ),
    "stress_test": TrayPatternSpec(
        pattern_id="stress_test",
        name="Batch Stress Test",
        description="4 missing caps and 4 crooked caps to test queue and batch-remediation stability.",
        missing_slots=("G02", "G09", "G17", "G28"),
        crooked_slots=("G04", "G12", "G19", "G25"),
        empty_slots=(),
        table_caps=4,
    ),
}


def parse_slot_list(value: Optional[str]) -> Tuple[str, ...]:
    """Parse a comma-separated list of slots like 'G01,G05,G12'."""
    if not value or not value.strip():
        return ()
    slots = []
    for item in value.split(","):
        s = item.strip().upper()
        if not s:
            continue
        if not s.startswith("G"):
            s = f"G{int(s):02d}"
        if s not in ALL_SLOTS:
            raise ValueError(f"Invalid slot ID: '{item}' (must be G01..G30)")
        slots.append(s)
    return tuple(dict.fromkeys(slots))  # preserve order, eliminate dupes


def resolve_tray_pattern(
    pattern_name: str = "canonical",
    missing_slots: Optional[str | List[str]] = None,
    crooked_slots: Optional[str | List[str]] = None,
    empty_slots: Optional[str | List[str]] = None,
    table_caps: Optional[int] = None,
) -> TrayPatternSpec:
    """Resolve a complete pattern, optionally overriding specific slots."""
    name_clean = pattern_name.lower().strip()
    if name_clean in PATTERNS:
        base = PATTERNS[name_clean]
    elif name_clean == "custom":
        base = TrayPatternSpec(
            pattern_id="custom",
            name="Custom User Pattern",
            description="Operator-defined custom pattern.",
            missing_slots=(),
            crooked_slots=(),
            empty_slots=(),
            table_caps=1,
        )
    else:
        available = ", ".join(sorted(PATTERNS.keys()))
        raise ValueError(f"Unknown pattern: '{pattern_name}'. Available: {available}, custom")

    m = parse_slot_list(missing_slots) if isinstance(missing_slots, str) else (
        tuple(missing_slots) if missing_slots is not None else base.missing_slots)
    c = parse_slot_list(crooked_slots) if isinstance(crooked_slots, str) else (
        tuple(crooked_slots) if crooked_slots is not None else base.crooked_slots)
    e = parse_slot_list(empty_slots) if isinstance(empty_slots, str) else (
        tuple(empty_slots) if empty_slots is not None else base.empty_slots)
    tc = int(table_caps) if table_caps is not None else (len(m) if m != base.missing_slots else base.table_caps)

    resolved = TrayPatternSpec(
        pattern_id=base.pattern_id if not (missing_slots or crooked_slots or empty_slots) else "custom",
        name=base.name if not (missing_slots or crooked_slots or empty_slots) else "Custom Slot Pattern",
        description=base.description,
        missing_slots=m,
        crooked_slots=c,
        empty_slots=e,
        table_caps=tc,
    )
    resolved.validate()
    return resolved


def render_ascii_tray(pattern: TrayPatternSpec) -> str:
    """Render an ASCII diagnostic visual representation of the 6x5 tray layout."""
    lines = [
        f"=== Tray Pattern: {pattern.name} ({pattern.pattern_id}) ===",
        "Legend: [O] Good Cap  [/] Crooked Cap  [X] Missing Cap  [.] Empty Aperture",
        "         Col 0    Col 1    Col 2    Col 3    Col 4",
        "       +--------+--------+--------+--------+--------+",
    ]
    for r in range(GRID_ROWS):
        cells = []
        for c in range(GRID_COLS):
            slot = slot_at(r, c)
            if slot in pattern.empty_slots:
                symbol = " . "
            elif slot in pattern.missing_slots:
                symbol = " X "
            elif slot in pattern.crooked_slots:
                symbol = " / "
            else:
                symbol = " O "
            cells.append(f"{slot}:{symbol}")
        lines.append(f" Row {r} | " + " | ".join(cells) + " |")
        lines.append("       +--------+--------+--------+--------+--------+")
    lines.append(
        f"Summary: {len(pattern.good_slots)} Good | {len(pattern.crooked_slots)} Crooked | "
        f"{len(pattern.missing_slots)} Missing | {len(pattern.empty_slots)} Empty | "
        f"{pattern.table_caps} Spare Table Caps"
    )
    return "\n".join(lines)
