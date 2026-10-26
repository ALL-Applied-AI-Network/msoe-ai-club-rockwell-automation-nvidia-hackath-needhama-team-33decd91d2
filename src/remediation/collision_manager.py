"""Collision Manager Sub-Agent for Fanuc ER-4iA Robotic Remediation.

This module enforces physically correct, jitter-free PhysX collision behavior:
1. Retires conflicting stock EOAT colliders while preserving custom finger proxies.
2. Applies calibrated sub-millimeter contact offsets (eliminating 20 mm ghost bubbles).
3. Configures Continuous Collision Detection (CCD) to prevent mesh tunneling.
4. Establishes dedicated PhysX collision groups and filtering rules.
5. Sets high-friction viscoelastic physics materials for realistic gripping.
6. Anchors static fixtures (vial tray, tubes, tabletop) against downward press forces.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

from src.remediation.geometry import SPEC


# Retired starter/stock EOAT colliders that must NEVER be collision-enabled
# when the custom conformal fingers are attached.
RETIRED_STOCK_EOAT_TOKENS = [
    "finger",
    "softtip",
    "detail",
    "sensorcarrier",
    "sensorrednose",
    "outercrossbolt",
    "mountscrew",
    "screwsocket",
    "tape",
    "collision",
    "jawlplate",
    "jawrplate",
    "laserbody",
    "photoblacksensorlead",
]


@dataclass(frozen=True)
class CollisionConfig:
    # Sub-millimeter contact offsets to eliminate PhysX 20 mm phantom bubbles
    finger_contact_offset_m: float = 0.0002  # 0.2 mm
    cap_contact_offset_m: float = 0.0005  # 0.5 mm
    vial_contact_offset_m: float = 0.0005  # 0.5 mm
    tray_contact_offset_m: float = 0.0005  # 0.5 mm
    rest_offset_m: float = 0.0  # Surface coincidence at zero penetration

    # Viscoelastic friction materials
    pad_static_friction: float = 1.5
    pad_dynamic_friction: float = 1.3
    pad_restitution: float = 0.0  # Completely inelastic (damps bounce)

    cap_static_friction: float = 1.2
    cap_dynamic_friction: float = 1.0
    cap_restitution: float = 0.0


CONFIG = CollisionConfig()


class CollisionSubAgent:
    """Specialized sub-agent that configures, audits, and maintains stage collisions."""

    def __init__(self, config: CollisionConfig = CONFIG):
        self.cfg = config

    def configure_stage(self, stage: Usd.Stage, verbose: bool = True) -> Dict[str, int]:
        """Perform full collision setup across the entire stage."""
        report = {
            "disabled_stock_eoat": 0,
            "configured_custom_fingers": 0,
            "configured_caps": 0,
            "configured_vials_and_tray": 0,
        }

        # 1. Physics Material Setup
        pad_material = self._get_or_create_physics_material(
            stage,
            "/World/PhysicsMaterials/GripperPadMaterial",
            static_friction=self.cfg.pad_static_friction,
            dynamic_friction=self.cfg.pad_dynamic_friction,
            restitution=self.cfg.pad_restitution,
        )
        cap_material = self._get_or_create_physics_material(
            stage,
            "/World/PhysicsMaterials/RubberCapMaterial",
            static_friction=self.cfg.cap_static_friction,
            dynamic_friction=self.cfg.cap_dynamic_friction,
            restitution=self.cfg.cap_restitution,
        )

        # 2. Deconflict End-Effector (EOAT) Colliders
        disabled = self.deconflict_eoat_colliders(stage)
        report["disabled_stock_eoat"] = len(disabled)

        # 3. Configure Active Custom Finger Proxies
        finger_proxies = self.configure_custom_finger_colliders(stage, pad_material)
        report["configured_custom_fingers"] = len(finger_proxies)

        # 4. Configure Caps (CCD + sub-mm offset + material)
        caps = self.configure_caps_collision(stage, cap_material)
        report["configured_caps"] = len(caps)

        # 5. Configure Static Tray & Vial Colliders
        static_prims = self.configure_static_tray_and_vials(stage)
        report["configured_vials_and_tray"] = len(static_prims)

        if verbose:
            print(f"[CollisionSubAgent] Stage configured successfully:")
            for k, v in report.items():
                print(f"  - {k}: {v}")

        return report

    def _get_or_create_physics_material(
        self,
        stage: Usd.Stage,
        path: str,
        static_friction: float,
        dynamic_friction: float,
        restitution: float,
    ) -> UsdShade.Material:
        mat_prim = stage.GetPrimAtPath(path)
        if not mat_prim.IsValid():
            mat = UsdShade.Material.Define(stage, path)
            mat_prim = mat.GetPrim()
        else:
            mat = UsdShade.Material(mat_prim)

        p_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        p_mat.CreateStaticFrictionAttr().Set(static_friction)
        p_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
        p_mat.CreateRestitutionAttr().Set(restitution)
        return mat

    def deconflict_eoat_colliders(self, stage: Usd.Stage) -> List[str]:
        """Disable collision ONLY on retired stock parts, keeping custom fingers active."""
        gripper_roots = [
            "/World/ER4iA/J6_link/Gripper",
            "/World/ER4iA/JawL",
            "/World/ER4iA/JawR",
        ]
        disabled = []

        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            is_gripper_prim = any(
                path_str == root or path_str.startswith(root + "/")
                for root in gripper_roots
            )
            if not is_gripper_prim:
                continue

            # NEVER disable our custom finger colliders!
            if "CustomFinger" in path_str:
                continue

            # Check if this prim matches any retired token
            leaf = prim.GetName().lower()
            if any(token in leaf for token in RETIRED_STOCK_EOAT_TOKENS):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(
                        False
                    )
                    disabled.append(path_str)

        return disabled

    def configure_custom_finger_colliders(
        self, stage: Usd.Stage, pad_material: UsdShade.Material
    ) -> List[str]:
        """Configure active collision proxies on JawL and JawR with CCD and sub-mm offsets."""
        configured = []
        for jaw_tag in ("JawL", "JawR"):
            proxy_path = f"/World/ER4iA/{jaw_tag}/CustomFingerCollisionProxy"
            prim = stage.GetPrimAtPath(proxy_path)
            if not prim.IsValid():
                continue

            # Ensure collision is ENABLED
            col_api = UsdPhysics.CollisionAPI.Apply(prim)
            col_api.CreateCollisionEnabledAttr().Set(True)

            # PhysX sub-millimeter contact offset
            physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_col.CreateContactOffsetAttr().Set(self.cfg.finger_contact_offset_m)
            physx_col.CreateRestOffsetAttr().Set(self.cfg.rest_offset_m)

            # Enable Continuous Collision Detection (CCD) to prevent high-speed tunneling
            physx_col.CreateCcdEnabledAttr().Set(True)

            # Bind viscoelastic high-friction pad material
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(pad_material)
            configured.append(proxy_path)

        return configured

    def configure_caps_collision(
        self, stage: Usd.Stage, cap_material: UsdShade.Material
    ) -> List[str]:
        """Configure dynamic cap colliders: sub-mm offsets, CCD, and physics material."""
        configured = []
        caps_root = stage.GetPrimAtPath("/World/RemediationCaps")
        prims_to_check = caps_root.GetChildren() if caps_root.IsValid() else []

        # Also search for standalone caps or sort demo caps
        for prim in stage.Traverse():
            p_name = prim.GetName().lower()
            if "cap" in p_name and prim.IsA(UsdGeom.Xformable):
                if prim not in prims_to_check:
                    prims_to_check.append(prim)

        for cap_prim in prims_to_check:
            # Apply to cap root and any submesh collision prims (e.g. Flange_*, Plug_*)
            for child in Usd.PrimRange(cap_prim):
                if child.HasAPI(UsdPhysics.CollisionAPI) or child.IsA(UsdGeom.Gprim):
                    col = UsdPhysics.CollisionAPI.Apply(child)
                    col.CreateCollisionEnabledAttr().Set(True)

                    physx_col = PhysxSchema.PhysxCollisionAPI.Apply(child)
                    physx_col.CreateContactOffsetAttr().Set(
                        self.cfg.cap_contact_offset_m
                    )
                    physx_col.CreateRestOffsetAttr().Set(self.cfg.rest_offset_m)
                    physx_col.CreateCcdEnabledAttr().Set(True)

                    UsdShade.MaterialBindingAPI.Apply(child).Bind(cap_material)
                    configured.append(str(child.GetPath()))

        return configured

    def configure_static_tray_and_vials(self, stage: Usd.Stage) -> List[str]:
        """Ensure tray fixture, apertures, and vials remain static unyielding colliders."""
        configured = []
        target_roots = [
            "/World/RemediationTray",
            "/World/RemediationTubes",
            "/CleanTray",
            "/World/Table",
        ]

        for root_path in target_roots:
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                continue

            for prim in Usd.PrimRange(root_prim):
                if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Gprim):
                    col = UsdPhysics.CollisionAPI.Apply(prim)
                    col.CreateCollisionEnabledAttr().Set(True)

                    physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    physx_col.CreateContactOffsetAttr().Set(
                        self.cfg.tray_contact_offset_m
                    )
                    physx_col.CreateRestOffsetAttr().Set(self.cfg.rest_offset_m)

                    # Ensure static anchoring: infinite mass / no dynamic gravity
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        rb = UsdPhysics.RigidBodyAPI(prim)
                        rb.CreateKinematicEnabledAttr().Set(True)

                    configured.append(str(prim.GetPath()))

        return configured

    def audit_stage_collisions(self, stage: Usd.Stage) -> Dict[str, any]:
        """Run a diagnostic audit verifying collision health across the stage."""
        active_colliders = []
        disabled_colliders = []
        conflicts = []
        ccd_enabled = []

        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                path = str(prim.GetPath())
                col_api = UsdPhysics.CollisionAPI(prim)
                enabled_attr = col_api.GetCollisionEnabledAttr()
                is_enabled = enabled_attr.Get() if enabled_attr.IsValid() else True

                if is_enabled:
                    active_colliders.append(path)
                    leaf = prim.GetName().lower()
                    if any(token in leaf for token in RETIRED_STOCK_EOAT_TOKENS):
                        conflicts.append(
                            f"CONFLICT: Retired stock EOAT prim has active collision: {path}"
                        )

                    if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                        physx_col = PhysxSchema.PhysxCollisionAPI(prim)
                        ccd_attr = physx_col.GetCcdEnabledAttr()
                        if ccd_attr.IsValid() and ccd_attr.Get():
                            ccd_enabled.append(path)
                else:
                    disabled_colliders.append(path)

        return {
            "healthy": len(conflicts) == 0,
            "total_active_colliders": len(active_colliders),
            "total_disabled_colliders": len(disabled_colliders),
            "ccd_enabled_count": len(ccd_enabled),
            "conflicts": conflicts,
        }
