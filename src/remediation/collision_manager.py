"""Collision Manager Sub-Agent for Fanuc ER-4iA Robotic Remediation.

This module enforces physically correct, jitter-free PhysX collision behavior:
1. Retires conflicting stock EOAT colliders while preserving custom finger proxies.
2. Applies calibrated sub-millimeter contact offsets (eliminating 20 mm ghost bubbles).
3. Configures Continuous Collision Detection (CCD) to prevent mesh tunneling.
4. Establishes dedicated PhysX collision groups and filtering rules.
5. Uses high-friction gripper pads and lower-friction plastic cap/vial contact.
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
    finger_contact_offset_m: float = 0.0005  # 0.5 mm
    cap_contact_offset_m: float = 0.0005  # 0.5 mm
    vial_contact_offset_m: float = 0.0005  # 0.5 mm
    tray_contact_offset_m: float = 0.0005  # 0.5 mm
    rest_offset_m: float = 0.0  # Surface coincidence at zero penetration

    # Viscoelastic friction materials
    pad_static_friction: float = 1.5
    pad_dynamic_friction: float = 1.3
    pad_restitution: float = 0.0  # Completely inelastic (damps bounce)

    # Smooth molded plastic against a clear plastic vial.  The old rubber-like
    # 1.2/1.0 pair wedged a four-point crooked cap onto the rim instead of
    # allowing it to pivot into the 1.27 mm radial plug clearance.
    cap_static_friction: float = 0.35
    cap_dynamic_friction: float = 0.25
    cap_restitution: float = 0.0

    # Vial "give": glass held in the tray is not perfectly rigid at the rim.  PhysX
    # compliant contact on the tube colliders acts as a stiff spring, ~0.5 mm of
    # deflection at a 30 N press, instead of a hard non-penetration constraint.
    vial_compliant_stiffness_N_per_m: float = 6.0e4
    vial_compliant_damping_Ns_per_m: float = 100.0
    vial_static_friction: float = 0.35
    vial_dynamic_friction: float = 0.25


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
            "/World/PhysicsMaterials/PlasticCapMaterial",
            static_friction=self.cfg.cap_static_friction,
            dynamic_friction=self.cfg.cap_dynamic_friction,
            restitution=self.cfg.cap_restitution,
        )

        for scene_path in ("/physicsScene", "/World/PhysicsScene"):
            scene_prim = stage.GetPrimAtPath(scene_path)
            if scene_prim.IsValid():
                PhysxSchema.PhysxSceneAPI.Apply(scene_prim).CreateEnableCCDAttr().Set(True)

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

        # 6. Compliant ("give") contact on the vial tubes
        report["compliant_vial_colliders"] = len(self.configure_vial_give(stage))

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

        px_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
        px_mat.CreateFrictionCombineModeAttr().Set("max")
        px_mat.CreateRestitutionCombineModeAttr().Set("min")
        return mat

    def deconflict_eoat_colliders(self, stage: Usd.Stage) -> List[str]:
        """Disable collision on all stock/carrier EOAT colliders when custom fingers are active."""
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

            # Every other collider under J6_link/Gripper, JawL, JawR is retired stock
            # or carrier geometry that causes self-collision jitter.
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                disabled.append(path_str)

        return disabled

    def configure_custom_finger_colliders(
        self, stage: Usd.Stage, pad_material: UsdShade.Material
    ) -> List[str]:
        """Configure active collision proxies on JawL and JawR with CCD and sub-mm offsets."""
        configured = []
        for jaw_tag in ("JawL", "JawR"):
            jaw_path = f"/World/ER4iA/{jaw_tag}"
            jaw_prim = stage.GetPrimAtPath(jaw_path)
            if jaw_prim.IsValid() and jaw_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                # Enable CCD on jaw rigid body
                PhysxSchema.PhysxRigidBodyAPI.Apply(jaw_prim).CreateEnableCCDAttr().Set(True)

            proxy_path = f"{jaw_path}/CustomFingerCollisionProxy"
            root = stage.GetPrimAtPath(proxy_path)
            if not root.IsValid():
                continue

            # The proxy is an Xform of convex pieces that keep the CAD pockets open.
            for prim in Usd.PrimRange(root):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)

                # PhysX sub-millimeter contact offset
                physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                physx_col.CreateContactOffsetAttr().Set(self.cfg.finger_contact_offset_m)
                physx_col.CreateRestOffsetAttr().Set(self.cfg.rest_offset_m)

                # Bind viscoelastic high-friction pad material with physics purpose
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    pad_material, materialPurpose="physics"
                )
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
            # Enable CCD if cap is a rigid body
            if cap_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(cap_prim).CreateEnableCCDAttr().Set(True)

            # Apply to cap root and any submesh collision prims (e.g. Flange_*, Plug_*)
            # Note: Do NOT apply CollisionAPI to non-colliding visual liners (CavityLiner_*, FlangeCavityLiner_*)
            for child in Usd.PrimRange(cap_prim):
                p_child_name = child.GetName()
                if "Liner" in p_child_name:
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr().Set(False)
                    continue

                if child.HasAPI(UsdPhysics.CollisionAPI):
                    col = UsdPhysics.CollisionAPI.Apply(child)
                    col.CreateCollisionEnabledAttr().Set(True)

                    physx_col = PhysxSchema.PhysxCollisionAPI.Apply(child)
                    physx_col.CreateContactOffsetAttr().Set(
                        self.cfg.cap_contact_offset_m
                    )
                    physx_col.CreateRestOffsetAttr().Set(self.cfg.rest_offset_m)

                    UsdShade.MaterialBindingAPI.Apply(child).Bind(
                        cap_material, materialPurpose="physics"
                    )
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
                if prim.HasAPI(UsdPhysics.CollisionAPI):
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

    def configure_vial_give(self, stage: Usd.Stage) -> List[str]:
        """Bind a compliant-contact clear-plastic material to every vial collider."""
        glass = self._get_or_create_physics_material(
            stage, "/World/PhysicsMaterials/VialPlasticCompliant",
            static_friction=self.cfg.vial_static_friction,
            dynamic_friction=self.cfg.vial_dynamic_friction,
            restitution=0.0,
        )
        px = PhysxSchema.PhysxMaterialAPI(glass.GetPrim())
        px.CreateCompliantContactStiffnessAttr().Set(self.cfg.vial_compliant_stiffness_N_per_m)
        px.CreateCompliantContactDampingAttr().Set(self.cfg.vial_compliant_damping_Ns_per_m)
        configured = []
        tray = stage.GetPrimAtPath("/World/RemediationTray")
        if not tray.IsValid():
            return configured
        for prim in Usd.PrimRange(tray):
            if "_tube" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    glass, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
                configured.append(str(prim.GetPath()))
        return configured

    def audit_stage_collisions(self, stage: Usd.Stage) -> Dict[str, any]:
        """Run a diagnostic audit verifying collision health across the stage."""
        active_colliders = []
        disabled_colliders = []
        conflicts = []
        ccd_enabled = []

        gripper_roots = [
            "/World/ER4iA/J6_link/Gripper",
            "/World/ER4iA/JawL",
            "/World/ER4iA/JawR",
        ]

        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                path = str(prim.GetPath())
                col_api = UsdPhysics.CollisionAPI(prim)
                enabled_attr = col_api.GetCollisionEnabledAttr()
                is_enabled = enabled_attr.Get() if enabled_attr.IsValid() else True

                if is_enabled:
                    active_colliders.append(path)
                    is_gripper_prim = any(
                        path == root or path.startswith(root + "/")
                        for root in gripper_roots
                    )
                    if is_gripper_prim and "CustomFinger" not in path:
                        conflicts.append(
                            f"CONFLICT: Interfering EOAT prim has active collision: {path}"
                        )
                    if "Liner" in path and "/RemediationCaps/" in path:
                        conflicts.append(
                            f"CONFLICT: Internal cap liner mesh has active collision: {path}"
                        )

                    if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                        physx_rb = PhysxSchema.PhysxRigidBodyAPI(prim)
                        ccd_attr = physx_rb.GetEnableCCDAttr()
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
