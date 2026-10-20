# Physical Specifications: Vials, Caps, and Tray Fixtures

This document records the exact physical, geometric, and dimensional specifications for the vials, caps, and tray fixtures used in the CSI Inspection & Fanuc Remediation Workcell.

---

## 1. Vial Dimensions (Glass Cylinder, Flat Bottom)
* **Inner Diameter (ID)**: $1.00\text{ in} = 0.02540\text{ m} = 25.40\text{ mm}$
  * Inner Radius: $0.50\text{ in} = 0.01270\text{ m} = 12.70\text{ mm}$
* **Outer Diameter (OD)**: $1.05\text{ in} = 0.02667\text{ m} = 26.67\text{ mm}$
  * Outer Radius: $0.525\text{ in} = 0.013335\text{ m} = 13.335\text{ mm}$
  * Wall Thickness: $\frac{1.05 - 1.00}{2} = 0.025\text{ in} = 0.635\text{ mm}$
* **Height**: $8.00\text{ in} = 0.20320\text{ m} = 203.20\text{ mm}$
* **Base**: Flat bottom (rests vertically inside tray fixture aperture).

---

## 2. Cap Dimensions (MOCAP Tapered Flanged Insert Plug)
* **Flange Outer Diameter (Top OD)**: $1.24\text{ in} = 0.031496\text{ m} = 31.50\text{ mm}$
  * Flange Outer Radius: $0.62\text{ in} = 0.015748\text{ m} = 15.75\text{ mm}$
* **Flange Thickness**: $0.08\text{ in} = 0.002032\text{ m} = 2.032\text{ mm}$
  * Rests flat on top of the vial lip ($1.24\text{ in} > 1.05\text{ in}$ vial OD).
* **Insert Plug Outer Bottom Diameter**: $0.90\text{ in} = 0.02286\text{ m} = 22.86\text{ mm}$
  * Bottom Plug Outer Radius: $0.45\text{ in} = 0.01143\text{ m} = 11.43\text{ mm}$
  * Fits inside the vial mouth ($0.90\text{ in} < 1.00\text{ in}$ vial ID).
* **Insert Plug Top Outer Diameter (under flange)**: $\approx 0.98\text{ in} - 1.00\text{ in}$ (tapered wedge fit into vial inner diameter).
* **Inner Hollow Diameter**: $0.80\text{ in} = 0.02032\text{ m} = 20.32\text{ mm}$
  * Inner Hollow Radius: $0.40\text{ in} = 0.01016\text{ m} = 10.16\text{ mm}$
* **Plug Depth**: $\approx 0.53\text{ in} = 0.01346\text{ m} = 13.46\text{ mm}$

---

## 3. Tray Fixture Specifications
* **Aperture Hole Diameter**: $1.25\text{ in} = 0.03175\text{ m} = 31.75\text{ mm}$
  * Clearance: $1.25\text{ in} - 1.05\text{ in} = 0.20\text{ in} = 5.08\text{ mm}$ diametral clearance around vial outer walls.
* **Flange Catch**:
  * Cap top flange ($1.24\text{ in}$) is smaller than the tray hole ($1.25\text{ in}$), allowing the vial and seated cap to be inserted and withdrawn vertically without mechanical interference with the fixture plate.
* **Physics Rig Body**: Static rigid body with collision meshes for fixture plate and aperture cutouts.
* **Layout**: 30 apertures arranged in a 6 $\times$ 5 grid (numbered `G01` through `G30`).

---

## 4. Gripper End-Effector Specifications (SMC MHZL2-16D2 Parallel Gripper)
* **Pneumatic Stroke**: $6.00\text{ mm}$ lateral motion per jaw $\to \mathbf{12.00\text{ mm}}$ total stroke between open and closed states.
  * `JAW_CLOSED`: $0.0\text{ mm}$ (minimum gap).
  * `JAW_OPEN`: $+6.0\text{ mm}$ per jaw ($+12.0\text{ mm}$ lateral opening).
* **Kinematic Stack & Tool Length**:
  * $J6$ wrist center to faceplate: $70\text{ mm}$.
  * Faceplate adapter, upper spacer, SMC body, and finger blades: $223\text{ mm}$.
  * Total tool length (`tip_offset` along local $+X$): $\mathbf{293.0\text{ mm}} = 0.293\text{ m}$.
* **Fingertip Grip Pads**:
  * Custom $34.0\text{ mm}$ wide conformal high-friction silicone pads spanning the $1.24\text{ in}$ ($31.50\text{ mm}$) cap flange with non-slip contact mechanics ($\mu_s \ge 3.0$).

---

## 5. Crooked Cap Resting Geometry
* **Resting Angle**: $45^\circ - 60^\circ$ tilt across the vial rim mouth.
* **Mechanism**: One edge of the $0.90\text{ in}$ plug dips slightly inside the $1.00\text{ in}$ vial mouth resting against the inner glass wall, while the opposing edge of the $1.24\text{ in}$ flange catches on the top glass rim.
* **Physical Elevation**: The elevated edge sits approximately $+15\text{ mm}$ to $+24\text{ mm}$ higher than the flush seated height ($Z \approx Z_{\text{vial\_mouth}} + 0.018\text{ m}$).

---

## 6. Physics Simulation Implementation
* **Robot Table Surface**: Static rigid body with high-friction physics material ($\mu_s = 0.8, \mu_k = 0.7$).
* **Tray & Vials**: Mesh and cylinder physics colliders (`UsdPhysics.CollisionAPI`).
* **Caps**: Compound multi-body colliders with PhysX convex decomposition:
  * Cylinder top flange ($D = 1.24\text{ in}, T = 0.08\text{ in}$).
  * Tapered insert plug ($D_{bot} = 0.90\text{ in}, D_{top} = 0.99\text{ in}$).
* **Gripper Finger Pads**: Elastic contact model with friction ($\mu_s = 1.2$) for non-slip physical pinch grasps without numerical jitter.
