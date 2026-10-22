# Physical Specifications: Vials, Caps, and Tray Fixtures

This document records the exact physical, geometric, and dimensional specifications for the vials, caps, and tray fixtures used in the CSI Inspection & Fanuc Remediation Workcell.

**Implementation validation note (October 21, 2026):** The canonical remediation
implementation in `src/remediation/geometry.py` uses the vial/cap dimensions below.
Its tray slot locations and support heights come from a read-only measurement of
the supplied Dataset 1 USD, recorded in `src/remediation/config/dataset1_layout.json`.
Those source-asset dimensions are not an independent physical survey. The source
tray places the vial bases about 69.6 mm above the tabletop; with 203.2 mm vials,
the rims are about 272.8 mm above it. Transit clearance must account for this
fixture height as well as the carried cap.

The team confirmed custom fingers with recessed profiles for a loose cap, an
uncapped vial, or a capped vial. The supplied `data/gripper.step` is an AP242
metre-based CAD model, but its mounting frame, TCP, grasp surfaces, and collision
proxies have not been validated. The 34 mm pad width and 6 mm/jaw actuator travel
do not establish the groove depths, usable opening, contact offsets, or final TCP.
Qualification renders currently show the stock starter fingers; they are not a
validated model of the planned custom tool.

The original equilibrium/contact claims below are retained as reference targets.
New gravity-settling trials with a hollow 0.99-inch-top tapered plug produced
stable crooked angles around 40–44°, rather than 31.5–34°. The report records the
measured angles and whether they fall in the documented band. The difference
remains unresolved; no real cap/contact calibration has been performed. Friction
1.2/1.0 is a simulation assumption, not a measured silicone/glass property.

**Cap-wall update (October 21, 2026):** The plug is modeled as a hollow taper
with a 2.0 mm wall, per the team update. The inner wall follows the outer taper
instead of using the prior constant 0.80-inch inner-diameter approximation. This
changes the physical configuration hash, so the synthetic dataset and checkpoint
must be regenerated before a later release decision.

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
* **Plug Wall Thickness**: $2.00\text{ mm}$ around the tapered plug and through
  the flange. The cap is an open-through sleeve, not a solid-topped stopper. Its
  top opening is approximately $0.832\text{ in}$ ($21.15\text{ mm}$); the
  inner diameter immediately above the closed base is approximately $0.742\text{ in}$
  ($18.86\text{ mm}$).
* **Plug Bottom Face**: A flat $2.00\text{ mm}$ thick base closes the bottom of
  the plug. The cavity is open at the flange/top and terminates at this base.
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

## 5. Crooked Cap Physics & Deterministic Setup

### Mechanical Wedging Equilibrium
* **Equilibrium Tilt Angle**: $31.5^\circ - 34.0^\circ$ (natural attractor state).
* **Two-Point Contact Geometry**:
  * One edge of the $0.90\text{ in}$ ($22.86\text{ mm}$) plug enters the $1.00\text{ in}$ ($25.40\text{ mm}$) tube mouth and catches against the inner glass wall.
  * The opposing edge of the $1.24\text{ in}$ ($31.50\text{ mm}$) flange overhang ($2.415\text{ mm}$ per side) rests on the outer rim lip ($26.67\text{ mm}$ OD).
* **Resting Elevation**: The cap center sits at $+5.0\text{ mm}$ to $+5.8\text{ mm}$ above the rim ($Z_{\text{settled}} \approx Z_{\text{rim}} + 0.0053\text{ m}$), with the high flange edge elevated $\approx +14\text{ mm}$ above the rim.

### Clean-Air Zero-Phasing Spawn Formula
To eliminate mesh interpenetration or solver clipping with the $0.635\text{ mm}$ glass vial walls at initialization, caps must be spawned in free air:
* **Spawn Tilt**: Initial pitch $\theta \in [25.0^\circ, 28.0^\circ]$ around local $Y$.
* **Center Elevation**:
  $$Z_{\text{spawn}} = Z_{\text{rim}} + \text{lowest\_point}(\theta) + \delta_z$$
  where:
  $$\text{lowest\_point}(\theta) = H_{\text{plug}} \cos(\theta) + R_{\text{flange}} \sin(\theta)$$
  and $\delta_z = +0.003\text{ m}$ ($+3\text{ mm}$ strictly above the glass rim in completely free air).
* **Lateral Offset**:
  * Columns 0–3: $\Delta x = -0.0015\text{ m}$ (bias toward the trailing rim edge).
  * Column 4 (Tray Edge): $\Delta x = -0.0005\text{ m}$ with $\theta = 28.0^\circ$.

### Contact Physics & Settling Damping
* **Sub-Millimeter Contact Offsets**:
  * Gripper jaws: `contactOffset = 0.0002m` ($0.2\text{ mm}$), `restOffset = 0.0`.
  * Dynamic Caps (`CapCrooked_*` and `PhysicalCap`): `contactOffset = 0.0005m` ($0.5\text{ mm}$), `restOffset = 0.0`.
  * Tube Colliders (`clean_vial_tray.usda`): Native collision offset.
  * *Ghost Wiggling Prevention*: Applying `contactOffset = 0.0005m` to the cap geometries eliminates PhysX's $20\text{ mm}$ default collision bubble, preventing mid-air repulsive chatter before physical contact.
* **Viscoelastic Material**:
  * Damping: `linear_damping = 1.2`, `angular_damping = 2.0`.
  * Friction: $\mu_s = 1.2, \mu_k = 1.0$.
  * Restitution: $0.0$ (`min` combine mode).
* **Warmup**: $60-90$ frames under gravity to allow the cap to contact the rim and settle into the $32^\circ$ static cradle prior to perception or manipulation.

---

## 6. Physics Simulation Implementation
* **Robot Table Surface**: Static rigid body with high-friction physics material ($\mu_s = 0.8, \mu_k = 0.7$).
* **Tray & Vials**: Mesh and cylinder physics colliders (`UsdPhysics.CollisionAPI`):
  * Vials 01–10: 10 dynamic crooked caps wedged at $32^\circ$ equilibrium.
  * Vials 11–20: 10 empty vial apertures (Target: Slot 15 / `G15` at Row 2, Col 4).
  * Vials 21–30: Native static seated caps from `clean_vial_tray.usda` (zero solver chatter).
* **Caps**: Compound multi-body colliders with PhysX convex decomposition:
  * Cylinder top flange ($D = 1.24\text{ in}, T = 0.08\text{ in}$).
  * Tapered insert plug ($D_{bot} = 0.90\text{ in}, D_{top} = 0.99\text{ in}$).
* **Gripper Finger Pads**: Elastic contact model with friction ($\mu_s = 1.2$) for non-slip physical pinch grasps without numerical jitter.
