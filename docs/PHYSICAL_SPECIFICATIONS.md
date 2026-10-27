# Physical Specifications: Vials, Caps, and Tray Fixtures

This document records the exact physical, geometric, and dimensional specifications for the vials, caps, and tray fixtures used in the CSI Inspection & Fanuc Remediation Workcell.

**Implementation validation note (October 21, 2026):** The canonical remediation
implementation in `src/remediation/geometry.py` uses the vial/cap dimensions below.
Dataset 1 supplies the 6 × 5 XY slot pattern only. Per the team fixture update on
October 22, 2026, the remediation tray uses a 10 mm black base, a 3 mm silver
perforated plate, and 80 mm standoffs from the top of the base to the underside
of the perforated plate. Vials rest on the base, so their rims are 213.2 mm above
the work surface. Transit clearance must account for this fixture height as well
as the carried cap.

**Current simulation metrology (October 22, 2026):** The mounted custom-finger
CAD envelope has an Isaac-measured visual grasp-tip offset of **284.30 mm** from
the robot flange. The original 293.00 mm stock-tool planner value was therefore
8.70 mm too long for that visual assembly. This is not a physical TCP
calibration or cap-contact offset. It made current TableCam missing-vial targets
unreachable in fixed-down IK, so the present best-effort TableCam demonstration
uses an explicit provisional **100 mm simulation-only** offset. Neither value
may be used for physical contact until measured TableCam/TCP and cap-contact
calibration is recorded.

**Accepted custom-finger visual mount (October 22, 2026):** Each supplied
custom-finger half is mounted with local **−Z** in the grasp direction and the
recessed **+Y** attachment face inward. The paired fingers use a 3--15 mm
mesh-gap range over the SMC's 6 mm per-jaw stroke. The reviewed mount places
the fingers **18.5 mm forward** of the baseline carrier reference and
**6.5 mm downward**; their two inset attachment holes align to the retained
black SMC carrier mounting faces. These are accepted simulation geometry
values. Moving the geometry invalidates the earlier 284.30 mm CAD-envelope
measurement, so the full-remediation trial treats its tool offset as a
best-effort simulation value and does not promote it to physical TCP data.

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

**Materials and priority (October 26, 2026):** This document takes priority over the
CAD models in `cad_models/` wherever they disagree (the CAD vial is 1.070/1.038 in
OD/ID and the CAD flange 1.25 in). Simulation materials are a black tray base and
standoffs, a silver perforated plate, clear glass vials, and semi-smooth (satin)
caps. A crooked cap is defined by how it rests (see section 5), not by a tilt angle.

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
* **Bottom Plate**: Black, 10.00 mm thick. Vials rest on its upper face.
* **Standoffs**: Four black supports hold the top plate 80.00 mm above the
  upper face of the bottom plate.
* **Top Plate**: Silver metal, 3.00 mm thick, perforated at every vial slot.
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
  * **Custom-finger visual CAD envelope**: $284.30\text{ mm}$ from flange,
    measured against the imported geometry in Isaac. It is retained as
    metrology evidence only; current TableCam best-effort IK uses a provisional
    $100.00\text{ mm}$ simulation-only offset because the CAD-envelope number
    cannot yet serve as a contact TCP. Do not use either number as a physical
    pick/place/press contact offset.
* **Fingertip Grip Pads**:
  * Custom $34.0\text{ mm}$ wide conformal high-friction silicone pads spanning the $1.24\text{ in}$ ($31.50\text{ mm}$) cap flange with non-slip contact mechanics ($\mu_s \ge 3.0$).

---

## 5. Crooked Cap Physics & Deterministic Setup

### Crooked Cap Definition: Balanced on the Vial Edge (team stipulation, October 26, 2026)
A crooked cap is defined by **how it rests, not by an exact tilt**:
* The plug's **bottom edge is balanced on the edge of the glass rim** (on the inner rim corner).
* The opposite side of the plug is **down inside the vial mouth**, leaning on the inner wall.
* The **whole flange stays above the rim plane**. A cap whose flange edge has dropped below the rim (resting on the plug side or plug-to-flange corner) has tipped past balance and is not a valid crooked cap.

The tilt follows from the geometry. With sections 1-2 the balanced pose is about **22 degrees** (bisected as the largest tilt that clears the glass, `perched_cap_pose` in `src/remediation/geometry.py`); under PhysX it held at 21.9 degrees with zero drift over 360 steps. The classifier (`cap_state`) checks the three conditions above and has no tilt band.

The earlier flange-on-outer-rim description (31.5-34.0 degrees) is geometrically impossible with these dimensions: the plug top (25.15 mm) fits the mouth (25.40 mm) with 0.13 mm radial clearance, so the cap cannot shift the 2.4 mm needed for the flange edge to reach the outer rim. The previous free-settling result (~45 degrees) had tipped past balance onto the plug corner.

### Crooked-Cap Spawn
Crooked caps are spawned directly in the balanced pose with a 0.2 mm gap above the rim contact, then settle under gravity. This replaces the earlier 25-28 degree free-air drop formula.

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

---

## 7. Simulation Materials, Source Priority & Run Conventions (October 26, 2026)

### Source priority
1. This document is authoritative. Vial rim height uses the 10 mm base: `rim_base_z()` = 0.2081 m in the robot base frame (213.2 mm above the work surface). `src/remediation/geometry.py` (`SPEC`) mirrors sections 1-3.
2. The CAD models in `cad_models/` are used only where this document is silent: the vial inner floor (3.327 mm, CAD 0890-01) and visual appearance reference.
3. Where they disagree, this document wins. CAD values that were *not* adopted: vial 1.070 in OD / 1.038 in ID / 7.992 in length, and flange 1.250 in.

### Materials
| Part | Material | Settings |
|---|---|---|
| Tray base (10 mm) and four 80 mm standoffs | Black | roughness 0.42 |
| Perforated top plate (3 mm) | Silver metal | metallic 1.0, roughness 0.30 |
| Vials | Clear glass | opacity 0.08, roughness 0.02, IOR 1.5 |
| Caps | Semi-smooth (satin) silicone | roughness 0.45, cavity roughness 0.55 |

Physics friction is unchanged (section 5: mu_s 1.2, mu_k 1.0, restitution 0). Visual values are defined in `src/remediation/scene.py`.

### Measured with these dimensions
- A crooked cap is balanced on the vial edge (section 5); with these dimensions that is about 22 degrees, held under PhysX.
- Vial rim height is 213.2 mm above the work surface (10 mm base + 203.2 mm vial).
- The YOLO checkpoint predates these dimensions and materials; retrain it before treating detection results as meaningful (see the cap-wall update above).

### Run conventions
- All outputs go under `shared/runs/YYYY-MM-DD_<task>/` (the pipeline script forces this).
- Build the scene with `qualify_scene.py --fixed --scene-out shared/runs/<run>/scene.usda`, then pass it as the first argument to `scripts/run_compliant_breakable_pipeline.slurm`.
- Reference run: `shared/runs/2026-10-26_docspec_compliant_remediation/`.

### Remediation motion results (October 26, 2026, scene `shared/runs/2026-10-26_docspec_blacksilver_balanced/`)
- **Missing cap (physics pick-and-place):** seated in 3/3 repeats (tilt <= 0.7 deg, flush), peak measured vial load 3.6-5.6 N.
- **Crooked cap (45 deg tilted press, back-face fingertip edge):** seated in 1/3 repeats. A balanced cap rests on its high-side plug edge with the whole flange inboard of that rim support (flange edge 10.9 mm vs rim 12.7 mm from the axis), so a downward push tips it further; the rigid sim has no cap compliance to help it slide in. Side-face (notched) and back-face-along-tilt variants were also tried; the cap pocket mouth at the fingertip captured the cap. Runs: `shared/runs/2026-10-26_physics_remediation_*`.
- 90 deg (flat face horizontal) is not usable inside the tray: the SMC body hangs 18-24 mm below the finger faces and hits neighbouring vials.

