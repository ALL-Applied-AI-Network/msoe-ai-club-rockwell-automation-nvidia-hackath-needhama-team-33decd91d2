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
standoffs, a satin silver perforated plate, clear plastic vials, and semi-gloss
plastic caps (surface roughness 0.30; darker cavity roughness 0.42). The clear-plastic render is an appearance approximation using the
YOLO training shader (IOR 1.57, frosting roughness 0.035); the resin has not been
measured. The silver plate uses the aluminum roughness and normal maps. These
visual settings do not change collision geometry or contact parameters. A crooked
cap is defined by how it rests (see section 5), not by a tilt angle.

---

## 1. Vial Dimensions (Clear Plastic Cylinder, Flat Bottom)
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

### Crooked Cap Definition: Leaning on the Rim (team observation, October 26, 2026)
A crooked cap is defined by **how it rests, not by an exact tilt** (photo of the real cap: `Media.jpg`):
* On the **low side** the wide flange dips **inside the vial mouth**; its outer edge rests on the inner edge of the glass rim at two points.
* On the **high side** the plug's **bottom outer corner rests on the rim** at two points, with the plug overhanging the outside of the vial.

PhysX contact reports confirm this four-point support: flange edge (cap r = 15.9 mm) on the rim inner edge on the low side, plug bottom corner (cap r = 11.4 mm, z = -13.5 mm) on the rim on the high side. `leaning_cap_pose` in `src/remediation/geometry.py` constructs the pose for a given tilt; `cap_state` classifies a cap as crooked when its flange dips into the mouth and its plug bottom rests on the rim (no tilt band).

**Tilt range (`CROOKED_TILT_RANGE_DEG = (48, 77)`).** Scenes draw a uniform random tilt from this range. It comes from a PhysX sweep of 28 tilts x 3 seeds (`shared/runs/2026-10-26_crooked_tilt_sweep_v2_seed*`):
* Below 48 deg the pose is geometrically impossible (the flange cannot dip into the mouth while the plug corner reaches the rim).
* 48-77.6 deg held in every seed (settled 47-75 deg, drift <= 0.3 mm).
* 79 deg failed in all three seeds (one cap fell into the vial); 80-86 deg slid down to ~77-81 deg.

The real cap in the photo is at roughly 55-60 deg. The earlier "balanced on the plug edge at ~22 deg" definition and the 31.5-34 deg band are superseded.

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
- A crooked cap leans on the rim (section 5) at a random tilt in 48-77 degrees.
- Vial rim height is 213.2 mm above the work surface (10 mm base + 203.2 mm vial).
- The YOLO checkpoint predates these dimensions and materials; retrain it before treating detection results as meaningful (see the cap-wall update above).

### Run conventions
- All outputs go under `shared/runs/YYYY-MM-DD_<task>/` (the pipeline script forces this).
- Build the scene with `qualify_scene.py --fixed --scene-out shared/runs/<run>/scene.usda`, then pass it as the first argument to `scripts/run_compliant_breakable_pipeline.slurm`.
- Reference run: `shared/runs/2026-10-26_docspec_compliant_remediation/`.

### Legacy remediation motion results (October 26–27, 2026)

The press, wrist-servo and regrip results below are retained experimental history.
They are not the active shared physical/simulation policy.
- **Missing cap (physics pick-and-place):** seated in 3/3 repeats (tilt <= 0.7 deg, flush), peak measured vial load 3.6-5.6 N.
- **Crooked cap, superseded balanced-pose scene (45 deg tilted press, back-face fingertip edge):** seated in 1/3 repeats. A balanced cap rests on its high-side plug edge with the whole flange inboard of that rim support (flange edge 10.9 mm vs rim 12.7 mm from the axis), so a downward push tips it further; the rigid sim has no cap compliance to help it slide in. Side-face (notched) and back-face-along-tilt variants were also tried; the cap pocket mouth at the fingertip captured the cap. Runs: `shared/runs/2026-10-26_physics_remediation_*`.
- 90 deg (flat face horizontal) is not usable inside the tray: the SMC body hangs 18-24 mm below the finger faces and hits neighbouring vials.
- **Crooked cap, leaning pose (scene `shared/runs/2026-10-26_docspec_blacksilver_leaning/`, G09 at 47.8 deg):** 0/3 seated with a straight-down press beside the raised flange edge. One run pushed the cap into the vial (68 N measured, vial fractured), one levelled it to ~10 deg but the cap was carried off on retreat, one was carried off before levelling. The TableCam depth plane fit misread the lean (21 deg tilt / 53 deg heading vs true 47.8 deg / 42.5 deg) because it assumes a flat cap top. Runs: `shared/runs/2026-10-26_physics_remediation_leaning_repeat*`.
- **Same, with vial give** (compliant glass contact, 6e4 N/m / 100 N s/m, ~0.5 mm at 30 N; `collision_manager.configure_vial_give`): no fractures (peak 14-36 N vs 68 N rigid), but 0/3 seated. In every repeat the cap is carried out on retreat (see the back-face result below for the cause). Runs: `shared/runs/2026-10-26_physics_remediation_vialgive_repeat*`.
- **Same, pressing with the solid outer back face 10 mm up from the tip** (`PRESS_FACE_UP_M`, path checked offline to stay clear of all glass): 0/3 seated, no fractures (13-28 N). The cap's open top (21.15 mm through the flange) faces up toward the tool in this lean; the finger slides into it and the cap rides out on the finger like a thimble. This is also what happened in the earlier fingertip presses (an earlier note attributing it to the finger pocket was wrong). Runs: `shared/runs/2026-10-26_physics_remediation_backface_repeat*`.
- **Straight-down press with the closed fingertips** (legacy `--controller-policy legacy --crooked-press vertical`): 0/3 seated centred on the vial (the leaning cap slides up into the fingertip pocket; at the tip the pocket mouth, r 13.6 mm, is wider than the finger's 11.1 mm half-width, leaving only two solid crescents) and 0/3 with a crescent offset over the raised flange edge (tilt fell from ~45 to ~38-40 deg, then stalled). The cap stays in its vial; peak loads 12-18 N. Runs: `shared/runs/2026-10-26_physics_remediation_{vertical,crescent}_repeat*`.

### Wrist-camera servoing onto the crooked cap axis (October 26, 2026)
`src/remediation/servo_crooked_cap.py` (maths in `wrist_servo.py`). TableCam supplies only the vial position: its depth-based lean estimate was unreliable (heading wrong by 142 deg on one cap, tilt 21-33 deg against 48-71 deg true). The wrist camera first looks straight down over the vial, fits an ellipse to the inner edge of the cap's lit flange ring (the 21.15 mm opening), computes the two possible circle-plane normals in closed form, and picks one using the wrist depth gradient across the ring. It then points the camera down that axis and stops at the first view where the opening is round (minor/major >= 0.995), reporting that camera axis.

| Cap | True tilt | Views | Final axis error |
|---|---|---|---|
| G09 | 47.7 deg | 2 | 2.2 deg |
| G05 | 57.9 deg | 2 | 0.3 deg |
| G30 | 67.3 deg | 2 | 0.4 deg |
| G04 | 70.7 deg | 5 | 0.5 deg |

The closed-form estimate is accurate from an oblique view (0.3-2.2 deg from the first, straight-down image) but noisy once the opening is nearly round (up to ~14 deg), so the round view's camera axis is used rather than its image estimate. Sim images are noise-free; real wrist RGB-D has not been tested. Runs: `shared/runs/2026-10-26_wrist_servo_v4_G09`, `shared/runs/2026-10-26_servo_test_seed*/servo_v4`.

**Legacy real wrist-camera model (October 27, 2026).** The retained servo sees the wrist camera as the cell delivers it (`src/remediation/wrist_camera_model.py`, from `PhysicalAI-Hackathon-2026/calibration/cameras.json`). Colour: 848 x 480 sensor served at 1280 x 720, fx 619.6 px, radial k1 = -0.0196. Depth: a separate camera at its own calibrated pose (21 mm and 2.9 deg from the colour camera), 848 x 480, fx 416.9 px, principal point (421.35, 239.34), uint16 mm with 0 = no depth. Both poses are set from the calibration at runtime, not taken from the scene. The ellipse is fitted to the undistorted opening edge, and depth is registered onto the colour pixels before the ring's depth gradient is taken. Depth noise is an assumption, not a measurement: the calibration's 3.0 mm plane RMS at 0.363 m, scaled with range^2. No colour sensor noise is modelled. Ray-traced depth still returns transparent plastic as a surface.

| Scene (cap, tilt) | Ideal camera error | Real-camera model error |
|---|---|---|
| seed 21 (G07, 53.0 deg) | 1.5 deg | 1.3 deg |
| seed 22 (G06, 60.8 deg) | 0.3 deg | 0.2 deg (second view unreachable in both) |
| seed 25 (G11, 64.7 deg) | 0.6 deg | 1.0 deg |

Runs: `shared/runs/2026-10-27_wrist_camera_model/servo_seed*`; ideal-camera figures from `shared/runs/2026-10-27_pipeline_seed*`.

Registering the depth onto the colour pixels needs each depth sample splatted over 3 x 3 colour pixels (nearest wins). With one pixel per sample, the table showed through the gaps between the cap's samples: over 5% of the ring's depth readings came from 0.31-0.41 m. That flipped the depth gradient and made the servo pick the wrong normal on the first view (G05, G04; first pipeline batch, `shared/runs/2026-10-27_regrip_final_results_attempt1_splatbug`).

Full pipeline with the real-camera model, one run per scene (`shared/runs/2026-10-27_regrip_final_results/<cap>/`, videos in `docs/media/regrip_final_results/`):

| Cap (tilt) | Servo axis error | Crooked cap seated | Missing cap seated | Vials |
|---|---|---|---|---|
| G04 (70.7 deg) | 0.6 deg | yes | no missing-cap detection | intact |
| G30 (67.3 deg) | 0.4 deg | no: 1.1 mm high, 1.9 deg tilt | yes | intact |
| G05 (57.9 deg) | 0.4 deg | yes | yes | intact |
| G07 (53.0 deg) | 1.6 deg | yes | yes | intact |
| G09 (47.7 deg) | 3.0 deg | yes | no: released 1 mm high, carried off on retreat (before any wrist view) | intact |


### Legacy crooked-cap wrist-servoed grip, table regrasp and re-place (October 27, 2026)
Legacy comparison strategy (`--controller-policy legacy --crooked-press regrip`). Sequence:
1. Wrist-camera servo onto the cap axis (above).
2. Open fingers approach down that axis, spread axis horizontal, and close on the flange. Near the robot base a cap leaning toward the robot needs J3 past its -69 deg limit for a coaxial grip (G09: -72.7 deg), so the tool may lean up to 10 deg less than the cap.
3. The cap is lifted straight up and turned upright in free air.
4. **Regrasp on the table.** The side grip leaves the cap up to 2.5 mm off the pocket axis and 15 deg rocked: the flange slot in the finger pocket is 4.76 mm tall against the 2.03 mm flange, and the plug top has only 0.13 mm radial clearance in the mouth, so placing it directly caught on the rim. The cap is pressed 1.5 mm into the table (squaring it in the slot), released on the vacated spare-cap spot (or a bare spot found in TableCam depth), located with a straight-down wrist view (opening centre back-projected onto the flange-top plane; roundness < 0.95 = fallen over, reported as a failure), and picked again like the spare.
5. Placed and pressed like the missing cap.

Supporting changes: the tray-grid fit rejects outliers (a false crooked detection 150 mm off the tray had shifted the grid ~5 mm) and uses TableCam-depth flange-top centroids instead of box centres (box centres are biased ~1.4 mm; the empty-bore correction added a +1.3 mm y bias), putting targets within 0.7 mm. Held caps are clamped 0.4 mm past the coaxial jaw value (~11 N per jaw) so they cannot rock onto one lip. The plug enters the mouth at 0.2 mm/step. With no missing-cap detection only the crooked cap is handled.

| Slot (tilt) | Crooked cap seated | Missing cap seated |
|---|---|---|
| G05 (57.9 deg) | 2/2 | 2/2 |
| G09 (47.7 deg) | 2/2 | 2/2 |
| G30 (67.3 deg) | 2/2 | 2/2 |
| G04 (70.7 deg) | 1/3 (2 fell over on the table during the regrasp) | no missing-cap detection |

Seated caps finish at <0.002 deg tilt and <0.13 mm off the rim; peak measured vial loads 4-7 N. Final runs: `shared/runs/2026-10-27_regrip_*_v9*`; development history `_v1`-`_v8`.
