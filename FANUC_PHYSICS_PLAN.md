# FANUC ER-4iA Simulation: Physics Implementation Plan & Architecture Review

**Branch:** `playground/fanuc-physics`  
**Date:** October 2026  
**Status:** Audit & Architecture Completed; Implementation Pending Review  

---

## 1. Executive Summary & Audit of Current Simulation State

In prior simulation runs (`src/remediation/run_best_effort_closed_loop.py` and `render_prediction_driven_dry_run.py`), the Fanuc ER-4iA arm moved kinematically rather than dynamically. The primary root causes and constraints identified during testing include:

1. **Kinematic Teleportation Bypassing PhysX:**
   - Joint states were set inside the stepping loop via `arm.set_joint_positions(positions)` followed by `world.step()`.
   - `set_joint_positions()` overrides PhysX dynamics and instantaneously sets joint coordinates. It bypasses actuator joint drives (PD controllers) and torque/force computation. When a kinematically teleported body contacts a dynamic rigid body, the infinite instantaneous acceleration often causes contact penetration, mesh clipping, or violent physics explosions.


2. **Conflicting EOAT Colliders & Self-Collision Explosions:**
   - When the custom grooved fingers (`data/gripper.usd` / `custom_gripper.py`) were mounted onto the arm, the retired stock finger parts and plates remained collision-enabled.
   - Overlapping geometry caused severe self-contact impulses ("wrist shaking"), prompting developers to disable colliders (`disable_conflicting_eoat_colliders`) and operate in non-contact review mode.


3. **Asset & Texture Path Resolution:**
   - The pre-built USD scene previously hardcoded absolute paths pointing to an external user home directory.
   - On the `playground/fanuc-physics` branch, local textures have been symlinked to `shared/data/fanuc_er4ia_sim/textures/` to eliminate material shader runtime errors.


4. **Cluster Baseline Verification (Slurm Job #287452 on `dh-node1`):**
   - **Robot Prim:** `/World/ER4iA` (referenced from Isaac asset root)
   - **Actuated DOFs (8):** `['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'jaw_l', 'jaw_r']`
   - **Arm Joint Drives:** Angular drives on J1–J6 ($K_p = 2500$, $K_d = 100$, $F_{\max} = 20\text{--}400\,\text{N}\cdot\text{m}$)
   - **Gripper Joint Drives:** Linear prismatic drives on `jaw_l` and `jaw_r` ($K_p = 40000$, $K_d = 3000$, $F_{\max} = 180\,\text{N}$)
   - **Active Collision Prims:** 40 colliders across base, arm links, camera bracket, and jaws.
   - **Static Joint Drift:** Measured at $0.000115\,\text{rad}$ over 30 steps with unactuated hold.

---

## 2. Potential Architectural Approaches to Adding Physics

### Option A: Full-PhysX Dynamic Articulation & Compliant Contact (Pure Physics)
* **Mechanics:**
  - Arm is actuated strictly via `arm.apply_action(ArticulationAction(joint_positions=q_targets))` or effort commands.
  - Multi-body tree dynamics solved by PhysX via Featherstone's reduced-coordinate formulation.
  - Custom fingers receive clean convex-hull / compound-box collision proxies (all stock finger colliders deactivated).
  - Vials, caps, and workpieces carry `RigidBodyAPI`, `MassAPI`, and viscoelastic physics materials (restitution = 0, high friction).
* **Pros:**
  - Physically faithful. Allows true force interaction (e.g. downward pressing force when reseating crooked caps).
  - Responds realistically to misalignments or mechanical obstacles.
* **Cons & Considerations:**
  - Standard PhysX Projected Gauss-Seidel (`PGS`) solver can experience micro-slip or contact chatter when gripping small objects during fast arm travel. Requires tuning Temporal Gauss-Seidel (`TGS`) solver with $\ge 16$ position iterations and $\ge 4$ velocity iterations.

---

### Option B: Hybrid Kinematic-Arm + Dynamic Gripper & Workpiece (Digital Twin Industry Standard)
* **Mechanics:**
  - Arm links (J1 through J6) are set as kinematic rigid bodies (`CreateKinematicEnabledAttr().Set(True)`) or commanded along exact IK trajectory curves.
  - Gripper jaws (`JawL`, `JawR`), caps, and vials remain 100% dynamic rigid bodies with active PhysX collisions and gravity.
* **Pros:**
  - Zero arm sagging, joint droop, or trajectory tracking errors regardless of payload.
  - Cap reseating and gripping remain dynamic (finger physically pushes the cap; cap contacts vial throat).
* **Cons:**
  - Arm has infinite effective momentum; if an IK path commands penetration into an immovable surface (e.g. the aluminum table), the arm will not stall under torque limits unless software collision detection is guarded.

---

### Option C: Dynamic Articulation + Virtual Attachment Joint for Pick & Place (Hybrid Physics)
* **Mechanics:**
  - Complete PhysX articulation and collision contacts for the arm, jaws, and caps.
  - **Crooked Cap Reseating:** Pure physical downward contact force (flat finger face presses cap rim until flush).
  - **Buffer Cap Pick-and-Place:** Jaws close dynamically until normal contact is registered. Upon contact within a distance threshold, a dynamic `UsdPhysics.FixedJoint` (weld constraint) attaches the cap to `J6_link`. Upon opening the jaws over the target slot, the joint is deleted/broken, allowing the cap to physically drop/settle into the vial throat.
* **Pros:**
  - Eliminates micro-slip drop failures during high-speed transits while preserving authentic physical contact dynamics during cap pressing and settling.
* **Cons:**
  - Requires maintaining joint creation and destruction lifecycle hooks in the controller.

---

### Option D: Isaac Sim `SurfaceGripper` / Gripper Actuator Component
* **Mechanics:**
  - Uses NVIDIA's `isaacsim.robot.surface_gripper` extension.
  - Places a contact/suction point at the TCP that binds overlapping rigid bodies upon trigger.
* **Pros:**
  - Standardized NVIDIA API with configurable break force and proximity thresholds.
* **Cons:**
  - Tailored primarily for suction cups or magnetic pickers rather than parallel grooved jaws.

---

## 3. Physical & Numerical Nuances (Crucial Implementation Details)

1. **Elimination of the $20\text{ mm}$ PhysX "Ghost Contact Bubble":**
   - By default in Isaac Sim / PhysX, unconfigured meshes use `contactOffset = 0.02\text{ m}` ($20\text{ mm}$). This causes objects to exert repulsive forces when they are still 2 centimeters apart in mid-air.
   - **Resolution:** Explicitly set sub-millimeter offsets:
     - Gripper jaws: `contactOffset = 0.0002` ($0.2\text{ mm}$), `restOffset = 0.0`
     - Dynamic Caps: `contactOffset = 0.0005` ($0.5\text{ mm}$), `restOffset = 0.0`

2. **Continuous Collision Detection (CCD) to Prevent Tunneling:**
   - Small, thin features (the $0.635\text{ mm}$ vial wall and $2.03\text{ mm}$ cap flange) will phase through each other during rapid movements if CCD is disabled.
   - **Resolution:** Invoke `cell.enable_ccd(world)` and verify `PhysxSchema.PhysxCollisionAPI(prim).CreateCcdEnabledAttr().Set(True)` on the dynamic cap and gripper finger bodies.

3. **Cap Geometry: Multi-Body Compound vs Single Convex Hull:**
   - A single convex hull of the cap bridges the flange to the plug base, creating a solid wedge that cannot enter the $25.40\text{ mm}$ tube ID.
   - **Resolution:** The cap must be composed of two distinct collision primitives:
     - Flange: Thin cylinder ($D = 31.50\text{ mm}$, $H = 2.03\text{ mm}$)
     - Plug: Tapered cylinder / cone ($D_{\text{bottom}} = 22.86\text{ mm}$, $D_{\text{top}} = 25.10\text{ mm}$, $H = 13.46\text{ mm}$) or convex sectors preserving the void.

4. **Vial Stability in Tray:**
   - In `clean_vial_tray.usda`, the tray base plate, standoff supports, top perforated plate, and vial tubes must remain **static colliders** (infinite mass).
   - Only the caps and robot arm are dynamic. If the glass vials are dynamic without complex multi-body nesting constraints, downward press forces from the robot will pop the vials out of the fixture or drive them through the floor.

5. **Tool Center Point (TCP) & IK Reachability:**
   - The stock tool length was $293.00\text{ mm}$ from flange.
   - Isaac CAD metrology measured the custom-finger visual grasp tip at $284.30\text{ mm}$ (an $8.70\text{ mm}$ difference).
   - In fixed-down IK solving, an incorrect tool offset will either command unattainable joint configurations or push the flange into the tray. Set the IK tool offset accurately to reflect the active finger tip ($284.30\text{ mm}$).

6. **Trajectory Interpolation vs Sudden Step Jumps:**
   - Moving from pose A to pose B in a single physics tick causes enormous velocity derivatives ($\Delta q / \Delta t$), triggering torque limit saturation and severe overshoot.
   - **Resolution:** Generate intermediate joint-space waypoints (trapezoidal velocity or cubic spline profiles) sampled at the physics rate ($120\text{ Hz}$).

7. **Physics Substepping Ratio:**
   - Maintain `physics_dt = 1/120` ($120\text{ Hz}$) and `rendering_dt = 1/30` ($30\text{ fps}$).
   - This ensures 4 physics substeps per render frame, providing numerical stability for contact constraints without slowing down video production.

---

## 4. Recommended Implementation Strategy & Blueprint

**Recommendation:** **Option A** for crooked-cap reseating (direct physical pressing), with an optional **Option C** fallback for buffer cap transport if friction grasping exhibits micro-slip under acceleration.

### Step-by-Step Implementation Roadmap:

1. **Step 1: Clean EOAT Collision Structure (`src/remediation/custom_gripper.py`)**
   - Permanently disable collisions on retired stock finger parts (`/World/ER4iA/J6_link/Gripper/Jaw{L,R}Plate` and old finger brackets).
   - Ensure custom CAD finger proxies are attached to `JawL` and `JawR` using convex hull approximations (`MeshCollisionAPI` with `convexHull` or compound box colliders).
   - Validate zero self-intersection across full jaw travel ($3.0\,\text{mm}$ to $15.0\,\text{mm}$ gap).

2. **Step 2: PhysX Scene & Solver Configuration (`setup_physics_scene`)**
   - Switch solver type to `TGS` (Temporal Gauss-Seidel) and increase iteration counts:
     ```python
     physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
     physx_scene.CreateSolverTypeAttr("TGS")
     physx_scene.CreateTimeStepsPerSecondAttr(120)  # 1/120s substepping
     ```
   - Configure solver iterations: $\ge 16$ position iterations, $\ge 4$ velocity iterations.

3. **Step 3: Polymer & Pad Contact Physics Materials (`setup_gripper_contact_physics`)**
   - Create and bind a dedicated `PhysicsMaterial`:
     - `dynamicFriction`: $1.2$
     - `staticFriction`: $1.5$
     - `restitution`: $0.0$ (absorbs contact bounce)
   - Bind to gripper conformal pads and cap geometry.

4. **Step 4: Actuator Drive Tuning (`configure_robot_articulation_drives`)**
   - Replace kinematic `set_joint_positions()` with `arm.apply_action(ArticulationAction(joint_positions=q))` paired with continuous trajectory interpolation (e.g. cubic or trapezoidal joint-space waypoints).
   - Tune J1–J6 drive gains: $K_p \approx 4000\text{--}6000\,\text{N}\cdot\text{m/rad}$, $K_d \approx 200\text{--}300$ for crisp positioning without droop.
   - For cap pressing: generate vertical approach $\to$ contact $\to$ press depth ($12\,\text{mm}$ stroke) trajectory.

---

## 5. Test Environment on `playground/fanuc-physics`

- **Branch:** `playground/fanuc-physics`
- **Runner Script:** `scripts/fanuc_physics_playground.py`
  - Contains modular placeholder hooks:
    - `setup_physics_scene(world, stage)`
    - `configure_robot_articulation_drives(arm, stage)`
    - `setup_gripper_contact_physics(stage, finger_type)`
- **Execution Command:**
  ```bash
  sbatch scripts/submit_isaac_job.slurm scripts/fanuc_physics_playground.py --steps 60
  ```
