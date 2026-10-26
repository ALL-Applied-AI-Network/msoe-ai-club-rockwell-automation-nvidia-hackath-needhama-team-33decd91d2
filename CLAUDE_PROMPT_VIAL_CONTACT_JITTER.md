# TASK: Eliminate Robotic Arm Wrist Jitter During Vial Top Contact in Isaac Sim Simulation

## Context & Repository Architecture
You are working on an autonomous Fanuc ER-4iA robotic remediation digital twin in NVIDIA Isaac Sim 4.x/6.1.
- **Repository Root**: `/home/ad.msoe.edu/cojonkeddiej/msoe-ai-club-rockwell-automation-nvidia-hackath-needhama-team-33decd91d2`
- **Key Runner Script**: `src/remediation/run_compliant_breakable_remediation.py`
- **Collision Configuration**: `src/remediation/collision_manager.py`
- **Custom Gripper Mounting & Proxies**: `src/remediation/custom_gripper.py`
- **Physical Specifications**: `src/remediation/geometry.py`
- **Active USD Scene**: `shared/scenes/remediation_dataset1_v1.usda`
- **Active Detections**: `compliant_remediation_test/perception/current_scene_detections.json`
- **Output Directory**: `compliant_remediation_test/action/` (frames saved to `frames/`, encoded to `compliant_remediation.mp4`)

---

## The Specific Problem Reported by the User
> "The jittering is still happening. It starts specifically around collision with the top of the vial. Look into problems that could happen with this first"

While free-air transit, S-curve interpolation, and waypoint generation are smooth, **wrist/arm jittering is triggered specifically when the gripper descends to interact with the top of the vial** (both during spare cap insertion at slot G11 and crooked cap reseating at slot G21).

---

## Physical Dimensions & Coordinate Reference Frame
- **Robot Base**: `(0, 0, 0.900)` in World frame ($Z_{\text{base}} = Z_{\text{world}} - 0.900\,\text{m}$).
- **Vials**: 
  - Outer Diameter (OD): $27.178\,\text{mm}$ ($1.070\,\text{in}$).
  - Inner Diameter (ID): $26.365\,\text{mm}$ ($1.038\,\text{in}$).
  - Height: $203.20\,\text{mm}$ ($8.00\,\text{in}$).
  - Vials sit on the tray support plate at $Z_{\text{base}} = 0.0645\,\text{m}$.
  - **Top Rim Elevation**: $Z_{\text{base}} = 0.0645 + 0.2032 = \mathbf{0.2677\,\text{m}}$ ($Z_{\text{world}} = 1.1677\,\text{m}$).
- **Caps (MOCAP TWF)**:
  - Flange Diameter: $31.75\,\text{mm}$ ($1.25\,\text{in}$).
  - Flange Thickness: $2.235\,\text{mm}$ ($0.088\,\text{in}$).
  - Plug Depth: $13.462\,\text{mm}$ ($0.530\,\text{in}$).
  - Plug Top OD: $23.62\,\text{mm}$; Plug Bottom OD: $20.83\,\text{mm}$.
- **Custom Gripper & Finger Proxies**:
  - SMC MHZL2-16D2 pneumatic carrier mounted on Fanuc J6 flange.
  - Active Custom Fingers: `/World/ER4iA/JawL/CustomFingerCollisionProxy` and `/World/ER4iA/JawR/CustomFingerCollisionProxy`.
  - True Finger TCP: `(0.284302, 0.0, -0.0243)\,\text{m}` relative to J6 flange.
  - Finger dimensions: Length $= 91.5\,\text{mm}$, Width $= 22.3\,\text{mm}$, Height $= 35.6\,\text{mm}$.
- **Slots Handled in Current Scene**:
  - Missing Cap: Slot **`G11`** at base `(0.35981, -0.08692, 0.267896)`.
  - Crooked Cap: Slot **`G21`** at base `(0.46657, -0.08962, 0.276908)`.

---

## Priority Areas to Investigate First

1. **Finger Collision Proxies Interpenetrating the Glass Vial Rims**:
   - The custom finger collision proxies (`CustomFingerCollisionProxy`) are rectangular boxes attached to `JawL` and `JawR`.
   - When the robot descends to seat the cap at $Z = 0.25934\,\text{m}$ (where the fingertips are $\approx 8.4\,\text{mm}$ below the top rim $0.2677\,\text{m}$), check whether the rigid box proxies penetrate the static collision mesh of `/World/RemediationTray/G11_tube/Wall` or `/World/RemediationTray/G21_tube/Wall`.
   - In PhysX, static meshes have infinite mass. If an articulation link or collision proxy penetrates a static mesh, the contact solver generates massive corrective contact impulses fighting the joint PD controller, creating violent wrist vibration/chatter.

2. **Collision Filtering / Disable Pairs Between Gripper and Tray**:
   - In `src/remediation/collision_manager.py`, inspect whether collision filtering or collision groups exist between the robot gripper (`/World/ER4iA/*`) and the static fixture (`/World/RemediationTray/*`).
   - If the finger proxies do not need to contact the glass tube itself (only the rubber cap), examine whether contact between the gripper proxies and the static tube colliders should be filtered or whether the finger proxy dimensions/offsets must be adjusted so they cannot collide with the tube rim.

3. **Cap Collision State During Insertion / Reseating**:
   - When the cap is being inserted into the vial bore, is `set_cap_collision()` enabled or disabled?
   - The vial tube wall has an inner bore of $26.365\,\text{mm}$ ID, while the cap plug has an outer diameter of $23.62\,\text{mm}$. If the cap mesh approximation in PhysX uses a bounding box, convex hull, or has contact offsets ($\ge 0.5\,\text{mm}$), it may generate continuous contact chatter against the vial tube inner wall.
   - Check the contact offset and rest offset on the caps and vials (`contactOffset`, `restOffset`).

4. **Actuator Stepping vs. Physics Solver Interaction at Contact**:
   - During `staged_insertion_missing` and `staged_reseat_crooked`:
     Look at how `arm.set_joint_positions(positions)`, `arm.set_joint_velocities(velocities)`, and `arm.apply_action(...)` are executed when external physical contact occurs.
     If `set_joint_positions` is called while the gripper or cap is in contact with another collider, it forces instantaneous kinematic penetration, causing PhysX to push back violently on the next `world.step()`.

---

## Execution Environment & Commands
Execute all simulations on the Slurm cluster:
```bash
# 1. Run simulation on GPU:
srun --partition=teaching --gres=gpu:1 --time=00:10:00 ./scripts/run_isaac.sh src/remediation/run_compliant_breakable_remediation.py \
    --scene shared/scenes/remediation_dataset1_v1.usda \
    --detections compliant_remediation_test/perception/current_scene_detections.json \
    --out compliant_remediation_test/action

# 2. Encode frames to MP4:
.conda_env/bin/python scripts/encode_frames.py \
    --frames compliant_remediation_test/action/frames \
    --out compliant_remediation_test/compliant_remediation.mp4 \
    --fps 25 && \
cp -f compliant_remediation_test/compliant_remediation.mp4 compliant_remediation_test/test_demo.mp4
```

Please investigate the physics contact behavior at the vial rim, resolve the collision interpenetration or controller conflict, and ensure the simulation runs jitter-free during contact.
