# Remediation audit and implementation plan

Review date: October 21, 2026. Status: implementation in progress; this document preserves the original audit and acceptance gates.

## Implementation status — October 21, 2026

The original Dataset 1 station geometry was measured read-only through Slurm and recorded in `src/remediation/config/dataset1_layout.json`.  A canonical builder now uses those slot/support locations with the verified physical vial and cap dimensions, actual aperture openings, physics settling, dense cap-surface penetration checks, and instance-mask-derived labels.  The scene qualification reports are archived under `runs/archive/2026-10-21_remediation/`.

A 10-scene Slurm pilot completed with all scenes accepted and produced the four required classes.  Its tiny 8/1/1 train/validation/test split is only a plumbing check: its crooked-cap precision was 9.0% on the one-scene test set, so its checkpoint is deliberately not approved for robot use.  The controller and physical policy require an explicitly validated detector/localization/tool chain before contact motion; by default the physical policy only holds the observed joint position.

A varied 100-scene qualification run and dependent training run completed as Slurm jobs 286916 and 286917.  All 100 scenes were accepted, with 3,257 labels and independently sampled inventory (271 crooked, 259 missing, and 257 table caps requested).  The held-out synthetic test reported crooked and missing-cap recall of 1.0 and table-cap precision of 0.996.  This is a small synthetic holdout, not a real-world release result: the release checker still blocks contact motion pending independently reviewed real images, measured contact-localization accuracy, and custom-finger CAD/contact validation.

The custom grooved fingers are now represented by `src/remediation/config/custom_finger_contract.json`. The user-supplied `data/gripper.step` is recorded there with its AP242 metre declaration, checksum, and raw coordinate envelope; `src/remediation/inspect_gripper_step.py` reproduces that record without a CAD kernel. The contract remains deliberately blocked until the imported model's mounting pose, dimensions, collision proxies, TableCam-to-TCP calibration, Isaac contact trials, and supervised physical grasp trials are recorded. A STEP bounding box does not establish any of those properties.

The visual mount has now been reviewed against the supplied finger orientation. The STEP is mirrored as a left/right pair with local -Z pointing toward the grasped object and the inset +Y face toward the inside of each pneumatic jaw. Relative to the initial mount, the pair is extended 20 mm along the tool axis and raised 6 mm; the jaws start fully extended at 6 mm per side. The measured mesh envelope is set to 3 mm minimum separation when closed and 15 mm when fully open. The moving `SMCJaw` bodies remain visible; the old long extenders, sensor carrier/red nose, outer cross-bolt, laser body, and black sensor lead are hidden from the visual preview. Top, front, side, and perspective renders are archived under `runs/archive/2026-10-21_remediation/mounted_custom_gripper_views_open15`. This is still a visual pose review: collision/contact, TCP calibration, and physical grasp validation remain release gates.

TableCam qualification captures now save aligned RGB and metric depth (`table.png` and `table_depth_m.npy`). YOLO uses RGB for class/box predictions and records a robust depth observation per box; the depth observation does not make a target contact-qualified. `src/inspection/evaluate_tablecam_localization.py` measures predicted locations against scene truth strictly as an offline evaluation.

The release check was rerun after adding the collision proxies. Contact remains correctly denied: the pilot checkpoint is stale, the pilot crooked/missing-cap metrics are below threshold, no real-image validation or contact-localization error budget is recorded, and custom-finger contact measurements are incomplete. The next implementation gate is a proxy-only swept-motion test across the 3--15 mm jaw envelope, followed by TableCam-to-TCP calibration and object-specific contact trials.

The requested workflow is feasible in structure, but the current repository does not yet demonstrate an end-to-end, perception-driven remediation process. Scene generation, training, localization, simulation control, and the physical-cell policy currently have separate assumptions and incomplete connections. More training on the current images should wait until scene physics and labels are corrected.

## Scope and evidence

Reviewed `AGENTS.md`, all four existing documents in `docs/`, the generation/inspection/remediation source, launch scripts, the physical-cell client and calibration, shared scene assets, Dataset 1 documentation and preview, a generated remediation image, saved dataset labels, training results, and recent generation logs. Raw shared data was only read.

Lightweight checks performed:

- Parsed all 22 project Python files under `src/` and the physical client/calibration directories using Python 3.10; syntax passed. Environment/vendor packages were excluded.
- Called the policy locally with empty images: it still produced a 13-action remediation sequence. No commands were sent to a robot or server.
- Checked the table-camera projection mathematically: native-resolution round-trip error was effectively zero. Supplying a pixel scaled to the policy's 640 × 360 image size without adjusting its 1920 × 1080 calibration produced approximately **766 mm** of position error for one representative vial-rim point. This demonstrates a latent integration bug, not measured robot error.
- Counted **8 training and 2 validation images**, with 320 and 80 labels respectively, in the remediation dataset.
- Inspected the saved training CSV: **23 recorded epochs**, with precision, recall, mAP50, and mAP50–95 all zero throughout. Local best/last checkpoints exist, but the shared `remediation_det_best.pt` export was absent at inspection time. These artifacts do not establish a successful completed training run.

No new Isaac simulation, GPU training, physical-cell execution, or submission was run. The audited host interpreter has no `pxr`; USD findings below come from source/asset text and existing images, not a fresh composed-stage or physics validation. Existing videos and success captions do not substitute for measured execution reports.

## Current readiness against the requested process

| Step | Existing foundation | Blocking gap |
|---|---|---|
| 1. Physically accurate synthetic data | Shared metric vial scene, randomized builders, four-class image generator | Conflicting geometry/frames, initial table penetration, no verified settling or state-derived labels |
| 2. Train YOLO | Dedicated four-class trainer and Slurm training wrapper | Tiny dataset; recorded metrics are zero; no held-out real cap-defect evaluation |
| 3. Locate every defect and spare | Camera calibration and ray-plane helpers | YOLO is not connected to remediation; tray classifier is a stub; fixed target coordinates remain |
| 4. Repair in Isaac Sim | Contact-based scripted press/pick/place demo and IK | Demo targets come from known scene coordinates; randomized video script only renders; no closed-loop success verification |
| 5. Transfer to physical cell | Observation/action client, UF0 conversion, mock and submission files | Policy ignores checkpoint/images; hardware mismatch; resolution/frame handling and actual grasp/press behavior unvalidated |

## Findings that change the implementation order

### 1. Scene geometry and frames must be unified first

`src/remediation/build_randomized_remediation_scene.py:58` places the tray at world Z = 0.002 m and spares near −0.0031 m, although its imported shared FANUC cell places the work surface at **world Z = 0.8949 m**. The latter corresponds to **robot-base Z = −0.0051 m**, because the robot base is at world Z = 0.90 m. These are different coordinate frames.

That builder uses **40 × 38 mm** slot pitch, a **5.5 mm** cylindrical plug, and a **4 mm** flange. The training generator and shared clean tray use **44.45 × 44.45 mm** pitch, **13.462 mm** plug depth, and **2.032 mm** flange thickness. The specification calls for a tapered, hollow plug; the generated caps use solid cylinders. Randomized crooked caps therefore do not align with all referenced vial mouths, and their contact geometry differs from the stated design. Hiding native caps in the randomized builder also leaves their collision geometry enabled. Its `num_good` parameter is not used to enforce the requested count, and the purported minimum separation of table caps is not implemented.

The current `shared/scenes/clean_vial_tray.usda` contains a solid Cube plate with a collider, rather than modeled aperture cutouts. It is a simplified tray with hollow vial meshes; the Dataset 1 preview includes an elevated fixture, support structure, and aperture pattern. `generate_clean_tray_usd.py` copies the original station geometry, while the current shared asset has different procedural structure; regeneration provenance is unclear. Exact Dataset 1 pitch, support elevations, and tube seating depth need a measured asset comparison before calling this a faithful physical layout.

### 2. Replacement caps start inside the table

In `src/data_generation/generate_remediation_dataset.py:196`, the cap origin is placed at table height + flange thickness + 2 mm. Its downward plug reaches 13.462 mm below that origin. Consequently the lowest point starts **9.430 mm below the table**:

`2.032 + 2.000 − 13.462 = −9.430 mm`.

Use the transformed geometry's lowest support point to calculate spawn height, for every orientation. A plug-down cap on a flat table rests on its plug, raising its flange; a flange-down cap has a different grasp and may need reorientation. The existing demo instead uses an elevated supply nest. Those are different manipulation tasks and must not be treated as equivalent.

### 3. Crooked labels describe intended spawning, not verified resting state

`src/data_generation/generate_remediation_dataset.py:224` performs render convergence with `delta_time=0.0`; it does not implement an explicit advancing-physics settle loop or measure stability. Every requested crooked object is labeled crooked regardless of whether it is floating, seated, fallen, or occluded. Boxes are fixed **42 × 42 pixels** around projected origins, with no visible-extent or occlusion validation.

The document's claimed 31.5–34° equilibrium must be reproduced for the actual tapered cap, mouth collider, and material configuration. It cannot be inferred from a 26° spawn angle. The random yaw also needs the lateral spawn bias rotated consistently with the tilt direction.

### 4. Current detection pipelines are not interchangeable

The original CapSealCam detector uses `pass/crooked_cap/missing_cap/damaged_vial`. The remediation detector uses `good_cap/crooked_cap/missing_cap/table_cap`. The first uses a different camera and fixed pixel slot centers. Its unmatched-slot logic emits `no_vial` with confidence 0.99, so a missed detection can become a false robot task. Unknown classes fall through to pass.

`src/inspection/vision_pipeline.py:117` never implements tray classification: its loop ends with `pass`. Its spare-cap detector is a yellow-color heuristic. Neither this module nor the physical policy loads the remediation YOLO checkpoint.

Dataset 1's `cap_seal` images are **unlabeled** according to the supplied dataset README. The supplied labels describe vial-fill recipes, not cap defects. Real cap-defect validation requires an independently reviewed annotation set. Dataset 1's camera also does not establish performance on the FANUC TableCam or on loose replacement caps.

### 5. Existing motion is demonstration code

`src/remediation/simulate_remediation_video.py:295` projects known G05/G15 coordinates and backprojects them, then builds motion directly from the known world positions. This is not image-driven localization and conflicts with the repository's trajectory rule. It services one example of each defect, and its success caption is selected by elapsed frame count, without verifying a repaired cap. IK failures retain the previous solution while the sequence continues.

`src/remediation/simulate_randomized_remediation_video.py:115` only advances the world and records frames. Its joint helper is never called and it performs no inference or remediation. It also builds a base cell and references another complete cell beneath a new root, requiring composition checks for duplicated robots, tables, and cameras.

The demo imports the modified local `data/fanuc_er4ia_sim` preferentially; newer scripts import immutable `shared/data/fanuc_er4ia_sim`. Only the local `cell.build` supports `finger_type="cap"`. This split makes the demo's gripper behavior non-reproducible through the shared starter alone.

### 6. Physical-cell integration is incomplete

`PhysicalAI-Hackathon-2026/policy.py:43` ignores `checkpoint`; its planner uses hardcoded tray and spare coordinates. It requests 640 × 360 images but its projection helper uses native 1920 × 1080 calibration. It plans once and does not inspect outcomes. The move toward the spare descends diagonally to 120 mm above the table rather than maintaining the required transit height. Clipping waypoints to workspace bounds silently changes the intended target and does not check the swept path.

The stock finger calibration documents approximately **16 mm maximum hooked-tip gap**; the cap flange is **31.5 mm** wide. The demo uses custom cap fingers, while `submission/submission.yaml` declares `custom_hardware: none`. Pad width is not the same as jaw opening. Required opening, usable grip depth, neighboring-vial clearance, and actual hardware availability are release-blocking questions.

`submission/placement.yaml` lists loose caps but no tray pose, vial inventory, or defect setup. The client environment also does not declare the detector runtime dependencies. A mock pass checks formats and limits, not camera accuracy, reachability, contact forces, or grasp success. The documented observation schema supplies no force measurement; a force-controlled reseat cannot be assumed available.

### 7. Reproducibility and documentation lag the code

The dedicated remediation generator, trainer, randomized video script, and training Slurm wrapper were untracked at audit start. `scripts/run_all.sh` still runs the older CapSealCam workflow and calls the Isaac runner directly. It is not the requested five-step Slurm workflow. Generation uses unchecked `os.system` child execution; stale samples and overwritten run names can hide failed reruns.

The existing architecture guide describes classifier/VLM scripts that are absent and reports accuracy/latency claims without a matching evaluation artifact for this workflow. The physical specifications contain inconsistent gripper friction values (≥3.0 versus 1.2) and mix measured dimensions with simulation equilibrium claims. `setup.sh` bootstraps inside immutable `shared/data`; generated textures and other mutable dependencies need to be redirected. Raw data currently has mode 0555. The permissions helper was inspected but not executed during this read-only audit because it recursively changes shared permissions.

## Proposed implementation sequence and acceptance gates

The gates below are proposed engineering criteria, not results already achieved. Scope is cap reseating and missing-cap replacement on a stationary tray. Pump control, damaged-vial sorting, and empty-vial backfill remain separate projects.

### Phase A — Establish one scene and hardware contract

1. Create one versioned physical configuration for vial dimensions, cap taper/hollow geometry, fixture pitch/support/seating geometry, tool geometry, contact properties, and coordinate transforms. Record each value's source and distinguish measured values from estimates.
2. Compare the original Dataset 1 USD, real reference images, and shared clean tray. Preserve verified tray structure and G01–G30 ordering while adding appropriate collision geometry. Geometry coordinates may be used for scene construction and evaluation; they must not supply runtime robot targets.
3. Consolidate scene construction into one reusable implementation used by data generation and remediation simulation. Use a maintained writable robot adapter rather than editing `shared/data` or relying on two divergent starter copies. Store canonical scene assets in `shared/scenes` and generated dependencies in `shared/cache`.
4. Resolve the actual cap-gripping hardware and loose-cap orientation. Keep loose tabletop caps as the default requested scope; do not silently replace them with a nest. If a nest or custom fingers are necessary, document the concrete geometry and proposed physical setup for review.
5. Implement explicit transforms: world → robot base subtracts 0.90 m from Z in the current cell; base metres → UF0 millimetres multiplies by 1000 and subtracts 330 mm from Z. Derive rim height from actual tray support and vial seating, not the current fixed 0.2052 m.

**Gate:** a dimension/composition report and table/side reference images show correct vial and cap geometry, real aperture cutouts, support heights, a single robot/cell, and feasible gripper clearance. Dataset 1 layout discrepancies are resolved or explicitly documented with their effect on the task.

### Phase B — Verify contact physics before collecting training images

1. Spawn caps above the highest relevant support using transformed mesh bounds and positive clearance. Enforce cap separation and reachable placement regions.
2. Disable both visual geometry and collisions for removed caps. Use hollow vial colliders that preserve the mouth and tapered cap colliders that reproduce rim contact. Check solver settings, thin-wall contacts, and the need for continuous collision detection.
3. Advance physics for a minimum warmup, then require low linear/angular velocity and stable pose over a measurement window. Treat the documented 60–90 frames as a starting point, with timeout/rejection for unsettled scenes.
4. Classify final state using measured cap normal, rim-relative height, support/contact, and vial association. Reject or relabel caps that become seated, fall off, or leave their intended vial. Define a deliberate ambiguous range rather than forcing every object into crooked/good.
5. Sweep all 30 slots, tilt directions, tray poses, and material variations. Export per-object state, support/penetration checks, and settling success statistics; inspect side-view evidence as well as overhead renders.

**Gate:** an initial 100-seed qualification set has no accepted unsupported/interpenetrating caps, and every accepted crooked label corresponds to a stable cap supported by its vial. Record rejection rates and tolerances. The acceptable penetration tolerance must be small relative to the 0.635 mm vial wall; propose ≤0.1 mm and verify that the chosen solver/collider configuration can support it. Failure should block dataset acceptance rather than be hidden by labels.

### Phase C — Generate and validate the four-class dataset

1. Fix a shared class contract: `good_cap` = seated cap on vial; `crooked_cap` = supported misseated cap; `missing_cap` = visible uncapped vial mouth; `table_cap` = usable replacement cap. Empty holes, uncertain/occluded objects, and unusable spares must not become robot targets by default. Add an orientation/eligibility check for table caps if their orientations vary.
2. Generate boxes from rendered object extents/instance masks after settling. For missing caps, annotate the vial mouth region. Save segmentation/keypoint information where useful for later grasp-center refinement. Apply visibility, image-boundary, and occlusion checks.
3. Start with a 100-scene pilot, inspect overlays and class counts, then scale to approximately 2,000–5,000 independent scenes if the pilot passes. Randomize defect counts/locations, cap orientation, tray pose, lighting, materials, backgrounds, camera variation, and partial occlusion; include all-good and no-spare cases.
4. Separate training, validation, and test by scene seed/configuration. Keep related frames in one split. Build a human-reviewed real-image holdout grouped by source video/session, including eventual real TableCam spare-cap images. Keep annotations outside immutable data.
5. Use versioned output directories, checked subprocess exits, expected sample counts, and manifests containing seed, scene/config hashes, calibration, class mapping, and quality results. A failed child or incomplete render must fail the generation job.

**Gate:** every accepted image passes label and physics checks, and a reviewed contact sheet covers every class, tray region, and difficult case. Split integrity and a complete reproducible manifest are required before training.

### Phase D — Train YOLO and make predictions usable for motion

1. Run the existing YOLO11 baseline through Slurm on the corrected pilot. Confirm it can learn a small clean set before committing to a full training run. Track per-class precision/recall, confusion, and failures; do not promote a model solely because `best.pt` exists.
2. Train the larger dataset and evaluate synthetic test scenes and independent real images separately. Use real training examples for fine-tuning if required, preserving the real holdout. Compare deployed input resolutions: the current 640 × 360 policy downsamples a 42-pixel cap to roughly 14 pixels.
3. Build one perception service that loads the checkpoint and produces image-space detections plus confidence, class, timestamp, and refinement quality. Determine vial-mouth/press centers and grasp centers from pixels; tilted-cap box centers are not necessarily vial centers.
4. Use calibrated TableCam ray-plane backprojection with explicit image resize/letterbox reversal and lens handling. Use the correct class/feature height plane. If depth is used, transform its separate calibration into the RGB/base frame; do not index the depth stream using color-image coordinates.
5. Return base-frame positions and uncertainty. Build any tray grid from observed features and fit confidence; use nominal spacing only as a consistency prior. No detections, stale observations, ambiguity, and out-of-workspace targets must produce a hold/reinspection result.

**Gate:** proposed detection targets are ≥95% recall for crooked/missing caps and ≥99% precision for actionable replacement caps, reported separately on synthetic and real holdouts with counts and uncertainty. Geometry-only projection tests should be submillimetre. Final contact localization must meet a measured insertion/grasp error budget; the nominal plug-bottom radial clearance is only **1.27 mm**, so a coarse box center is insufficient evidence. Establish the required accuracy/compliance with the actual tapered cap before authorizing insertion.

### Phase E — Execute and verify all repairs in Isaac Sim

1. Implement a shared state machine: inspect → choose supported task → approach → act → retract → reinspect → mark success/retry/fail. Continue until all observable defects are repaired or explicitly unresolved, with bounded retries and supply inventory tracking.
2. Reseat crooked caps with a validated contact approach. Pick usable loose spares, verify pickup, transfer, place/seat, and verify the result. Add orientation handling if a cap cannot be inserted in its resting table orientation.
3. Keep lateral travel above the required 0.26 m clearance plane, explicitly defined relative to the work surface, and also above the highest obstacle plus the carried-object/tool envelope. Raise before crossing the tray and descend only over the observed target. Validate the swept gripper/arm path and all slot reachability; handle failed IK by stopping/replanning.
4. Use images and robot observations for online decisions. Keep simulator truth in a separate evaluator for contact, pose, collision, pickup, and final-seating measurements. Never use truth to correct predicted targets or animate a successful repair.
5. Reuse the same perception and policy logic that will run against the physical client. Supply an Isaac adapter for observations/actions. Capture videos together with machine-readable execution reports.

**Gate:** pass isolated reseat and replacement trials, then the 100-seed randomized suite. Proposed target: ≥95% complete-tray success, no table penetration or unintended collisions, and zero false success reports. Include perturbation, missed detections, exhausted spares, failed pickup, unreachable target, and reinspection tests. Report per-action and full-tray results separately.

### Phase F — Validate the physical-cell package and transfer

1. Replace the fixed example sequence in `PhysicalAI-Hackathon-2026/policy.py` with the shared perception/state machine. Load the checkpoint, validate class mapping, support actual observation sizes, and reset cleanly between rollouts. Declare and pin runtime dependencies.
2. Test `run.py --mock` for contract/reset/error behavior, plus image-replay tests for meaningful perception. Passing the mock alone is not a manipulation result.
3. Update submission hardware, tray/vial/spare inventory, defect definitions, placements, and success criteria to match the tested task. Confirm actual custom-finger availability and permitted press/control behavior with the cell operator before physical evaluation. Do not assume force feedback or compliance from the current API.
4. Validate real camera/TCP/frame alignment using known references, then run perception without motion, above-object approach checks, one supervised spare pickup, one missing-cap repair, one reseat, and finally full-tray trials.
5. Evaluate the requested 10-rollout physical run and archive detections, actions, before/after images, faults, interventions, and task outcomes. Tune using separate development runs rather than the final evaluation set.

**Gate:** physical compatibility is claimed only after measured real-camera localization and supervised repair success on the actual hardware, followed by the agreed rollout evaluation. Simulation performance and Dataset 1 accuracy alone cannot guarantee real-robot success.

## Execution and deliverables

Sequence: **scene/hardware contract → physics qualification → label pilot → training/localization → closed-loop simulation → physical-cell validation**. Resolve the hardware and API questions early because they affect geometry and grasp strategy.

All heavy Isaac work must use `sbatch scripts/submit_isaac_job.slurm <target_script> [arguments]`; training uses `scripts/submit_training_job.slurm`. Add a preflight and dependent Slurm stages so training starts only after the dataset passes acceptance checks. Create log directories before submission. Move debug crops, qualification reports, and checkpoints into `runs/archive/YYYY-MM-DD_<task>/`; keep canonical scene assets in `shared/scenes` and public demo videos in `runs/` or `docs/media/`. Preserve immutable `shared/data`. Use the permission helper during shared-asset setup, verifying its effects rather than trusting its unconditional success message.

Deliverables for review at each gate: scene/dimension report; settling/contact report and videos; labeled dataset pilot and manifest; model evaluation and localization error report; randomized execution report; reproducible physical-cell package and real-trial results. Refresh README, architecture documentation, and physical specifications to describe the tested workflow and distinguish measured results from planned targets.

Recommended first implementation milestone: **one faithful, reproducible scene with a stable crooked cap, an uncapped vial, and a nonpenetrating loose replacement cap; correct labels and qualification evidence before bulk generation.**
