# Demonstration Results

This directory contains presentation-ready evidence from the physical and simulated vial-remediation workflows. Together, the artifacts demonstrate a generalizable manufacturing-defect workflow: build a digital twin, use synthetic data to train visual inspection, refine targets from in-context wrist-camera observations, and command a physical system to correct detected defects.

The recorded scenario includes a loose cap, an uncapped vial, and a crooked cap. The robot picks up and installs the loose cap, then presses the crooked cap into place. The simulated run reproduces the same workflow with collision-aware PhysX contact; caps are moved by contact rather than by directly setting their poses.

## Videos

| View | Physical run | Simulated run | What it shows |
| --- | --- | --- | --- |
| Table camera | [real_table_cam.mp4](real_table_cam.mp4) | [simulated_table_cam.mp4](simulated_table_cam.mp4) | Full-cell inspection and remediation from the calibrated overhead view |
| Wrist camera | [real_wrist_cam.mp4](real_wrist_cam.mp4) | [simulated_wrist_cam.mp4](simulated_wrist_cam.mp4) | Close-range target refinement during pickup and crooked-cap correction |
| Side camera | [real_side_cam.mp4](real_side_cam.mp4) | [simulated_side_cam.mp4](simulated_side_cam.mp4) | Robot motion, clearance, pickup, placement, and contact from an external view |

The real and simulated videos are separate executions of the same overall remediation workflow. They are intended for qualitative comparison, not synchronized frame-by-frame playback.

## Complete physical-policy run

[`2026-10-30_16-21-27_physical_policy/`](2026-10-30_16-21-27_physical_policy/) is the complete machine-readable capture from the successful physical demonstration. The videos and top-level images are the smaller, presentation-oriented result set; this directory preserves the underlying sensor observations, model outputs, configuration, and controller trace for review.

| Path | Contents |
| --- | --- |
| [`tablecam_raw/`](2026-10-30_16-21-27_physical_policy/tablecam_raw/) | 39 full-resolution physical TableCam RGB observations |
| [`tablecam_annotated/`](2026-10-30_16-21-27_physical_policy/tablecam_annotated/) | The same 39 observations with YOLO detections where inference was requested |
| [`depth_raw/`](2026-10-30_16-21-27_physical_policy/depth_raw/) | 39 raw TableCam depth captures |
| [`depth/`](2026-10-30_16-21-27_physical_policy/depth/) | 39 colorized depth visualizations for human review |
| [`wristcam_raw/`](2026-10-30_16-21-27_physical_policy/wristcam_raw/) | Raw close-range views of the loose cap, uncapped vial, and crooked cap |
| [`wristcam_annotated/`](2026-10-30_16-21-27_physical_policy/wristcam_annotated/) | Wrist views showing the coarse projected target and refined visual target |
| [`wristcam_results/`](2026-10-30_16-21-27_physical_policy/wristcam_results/) | LocateAnything request, bounding boxes, latency, backprojection, and base-frame correction for each wrist observation |
| [`policy_config.json`](2026-10-30_16-21-27_physical_policy/policy_config.json) | Exact runtime mode, checkpoint, motion parameters, offsets, and camera-refinement settings |
| [`height_planes.json`](2026-10-30_16-21-27_physical_policy/height_planes.json) | RGB-D work-surface fit and the derived vial-rim plane used by the policy |
| [`trace.jsonl`](2026-10-30_16-21-27_physical_policy/trace.jsonl) | 117 timestamped policy, motion, detection, refinement, repair, and verification records |

The run used the full physical policy with YOLO inspection, RGB-D localization, wrist refinement, slide-in cap pickup, missing-cap placement, and cone pressing for the crooked cap. The fitted work-surface plane was derived from 6,584 depth inliers with a 1.91 mm RMS residual. The wrist model localized all three task targets; its saved inference latencies range from approximately 2.6 to 5.0 seconds.

The raw archive is retained for auditability. It contains deployment-specific absolute paths and calibrated offsets, so it should be treated as evidence of this run rather than as a portable configuration template.

## Inspection and perception images

### Table camera

- [simulated_table_cam_annotated_example.png](simulated_table_cam_annotated_example.png) — example YOLO predictions on a synthetic TableCam image from the digital twin.
- [real_table_annotated_initial.png](real_table_annotated_initial.png) — initial physical inspection showing the defects and loose replacement cap.
- [real_table_annotated_pickup_verification.png](real_table_annotated_pickup_verification.png) — TableCam verification after the replacement cap is picked up.
- [real_table_annotated_placement_verification.png](real_table_annotated_placement_verification.png) — verification after the replacement cap is placed on the uncapped vial.
- [real_table_annotated_final_verification.png](real_table_annotated_final_verification.png) — final inspection after the crooked-cap correction.
- [real_table_depth.png](real_table_depth.png) — colorized physical TableCam depth capture used with camera calibration to localize detections in robot-base coordinates.

The selected physical stills above are convenient copies from the complete physical-policy run directory.

The detector labels four task-relevant visual states:

- `good_cap`: a normally seated cap
- `crooked_cap`: a cap requiring corrective pressing
- `missing_cap`: an uncapped vial position
- `table_cap`: a loose replacement cap available for pickup

### Wrist camera

- [real_wrist_table_cap_annotated.png](real_wrist_table_cap_annotated.png) — in-context refinement of the loose-cap pickup location.
- [real_wrist_missing_cap_annotated.png](real_wrist_missing_cap_annotated.png) — refinement of the uncapped vial target.
- [real_wrist_crooked_cap_annotated.png](real_wrist_crooked_cap_annotated.png) — refinement of the crooked-cap target before correction.

These images visualize the correction between the coarse TableCam-projected target and the close-range target inferred from the wrist image. This refinement uses visual context rather than fixed tray coordinates.

## System represented by these results

1. A calibrated overhead RGB-D camera observes the manufacturing fixture.
2. A YOLO detector trained with digital-twin imagery identifies defects and available remediation objects.
3. Depth and camera calibration backproject image detections into robot-base coordinates.
4. The wrist camera provides a close-range observation for in-context target refinement.
5. A shared remediation policy generates the physical or simulated robot motion.
6. Collision-aware contact performs the pickup, placement, and crooked-cap press.

This vial-and-cap task is a concrete demonstration, not the limit of the approach. The larger goal is to shorten deployment time for new manufacturing defects by combining synthetic data from digital twins, learned perception, in-context refinement, and reusable physical-control workflows.

## Reproducing the simulation

The supported simulation path is the compliant, collision-aware pipeline documented in [`../docs/REMEDIATION_RUNBOOK.md`](../docs/REMEDIATION_RUNBOOK.md). Its entry point is:

```bash
sbatch scripts/run_compliant_breakable_pipeline.slurm \
  [scene.usda] <yolo_checkpoint.pt> <output_directory>
```

Physical deployment and calibration are documented in [`../docs/PHYSICAL_ROBOT_DEPLOYMENT.md`](../docs/PHYSICAL_ROBOT_DEPLOYMENT.md). Verified object and fixture dimensions are recorded in [`../docs/PHYSICAL_SPECIFICATIONS.md`](../docs/PHYSICAL_SPECIFICATIONS.md).

## Interpretation notes

- The videos are demonstration artifacts; quantitative detector evaluation belongs in the corresponding evaluation run reports and is not inferred from these selected images.
- The simulated result uses real rigid-body contact. It does not kinematically snap caps into their final poses.
- Visual appearance, sensor noise, timing, and robot dynamics are approximations, so agreement between simulation and the physical cell should be judged by task behavior and geometry rather than pixel-perfect rendering.
