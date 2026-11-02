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

## Inspection and perception images

### Table camera

- [simulated_table_cam_annotated_example.png](simulated_table_cam_annotated_example.png) — example YOLO predictions on a synthetic TableCam image from the digital twin.
- [real_table_annotated_01.png](real_table_annotated_01.png)
- [real_table_annotated_12.png](real_table_annotated_12.png)
- [real_table_annotated_22.png](real_table_annotated_22.png)
- [real_table_annotated_38.png](real_table_annotated_38.png) — selected physical TableCam frames with YOLO detections at different points in the run.
- [real_table_depth.png](real_table_depth.png) — colorized physical TableCam depth capture used with camera calibration to localize detections in robot-base coordinates.

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
