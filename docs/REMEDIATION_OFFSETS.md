# Remediation offset reference

This document is the field reference for every setting in
`PhysicalAI-Hackathon-2026/config/remediation_offsets.json`.

All distances are in **metres** and all XY values use the **robot-base frame**,
not image left/right. The file is read when the policy starts, so restart the
policy after editing it. Every physical run copies the effective values into
`policy_config.json`; always inspect that file before interpreting a result.

## Z planes and action heights

### `pickup_z_offset_m`

- Default: `0.003`
- Meaning: absolute fingertip clearance above the locally fitted table/paper
  support plane when the gripper closes on a loose cap.
- Direction: a larger positive value keeps the fingertip **higher**. A smaller
  value descends closer to the table.
- Important: `0.003` means “fingertip 3 mm above the table,” not “move down an
  additional 3 mm.”
- Affects: loose `table_cap` pickup only.

### `work_surface_z_offset_m`

- Default: `0.0`
- Meaning: signed correction added to the once-per-rollout TableCam depth fit
  of the work surface. It is also applied to a valid local wrist-depth support
  fit used for pickup.
- Direction: positive is **up** in robot-base Z; negative is down.
- Affects: pickup support height and the automatically derived vial-rim plane.
- Use for: a repeatable global Z calibration error affecting the whole work
  surface. Do not use it to fix only one action.

### `vial_rim_plane_offset_m`

- Default: `0.0`
- Meaning: signed residual added after rim Z is derived from:

  ```text
  fitted work surface + 10 mm tray base + effective vial height
  ```

- Direction: positive moves the derived rim plane **up**; negative moves it
  down.
- Affects: missing-cap placement height and cone-contact height.
- Use for: a repeatable rim-specific error when the fitted table plane itself
  is correct.

### `vial_rim_plane_override_m`

- Default: `0.0`
- Meaning: absolute vial-rim Z in the robot-base frame.
- Direction: this is an absolute coordinate, not a signed correction.
- `0.0` disables the override and keeps automatic rim derivation.
- Affects: all actions that use the vial-rim plane.
- Use only when automatic derivation is intentionally being bypassed. A
  nonzero value takes precedence over the fitted/derived rim and its residual.

### `vial_height_override_m`

- Default: `0.0`
- Meaning: absolute physical vial height used to derive the rim plane.
- Direction: this is a positive physical dimension, not an up/down offset.
- `0.0` uses the verified `0.2032 m` vial specification.
- Affects: automatically derived vial-rim Z.
- Example: a measured 202.7 mm vial is entered as `0.2027`, not `-0.0005`.

### `missing_release_z_offset_m`

- Default: `0.0`
- Meaning: residual added to the final gripper-open position over a missing
  vial.
- Direction: positive is **up/shallower**; negative is down/deeper.
- Affects: missing-cap release only.
- For first physical clearance checks, start positive and reduce it in small
  steps.

### `cone_press_z_offset_m`

- Default: `0.0`
- Meaning: residual added to the cone-tip contact endpoint.
- Direction: positive is **up/shallower**; negative is down/deeper.
- Affects: crooked-cap cone press only.
- It combines with the CLI motion parameter `--cone-press-below-rim-m` as:

  ```text
  cone tip Z = effective rim Z
               - cone_press_below_rim_m
               + cone_press_z_offset_m
  ```

  Therefore `--cone-press-below-rim-m` has the opposite convention: positive
  means below the rim, while positive `cone_press_z_offset_m` raises the tip.

## Wrist-camera XY residuals

These offsets are applied **after** the wrist camera has refined the TableCam
target. They compensate residual wrist hand-eye calibration and tool-center
errors. Each value is `[robot_base_X, robot_base_Y]`.

For the controller's normal flat, downward-looking wrist-camera pose, the
directions appear in the saved RGB image as follows (the small hand-eye skew is
about 1--2 degrees):

| Offset direction | Direction in wrist RGB |
| --- | --- |
| `+X` | up |
| `-X` | down |
| `+Y` | left |
| `-Y` | right |

Equivalently, image-right is approximately robot-base `-Y`, and image-down is
approximately robot-base `-X`. This mapping was checked at the saved loose-cap,
missing-vial, and crooked-cap view poses. The values in the JSON remain
robot-base offsets; the wrist projection itself uses the measured snapshot
joints, so do not reinterpret the JSON fields as camera-local axes if the
camera pose is changed.

If the robot lands `+2 mm` in robot-base X from the desired point, start by
changing that action's X residual by `-0.002`. Measure error in the robot-base
frame; do not copy image-pixel left/right into this file.

### `wrist_table_cap_xy_offset_m`

- Default: `[0.0, 0.0]`
- Affects: loose-cap pickup after wrist-camera refinement.
- Does not affect missing-vial placement or cone pressing.

### `wrist_missing_cap_xy_offset_m`

- Default: `[0.0, 0.0]`
- Affects: cap placement over the wrist-refined missing-vial mouth.
- Does not affect pickup or cone pressing.

### `wrist_crooked_cap_xy_offset_m`

- Default: `[0.0, 0.0]`
- Affects: cone-tip XY over wrist-refined crooked caps.
- This must be tuned for the physical side-mounted cone; do not copy a jaw or
  TableCam offset into it.

## Missing-cap release clearance

These settings are nonnegative movement magnitudes. They are not generic
robot-base XY calibration offsets.

### `missing_release_side_step_m`

- Default: `0.001`
- Meaning: after opening once over the vial, move a small distance along the
  gripper jaw-spread axis to clear one flange lip.
- Direction: for the normal vertical placement W/P/R `(-180, 0, 0)`, a positive
  magnitude moves toward robot-base **-Y**.
- Affects: the short, level post-release clearance move only.

### `missing_release_exit_step_m`

- Default: `0.002`
- Meaning: horizontal component of the following rising exit, in the opposite
  jaw-axis direction.
- Direction: for normal vertical placement, a positive magnitude contributes
  robot-base **+Y** while the TCP also rises 25 mm.
- Affects: the post-release retreat only.

The defaults intentionally produce a gentle 1 mm side-clear followed by a 2 mm
opposite rising exit. Increase them cautiously; overly large values can knock a
newly placed cap off the vial.

## Recommended tuning discipline

1. Keep both absolute overrides at zero unless there is a documented reason.
2. Tune wrist XY separately for pickup, placement, and cone contact.
3. Tune `pickup_z_offset_m` as an absolute table clearance.
4. Tune missing-release Z and cone-contact Z independently.
5. Change only one parameter between physical trials.
6. Save `remediation_offsets.json`, `policy_config.json`, and
   `height_planes.json` with each result.
7. Never compensate one physical error in several offsets at once.

For the complete launch and safety sequence, see
[PHYSICAL_ROBOT_DEPLOYMENT.md](PHYSICAL_ROBOT_DEPLOYMENT.md).
