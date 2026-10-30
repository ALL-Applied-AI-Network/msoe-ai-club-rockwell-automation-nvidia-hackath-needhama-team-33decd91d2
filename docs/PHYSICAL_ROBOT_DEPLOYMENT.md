# Physical robot deployment

This is the operator checklist for the cap-remediation policy. The deployed
perception stack is YOLO26x plus FP16 NVIDIA LocateAnything-3B on one Tesla T4.
The standalone Qwen3.8-27B GGUF and llama.cpp server are not part of the robot
runtime. LocateAnything internally contains its own packaged Qwen2 language
component; it loads as part of LocateAnything and is not a separate service.

The policy always handles missing caps before crooked caps. It uses TableCam
for classification and coarse targets, then wrist RGB for local XY correction.
TableCam depth fits one work-surface Z plane per rollout. Crooked caps are
pressed as one cached batch: one flat wrist snapshot, one clockwise wrist roll,
one 45-degree cone posture, all down/up presses, then one clear verification.

## 1. Before entering the cell

From the repository root, submit the non-contact perception smoke test:

```bash
sbatch scripts/smoke_physical_deployment.slurm
```

This requests exactly one T4, stays offline, never imports the robot client,
and keeps YOLO resident while loading and running LocateAnything. A passing log
ends with:

```text
PASS: YOLO + LocateAnything initialized and inferred together on one T4.
```

Inspect `logs/physical-stack-smoke-<job>.out` and the generated
`shared/runs/<date>_physical_stack_smoke_<job>/deployment_smoke_report.json`.
Do not proceed if the class map is not exactly:

```text
0 good_cap, 1 crooked_cap, 2 missing_cap, 3 table_cap
```

Then run the API/cell-contract mock, still without a robot:

```bash
sbatch scripts/run_physical_policy.slurm \
  --mock --trials 1 --trial-s 8 \
  --mode full --allow-provisional-contact \
  --crooked-action cone-press --wrist-refinement
```

It must end with `Ready for the real robot` and zero rejected actions.

## 2. Robot and scene preflight

- Confirm the approved request/token is available. Keep `.request.json` and
  `PHYSICAL_CELL_TOKEN` private; never put a token in a command line or log.
- Use the exact physical gripper/cone represented by the simulation STEP/USD.
- Confirm the wrist camera is mounted in front of the arm in the verified home
  configuration and that its cables follow the approved route.
- Jog to the approved straight-down home. The expected starting W/P/R is
  `(-180, 0, 0)` degrees. The policy rejects a start more than 20 degrees away.
- Confirm the cone is 45 degrees from the gripper centerline and its tip is not
  damaged or flexed.
- Make the TableCam view clear. Do not move, crop, rotate, resize, or recalibrate
  TableCam for this policy; it remains native 1920x1080 and YOLO letterboxes to
  1280 internally.
- Verify the tray, paper, loose caps, vials, and work surface cannot move during
  the rollout. Loose caps must be upright and separated.
- Keep the cell's physical speed limits in place. Do not increase motion speed
  during first-contact tuning.

## 3. Offset file and sign conventions

Normal deployment tuning lives in
`PhysicalAI-Hackathon-2026/config/remediation_offsets.json`. Change one value at
a time and record the exact file with every physical trial. The canonical,
key-by-key field reference is [REMEDIATION_OFFSETS.md](REMEDIATION_OFFSETS.md).

| Setting | Meaning | Default / sign |
|---|---|---|
| `pickup_z_offset_m` | Fingertip clearance above the locally fitted table/paper plane while closing on a loose cap | `0.003`; larger is higher, smaller descends farther |
| `work_surface_z_offset_m` | Global correction added to the fitted work-surface plane and therefore its derived rim plane | `0`; positive is up |
| `vial_rim_plane_offset_m` | Residual added only after deriving rim Z from surface + tray base + vial height | `0`; positive is up |
| `vial_rim_plane_override_m` | Absolute robot-base rim Z | `0` means automatic; use only when deliberately disabling derived rim Z |
| `vial_height_override_m` | Absolute measured vial height used in rim derivation | `0` uses the verified 0.2032 m specification |
| `missing_release_z_offset_m` | Final cap-release height over a missing vial | `0`; positive is up/safer |
| `cone_press_z_offset_m` | Residual at the cone contact endpoint | `0`; positive is up/shallower |
| `wrist_table_cap_xy_offset_m` | Residual after wrist correction for pickup | robot-base `[X,Y]`; defaults `[0,0]` |
| `wrist_missing_cap_xy_offset_m` | Residual after wrist correction for placement | robot-base `[X,Y]`; defaults `[0,0]` |
| `wrist_crooked_cap_xy_offset_m` | Residual after wrist correction for the cone tip | robot-base `[X,Y]`; defaults `[0,0]` |
| `missing_release_side_step_m` | Small jaw-axis movement after opening | `0.001` m magnitude |
| `missing_release_exit_step_m` | Opposite jaw-axis component of the rising exit | `0.002` m magnitude |

The wrist XY offsets are added to the corrected target. If the robot lands
`+2 mm` in robot-base X from the desired point, start by changing that target's
X offset by `-0.002`. Do not enter camera-image left/right as robot-base XY.
Use the saved wrist annotation and measured physical miss to determine the sign.

`--cone-press-below-rim-m` remains a deliberate motion parameter rather than a
calibration residual. Its sign is the opposite of the positive-up offsets:
positive values put the cone endpoint below the fitted rim; negative values
leave clearance. The commanded cone-tip endpoint is effectively:

```text
fitted_rim_z - cone_press_below_rim_m + cone_press_z_offset_m
```

Avoid compensating one error in several places. In normal operation keep the
absolute rim override at zero, fit the surface automatically, use the rim
offset only for a repeatable derived-rim error, and reserve action offsets for
operation-specific residuals.

## 4. Safe tuning order

First verify camera-only targeting in hover mode:

```bash
sbatch scripts/run_physical_policy.slurm --mode hover
```

Confirm the annotated TableCam classes and coarse locations across the entire
work area. Then enable the full wrist workflow with generous contact clearance:

```bash
sbatch scripts/run_physical_policy.slurm \
  --mode full --allow-provisional-contact \
  --crooked-action cone-press --wrist-refinement \
  --pickup-z-offset-m 0.010 \
  --missing-release-z-offset-m 0.010 \
  --cone-press-z-offset-m 0.010 \
  --cone-press-below-rim-m=-0.005
```

That command is a clearance check, not a production cap cycle. Tune one action
at a time:

1. **Loose-cap pickup XY:** verify the green wrist/VLM center is on the cap.
   Adjust only `wrist_table_cap_xy_offset_m` for a repeatable residual.
2. **Loose-cap pickup Z:** reduce fingertip clearance from 10 mm toward 5 mm,
   then the 3 mm default. `pickup_z_offset_m=0.003` means the fingertip ends
   3 mm above the fitted local table plane; it is not “3 mm extra descent.”
3. **Missing-vial XY:** hold Z high and tune only
   `wrist_missing_cap_xy_offset_m`. The vial mouth is localized from flat wrist
   RGB; hollow-vial depth is not required.
4. **Missing release Z:** reduce `missing_release_z_offset_m` in small steps.
   The gripper opens once, moves sideways by the configured small step, then
   exits in the opposite direction while rising.
5. **Crooked-cap XY:** run one crooked cap with clearance and tune
   `wrist_crooked_cap_xy_offset_m` for the physical cone tip.
6. **Cone Z:** first use a negative `cone_press_below_rim_m` and positive
   `cone_press_z_offset_m`. Lower in 2–5 mm steps. The normal simulated endpoint
   was 3 mm below the rim, but physical tuning wins.
7. **Multiple defects:** only after each individual action works, test several
   missing caps, then several crooked caps. Missing replacements finish first;
   all remembered crooked targets reuse the same cone orientation before the
   arm clears TableCam for final verification.

Stop immediately if the wrist rolls around the wrong axis, the camera/cables
approach the tray, the fitted plane is implausible, or a commanded XY correction
has the wrong robot-base sign.

## 5. What to inspect after every run

Physical runs are written under `shared/runs/<timestamp>_physical_policy/` (or
the explicitly supplied trace directory). Check:

- `policy_config.json`: the exact checkpoint, action, and all effective offsets.
- `height_planes.json`: fitted work-surface Z, point counts, RMS, fallback
  reason, derived rim, and any active override. The fit is frozen per rollout.
- `tablecam_annotated/`: YOLO classes and coarse candidates.
- `wristcam_raw/`: the unmodified wrist RGB snapshots submitted to the VLM.
- `wristcam_annotated/`: selected VLM box and green refined centre, blue
  TableCam-derived target, their image-space residual, VLM latency, and the
  resulting base-frame correction. An image is written before geometric
  validation so a rejected correction remains diagnosable.
- `wristcam_results/`: VLM box, expected pixel, base-frame correction, latency,
  any wrist-depth support-plane fallback, and the corresponding annotated-image
  path.
- `trace.jsonl`: state transitions and commanded TCP poses.

Do not make contact if `height_planes.json` reports a nominal fallback without
an understood cause, the work-surface correction jumps several millimetres, or
the wrist correction exceeds the configured 40 mm bound.

## 6. Production launch

After offsets have been tuned and reviewed:

```bash
sbatch scripts/run_physical_policy.slurm \
  --mode full --allow-provisional-contact \
  --crooked-action cone-press --wrist-refinement
```

The launcher requests one Tesla T4. YOLO and LocateAnything share it. Isaac is
not part of physical deployment and still requires two T4s when wrist VLM is
enabled. No standalone Qwen or llama.cpp process should be started for a
physical run.
