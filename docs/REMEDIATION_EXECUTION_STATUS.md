# Remediation execution status

Archived status snapshot from October 22, 2026. It records the evidence available for the
black-and-silver remediation fixture. It does not promote the system for physical
robot contact. For the active scene and policy, use `REMEDIATION_RUNBOOK.md`.

## Fixture and scene

The field-ready list of physical measurements and capture requirements is
[`PHYSICAL_MEASUREMENT_CHECKLIST.md`](PHYSICAL_MEASUREMENT_CHECKLIST.md).
The reproducible command sequence is in
[`REMEDIATION_RUNBOOK.md`](REMEDIATION_RUNBOOK.md).

The canonical builder uses the Dataset 1 6 × 5 XY slot pattern with a fabricated
fixture: a 10 mm black bottom plate, four supports, and a 3 mm silver perforated
top plate. The underside of the silver plate is 80 mm above the top of the
black base. Vials rest on the base and pass through 31.75 mm apertures. The
latest single-scene qualification passed physics and label checks:

- `shared/scenes/black_silver_remediation_qualified_v3.usda`
- `shared/scenes/black_silver_remediation_qualified_v3.qualification.json`

The USD above is the canonical scene for compliant remediation. Do not use the
older `shared/scenes/remediation_dataset1_v2_cad.usda` for this workflow; it is
an earlier patched Dataset 1 scene rather than the qualified black-and-silver
fixture.

Caps use the hollow tapered geometry defined in `src/remediation/geometry.py`:
a 2 mm wall and a flat, closed 2 mm plug-bottom face.

## Synthetic data and YOLO

The current fixture configuration generated and qualified 100 scenes. The
generator can resume an interrupted Slurm run without replacing previously
qualified scenes. The resulting manifest and model evaluation are:

- `runs/archive/2026-10-22_black_silver_fixture_yolo/dataset/manifest.json`
- `runs/archive/2026-10-22_black_silver_fixture_yolo/training/evaluation.json`
- `runs/archive/2026-10-22_black_silver_fixture_yolo/training/weights/best.pt`

Held-out synthetic performance: crooked-cap recall 1.000, missing-cap recall
1.000, table-cap recall 1.000, and overall precision 0.9965. These are synthetic
metrics only; they do not establish real-camera or real-robot performance.

## Perception and motion handoff

`src/inspection/inspect_remediation_capture.py` runs the checkpoint on TableCam
RGB with aligned depth and produces base-frame candidate locations. The
prediction-driven Isaac renderer uses only those detections for selection. It
rejects a high-confidence target that has no fixed-down IK solution and selects
the next-highest-confidence reachable prediction. It does not replace targets
with known tray coordinates.

On held-out scene `001000`, offline evaluation against simulator truth reported:

| Class | Median XY error | Maximum XY error |
| --- | ---: | ---: |
| good cap | 0.78 mm | 1.18 mm |
| crooked cap | 5.30 mm | 5.30 mm |
| missing cap | 0.38 mm | 0.62 mm |
| table cap | 1.82 mm | 2.60 mm |

The crooked-cap estimate is not accurate enough for a press motion. Missing-vial
estimates are suitable for a non-contact approach in this simulated scene. See
`runs/archive/2026-10-22_remediation/black_silver_prediction_dry_run/localization_evaluation.json`.

The latest renderer reaches a vertically constrained approach 20 mm above the
selected missing-vial prediction. It holds the tool direction within 0.013° and
uses a 260 mm transit plane. Its report is:

- `runs/archive/2026-10-22_remediation/black_silver_prediction_dry_run/cad_tip_approach/dry_run_report.json`

This is deliberately a non-contact approach. It does not pick a spare, insert a
cap, reseat a crooked cap, or report a repair as successful.

## Current TableCam pipeline

The simulation pipeline is organized around the fixed TableCam that supplies
the initial decision. Its single Slurm submission script is
`scripts/run_compliant_breakable_pipeline.slurm`:

1. Isaac captures native 1920 × 1080 RGB, aligned depth, and camera metadata
   from the loaded remediation scene with `capture_live_tablecam.py`.
2. The project Conda environment runs the remediation YOLO checkpoint on that
   capture with `inspect_remediation_capture.py`.
3. Isaac selects only from that fresh prediction JSON and runs the compliant,
   breakable-vial PhysX simulation from TableCam with
   `run_compliant_breakable_remediation.py`.
4. The Conda environment runs YOLO on every rendered TableCam frame and writes
   the labelled review video.

Isaac Sim and Ultralytics run as separate sequential child processes within the
same Slurm job because their Python runtimes conflict over protobuf
registrations. This is a runtime isolation detail, not a split perception/action
workflow. A pipeline invocation is:

```bash
sbatch scripts/run_compliant_breakable_pipeline.slurm \
  shared/inspection_models/remediation_yolo_best.pt \
  shared/runs/$(date +%F)_compliant_remediation
```

The first frame inventories defects and starts one action plan. The current
per-frame labels are rendered evidence only: they do not yet interrupt, replan,
or verify the action. The required production behavior is instead
**inspect → choose one action → approach/act/retract → reinspect → confirm,
retry, or stop**. An action must never be marked successful from its rendered
motion alone.

## Custom fingers and tool offset

The imported custom finger CAD is mirrored onto the moving SMC jaws. Its commanded
minimum mesh separation has been checked at 3.03–14.95 mm. Isaac CAD metrology
previously measured a conservative visual-envelope grasp-tip offset of 284.302 mm from the flange:

- `runs/archive/2026-10-22_remediation/black_silver_prediction_dry_run/custom_gripper_tcp_metrology.json`

The prior 293 mm stock tool value differs by 8.70 mm. This visual measurement
is not a physical TCP and made current TableCam missing-vial predictions
unreachable in the fixed-down simulation solver. The best-effort TableCam run
therefore exposes a separate provisional 100 mm **simulation-only** offset on
its command line; it is used only to exercise the perception/action handoff,
not to claim contact accuracy. The custom-finger contract remains
`blocked_pending_contact_and_tcp_validation`; it has no measured physical TCP,
cap pick offset, place offset, press offset, contact trial, or physical grasp
evidence.

## Remaining work before full remediation

1. Add an Isaac observation/action adapter that executes the same state machine
   used by the physical policy: re-run YOLO after every short action chunk,
   require state change for success, and stop on stale, ambiguous, unreachable,
   or low-confidence observations.
2. Improve crooked-cap localization to the measured press-contact error budget;
   a bounding-box center is not enough for a press point.
3. Measure physical TableCam-to-base/TCP references and custom-finger pick,
   place, and press offsets. Replace the provisional simulation offset only
   after this calibration passes.
4. Qualify one isolated spare pickup, missing-cap placement, and crooked-cap
   reseat in Isaac using collision/contact geometry that does not overlap the
   retired stock-finger colliders.
5. Move the shared detector/state machine into
   `PhysicalAI-Hackathon-2026/policy.py`, request TableCam depth, use its
   real-cell calibration helpers, and pass `uv run run.py --mock`.
6. Validate real images and supervised hardware approaches before enabling
   physical contact or submitting a non-hold policy.
