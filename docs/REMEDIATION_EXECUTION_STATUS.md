# Remediation execution status

Updated October 22, 2026. This records the current implemented evidence for the
black-and-silver remediation fixture. It does not promote the system for physical
robot contact.

## Fixture and scene

The canonical builder uses the Dataset 1 6 × 5 XY slot pattern with a fabricated
fixture: a 10 mm black bottom plate, four supports, and a 3 mm silver perforated
top plate. The underside of the silver plate is 80 mm above the top of the
black base. Vials rest on the base and pass through 31.75 mm apertures. The
latest single-scene qualification passed physics and label checks:

- `runs/archive/2026-10-22_remediation/black_silver_tray_qualification_v3/qualification.json`
- `runs/archive/2026-10-22_remediation/black_silver_tray_qualification_v3/side.png`

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

## Custom fingers and tool offset

The imported custom finger CAD is mirrored onto the moving SMC jaws. Its commanded
minimum mesh separation has been checked at 3.03–14.95 mm. Isaac CAD metrology
measured a conservative grasp-tip offset of 284.302 mm from the flange:

- `runs/archive/2026-10-22_remediation/black_silver_prediction_dry_run/custom_gripper_tcp_metrology.json`

The prior 293 mm stock tool value differs by 8.70 mm. The non-contact renderer
uses the CAD-derived value. The custom-finger contract remains
`blocked_pending_contact_and_tcp_validation`; it has no measured physical TCP,
cap pick offset, place offset, press offset, contact trial, or physical grasp
evidence.

## Remaining work before full remediation

1. Improve crooked-cap localization to the measured contact error budget.
2. Measure actual physical TableCam-to-base/TCP references and custom-finger
   contact offsets.
3. Validate gripper collision/contact, spare pickup, placement, and crooked-cap
   press in Isaac using the measured tool geometry.
4. Add reinspection after each action and only mark repair success from new
   perception evidence.
5. Repeat the validated perception/controller path with supervised hardware
   trials before enabling physical contact in `PhysicalAI-Hackathon-2026/`.
