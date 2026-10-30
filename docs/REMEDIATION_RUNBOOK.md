# Remediation runbook

For the focused physical-cell operator checklist, one-T4 perception smoke test,
and offset-tuning sequence, see [PHYSICAL_ROBOT_DEPLOYMENT.md](PHYSICAL_ROBOT_DEPLOYMENT.md).

This is the reproducible path for the synthetic-data, TableCam perception, and
Isaac Sim remediation workflow. New generated files live under `shared/runs/`
so the team can inspect them; historical local archives remain under
`runs/archive/`. The source code, configuration, commands, and required reports
are tracked in this repository.

The current controller is a **best-effort simulation review**, not authorization
for physical robot contact. It uses a fresh TableCam capture to choose targets,
but does not yet verify a successful pickup, placement, or press after acting.

## 1. One-time environment checks

Run these from the repository root on the cluster login node:

```bash
test -x .conda_env/bin/python
test -x shared/isaacsim_6.1/python.sh
test -r shared/data/fanuc_er4ia_sim/cell.py
test -r data/gripper.step
test -r data/gripper.usd
scripts/fix_shared_permissions.sh
mkdir -p logs shared/runs
```

`shared/data/` is competition reference data. Never modify it. Isaac rendering,
physics qualification, and synthetic-data generation must be submitted through
Slurm; do not run them directly on the login node.

Choose a new descriptive output directory for every experiment:

```bash
export RUN_DIR="shared/runs/$(date +%F)_remediation_<description>"
```

Replace `<description>` with a short identifier such as `baseline_seed20261022`.
Do not reuse a directory containing an accepted dataset or completed run.

`data/gripper.usd` is the Isaac-readable conversion of the supplied custom
finger CAD. If it is missing after a fresh checkout, create it once through
Slurm, then preserve the conversion report with the run:

```bash
sbatch scripts/submit_isaac_job.slurm src/remediation/convert_gripper_step.py \
  --step data/gripper.step \
  --out data/gripper.usd \
  --report "$RUN_DIR/gripper_cad_conversion.json"
```

## 2. Build and qualify one deterministic remediation scene

This creates the black 10 mm base, 80 mm supports, 3 mm perforated silver
plate, vials, hollow caps, one crooked cap, one missing cap, and one loose
replacement cap. `--fixed` disables tray and camera randomization so the result
is reproducible from its seed.

```bash
sbatch scripts/submit_isaac_job.slurm src/remediation/qualify_scene.py \
  --out "$RUN_DIR/qualification" \
  --seed 20261022 \
  --crooked 1 --missing 1 --table-caps 1 \
  --paper-color 1.0,0.25,0.55 \
  --steps 360 --fixed
```

Loose replacement caps are staged on the right side of the calibrated
TableCam image, over the optional pink paper. Use the same paper placement in
the physical cell; it is a visual background aid, never a localization target.

After the job finishes, inspect:

```bash
cat "$RUN_DIR/qualification/qualification.json"
```

Continue only when `accepted` is `true`. Retain these files with the run:

- `qualification/scene.usda` — the qualified scene passed to remediation.
- `qualification/table.png` and `qualification/side.png` — review images.
- `qualification/table_depth_m.npy` — TableCam depth used for localization.
- `qualification/qualification.json` — inventory, physical configuration hash,
  camera metadata, and physics/label checks.

## 3. Generate a synthetic dataset and train YOLO

This helper submits a physics-qualified dataset-generation job followed by a
dependent training job. The training job starts only if generation succeeds.

```bash
scripts/training/run_remediation_pipeline.sh "$RUN_DIR/yolo" 100
```

The first argument must be a fresh directory; `100` is the scene count. The
generator produces four classes: `good_cap`, `crooked_cap`, `missing_cap`, and
`table_cap` (a usable loose replacement cap).

Dataset generation builds and qualifies its own randomized scenes. It does not
alter the deterministic scene created in step 2.

When both jobs are complete, review before using the checkpoint:

```bash
cat "$RUN_DIR/yolo/dataset/manifest.json"
cat "$RUN_DIR/yolo/training/evaluation.json"
```

Required checks:

1. Dataset `manifest.json` reports `accepted: true` and has no failed samples.
2. `data.yaml` names exactly the four classes above.
3. Evaluation reports per-class results; synthetic metrics are not evidence of
   performance on physical TableCam images.
4. The checkpoint exists at `training/weights/best.pt`. For a checkpoint that
   has passed team review, copy it to `shared/inspection_models/` and record
   its source path and SHA-256 checksum in
   [SHARED_MODEL_REGISTRY.md](SHARED_MODEL_REGISTRY.md). Do not put models in
   the read-only `shared/data/` directory.

## 4. Run the compliant TableCam-to-action simulation

For the fracture-aware compliant simulation, use the canonical shared fixture.
The scene argument is optional because the Slurm script selects it by default:

```bash
sbatch scripts/run_compliant_breakable_pipeline.slurm \
  shared/yolo26x/best.pt \
  shared/runs/$(date +%F)_compliant_remediation
```

To run a deliberate alternate scene, supply it as the first argument before the
checkpoint. The canonical scene is
`shared/scenes/tablecam_right_pink_spares_qualified_v2.usda`; do not substitute
`shared/scenes/remediation_dataset1_v2_cad.usda` for the black-and-silver test.

## 5. Physical-workcell preparation

Before any physical motion, complete
[PHYSICAL_MEASUREMENT_CHECKLIST.md](PHYSICAL_MEASUREMENT_CHECKLIST.md). The
minimum evidence is measured TableCam accuracy, physical gripper reference
points for pickup/placement/press, installed finger dimensions, a repeatable
tray location, and a defined replacement-cap fixture.

Enter six physical TableCam reference correspondences in
`src/remediation/config/tablecam_tcp_capture_template.json`, then validate them:

```bash
python3 src/remediation/validate_tablecam_tcp.py \
  --captures path/to/filled_tablecam_tcp_capture.json \
  --out "$RUN_DIR/tablecam_tcp/calibration_report.json"
```

The checker expects native TableCam pixels and measured robot-base positions;
simulator truth and stored tray coordinates are not valid calibration evidence.

## 6. PhysicalAI-Hackathon-2026 handoff

The physical policy must use the same state machine as the simulator:

```text
observe TableCam → run YOLO → choose one safe action → execute a short chunk
→ observe again → confirm, retry within a limit, or stop
```

The supported Isaac pipeline now selects `--controller-policy shared` and imports
the same `src.remediation.controller.RemediationController` used by the physical
client. The Isaac adapter renders fresh TableCam observations,
runs the configured YOLO checkpoint at decision gates, and translates only the
returned TCP/W/P/R/gripper command into IK. The controller accepts only:

- TableCam RGB-D detections.
- Current TCP pose, robot joints and gripper state.
- Configured calibration and pick/place/press offsets.
- TableCam verification of pickup and final repair.

USD prims, PhysX poses, true cap axes, contact reports, generated labels and vial
loads are rejected at the policy boundary. In simulation they are written only to
the evaluator artifacts after commands have been selected.

The shared missing-cap sequence is pick, camera-verify, place, release,
position-limited press and camera verification. For a crooked cap, the shared policy
keeps the claw open and descends as for a whole-vial pickup: the conformal pockets
surround the cap and vial while the solid internal mount shoulder presses the cap.
The CAD-measured shoulder depth is 32.6838 mm from the fingertip TCP; the configured
press offset is added as upward clearance. The policy then retreats vertically,
still open. The tool axis remains vertical throughout. Missing caps are handled first.

The Isaac adapter expands each TCP destination into a simultaneous Cartesian and
shortest-path wrist transition limited to 1 mm and 0.5 degree per 60 Hz physics
step. If otherwise-reachable endpoints cross an unsolvable intermediate wrist
orientation, it uses a cosine-eased endpoint joint transition limited to 0.5
degree per joint per step; PhysX collision checking remains active throughout.
Any Cartesian IK branch change that would exceed the same 0.5-degree joint
command limit is rejected before execution and routed through that fallback.
Shared-policy execution uses closed-loop following error: the 0.5-degree
trajectory streams below 2 degrees of error, pauses to recover below 0.5 degree
when that bound is reached, and requires each action endpoint to settle below
0.5 degree. It aborts if tracking cannot recover or either moving jaw
drifts more than 50 mm from its initialized flange-relative distance. An abrupt
wrist command or detached EOAT therefore cannot be encoded as a valid run.

For every shared-policy simulation run, review:

- `action/trace.json`: permitted policy fields, states and TCP actions only.
- `action/policy_decisions.json`: environment-neutral observe/decide/execute loop.
- `action/policy_observations/`: fresh RGB-D captures and YOLO detections.
- `action/oracle_evaluation_trace.json`: segregated simulator poses/contacts/loads.
- `action/attempt_report.json`: `force_available_to_controller` must be `false`,
  `press_control` must be `position-limited`, and every decision must report an
  empty `oracle_inputs_used` list.

Force and fracture estimates remain useful evaluation evidence, but they do not
stop or alter shared-policy motion. The optional legacy comparison mode is not used
by `run_compliant_breakable_pipeline.slurm`.

After a run, enforce the contract mechanically:

```bash
.conda_env/bin/python src/remediation/verify_policy_parity_run.py \
  shared/runs/YYYY-MM-DD_task/action \
  --out shared/runs/YYYY-MM-DD_task/action/policy_parity_verification.json
```

The required implementation location is `PhysicalAI-Hackathon-2026/policy.py`.
It loads the approved checkpoint, requests TableCam colour/depth, uses the
real-cell camera calibration in `PhysicalAI-Hackathon-2026/calibration/`, and
returns short, fixed-down TCP action chunks. Its default `hover` mode moves
only to 75 mm above a detected loose cap. `full` mode runs the supervised
pick/place/press sequence described below.

Validate the client contract locally before requesting a physical run:

```bash
cd PhysicalAI-Hackathon-2026
uv sync
uv run run.py --mock --device cpu --trials 1 --trial-s 25
```

The physical policy defaults to `shared/yolo26x/best.pt`, confidence `0.70`,
inference size `1280`, and CUDA device `0`. The checkpoint's four classes are
`good_cap`, `crooked_cap`, `missing_cap`, and `table_cap`. To override any inference
setting for an experiment:

```bash
uv run run.py --checkpoint /absolute/path/to/other.pt \
  --imgsz 1280 --confidence 0.70 --device 0
```

The mock validates client/API action formatting and limits. It does not validate
the camera, localization, gripper, pickup, seating, or safety around the actual
tray.

### Request a physical-test slot

Use the reviewed files `PhysicalAI-Hackathon-2026/submission/submission.yaml`
and `PhysicalAI-Hackathon-2026/submission/placement.yaml`. The current request
describes one supervised rollout: one loose replacement cap on the table, one
missing tray cap, and one visibly crooked tray cap. Prepare that arrangement
with the operator before the rollout.

From `PhysicalAI-Hackathon-2026`, send the request:

```bash
uv run run.py --submit
```

The client prints a request ID and writes its private request secret to
`.request.json`. Do not commit, delete, or share that file. Book a 30-minute
slot with the request ID at <https://hackathon.sidclab.org/#calendar>, then
check whether the request has been approved:

```bash
uv run run.py --status
```

### Run the approved physical test from a compute node

Do not keep the live client on a management node. Before the scheduled slot,
open an interactive compute-node shell that lasts longer than the reservation:

```bash
srun --partition=teaching --gres=gpu:t4:1 --cpus-per-task=4 --mem=16G --time=00:45:00 --pty bash
cd ~/Documents/GitHub/innovation-lab-fall-2026/PhysicalAI-Hackathon-2026
nvidia-smi
```

The policy refuses to initialize with the default `--device 0` unless CUDA is
available, preventing an accidentally CPU-bound YOLO26x run.

### Recommended first physical runs

Start with the contact-free hover policy. It keeps the gripper open and stops
75 mm above the highest-confidence detected table cap:

```bash
uv run run.py --mode hover
```

Confirm several hover targets across the table before trying full mode. If detections
are sparse, test `--confidence 0.60`; if false positives appear, test `0.80`. Keep
`--imgsz 1280` for the YOLO26x model unless an explicit speed/accuracy comparison
shows that a smaller aperture is acceptable.

Full mode uses unmeasured TCP offsets and therefore requires an explicit acknowledgement.
For the first non-contact route check, use +10 mm offsets; positive offsets move the
TCP upward relative to the RGB-D point:

```bash
uv run run.py --mode full --allow-provisional-contact \
  --pickup-z-offset-m 0.003 --work-surface-z-offset-m 0.000 \
  --missing-release-z-offset-m 0.010 --cone-press-z-offset-m 0.010 \
  --crooked-action cone-press --cone-press-below-rim-m=-0.005
```

Test one operation at a time over a clear work area. Suggested progression is
`0.010`, `0.005`, `0.002`, then `0.001` m. Do not move directly from +10 mm to zero
over the vial. The controls have distinct effects:

- `--pickup-z-offset-m`: fingertip/end clearance above the local table plane
  during loose-cap pickup. Default `0.003` m from
  `PhysicalAI-Hackathon-2026/config/remediation_offsets.json`.
- `--work-surface-z-offset-m`: signed correction added to both the once-per-run
  TableCam depth fit and local wrist-stereo support fits. Default `0`.
- `--pick-offset-m` and `--cone-table-pick-offset-m`: retained positive-up
  compatibility overrides for older launch commands. The Isaac adapter's
  STEP-derived default remains `-0.006` m.
- `--table-cap-xy-offset-m X Y` and `--missing-cap-xy-offset-m X Y`: retained
  zero-default fallback corrections for operation without wrist refinement.
- `--crooked-cap-xy-offset-m X Y`: robot-base XY correction added only to crooked-cap
  contact targets, including the compensated cone nose path. Its default remains
  `0 0` until the side cone is physically checked.

The Isaac runner exposes the same flags but defaults all three XY corrections to
zero because these values compensate residual calibration in the physical TableCam.
- `--work-surface-z-m`: work-surface Z in robot-base metres (default `-0.0051`).
  The default vial-rim plane is derived from this as work surface + 10 mm tray
  base + 203.2 mm vial height.
- `--vial-rim-plane-offset-m`: residual added to the rim derived from the fitted
  surface, tray base and vial height.
- `--vial-rim-plane-override-m`: absolute rim Z; zero keeps automatic fitting.
- `--vial-height-override-m`: measured vial height; zero uses 203.2 mm.
- `--missing-release-z-offset-m`: release height above a missing-cap target.
- `--cone-press-z-offset-m`: upward residual applied to the cone endpoint.
- `--cone-press-below-rim-m`: side-cone endpoint relative to the calibrated rim; positive is lower and negative is clearance.
- `--cone-tip-from-tcp-m X Y Z`: measured cone-nose offset from UTOOL1, replacing the CAD default.
- `--crooked-shoulder-depth-m`: fingertip-TCP-to-internal-shoulder distance; default is the CAD-measured `0.0326838` m.
- `--crooked-tap-depth-m`: position-controlled travel beyond the RGB-D-measured first shoulder contact; default is `0.005` m.
- `--crooked-wiggle-radius-m`: open-shoulder XY orbit radius at each of three incremental press depths; default is `0.00075` m.
- `--crooked-wiggle-cycles`: square orbits per press depth; default is `1`, and `0` disables the wiggle.
- `--pickup-wiggle-radius-m`: optional open-claw square centering orbit at the loose-cap pickup height; default `0` disables it. Start experiments at `0.0005` m.
- `--pickup-wiggle-cycles`: pickup centering orbits before closing; default `0`, with `1` recommended for the first comparison.
- `--crooked-action shoulder|shoulder-close|closed-tip|regrip|cone-press`: choose the shared camera-only crooked-cap sequence; default is `shoulder`. `shoulder-close` descends open, closes 3 mm above predicted shoulder contact, presses incrementally, opens at the bottom, and retreats open. Deployment of the side-mounted TPU cone must explicitly select `cone-press`.

Before physical execution, jog the robot to the approved vertical home with the
wrist camera in its verified physical-cell starting roll. The cable-safe shared-policy pose is
W/P/R `(-180, 0, 0)` degrees. The physical policy refuses to start more than
20 degrees from this orientation rather than commanding an automatic wrist unwind.

Full remediation observes and replans only from the shared camera-clear inspection
pose at base `(0.240, 0.250, transit Z)` m (the v47 starting pose). It runs YOLO,
executes one complete repair, returns to inspection, verifies, and then selects the
next target. Missing-cap pickup retains one intermediate inspection-pose snapshot
to verify that the loose cap left the table before placement. YOLO is not rerun
between ordinary motion waypoints.

At the unobstructed start of a cone batch, the controller also retains every
qualified crooked-cap camera observation. If the leaned arm masks a farther cap
from TableCam after the first press, that unattempted camera-derived XYZ, surface
height, and cap axis remain eligible for its own wrist-camera refinement. This is
short-lived perception memory; it does not look up a tray or G-slot coordinate.

The physical cone handoff remains vertical for the clockwise wrist roll
`(-180, 0, 0) -> (-180, 0, -90)`, then leans through FANUC W. The October 30
supervised trial verified that the physical mount needs the opposite W branch
from the retained Isaac visual: the nominal physical lean is `(-135, 0, -90)`,
while simulation retains `(135, 0, -90)`. The physical policy keeps the verified
fixed `-135` branch; simulation may reduce its positive-W lean using the observed
base-frame XYZ and its IK reachability check. The clockwise roll occurs once and
each cone-tip press remains vertically downward. Do not express this lean as
`P=45`: that bends the real arm across the wrong axis and invalidates the
cone-tip position compensation.

Loose `table_cap` depth is optional because the real depth camera may not return
pixels from the cap material. In that case pickup uses the YOLO box-centre ray
intersected with the calibrated cap plane. Depth remains required for missing and
crooked defect contact.

The physical adapter subdivides every Cartesian goal into a single queued chunk
of 0.1-second rows no longer than 9 mm (and no more than 5 degrees of W/P/R
change). Queuing the complete chunk allows continuous cell-side playback instead
of stopping for a network round trip after every row. It retains the goal and
does not advance the policy state after the chunk until measured `tcp_pose` is
within 1.5 mm and 2 degrees, with the requested jaw state reached. `trace.jsonl`
records the planned goal, chunk row count, and any correction chunk.

Use a positive press offset for alignment-only trials. With the 32.6838 mm shoulder
depth, `+0.010` m commands 22.6838 mm of descent past the observed cap surface and
leaves the shoulder 10 mm above nominal contact. A zero or negative press offset
can create contact and must be tried only at low speed with the operator ready to stop
the arm. The controller rejects offsets outside +/-25 mm, targets without projected
RGB-D points, detections below the confidence threshold, and targets outside the
workspace. It permits one attempt per target and visually checks pickup and repair.

### Review local decision evidence

Every physical-policy process prints a local trace directory when it starts.
By default it is a new directory under `shared/runs/` named
`YYYY-MM-DD_HH-MM-SS_physical_policy`. To choose an explicit directory for a
scheduled run, pass it on the command line:

```bash
uv run run.py --mode full --allow-provisional-contact \
  --pick-offset-m 0.010 --place-offset-m 0.010 --press-offset-m 0.010 \
  --trace-dir "../shared/runs/$(date +%F)_physical_remediation"
```

The trace directory contains:

- `tablecam_raw/` — the TableCam RGB image used for each saved decision.
- `tablecam_annotated/` — the same image with YOLO boxes, classes, and confidence.
- `wristcam_raw/` — unmodified physical wrist-camera snapshots sent to the VLM.
- `wristcam_annotated/` — VLM box/centre, projected coarse target, and correction
  diagnostic for each physical wrist refinement.
- `depth/` — viewable TableCam depth heatmaps: near surfaces are warm, far surfaces are cool, and depth discontinuities have white outlines.
- `depth_raw/` — unchanged aligned uint16 TableCam depth in millimetres for calibration and numerical inspection.
- `trace.jsonl` — one JSON record per TCP command: timestamp, controller state,
  current TCP pose, command, detections, and the corresponding frame name.

The trace saves image/depth triplets only when the commanded target or
controller state changes. It is capped at 600 decision frames so an unattended
run cannot fill the shared filesystem. Every command remains in `trace.jsonl`
after the image cap is reached.

To encode the labelled frames into an H.264 video after the run, execute this
from the repository root, replacing `TRACE_DIR` with the printed directory:

```bash
TRACE_DIR="shared/runs/YYYY-MM-DD_HH-MM-SS_physical_policy"
FFMPEG_BIN=$(./.conda_env/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FFMPEG_BIN" -y -framerate 60 -i "$TRACE_DIR/tablecam_annotated/%04d.png" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$TRACE_DIR/tablecam_yolo_trace.mp4"
```

Compare this video with the hackathon portal's physical-cell recording and use
matching timestamps in `trace.jsonl` to identify the detection and command
that preceded any unexpected motion.

## 7. What to preserve outside Git

Because `runs/` is not tracked, retain or publish the following for each review:

- Slurm job identifiers and `logs/slurm-<job-id>.out/.err`.
- The exact command, Git commit hash, and output-directory path.
- Qualified-scene `qualification.json` and `scene.usda`.
- Dataset `manifest.json`, `data.yaml`, and training `evaluation.json`.
- The checkpoint used for the run, with its checksum.
- Initial TableCam capture, detections JSON, action report, and labelled video.
- Physical calibration captures and reports once available.

Never copy a generated run into `shared/data/`. Promote only reviewed, reusable
scene assets to `shared/scenes/` and record their provenance in a tracked
document.
