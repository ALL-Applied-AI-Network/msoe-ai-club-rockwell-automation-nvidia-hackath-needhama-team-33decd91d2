# Remediation runbook

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
  --steps 360 --fixed
```

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
scripts/run_remediation_pipeline.sh "$RUN_DIR/yolo" 100
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

## 4. Run the current full TableCam-to-action simulation sequence

The command below submits **one Slurm job** that performs these sequential
stages using the exact same saved scene:

1. Open the qualified scene and capture current TableCam colour and depth data.
2. Run YOLO on that capture.
3. Select reachable predicted targets and render the attempted action through
   TableCam.
4. Run YOLO on every rendered TableCam frame and encode a labelled video.

```bash
sbatch scripts/run_closed_loop_pipeline.slurm \
  "$RUN_DIR/qualification/scene.usda" \
  "$RUN_DIR/yolo/training/weights/best.pt" \
  "$RUN_DIR/closed_loop"
```

Isaac Sim and YOLO are separate child processes within this single job because
their Python runtimes conflict when loaded together. They share the same job,
scene, output directory, and GPU allocation.

Review these outputs:

- `closed_loop/perception/current_scene_tablecam.png` — the initial image used
  to select targets.
- `closed_loop/perception/current_scene_detections.json` — YOLO boxes, classes,
  confidence, and derived robot-base candidate positions.
- `closed_loop/action/attempt_report.json` — selected/rejected targets, tool
  orientation check, and explicit limitations.
- `closed_loop/action/yolo_frames/` — every TableCam action frame with YOLO
  labels drawn on it.
- `closed_loop/tablecam_yolo_per_frame.mp4` — review video.

Do not infer successful contact from this video. The controller must eventually
reinspect after every pickup, placement, and press before it can claim repair.

### Compliant black-and-silver fixture simulation

For the fracture-aware compliant simulation, use the canonical shared fixture.
The scene argument is optional because the Slurm script selects it by default:

```bash
sbatch scripts/run_compliant_breakable_pipeline.slurm \
  shared/inspection_models/remediation_yolo_best.pt \
  shared/runs/$(date +%F)_compliant_remediation
```

To run a deliberate alternate scene, supply it as the first argument before the
checkpoint. The canonical scene is
`shared/scenes/black_silver_remediation_qualified_v3.usda`; do not substitute
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
REMEDIATION_MODE=full uv run run.py --mock --trials 1 --trial-s 25
```

The current reviewed checkpoint is
`shared/inspection_models/remediation_yolo_best.pt`. The physical policy uses
it by default. To override it for an experiment, pass an explicit checkpoint:

```bash
uv run run.py --checkpoint /absolute/path/to/other.pt
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
srun --partition=teaching --cpus-per-task=4 --mem=8G --time=00:45:00 --pty bash
cd ~/Documents/GitHub/innovation-lab-fall-2026/PhysicalAI-Hackathon-2026
```

Run the approved supervised remediation policy and leave that terminal open for
the full slot:

```bash
REMEDIATION_MODE=full uv run run.py
```

The policy takes a fresh TableCam snapshot, chooses predicted defects, raises
to the transit height, moves horizontally, picks a loose cap, places it on a
missing vial, then presses a crooked cap. It re-observes after pickup and after
each repair. It keeps the fingers straight down. It currently assumes the
custom-tool TCP is aligned to the CAD contact point, so the pickup, placement,
and press offsets are provisional zero offsets. Keep the operator ready to
stop the arm during the first descent.

### Review local decision evidence

Every physical-policy process prints a local trace directory when it starts.
By default it is a new directory under `shared/runs/` named
`YYYY-MM-DD_HH-MM-SS_physical_policy`. To choose an explicit directory for a
scheduled run, set `REMEDIATION_TRACE_DIR` before starting the client:

```bash
export REMEDIATION_TRACE_DIR="$PWD/../shared/runs/$(date +%F)_physical_remediation"
REMEDIATION_MODE=full uv run run.py
```

The trace directory contains:

- `tablecam_raw/` — the TableCam RGB image used for each saved decision.
- `tablecam_annotated/` — the same image with YOLO boxes, classes, and confidence.
- `depth/` — aligned TableCam depth PNGs.
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
"$FFMPEG_BIN" -y -framerate 20 -i "$TRACE_DIR/tablecam_annotated/%04d.png" \
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
