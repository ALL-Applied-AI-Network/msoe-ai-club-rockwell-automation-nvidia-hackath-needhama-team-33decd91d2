# Agent Guidelines & Repository Architecture

This document defines critical rules, environment constraints, and directory conventions that all AI agents and developers must adhere to when working on this repository.

---

## 1. Directory Structure & Permissions Policy

### `shared/` (`/data/ai_club/team_14_2026/`)
The `shared/` directory is symlinked to the cluster team workspace (`/data/ai_club/team_14_2026/`).

- **`shared/data/` (STRICTLY READ-ONLY)**:
  - Contains the raw competition data extracted from `data.zip` (provided by the competition organizers).
  - **NEVER write, edit, rename, or delete any files inside `shared/data/`.**
  - All agents must treat `shared/data/` as immutable reference data.
- **`shared/scenes/` (TEAM COLLABORATION & SHARED ASSETS)**:
  - Canonical simulation USD scenes, verified physical layouts (such as `clean_vial_tray.usda`), and reusable digital twin stages belong here.
  - Accessible to all members of the `ai_club` group.
- **`shared/cache/`**:
  - Replicable cache files, Warp kernels, and intermediate artifacts shared across nodes.

### `shared/runs/` (EXPERIMENT ARTIFACTS & ARCHIVES)
All generated simulation runs, debug crops, perception evaluations, and model checkpoints must be structured cleanly and shared with the team:
- **Never dump loose debug files into the root of `shared/runs/`**.
- Intermediate debug images and crops must be grouped into timestamped subdirectories:
  - `shared/runs/YYYY-MM-DD_<task_description>/`
- Output videos and public demos should be placed in `shared/runs/` or referenced directly from `docs/media/`.

---

## 2. Cluster & Slurm Execution Rules

1. **GPU Simulation Execution**:
   - Heavy simulation rendering, Isaac Sim runs, and synthetic dataset generation **must** be submitted via Slurm using:
     ```bash
     sbatch scripts/submit_isaac_job.slurm <target_script> [arguments]
     ```
   - Do not launch heavy Isaac Sim rendering processes directly on the login node.
2. **Permissions Helper**:
   - Run `scripts/fix_shared_permissions.sh` to ensure `shared/scenes/` is writable by `ai_club` and `shared/data/` remains read-only.

---

## 3. Physical Specifications & Digital Twin Simulation

When simulating the vial tray, Fanuc ER-4iA arm, and gripper:
- **Physical Specifications**: Consult [`docs/PHYSICAL_SPECIFICATIONS.md`](docs/PHYSICAL_SPECIFICATIONS.md) for verified dimensions:
  - Vials: $1.00\text{ in}$ ID ($25.40\text{ mm}$), $1.05\text{ in}$ OD ($26.67\text{ mm}$), $8.00\text{ in}$ H ($203.20\text{ mm}$).
  - Caps: $1.24\text{ in}$ flange ($31.50\text{ mm}$), $0.90\text{ in}$ plug ($22.86\text{ mm}$).
  - Gripper: SMC MHZL2-16D2 $12\text{ mm}$ stroke ($6\text{ mm}$/jaw) with $34\text{ mm}$ conformal pads.
- **Vision-Guided Remediation**:
  - Never hardcode tray aperture coordinates. All robot trajectories must be guided by perception via overhead camera (`TableCam`) ray-plane backprojection.
  - Clearance height across the tray must be $> 0.203\text{ m}$ (minimum $0.26\text{ m}$) to clear all 8-inch vials without clipping or phasing.

---

## 4. Ground-Truth Pipeline (Collision Physics)

The only supported remediation path is the compliant, collision-aware, fracture-aware simulation. Its reference output is checked in at `compliant_remediation_test/`.

Entry point: `sbatch scripts/run_compliant_breakable_pipeline.slurm [scene.usda] <yolo_checkpoint.pt> <output_dir>`
(default scene: `shared/scenes/black_silver_remediation_qualified_v3.usda`). Stages:

1. `src/remediation/capture_live_tablecam.py` (Isaac): captures TableCam colour + depth -> `perception/`.
2. `src/inspection/inspect_remediation_capture.py` (YOLO, `.conda_env`): detects defects and backprojects them to robot-base positions via `src/inspection/vision_pipeline.py`.
3. `src/remediation/run_compliant_breakable_remediation.py` (Isaac): PhysX-collision pivot-press simulation with breakable vial monitoring -> `action/attempt_report.json` and `action/frames/`.
4. `src/remediation/overlay_yolo_frames.py` (YOLO): per-frame overlays, then ffmpeg encodes the video.

Supporting modules: `collision_manager.py` (PhysX colliders, CCD, groups, materials), `custom_gripper.py` (gripper mount and collision proxies; `data/gripper.step`/`data/gripper.usd`), `geometry.py` (+ `config/dataset1_layout.json`), `render_perception_shadow_video.py` (overlay colours).

Script layout (`scripts/`):
- Top level (entry points and launchers): `run_compliant_breakable_pipeline.slurm` (the pipeline), `run_isaac.sh` (Isaac python launcher), `submit_isaac_job.slurm` (generic Isaac Slurm wrapper), `submit_training_job.slurm` (generic conda-python Slurm wrapper), `setup_conda_env.slurm`, `fix_shared_permissions.sh`.
- `scripts/datasets/`: synthetic dataset generation, merging and finalizing.
- `scripts/training/`: YOLO fine-tuning, evaluation, prediction renders, and `run_remediation_pipeline.sh` (generate dataset -> dependent training job).
- `scripts/tools/`: utilities (`encode_frames.py`).

Rules:
- Do not add parallel/alternative runners (kinematic, "best effort", "lifelike") or one-off prompt/plan markdown files. Extend the pipeline above instead.
- Tests: `tests/` (run with `.conda_env/bin/python -m unittest tests.test_remediation_geometry (etc.)`).
- Never modify `shared/` contents unless the rules in section 1 allow it.

---

## 5. Retained Reference Material (not part of the runtime pipeline)

Kept deliberately; do not delete without asking:
- `cad_models/`, `data/gripper.step`, `data/gripper.usd`, and the CAD/scene tooling in `src/remediation/` (`convert_gripper_step.py`, `inspect_*`, `measure_custom_gripper_tcp.py`, `render_*_gripper*.py`, `validate_*`, `qualify_scene.py`, `build_*_scene.py`, `patch_scene_to_cad.py`, `labels.py`).
- YOLO dataset generation and training: `src/data_generation/`, `src/inspection/train_*`, `build_*`, `evaluate_tablecam_localization.py`, and the matching `scripts/*.slurm|py` (finetune, train_yolo*, merge, prepare, evaluate, generate/finalize deployment dataset).
- `runs/archive/` datasets, training logs and probes.

---

## 6. Team Stipulations (apply to every run)

- **Fixture and materials:** black-and-silver tray (10 mm black base, black standoffs, silver perforated plate), clear glass vials (opacity ~0.08, roughness 0.02, IOR 1.5) and semi-smooth (satin, roughness 0.45) caps. Defined in `src/remediation/scene.py`.
- **Dimensions:** `docs/PHYSICAL_SPECIFICATIONS.md` wins over the CAD models. `src/remediation/geometry.py` `SPEC` mirrors the document; `cad_models/` is used only where the document is silent (vial inner floor, appearance).
- **Outputs:** everything goes under `shared/runs/YYYY-MM-DD_<task>/`. `run_compliant_breakable_pipeline.slurm` forces its output directory under `shared/runs/`.
- **Scenes:** build with `src/remediation/qualify_scene.py --fixed --scene-out shared/runs/<run>/scene.usda`; the pipeline accepts that scene as its first argument.
