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

### `./runs/` (EXPERIMENT ARTIFACTS & ARCHIVES)
All generated simulation runs, debug crops, perception evaluations, and model checkpoints must be structured cleanly:
- **Never dump loose debug files into the root of `./runs/`**.
- Intermediate debug images and crops must be grouped into timestamped subdirectories:
  - `runs/archive/YYYY-MM-DD_<task_description>/`
- Output videos and public demos should be placed in `runs/` or referenced directly from `docs/media/`.

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
- **Physical Specifications**: Consult [`docs/PHYSICAL_SPECIFICATIONS.md`](file:///home/ad.msoe.edu/needhama/Documents/GitHub/innovation-lab-fall-2026/docs/PHYSICAL_SPECIFICATIONS.md) for verified dimensions:
  - Vials: $1.00\text{ in}$ ID ($25.40\text{ mm}$), $1.05\text{ in}$ OD ($26.67\text{ mm}$), $8.00\text{ in}$ H ($203.20\text{ mm}$).
  - Caps: $1.24\text{ in}$ flange ($31.50\text{ mm}$), $0.90\text{ in}$ plug ($22.86\text{ mm}$).
  - Gripper: SMC MHZL2-16D2 $12\text{ mm}$ stroke ($6\text{ mm}$/jaw) with $34\text{ mm}$ conformal pads.
- **Vision-Guided Remediation**:
  - Never hardcode tray aperture coordinates. All robot trajectories must be guided by perception via overhead camera (`TableCam`) ray-plane backprojection.
  - Clearance height across the tray must be $> 0.203\text{ m}$ (minimum $0.26\text{ m}$) to clear all 8-inch vials without clipping or phasing.
