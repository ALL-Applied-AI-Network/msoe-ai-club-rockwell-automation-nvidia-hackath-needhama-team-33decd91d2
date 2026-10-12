# Rockwell Automation + NVIDIA Innovation Lab Fall 2026

A four-week physical AI hackathon with Rockwell Automation, NVIDIA, MSOE, and UWM's Connected Systems Institute (CSI).
- **Event Article:** <https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026>
- **Full Guide:** See [instructions.md](instructions.md)

---

## 1. Initial Setup (Always Run First)

`data.zip` is gitignored (too large for GitHub). If you just cloned the repository, ensure `data.zip` exists in the repo root first:
- **On Rosie (School Cluster):** If not already present in `/data/ai_club/team_14_2026/data/`, copy `data.zip` into the repository root.
- **On Brev:** Copy `data.zip` from your local machine using the Brev CLI:
  ```bash
  brev copy /path/to/data.zip <instance-name>:/home/ubuntu/<repo-folder>/
  ```

Once `data.zip` is in place, run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

`setup.sh` automatically:
- On Rosie, unpacks to `/data/ai_club/team_14_2026/data`, symlinks `team_14_2026 -> /data/ai_club/team_14_2026`, and sets `ai_club` group permissions for all teammates.
- On Brev, unpacks `data.zip` directly into `data/`.
- Bootstraps the FANUC robot simulation (creates Python 3.10 virtual environment, generates procedural textures, and runs verification tests).

---

## 2. Running Simulations on Rosie (School Cluster)

On Rosie, access data via `team_14_2026/data/` and run all simulations via Slurm:

```bash
# Example: Render the finished vial from the CSI digital twin
sbatch scripts/submit_isaac_job.slurm team_14_2026/data/csi_digital_twin_v1/tools/render_example.py --camera FillCam --frame 686 --out runs/fill.png

# Example: Build the FANUC robot arm simulation scene
sbatch scripts/submit_isaac_job.slurm team_14_2026/data/fanuc_er4ia_sim/build_scene.py --headless --save runs/fanuc_scene.usd
```

To monitor your jobs and view output logs:
```bash
squeue -u $USER
cat logs/slurm-<jobid>.out
```

*(Note: Downloaded Singularity layers and container images are automatically cached in `/data/ai_club/team_14_2026/cache` so teammates do not re-download them).*

---

## 3. Running Simulations on Brev (Cloud GPU)

On an [Isaac Sim Brev instance](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-35JP2ywERLgqtD0b0MIeK1HnF46), data is located directly in `data/`. Run jobs directly from your terminal:

```bash
# Example: Render the finished vial from the CSI digital twin
./scripts/run_isaac.sh data/csi_digital_twin_v1/tools/render_example.py --camera FillCam --frame 686 --out runs/fill.png

# Example: Build the FANUC robot arm simulation scene
./scripts/run_isaac.sh data/fanuc_er4ia_sim/build_scene.py --headless --save runs/fanuc_scene.usd
```

To download rendered images or videos from headless Brev to your laptop:
```bash
brev copy <instance-name>:/home/ubuntu/<repo-folder>/runs/fill.png ~/Downloads/
```