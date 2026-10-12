# Rockwell Automation + NVIDIA Innovation Lab Fall 2026

A four-week physical AI hackathon with Rockwell Automation, NVIDIA, MSOE, and UWM's Connected Systems Institute (CSI). Teams develop solutions for one of three challenges from a real factory testbed:
1. **Dataset 1: Final Vial Inspection** – Missing/crooked caps, damaged vials, empty fixtures.
2. **Dataset 2: Fluid Mixing Station 3** – Synchronized camera views to detect fill and color anomalies.
3. **Dataset 3: Grasping and Sorting** – Standalone FANUC arm sorting and digital twin.

Full event documentation and guide:
- **Event Article:** <https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026>
- **Local Documentation:** See [instructions.md](instructions.md)

---

## Setup & Data

The repository includes [`setup.sh`](setup.sh) which automatically handles data configuration depending on where it is run:

```bash
chmod +x setup.sh
./setup.sh
```

- **On the School Cluster (Rosie):**
  - Unpacks `data.zip` into the shared team directory `/data/ai_club/team_14_2026/data`.
  - Creates `/data/ai_club/team_14_2026/cache` for shared Singularity layer caching across all teammates.
  - Recursively assigns group ownership to `ai_club` with group read/write permissions (`chmod -R g+rwX`).
  - Creates a local `data` symlink in the repository pointing to the shared data directory.
- **On Brev / Standalone:**
  - Extracts `data.zip` directly into `./data/`.

Once extracted, the datasets and digital twins are accessible at `data/`:
- `data/csi_digital_twin_v1/`
- `data/csi_vial_fill_dataset/`
- `data/fanuc_er4ia_sim/`

---

## Running Isaac Sim (Brev & Cluster / Singularity)

A unified runner script [`scripts/run_isaac.sh`](scripts/run_isaac.sh) automatically adapts to your current compute platform:

### 1. On Brev Cloud GPU
Brev instances running the [Isaac Sim launchable](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-35JP2ywERLgqtD0b0MIeK1HnF46) have Isaac Sim pre-installed. The runner detects `/isaac-sim/python.sh` or `$ISAAC/python.sh` and runs natively:

```bash
# Render a sample frame from the CSI digital twin
./scripts/run_isaac.sh data/csi_digital_twin_v1/tools/render_example.py --camera FillCam --frame 686 --out fill.png

# Build or simulate the FANUC robot scene
./scripts/run_isaac.sh data/fanuc_er4ia_sim/build_scene.py --headless --save runs/scene.usd
```

To view images or videos generated on headless Brev, copy them to your local computer:
```bash
brev copy <instance-name>:/home/ubuntu/<repo-path>/fill.png ~/Downloads/
```

### 2. On School Cluster (Rosie / Slurm with Singularity)
On the cluster, the runner uses Singularity with GPU pass-through (`--nv`). Singularity cache and temporary layers are automatically routed to `/data/ai_club/team_14_2026/cache` so downloaded container layers are cached once and shared across all teammates.

#### Option A: Submit via Slurm Batch (`sbatch`)
Submit the pre-configured job template:
```bash
# Run default test render
sbatch scripts/submit_isaac_job.slurm

# Or pass custom render/sim arguments:
sbatch scripts/submit_isaac_job.slurm data/csi_digital_twin_v1/tools/render_example.py --camera FillCam --frame 686 --out fill_686.png
sbatch scripts/submit_isaac_job.slurm data/csi_digital_twin_v1/tools/render_example.py --camera CapSealCam --out cap_seal.png
```

Check job status and output:
```bash
squeue -u $USER
cat logs/slurm-<jobid>.out
```

#### Option B: Interactive GPU Session (`srun`)
Allocate an interactive GPU node to test commands directly:
```bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash

# Run directly inside your interactive shell
./scripts/run_isaac.sh data/csi_digital_twin_v1/tools/render_example.py --camera FillCam --frame 686 --out fill.png
```

#### Option C: Building the Shared `.sif` Container via Slurm (Recommended)
Building the container once into a `.sif` file saves ~5–10 minutes of layer extraction on subsequent runs and makes every future simulation start immediately.

Submit the build job with `sbatch`:
```bash
sbatch scripts/build_sif_job.slurm
```
This builds `/data/ai_club/team_14_2026/isaac-sim.sif` using the shared cache and automatically grants group execution permissions for all team members. Once completed, [`scripts/run_isaac.sh`](scripts/run_isaac.sh) automatically prioritizes this `.sif` file for all future runs.