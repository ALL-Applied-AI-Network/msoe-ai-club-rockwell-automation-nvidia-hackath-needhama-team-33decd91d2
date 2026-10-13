# Rockwell Automation + NVIDIA Innovation Lab Fall 2026

A four-week physical AI hackathon with Rockwell Automation, NVIDIA, MSOE, and UWM's Connected Systems Institute (CSI).
- **Event Article:** <https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026>
- **Full Guide:** See [instructions.md](instructions.md)

---

## 1. Initial Setup (Always Run First)

`data.zip` contains the digital twin and asset models (gitignored due to size).

1. Ensure `data.zip` is in the repository root (or on Rosie, data is pre-populated in `/data/ai_club/team_14_2026/data/`).
2. Run the repository setup:

```bash
chmod +x setup.sh
./setup.sh
```

`setup.sh` standardizes data access under `./shared/data/` across all platforms:
- **On Rosie:** Symlinks `shared -> /data/ai_club/team_14_2026` so all teammates share storage, assets, and Isaac Sim installations.
- **On GB10 / Brev / Local:** Extracts `data.zip` into `./shared/data/`.
- Bootstraps the FANUC robot arm simulation (generates procedural textures and runs unit tests).

---

## 2. Running on Rosie (HPC Cluster)

On Rosie, jobs run on Tesla T4 compute nodes via Slurm using the team's shared standalone **Isaac Sim 6.1.0** install (`/data/ai_club/team_14_2026/isaacsim_6.1`).

### Submitting Batch Jobs
Submit batch jobs with `sbatch`. Because jobs run in the background on compute nodes, you can safely disconnect or close your session:

```bash
# Default test render (FillStation, FillCam, Frame 686 -> runs/fill_test.png)
sbatch scripts/submit_isaac_job.slurm

# Render a specific station camera
sbatch scripts/submit_isaac_job.slurm shared/data/csi_digital_twin_v1/tools/render_example.py \
    --camera CapSealCam \
    --stage stations/tray_inspection/tray_inspection.usdc \
    --out runs/cap_seal.png

# Build the FANUC robot arm simulation scene
sbatch scripts/submit_isaac_job.slurm shared/data/fanuc_er4ia_sim/build_scene.py --headless --save runs/fanuc_scene.usd
```

### Monitoring Jobs & Viewing Outputs
```bash
# Check queue status
squeue --me

# Stream job logs
tail -f logs/slurm-<jobid>.out
tail -f logs/slurm-<jobid>.err
```
Rendered outputs are written to the `runs/` directory.

---

## 3. Running on GB10 / Brev / Dedicated Cloud GPU

On dedicated GPU instances (e.g., NVIDIA GB10 / Grace Blackwell `aarch64` or Brev `x86_64`), you have direct interactive shell access without Slurm queues.

### A. One-Time Isaac Sim 6.1 Installation
Run the installer script (it automatically detects `x86_64` vs `aarch64` / ARM, downloads the standalone package, and installs into `./shared/isaacsim_6.1` by default):

```bash
# Automatically detects architecture (x86_64 vs aarch64) and installs to ./shared/isaacsim_6.1
./scripts/install_isaac_standalone.sh
```

### B. Running Headless Simulations Directly
Run simulations directly via `./scripts/run_isaac.sh` (it automatically discovers `./shared/isaacsim_6.1` or `$ISAAC`):

```bash
# Example: Render the FillStation camera
./scripts/run_isaac.sh shared/data/csi_digital_twin_v1/tools/render_example.py \
    --camera FillCam \
    --stage stations/fill_station/fill_station.usda \
    --frame 686 \
    --out runs/fill_test.png

# Example: Build the FANUC robot arm simulation scene
./scripts/run_isaac.sh shared/data/fanuc_er4ia_sim/build_scene.py --headless --save runs/fanuc_scene.usd
```

### C. Transferring Rendered Outputs to Your Laptop
- **From Brev:**
  ```bash
  brev copy <instance-name>:/home/ubuntu/innovation-lab-fall-2026/runs/fill_test.png ~/Downloads/
  ```
- **From GB10 (SCP/RSYNC):**
  ```bash
  scp user@<gb10-host>:~/innovation-lab-fall-2026/runs/fill_test.png ~/Downloads/
  ```