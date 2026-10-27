# Rockwell Automation + NVIDIA Innovation Lab Fall 2026

A four-week physical AI hackathon with Rockwell Automation, NVIDIA, MSOE, and UWM's Connected Systems Institute (CSI).
- **Event Article:** <https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026>
- **Full Guide:** See [instructions.md](instructions.md)

---

## 1. Initial Setup (Always Run First)

All assets and dependencies are pre-configured. On any platform (Rosie, GB10, or Brev), simply run:

```bash
./setup.sh
```

`setup.sh` standardizes data access under `./shared/data/`:
- **On Rosie:** Symlinks `shared -> /data/ai_club/team_14_2026` so all teammates share storage, assets, and Isaac Sim installations.
- **On GB10 / Brev / Local:** Links or unpacks existing data assets.
- Bootstraps the FANUC robot arm simulation (generates procedural textures and prebuilds the persistent scene).

---

## 2. Running the Remediation Pipeline

The ground-truth path is the collision-physics, fracture-aware remediation simulation. Reference output: `compliant_remediation_test/`.

```bash
# One Slurm job: TableCam capture -> YOLO -> shared-policy PhysX remediation -> per-frame YOLO overlay -> mp4
sbatch scripts/run_compliant_breakable_pipeline.slurm \
    shared/yolo26x/best.pt \
    shared/runs/$(date +%F)_compliant_remediation

# Optional: supply a different scene as the first argument
sbatch scripts/run_compliant_breakable_pipeline.slurm <scene.usda> <checkpoint.pt> <out_dir>
```

Default scene: `shared/runs/2026-10-27_clear_polymer_textured_tray_glossy_caps_qualification/scene.usda`. It uses clear-plastic vials, textured satin aluminum, and semi-gloss plastic caps. Monitor with `squeue --me` and `tail -f logs/slurm-<jobid>.out`.

Outputs in `<out_dir>`:
- `perception/` - TableCam image, depth, `current_scene_detections.json`
- `action/trace.json` and `action/policy_decisions.json` - shared-policy inputs/actions
- `action/oracle_evaluation_trace.json` - segregated PhysX evaluation only
- `action/attempt_report.json` and `action/frames/` - simulation result and frames
- `action/yolo_frames/` and `tablecam_yolo_compliant_remediation.mp4` - annotated review video

Current crooked-cap validation: `shared/runs/2026-10-27_shared_policy_open_shoulder_press_v14/`. The shared controller keeps the claw open, descends the CAD-measured 32.6838 mm from the fingertip TCP so the internal shoulder presses the cap, then retreats open. The older [regrip video](docs/media/physics_remediation_regrip.mp4) is retained as a legacy comparison, not the active policy.

Any other Isaac script: `./run.sh <script.py> [args]` (submits via Slurm on the login node). See `AGENTS.md` for architecture and rules, and `docs/REMEDIATION_RUNBOOK.md` for scene qualification and dataset/training steps.

---

## 3. Platform Details

### A. Rosie (Slurm HPC Cluster)
Add the host alias to your SSH config (`notepad $env:USERPROFILE\.ssh\config`):
```sshconfig
Host ROSIE
    HostName dh-mgmt1.hpc.msoe.edu
    User <your-msoe-username>@ad.msoe.edu
    ForwardAgent yes
```
Connect anytime via `ssh ROSIE`. When running `./run.sh` on Rosie, jobs execute on compute nodes via Slurm. Monitor them with:
```bash
squeue --me                       # Check queue status
tail -f logs/slurm-<jobid>.out    # Stream job logs
```

### B. Connecting to GB10 (Grace Blackwell)
1. In Windows PowerShell:
   ```powershell
   type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh hackathon3@spark3.ai.msoe.edu "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
   ```
2. Add to your Windows SSH config (`notepad $env:USERPROFILE\.ssh\config`):
   ```sshconfig
   Host GB10
       HostName spark3.ai.msoe.edu
       User hackathon3
       IdentityFile ~/.ssh/id_ed25519
       ForwardAgent yes
   ```
3. Connect: `ssh GB10`

### C. Connecting to Brev Cloud GPU
1. In WSL terminal:
   ```bash
   bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"
   brev login
   ```
2. Copy key to Windows (PowerShell):
   ```powershell
   mkdir -Force $env:USERPROFILE\.brev
   wsl cp ~/.brev/brev.pem /mnt/c/Users/$env:USERNAME/.brev/brev.pem
   ```
3. Add to Windows SSH config (`code $env:USERPROFILE\.ssh\config`):
   ```sshconfig
   Host brev
       HostName <hostname-from-cat-brev-ssh_config>
       User ubuntu
       Port 22
       IdentityFile ~/.brev/brev.pem
       ForwardAgent yes
       StrictHostKeyChecking no
   ```
4. Connect: `ssh brev` (or in VS Code: `Remote-SSH: Connect to Host...` $\rightarrow$ `brev`).
5. Install host library on first run:
   ```bash
   sudo apt-get update && sudo apt-get install -y libxt6
   ```

### D. Transferring Renders to Your Laptop
Because `GB10` and `brev` are configured as SSH host aliases, you can use standard `scp` for all of them:

```bash
# Using SSH host alias (ROSIE, GB10, or brev):
scp -r brev:~/innovation-lab-fall-2026/runs/<run_folder> ~/Downloads/
scp -r GB10:~/innovation-lab-fall-2026/runs/<run_folder> ~/Downloads/
scp -r ROSIE:~/Documents/GitHub/innovation-lab-fall-2026/runs/<run_folder> ~/Downloads/

# Alternative for Brev via the Brev CLI:
brev copy <instance-name>:/home/ubuntu/innovation-lab-fall-2026/runs/<run_folder> ~/Downloads/
```

---

## Appendix: Manual Asset Upload (`data.zip`)

*(Backup reference only — data assets have already been seeded on the team instances).*

If bootstrapping a brand new remote instance from scratch:
```bash
# Upload data.zip from your laptop to Brev or GB10:
scp data.zip brev:~/innovation-lab-fall-2026/
scp data.zip GB10:~/innovation-lab-fall-2026/
```
Once uploaded, running `./setup.sh` unpacks the assets into `shared/data/`.
