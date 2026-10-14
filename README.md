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

## 2. Rendering Simulations (`./render.sh`)

`./render.sh` works identically everywhere:
- **On Rosie (Slurm login node):** Automatically submits a background GPU batch job via `sbatch`.
- **On GB10 / Brev / Local:** Automatically executes directly on the GPU.

Outputs are written to timestamped directories: `runs/<YYYY-MM-DD_HH-MM-SS>_<target>/` (`render.png` or `frame_*.png` and `meta.json`).

```bash
# CSI Digital Twin (Fill Station)
./render.sh --target vial --state filled                                                # Single frame (filled vial, frame 686)
./render.sh --target vial --state empty                                                 # Single frame (empty vial, frame 1)
./render.sh --target vial --state sequence --start-frame 100 --end-frame 500 --step 20  # Multi-frame sequence

# CSI Digital Twin (Cap & Seal Station)
./render.sh --target cap_seal

# FANUC ER-4iA Robotic Arm Cell
./render.sh --target fanuc --camera side     # Side camera view
./render.sh --target fanuc --camera table    # Table overview
./render.sh --target fanuc --camera wrist    # End-effector camera
```

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
Connect anytime via `ssh ROSIE`. When running `./render.sh` on Rosie, jobs execute on compute nodes via Slurm. Monitor them with:
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